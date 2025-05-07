import asyncio
from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    TypeVar,
)

from aidial_rag.batched import TqdmProgressBar
from aidial_rag.content_stream import SupportsWriteStr
from aidial_rag.dial_user_limits import UserLimitsForModel
from aidial_rag.errors import NotEnoughDailyTokensError
from aidial_rag.resource_counter import ResourceCounter
from aidial_rag.utils import timeout


class DialLimitedResources:
    def __init__(
        self,
        get_user_limits_coro: Callable[[str], Awaitable[UserLimitsForModel]],
    ):
        self._get_user_limits_coro = get_user_limits_coro
        self._counters = {}
        self._user_limits = {}
        self._reserved_tokens = defaultdict(int)
        self._locks = {}

    async def _get_user_limits(self, model_name: str) -> UserLimitsForModel:
        # We do not want to expose the number of used tokens, because we want users
        # to call reserve_daily_tokens() to account for the tokens they are going to use
        # instead of checking the limits separately.
        if model_name not in self._user_limits:
            self._user_limits[model_name] = await self._get_user_limits_coro(
                model_name
            )
        return self._user_limits[model_name]

    async def get_minute_token_limit(self, model_name: str) -> int:
        user_limits = await self._get_user_limits(model_name)
        return user_limits.minute_token_stats.total

    async def get_day_token_limit(self, model_name: str) -> int:
        user_limits = await self._get_user_limits(model_name)
        return user_limits.day_token_stats.total

    async def reserve_daily_tokens(self, model_name, expected_tokens):
        user_limits = await self._get_user_limits(model_name)
        reserved_tokens = self._reserved_tokens[model_name]

        available_tokens = (
            user_limits.day_token_stats.total - user_limits.day_token_stats.used
        )
        if expected_tokens + reserved_tokens > available_tokens:
            raise NotEnoughDailyTokensError(
                model_name=model_name,
                expected=expected_tokens,
                reserved=reserved_tokens,
                used=user_limits.day_token_stats.used,
                total=user_limits.day_token_stats.total,
            )

        self._reserved_tokens[model_name] += expected_tokens

    async def get_counter(self, model_name: str) -> ResourceCounter:
        if model_name not in self._counters:
            minute_limits = await self.get_minute_token_limit(model_name)
            self._counters[model_name] = ResourceCounter(minute_limits)
        return self._counters[model_name]

    def get_lock(self, model_name: str) -> asyncio.Lock:
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
        return self._locks[model_name]


T = TypeVar("T")
U = TypeVar("U")


class AsyncGeneratorWithTotal(Generic[T]):
    agen: AsyncGenerator[T, None]
    total: int

    def __init__(self, agen: AsyncGenerator[T, None], total: int):
        self.agen = agen
        self.total = total


async def map_with_resource_limits(
    dial_limited_resources: DialLimitedResources,
    items: AsyncGeneratorWithTotal[T],
    coro_func: Callable[[T], Coroutine[None, None, U]],
    estimated_task_tokens: int,
    model_name: str,
    file: SupportsWriteStr,
    time_limit_multiplier: float = 1.5,
    min_time_limit_sec: float = timedelta(minutes=5).total_seconds(),
) -> Sequence[U]:
    total_tokens = estimated_task_tokens * items.total
    await dial_limited_resources.reserve_daily_tokens(model_name, total_tokens)

    minute_token_limit = await dial_limited_resources.get_minute_token_limit(
        model_name
    )
    estimated_time = timedelta(minutes=float(total_tokens) / minute_token_limit)
    if estimated_time > timedelta(minutes=1):
        # The processing will not fit into a minute token limit and be rate limited
        print(
            f"Estimated processing time is {estimated_time}"
            f" due to the limit for the {model_name}"
            f" is {minute_token_limit} tokens per minute.\n\n",
            file=file,
        )

    time_limit_sec = max(
        estimated_time.total_seconds() * time_limit_multiplier,
        min_time_limit_sec,
    )
    resource_counter = await dial_limited_resources.get_counter(model_name)
    tasks = []

    # Need to wrap timeout with a lock to run the map operations one by one
    # because the timeout was calculated for this map operation
    # If the other map operations use the same resource, the time limit might be exceeded
    async with dial_limited_resources.get_lock(model_name):
        async with timeout(time_limit_sec):
            with TqdmProgressBar(total=items.total, file=file) as pbar:
                async with asyncio.TaskGroup() as task_group:
                    async for item in items.agen:
                        task = await resource_counter.acquire_and_create_task(
                            coro_func(item), estimated_task_tokens, task_group
                        )
                        task.add_done_callback(lambda _: pbar.update())
                        tasks.append(task)

    return [task.result() for task in tasks]
