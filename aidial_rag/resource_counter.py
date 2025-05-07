import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Coroutine, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class ResourceCounter:
    """Based on the asyncio.Semaphore, but with the ability to acquire several resources at once"""

    @dataclass
    class _WaitItem:
        future: asyncio.Future
        count: int

    def __init__(self, total: int):
        if total <= 0:
            raise ValueError("ResourceCounter total value must be > 0")
        self._loop = asyncio.get_event_loop()
        self._waiters = deque()
        self._total = total
        self._value = total

    def locked(self, count: int = 1):
        # If anyone is waiting, we should try to give them the resources first
        return self._value < count or any(
            not w.future.cancelled() for w in self._waiters
        )

    async def acquire(self, count: int):
        if count > self._total:
            raise ValueError("Requested count is more than total")

        if not self.locked(count) and all(
            w.future.cancelled() for w in self._waiters
        ):
            self._value -= count
            return

        wait_item = ResourceCounter._WaitItem(self._loop.create_future(), count)
        self._waiters.append(wait_item)
        try:
            try:
                await wait_item.future
            finally:
                self._waiters.remove(wait_item)
        except asyncio.CancelledError:
            # We may get CancelledError after the value was decreased in _wake_up_next()
            if not wait_item.future.cancelled():
                self._value += count
            raise

    def release(self, count: int):
        if self._value >= self._total:
            raise ValueError("ResourceCounter released more than total")
        self._value += count
        self._wake_up_next()

    def _wake_up_next(self):
        if not self._waiters:
            return

        for wait_item in self._waiters:
            # We may have done future, because it is removed from the waiters list in acquire()
            if wait_item.future.done():
                continue

            # Trying to wake up the several waiters, until we meet the first one that does not fit
            if wait_item.count > self._value:
                break

            self._value -= wait_item.count
            wait_item.future.set_result(None)

    @asynccontextmanager
    async def acquire_context(self, count: int):
        await self.acquire(count)
        try:
            yield
        finally:
            self.release(count)

    async def acquire_and_create_task(
        self,
        coro: Coroutine,
        count: int,
        task_group: asyncio.TaskGroup | None = None,
    ):
        create_task = (
            task_group.create_task if task_group else asyncio.create_task
        )
        await self.acquire(count)
        try:
            task = create_task(coro)
        except Exception:
            # If the task creation failed, we should release the resources immediately
            self.release(count)

        # If the task is created successfully, if should hold the resources until it is done
        task.add_done_callback(lambda _: self.release(count))
        return task
