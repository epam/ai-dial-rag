from typing import Awaitable, Callable, Iterable, List, TypeVar

from more_itertools import chunked, flatten
from tqdm.std import tqdm as std_tqdm

from aidial_rag.content_stream import SupportsWriteStr


class TqdmProgressBar(std_tqdm):
    def __init__(self, iterable=None, total=None, file=None):
        super().__init__(
            iterable=iterable,
            total=total,
            file=file,
            bar_format="{l_bar}{r_bar}\n",  # Skip the {bar} part, and add an extra \n for markdown
            mininterval=10,  # We do not want too frequent updates
            maxinterval=30,  # But we want an update at least every 30 seconds, to keep connection alive
            smoothing=0.5,  # More to the current speed, because we can have time fluctuations waiting for the parallel tasks
            position=0,  # Override auto-position, because we every progress bar is written to a separate file
        )

    @staticmethod
    def status_printer(file):
        def print_status(s: str) -> None:
            # Overriding the print_status to avoid adding "\r"
            file.write(s)

        return print_status


T = TypeVar("T")
U = TypeVar("U")


async def batched_map_with_progress(
    iterable: Iterable[T],
    coro_func: Callable[[List[T]], Awaitable[Iterable[U]]],
    batch_size: int,
    file: SupportsWriteStr,
) -> Iterable[U]:
    chunked_iterable = list(chunked(iterable, batch_size))
    # Do not use asyncio.gather here, because we want batches to be processed one-by-one
    # to avoid running several heavy tasks in parallel, and fairly distribute CPU usage
    # between several heavy tasks from different users
    return flatten(
        [
            await coro_func(batch)
            for batch in TqdmProgressBar(
                iterable=chunked_iterable,
                file=file,
            )
        ]
    )
