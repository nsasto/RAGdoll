"""Small internal async rate limiter used by downstream adapters."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable


class AsyncRateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.clock = clock
        self.sleep = sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self.clock()
            while now < self._next:
                await self.sleep(self._next - now)
                now = self.clock()
            self._next = max(now, self._next) + self.interval
