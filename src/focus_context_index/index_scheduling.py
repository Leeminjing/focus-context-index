# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
"""索引调用的共享名额：前台优先，后台最多一个；取消等待不占名额。"""

import asyncio
from contextlib import asynccontextmanager


class IndexModelSlots:
    def __init__(self, capacity):
        self._capacity = capacity
        self._condition = asyncio.Condition()
        self._busy = 0
        self._background = 0
        self._foreground_waiters = 0

    def would_wait(self, *, background=False):
        return self._busy >= self._capacity or (
            background and (self._background or self._foreground_waiters)
        )

    @asynccontextmanager
    async def slot(self, *, background=False):
        async with self._condition:
            if not background:
                self._foreground_waiters += 1
            try:
                await self._condition.wait_for(
                    lambda: (
                        self._busy < self._capacity
                        and (
                            not background
                            or (not self._background and not self._foreground_waiters)
                        )
                    )
                )
                self._busy += 1
                self._background += int(background)
            finally:
                if not background:
                    self._foreground_waiters -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._busy -= 1
                self._background -= int(background)
                self._condition.notify_all()


class SegmentLocks:
    """同服务局部缓存填充互斥；最后一个调用者离开时释放锁条目。"""

    def __init__(self):
        self._entries = {}

    @asynccontextmanager
    async def hold(self, key):
        entry = self._entries.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if not entry[1]:
                del self._entries[key]
