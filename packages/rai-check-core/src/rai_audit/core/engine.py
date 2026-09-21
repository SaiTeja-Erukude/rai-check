from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from rai_audit.core.findings import AuditReport


class BaseAudit(ABC):
    """Abstract base class for all audit modules."""

    @abstractmethod
    def run(self) -> AuditReport:
        """Execute the audit synchronously and return a report."""
        ...

    async def run_async(self) -> AuditReport:
        """Execute the audit asynchronously (default: runs sync in a thread pool)."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.run)
