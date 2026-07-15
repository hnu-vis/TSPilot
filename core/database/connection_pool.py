"""Connection pool implementation for database connectors."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .connector import DBConnector, QueryResult


@dataclass
class PoolHealth:
    """Connection pool health status."""
    healthy: bool
    total_connections: int
    idle_connections: int
    active_connections: int
    waiting_requests: int
    avg_wait_time_ms: float
    errors: list[str] = field(default_factory=list)


class ConnectionPool:
    """Connection pool for database connectors.

    Manages a pool of connections for efficient resource usage.
    """

    def __init__(
        self,
        connector: DBConnector,
        min_size: int = 5,
        max_size: int = 20,
        max_idle_time: int = 300,
        acquire_timeout: int = 30,
    ):
        self._connector = connector
        self._min_size = min_size
        self._max_size = max_size
        self._max_idle_time = max_idle_time
        self._acquire_timeout = acquire_timeout

        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._active_count = 0
        self._total_count = 0
        self._waiting = 0
        self._errors: list[str] = []
        self._lock = asyncio.Lock()
        self._wait_times: list[float] = []
        self._closed = False

    async def initialize(self) -> None:
        """Initialize the connection pool with minimum connections."""
        for _ in range(self._min_size):
            conn = await self._create_connection()
            if conn:
                await self._pool.put(conn)

    async def _create_connection(self) -> Any:
        """Create a new database connection."""
        try:
            connector = self._connector.__class__(self._connector.config)
            await connector.connect()
            self._total_count += 1
            self._active_count += 1
            return connector
        except Exception as e:
            self._errors.append(f"Connection creation failed: {str(e)}")
            return None

    async def acquire(self) -> Any:
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        start_time = time.time()
        self._waiting += 1

        try:
            while True:
                # Try to get an available connection
                try:
                    conn = await asyncio.wait_for(
                        self._pool.get(),
                        timeout=self._acquire_timeout
                    )

                    wait_time = (time.time() - start_time) * 1000
                    self._wait_times.append(wait_time)
                    self._waiting -= 1

                    # Check if connection is still valid
                    if await self._is_connection_valid(conn):
                        self._active_count += 1
                        return conn
                    else:
                        # Connection expired, create new one
                        await self._close_connection(conn)
                        conn = await self._create_connection()
                        if conn:
                            self._active_count += 1
                            return conn

                except asyncio.TimeoutError:
                    self._waiting -= 1
                    raise RuntimeError(
                        f"Timeout waiting for connection after {self._acquire_timeout}s"
                    )

        except Exception as e:
            self._waiting -= 1
            self._errors.append(f"Acquire error: {str(e)}")
            raise

    async def _is_connection_valid(self, conn: Any) -> bool:
        """Check if a connection is still valid."""
        try:
            return await conn.health_check()
        except Exception:
            return False

    async def _close_connection(self, conn: Any) -> None:
        """Close a connection."""
        try:
            if conn:
                await conn.disconnect()
        except Exception:
            pass
        finally:
            self._total_count -= 1

    async def release(self, conn: Any) -> None:
        """Return a connection to the pool."""
        if self._closed:
            await self._close_connection(conn)
            return

        self._active_count -= 1

        if conn and await self._is_connection_valid(conn):
            try:
                await self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                await self._close_connection(conn)
        else:
            await self._close_connection(conn)

    async def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True

        # Close all connections in pool
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await self._close_connection(conn)
            except asyncio.QueueEmpty:
                break

        self._active_count = 0

    async def health_check(self) -> PoolHealth:
        """Check pool health status."""
        avg_wait = sum(self._wait_times[-100:]) / len(self._wait_times[-100:]) if self._wait_times else 0.0

        # Check if connector itself is healthy
        try:
            temp_conn = self._connector.__class__(self._connector.config)
            is_healthy = await temp_conn.health_check()
        except Exception:
            is_healthy = False

        return PoolHealth(
            healthy=is_healthy and len(self._errors) < 10,
            total_connections=self._total_count,
            idle_connections=self._pool.qsize(),
            active_connections=self._active_count,
            waiting_requests=self._waiting,
            avg_wait_time_ms=avg_wait,
            errors=self._errors[-10:],
        )

    @property
    def stats(self) -> dict:
        """Get pool statistics."""
        return {
            "total": self._total_count,
            "active": self._active_count,
            "idle": self._pool.qsize(),
            "waiting": self._waiting,
            "min_size": self._min_size,
            "max_size": self._max_size,
        }
