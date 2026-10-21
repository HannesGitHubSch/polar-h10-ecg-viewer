"""
Thread-safe runner bridging PyQt6 GUI events with an asyncio background loop.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class RunnerSignals(QObject):
    """Signals routed across threads into the Qt event loop."""

    error = pyqtSignal(str, str)  # (operation, error_message)
    callback_ready = pyqtSignal(object, object)  # (callable, result)


class AsyncRunner(QObject):
    """
    Manages a dedicated worker thread with an asyncio event loop, safely
    dispatching coroutines and routing completions and exceptions back to Qt.
    """

    def __init__(self):
        super().__init__()
        self.signals = RunnerSignals()
        self.signals.callback_ready.connect(self._dispatch_callback)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("AsyncRunner event loop is not running")
        return self._loop

    def start(self) -> None:
        """Start worker thread and initialize event loop."""
        if self._thread is not None and self._thread.is_alive():
            return

        def _worker():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready_event.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_worker, name="polar_ecg_asyncio", daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=3.0):
            raise RuntimeError("AsyncRunner failed to start asyncio event loop within 3.0s")

    def submit(
        self,
        coro: Any,
        op_name: str,
        on_done: Callable[[Any], None] | None = None,
    ) -> Future:
        """
        Submit a coroutine to the background loop.
        Exceptions are emitted via `signals.error(op_name, message)`.
        Successful return values are safely dispatched to `on_done` on the Qt main thread.
        """
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)

        def _on_future_complete(f: Future):
            try:
                res = f.result()
                if on_done:
                    self.signals.callback_ready.emit(on_done, res)
            except asyncio.CancelledError:
                logger.debug("Async operation '%s' cancelled", op_name)
            except Exception as exc:
                logger.exception("Async operation '%s' failed", op_name)
                self.signals.error.emit(op_name, str(exc))

        fut.add_done_callback(_on_future_complete)
        return fut

    def _dispatch_callback(self, cb: Callable[[Any], None], res: Any) -> None:
        """Executed on Qt main thread via signal dispatch."""
        try:
            cb(res)
        except Exception:
            logger.exception("Error in AsyncRunner callback")

    def shutdown(self, session: Any = None, timeout: float = 5.0) -> None:
        """
        Gracefully shut down active session, stop event loop, and join thread.
        """
        if self._loop and self._loop.is_running():
            if session is not None:
                try:
                    fut = asyncio.run_coroutine_threadsafe(session.shutdown(), self._loop)
                    fut.result(timeout=timeout)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Error during session shutdown: %s", e)

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None
            self._loop = None
