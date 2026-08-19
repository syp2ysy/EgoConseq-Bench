"""Small process-group lifecycle helpers for isolated OmniGibson workers."""

from __future__ import annotations

import os
import signal
import subprocess
import time


def _terminate_and_reap(process) -> None:
    """Terminate one child process group and wait until it is gone."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_isolated_process(
        process, *, timeout_s: int, heartbeat_interval_s: float = None,
        on_heartbeat=None) -> int:
    """Wait for a process-group leader and reap it on any supervisor exit."""
    try:
        timeout = float(timeout_s)
        if heartbeat_interval_s is None:
            return int(process.wait(timeout=timeout))
        interval = float(heartbeat_interval_s)
        if interval <= 0.0:
            raise ValueError("heartbeat interval must be positive")
        started = time.monotonic()
        remaining = timeout
        while True:
            try:
                return int(process.wait(timeout=min(interval, remaining)))
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise
                if on_heartbeat is not None:
                    on_heartbeat(float(elapsed))
                remaining = timeout - elapsed
    except BaseException:
        try:
            _terminate_and_reap(process)
        except BaseException:
            # Preserve the timeout, interrupt, or supervisor failure that made
            # cleanup necessary. The local single-user threat model does not
            # require replacing that evidence with a secondary cleanup error.
            pass
        raise
