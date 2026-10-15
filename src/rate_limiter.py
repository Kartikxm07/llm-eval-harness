"""Thread-safe rate limiter and request pacer to prevent API throttling."""

import threading
import time
from typing import Optional


class RateLimiter:
    """Enforces minimum time intervals between API calls to stay within provider limits."""

    def __init__(self, min_interval: float = 1.0, max_requests_per_minute: Optional[int] = None):
        """Initialize the rate limiter.

        Args:
            min_interval: Minimum time (seconds) to wait between calls.
            max_requests_per_minute: Optional ceiling on requests in any rolling 60s window.
        """
        self.min_interval = max(0.0, min_interval)
        self.max_rpm = max_requests_per_minute
        self.last_call_timestamp: float = 0.0
        self.call_timestamps = []
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Wait if necessary before making the next API call.

        Returns:
            The number of seconds waited.
        """
        with self._lock:
            now = time.time()
            waited = 0.0

            # 1. Enforce min_interval spacing
            elapsed = now - self.last_call_timestamp
            if elapsed < self.min_interval:
                sleep_needed = self.min_interval - elapsed
                time.sleep(sleep_needed)
                waited += sleep_needed
                now = time.time()

            # 2. Enforce rolling RPM window if configured
            if self.max_rpm:
                window_start = now - 60.0
                self.call_timestamps = [ts for ts in self.call_timestamps if ts > window_start]
                if len(self.call_timestamps) >= self.max_rpm:
                    oldest = self.call_timestamps[0]
                    sleep_needed = 60.0 - (now - oldest)
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)
                        waited += sleep_needed
                        now = time.time()

            self.last_call_timestamp = now
            self.call_timestamps.append(now)
            return waited

    def reset(self):
        """Reset internal rate tracking state."""
        with self._lock:
            self.last_call_timestamp = 0.0
            self.call_timestamps.clear()
