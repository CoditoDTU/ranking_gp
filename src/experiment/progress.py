"""
Progress tracking and timing utilities for experiments.
"""
import time
from tqdm import tqdm


class ProgressTracker:
    """Tracks experiment progress with a progress bar and timing."""

    def __init__(self, total: int, quiet: bool = False, desc: str = "Training"):
        """
        Initialize the progress tracker.

        Args:
            total: Total number of items to process.
            quiet: If True, disable progress bar output.
            desc: Description for the progress bar.
        """
        self.total = total
        self.quiet = quiet
        self.start_time = None
        self.end_time = None
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit="model",
            disable=quiet,
        )

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        if not self.quiet:
            tqdm.write(f"\nStarting {self.total} training runs...")

    def set_status(self, description: str):
        """Set the current status text without incrementing progress."""
        self.pbar.set_postfix_str(description)

    def update(self, n: int = 1):
        """Increment progress by n steps (default 1)."""
        self.pbar.update(n)

    def write(self, message: str):
        """Write a message without breaking the progress bar."""
        tqdm.write(message)

    def finish(self) -> float:
        """
        Finish progress tracking and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        self.pbar.close()
        self.end_time = time.time()
        return self.elapsed_time

    @property
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def elapsed_str(self) -> str:
        """Return elapsed time as a formatted string."""
        return format_duration(self.elapsed_time)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s" or "45.2s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"
