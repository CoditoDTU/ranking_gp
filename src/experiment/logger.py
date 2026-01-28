"""
Logger that tees stdout to both terminal and log file.
Extracted from module_1.py lines 37-48.
"""
import sys


class Logger:
    """Tee stdout to both terminal and a log file."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file."""
        self.log.close()
