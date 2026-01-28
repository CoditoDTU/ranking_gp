"""
Logger that tees stdout to both terminal and log file.
Extracted from module_1.py lines 37-48.
"""
import sys
import warnings


class Logger:
    """Redirect stdout to a log file, optionally also to terminal."""

    def __init__(self, filename: str, quiet: bool = False):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.quiet = quiet

    def write(self, message: str):
        if not self.quiet:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        if not self.quiet:
            self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file."""
        self.log.close()


def setup_logging(log_path: str, quiet: bool = False):
    """
    Setup logging to file, optionally suppressing terminal output.

    Also redirects warnings to the log file.
    """
    logger = Logger(log_path, quiet=quiet)
    sys.stdout = logger
    sys.stderr = logger

    # Redirect warnings to the log
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warning_handler

    return logger
