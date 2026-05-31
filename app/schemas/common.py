from loguru import logger
from pydantic import BaseModel


class Statistics(BaseModel):
    total: int
    waiting: int = 0
    running: int = 0
    success: int = 0
    error: int = 0

    def add_running(self) -> None:
        """Increase the running count."""
        if self.waiting > 0:
            self.waiting -= 1
            self.running += 1
        else:
            logger.warning('Attempt to increase the running count, but there is currently no task in a waiting state.')
            self.running += 1

    def add_success(self) -> None:
        """Increase the success count."""
        if self.running > 0:
            self.running -= 1
            self.success += 1
        else:
            logger.warning('Attempt to increase the success count, but there is currently no task in a running state.')
            self.success += 1

    def add_error(self) -> None:
        """Increase the error count."""
        if self.running > 0:
            self.running -= 1
            self.error += 1
        else:
            logger.warning('Attempt to increase the error count, but there is currently no task in a running state.')
            self.error += 1
