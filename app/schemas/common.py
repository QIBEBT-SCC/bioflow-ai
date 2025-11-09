from typing import Optional

from pydantic import BaseModel


class Statistics(BaseModel):
    total: int
    waiting: Optional[int] = 0
    running: Optional[int] = 0
    success: Optional[int] = 0
    error: Optional[int] = 0

    def add_running(self):
        assert self.waiting > 0
        self.waiting -= 1
        self.running += 1

    def add_success(self):
        assert self.running > 0
        self.running -= 1
        self.success += 1

    def add_error(self):
        assert self.running > 0
        self.running -= 1
        self.error += 1
