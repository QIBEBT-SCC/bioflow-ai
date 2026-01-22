from typing import Optional

from loguru import logger
from pydantic import BaseModel


class Statistics(BaseModel):
    total: int
    waiting: Optional[int] = 0
    running: Optional[int] = 0
    success: Optional[int] = 0
    error: Optional[int] = 0

    def add_running(self):
        """将一个任务从等待状态转为运行状态"""
        if self.waiting > 0:
            self.waiting -= 1
            self.running += 1
        else:
            logger.warning(f'尝试增加 running 计数，但 waiting={self.waiting}')
            self.running += 1

    def add_success(self):
        """将一个任务从运行状态转为成功状态"""
        if self.running > 0:
            self.running -= 1
            self.success += 1
        else:
            logger.warning(f'尝试增加 success 计数，但 running={self.running}')
            self.success += 1

    def add_error(self):
        """将一个任务从运行状态转为错误状态"""
        if self.running > 0:
            self.running -= 1
            self.error += 1
        else:
            logger.warning(f'尝试增加 error 计数，但 running={self.running}')
            self.error += 1
