import logging
import sys
from types import FrameType
from typing import cast

from loguru import logger
from datetime import datetime

from app.utils.config import get_settings


class InterceptHandler(logging.Handler):
    """
    拦截标准库logging的日志处理器，将其重定向到loguru
    """

    def __init__(self, logger_type: str = 'default'):
        """
        初始化拦截处理器

        Args:
            logger_type: 日志类型，用于分类存储，可选值: "default", "fastapi", "toil", "user"
        """
        super().__init__()
        self.logger_type = logger_type

    def emit(self, record: logging.LogRecord) -> None:
        # 获取对应的 Loguru 日志级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # 查找日志消息的调用者
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        # 提取额外的上下文信息
        extra = {
            'logger_type': self.logger_type,
            'logger_name': record.name,
            'pathname': record.pathname,
            'funcName': record.funcName,
            'created': record.created,
            'thread': record.thread,
            'threadName': record.threadName,
            'process': record.process,
            'processName': record.processName,
        }

        # 使用 bind 方法添加上下文信息
        bound_logger = logger.bind(**extra)

        # 特殊处理认证相关日志
        if '/api/v1/auth' in record.getMessage():
            bound_logger.opt(depth=depth, exception=record.exc_info).log(
                'DEBUG', record.getMessage()
            )
        else:
            bound_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


now = datetime.now()
year = now.year
month = now.month
day = now.day

# 获取配置
settings = get_settings()
log_config = settings.logger


# 基础日志配置
loguru_config = {
    'handlers': [
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold>{extra[logger_type]: <7}</bold><red>|</red> '
            '<cyan>{module}</cyan><red>:</red><cyan>{file.name}</cyan><red>:</red><cyan>{line}</cyan> <red>-</red> <level>{message}</level>',
            'filter': lambda record: 'logger_type' in record['extra'],
            'colorize': True,
        },
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold>default</bold><red>|</red> '
            '<cyan>{module}</cyan><red>:</red><cyan>{file.name}</cyan><red>:</red><cyan>{line}</cyan> <red>-</red> <level>{message}</level>',
            'filter': lambda record: 'logger_type' not in record['extra'],
            'colorize': True,
        },
        # 默认运行时日志
        {
            'sink': f'log/runtime_{year}{month:02d}{day:02d}.log',
            'level': log_config.LEVEL,
            'rotation': log_config.LOG_ROTATION,
            'retention': log_config.LOG_RETENTION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} |{level: <8}| {file.name: <15} | {function}:{line} - {message}',
        },
        # FastAPI日志
        {
            'sink': f'log/fastapi/fastapi_{year}{month:02d}{day:02d}.log',
            'level': log_config.FASTAPI_LEVEL,
            'rotation': log_config.LOG_ROTATION,
            'retention': log_config.LOG_RETENTION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | fastapi | {function}:{line} - {message}',
            'filter': lambda record: record['extra'].get('logger_type') == 'fastapi',
        },
        # Toil日志
        {
            'sink': f'log/toil/toil_{year}{month:02d}{day:02d}.log',
            'level': log_config.TOIL_LEVEL,
            'rotation': log_config.LOG_ROTATION,
            'retention': log_config.LOG_RETENTION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} |{level: <8}| {module}:{file.name}:{line} - {message}',
            'filter': lambda record: record['extra'].get('logger_type') == 'toil',
        },
        # 错误日志（所有类型的错误都会记录）
        {
            'sink': f'log/error/error_{year}{month:02d}{day:02d}.log',
            'level': log_config.ERROR_LEVEL,
            'retention': log_config.LOG_RETENTION,
            'rotation': log_config.LOG_ROTATION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[logger_type]: <8} | {file.name}:{line} - {message}\n{exception}',
        },
    ]
}


def init_logging():
    """初始化日志系统，拦截所有库的日志并重定向到loguru"""

    # 设置默认的日志处理器
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger().setLevel(logging.INFO)

    # FastAPI相关日志
    fastapi_loggers = ['uvicorn.asgi', 'uvicorn.access', 'uvicorn', 'fastapi']
    for logger_name in fastapi_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(logger_type='fastapi')]
        logging_logger.propagate = False

    # Toil相关日志
    toil_loggers = ['toil']
    for logger_name in toil_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(logger_type='toil')]
        logging_logger.propagate = False

    # 配置loguru
    logger.configure(**loguru_config)
