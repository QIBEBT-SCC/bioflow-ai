import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.observability.execution_logging import agent_record_visible

if TYPE_CHECKING:
    from types import FrameType


class InterceptHandler(logging.Handler):
    """拦截标准库logging的日志处理器, 将其重定向到loguru."""

    def __init__(self, logger_type: str = 'other'):
        """初始化拦截处理器.

        Args:
            logger_type: 日志类型, 用于分类存储
        """
        super().__init__()
        self.logger_type = logger_type

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a standard logging record to its categorized Loguru sink."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        extra = {
            'logger_type': self.logger_type,
            'name': record.name,
            'module': record.module,
            'filename': record.filename,
            'funcName': record.funcName,
            'lineno': record.lineno,
        }

        bound_logger = logger.bind(**extra)
        bound_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


now = datetime.now(UTC)
year = now.year
month = now.month
day = now.day

# 获取配置
settings = get_settings()
log_config = settings.logger


# 基础日志配置
loguru_config = {
    'handlers': [
        # fastapi/其它库重定向 stdout
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold><light-blue>{extra[logger_type]: <7}</light-blue></bold><red>|</red> '
            '<cyan>{extra[name]}</cyan><red>:</red><cyan>{extra[funcName]}</cyan><red>:</red><cyan>{extra[lineno]}</cyan> <red>-</red> <level>{message}</level>',
            'filter': lambda record: 'logger_type' in record['extra'] and agent_record_visible(record),
            'colorize': True,
        },
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold>default</bold><red>|</red> '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> <red>-</red> <level>{message}</level>',
            'filter': lambda record: 'logger_type' not in record['extra'] and agent_record_visible(record),
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
            'filter': lambda record: record['extra'].get('logger_type', '') == 'fastapi',
        },
        # Celery日志
        {
            'sink': f'log/celery/celery_{year}{month:02d}{day:02d}.log',
            'level': log_config.CELERY_LEVEL,
            'rotation': log_config.LOG_ROTATION,
            'retention': log_config.LOG_RETENTION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | celery | {function}:{line} - {message}',
            'filter': lambda record: record['extra'].get('logger_type', '') == 'celery',
        },
        # S3/boto3日志
        {
            'sink': f'log/s3/s3_{year}{month:02d}{day:02d}.log',
            'level': log_config.LEVEL,
            'rotation': log_config.LOG_ROTATION,
            'retention': log_config.LOG_RETENTION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | s3 | {function}:{line} - {message}',
            'filter': lambda record: record['extra'].get('logger_type', '') == 's3',
        },
        # 错误日志 (所有类型的错误都会记录)
        {
            'sink': f'log/error/error_{year}{month:02d}{day:02d}.log',
            'level': log_config.ERROR_LEVEL,
            'retention': log_config.LOG_RETENTION,
            'rotation': log_config.LOG_ROTATION,
            'encoding': 'utf-8',
            'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file.name}:{line} - {message}\n{exception}',
        },
    ]
}


def init_logging() -> None:
    """初始化日志系统, 拦截所有库的日志并重定向到loguru."""
    # 设置默认的日志处理器, level 设为 DEBUG 让 loguru 自己过滤
    logging.getLogger().handlers = [InterceptHandler('other')]
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().propagate = False

    # FastAPI相关日志
    fastapi_level = getattr(logging, log_config.FASTAPI_LEVEL)
    fastapi_loggers = ['uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.asgi', 'fastapi']
    for logger_name in fastapi_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(logger_type='fastapi')]
        logging_logger.setLevel(fastapi_level)
        logging_logger.propagate = False

    # Celery相关日志
    celery_level = getattr(logging, log_config.CELERY_LEVEL)
    celery_loggers = ['celery', 'celery.app.trace', 'celery.worker', 'celery.worker.strategy']
    for logger_name in celery_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(logger_type='celery')]
        logging_logger.setLevel(celery_level)
        logging_logger.propagate = False

    # boto3/botocore/s3transfer相关日志
    s3_level = getattr(logging, log_config.BOTO3_LEVEL)
    for boto_logger_name in ('boto3', 'botocore', 's3transfer'):
        boto_logger = logging.getLogger(boto_logger_name)
        boto_logger.handlers = [InterceptHandler(logger_type='s3')]
        boto_logger.setLevel(s3_level)
        boto_logger.propagate = False

    # httpx/httpcore相关日志 (降至WARNING避免噪音)
    for http_logger_name in ('httpx', 'httpcore'):
        http_logger = logging.getLogger(http_logger_name)
        http_logger.handlers = [InterceptHandler(logger_type='httpx')]
        http_logger.setLevel(logging.WARNING)
        http_logger.propagate = False

    # 配置loguru
    logger.configure(**loguru_config)  # ty: ignore[invalid-argument-type]
