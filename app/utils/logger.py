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
            logger_type: 日志类型，用于分类存储，可选值: "default", "fastapi"
        """
        super().__init__()
        self.logger_type = logger_type

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        extra = {
            'logger_type': self.logger_type,
            'name': record.name,
            'module': record.module,
            'filename': record.filename,
            'funcName': record.funcName,
            'lineno': record.lineno,
        }

        if 'flow_runs' in record.name:
            extra['logger_type'] = 'prefect.work'
            extra['runDes'] = 'Flow run'
            extra['runName'] = record.flow_run_name
        elif 'task_runs' in record.name:
            extra['logger_type'] = 'prefect.work'
            extra['runDes'] = 'Task run'
            extra['runName'] = record.task_run_name

        bound_logger = logger.bind(**extra)

        # print(record.__dict__.items())
        bound_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class IgnoreHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record: logging.LogRecord):
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        bound_logger = logger.bind(logger_type='httpx')
        bound_logger.opt(depth=depth, exception=record.exc_info).debug(record.getMessage())


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
        # 其它普通重定向
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold>{extra[logger_type]: <7}</bold><red>|</red> '
            '<cyan>{extra[name]}</cyan><red>:</red><cyan>{extra[funcName]}</cyan><red>:</red><cyan>{extra[lineno]}</cyan> <red>-</red> <level>{message}</level>',
            'filter': lambda record: 'logger_type' in record['extra']
            and record['extra']['logger_type'] != 'prefect.work',
            'colorize': True,
        },
        # prefect 重定向
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold><yellow>prefect</yellow></bold><red>|</red> '
            "{extra[runDes]} '<magenta>{extra[runName]}</magenta>' "
            '<red>-</red> <level>{message}</level>',
            'filter': lambda record: 'logger_type' in record['extra']
            and record['extra']['logger_type'] == 'prefect.work',
            'colorize': True,
        },
        {
            'sink': sys.stdout,
            'level': log_config.LEVEL,
            'format': '<green>{time:HH:mm:ss}</green> <red>|</red> '
            '<level>{level: <8}</level> <red>|</red>'
            '<bold>default</bold><red>|</red> '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> <red>-</red> <level>{message}</level>',
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
            'filter': lambda record: record['extra'].get('logger_type', '') == 'fastapi',
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
    logging.getLogger().handlers = [InterceptHandler('other')]
    # logging.getLogger().propagate = False

    # FastAPI相关日志
    fastapi_loggers = ['uvicorn.asgi', 'uvicorn.access', 'uvicorn', 'fastapi']
    for logger_name in fastapi_loggers:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(logger_type='fastapi')]
        logging_logger.propagate = False

    # prefect日志重定向
    prefect_logger = logging.getLogger('prefect')
    prefect_logger.handlers = [InterceptHandler(logger_type='prefect')]
    prefect_logger.propagate = False

    # httpx日志重定向
    httpx_logger = logging.getLogger('httpx')
    # httpx_logger.handlers = [IgnoreHandler()]
    httpx_logger.handlers = [InterceptHandler(logger_type='httpx')]
    httpx_logger.propagate = False

    # 配置loguru
    logger.configure(**loguru_config)
