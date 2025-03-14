import asyncio

from app.bioinfo.workflows import main_flow
from app.utils.logger import init_logging


if __name__ == '__main__':
    init_logging()
    asyncio.run(main_flow())
