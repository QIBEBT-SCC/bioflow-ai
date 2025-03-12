import os

from prefect import flow

from app.bioinfo.tasks.qc import fastp
from app.utils.logger import init_logging

@flow
def main_flow() -> None:
    fastp('test/raw/M3_1.fq.gz', 'test/raw/M3_2.fq.gz', 'test/clean')


if __name__ == '__main__':
    init_logging()
    main_flow()
