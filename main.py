from prefect import flow

from app.bioinfo.tasks.qc import fastp
from app.utils.logger import init_logging


@flow(log_prints=True)
def main_flow() -> None:
    clean_result = fastp('test/raw/M3_1.fq.gz', 'test/raw/M3_2.fq.gz', output_dir='test/clean')


if __name__ == '__main__':
    init_logging()
    main_flow()
