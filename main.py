from app.bioinfo.workflows.qc import run_fastp
from app.utils.logger import init_logging


# MetaPhlAn数据库路径
METAPHLAN_DB = '/mnt/data01/metaphlan_db'


def main() -> None:
    init_logging()

    run_fastp('test/clean', 'test/raw/M3_1.fq.gz', 'test/raw/M3_2.fq.gz')


if __name__ == '__main__':
    main()
