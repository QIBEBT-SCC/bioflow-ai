import os

from loguru import logger
from toil.common import Toil
from toil.job import Job
from toil.lib.io import mkdtemp

from app.bioinfo.qc import run_fastqc


def main() -> None:
    jobstore: str = mkdtemp('tutorial_docker')
    os.rmdir(jobstore)
    options = Job.Runner.getDefaultOptions(jobstore)
    options.logLevel = 'INFO'
    options.clean = 'always'

    output_dir = 'test/clean'
    os.makedirs(output_dir, exist_ok=True)

    with Toil(options) as toil:
        r1_id = toil.import_file(f'file://{os.path.abspath("test/raw/M3_1.fq.gz")}')
        r2_id = toil.import_file(f'file://{os.path.abspath("test/raw/M3_2.fq.gz")}')
        fastp = Job.wrapJobFn(run_fastqc, r1_id=r1_id, r2_id=r2_id)
        output_ids = toil.start(fastp)

        # 将处理后的文件导出到test/clean目录
        for file_name, file_id in output_ids.items():
            output_path = os.path.join(output_dir, file_name)
            toil.export_file(file_id, f'file://{os.path.abspath(output_path)}')
            logger.info(f'已导出文件: {output_path}')


if __name__ == '__main__':
    main()
