import os
import shutil
from pathlib import Path
from typing import Any, Optional

from toil.common import Toil
from toil.fileStores.abstractFileStore import AbstractFileStore
from toil.job import Job, logger
from toil.lib.docker import apiDockerCall
from toil.lib.io import mkdtemp

from app.bioinfo import FASTP_IMAGE


class FastpWorkflow(Job):
    def __init__(self, output_path: str | os.PathLike, r1_id: str, r2_id: Optional[str] = None):
        super().__init__()

        self.r1_id = r1_id
        self.r2_id = r2_id
        if r2_id:
            self.sample_name = os.path.commonprefix([Path(r1_id).name, Path(r2_id).name]).strip('_')
        else:
            self.sample_name = Path(r1_id).name.rstrip('.gz').rstrip('.fq').rstrip('.fastq')

        self.output_path = output_path if isinstance(output_path, Path) else Path(output_path)

    def run(self, file_store: AbstractFileStore) -> Any:
        file_store.readGlobalFile(
            self.r1_id, str(Path(self.tempDir) / f'{self.sample_name}_R1.fq.gz')
        )
        parameters = [
            'fastp',
            '-i',
            f'/data/{self.sample_name}_R1.fq.gz',
            '-o',
            f'/data/{self.sample_name}_fastp_R1.fq.gz',
        ]
        output_files = [f'{self.sample_name}_fastp_R1.fq.gz']
        if self.r2_id:
            file_store.readGlobalFile(
                self.r2_id, str(Path(self.tempDir) / f'{self.sample_name}_R2.fq.gz')
            )
            parameters.extend(
                [
                    '-I',
                    f'/data/{self.sample_name}_R2.fq.gz',
                    '-O',
                    f'/data/{self.sample_name}_fastp_R2.fq.gz',
                ]
            )
            output_files.append(f'{self.sample_name}_fastp_R2.fq.gz')

        parameters.extend(['-h', '/data/fastp.html', '-j', '/data/fastp.json', '-w', '8'])
        output_files.extend(['fastp.html', 'fastp.json'])

        apiDockerCall(job=self, image=FASTP_IMAGE, working_dir=self.tempDir, parameters=parameters)

        output_ids = {}
        for output_file in output_files:
            file_path = Path(self.tempDir) / output_file
            if os.path.exists(file_path):
                file_id = file_store.writeGlobalFile(str(file_path.absolute()))
                output_ids[output_file] = file_id

        return output_ids


def run_fastp(
    output_path: str | os.PathLike, r1_file: str | os.PathLike, r2_file: Optional[str | os.PathLike]
) -> dict[str, str]:
    """

    Args:
        output_path: fastp输出文件夹
        r1_file: pair-end file1
        r2_file: pair-end file2

    Returns:
        工作流返回文件列表
    """
    job_store: str = mkdtemp('toil_fastp')
    shutil.rmtree(job_store)
    options = Job.Runner.getDefaultOptions(job_store)
    options.logLevel = 'INFO'
    options.clean = 'always'

    output_dir = Path(output_path) if isinstance(output_path, str) else output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_path = Path(r1_file) if isinstance(r1_file, str) else r1_file
    r2_path = Path(r2_file) if isinstance(r2_file, str) else r2_file

    with Toil(options) as toil:
        logger.info('导入原始FASTQ文件...')
        r1_id = toil.importFile(f'file://{r1_path.absolute()}')
        r2_id = toil.importFile(f'file://{r2_path.absolute()}')

        fastp_flow = FastpWorkflow(output_dir, r1_id, r2_id)

        result = toil.start(fastp_flow)
        logger.info('导出QC结果文件...')
        for file_name, file_id in result.items():
            output_path = os.path.join(output_dir, file_name)
            toil.exportFile(file_id, f'file://{os.path.abspath(output_path)}')
            logger.info(f'已导出文件: {output_path}')

        return result
