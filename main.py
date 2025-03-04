import os

from loguru import logger
from toil.common import Toil
from toil.job import Job
from toil.lib.io import mkdtemp

from app.bioinfo.qc import run_fastqc
from app.bioinfo.metaphlan import run_metaphlan
from app.utils.logger import init_logging


# MetaPhlAn数据库路径
METAPHLAN_DB = '/mnt/data01/metaphlan_db'


def metaphlan_fn(job, qc_results):
    """
    在QC完成后运行MetaPhlAn分析

    Args:
        job: Toil作业对象
        qc_results: QC作业的结果（文件ID字典）

    Returns:
        MetaPhlAn分析结果（文件ID字典）
    """
    return job.addChildJobFn(
        run_metaphlan,
        r1_id=qc_results['R1_clean.fq.gz'],
        r2_id=qc_results['R2_clean.fq.gz'] if 'R2_clean.fq.gz' in qc_results else None,
        bowtie_db=METAPHLAN_DB,
    ).rv()


def workflow(job, r1_id, r2_id):
    """
    定义整个工作流程

    Args:
        job: Toil作业对象
        r1_id: 第一个FASTQ文件的文件ID
        r2_id: 第二个FASTQ文件的文件ID

    Returns:
        包含QC结果和MetaPhlAn结果的元组
    """
    # 运行QC
    fastp_job = job.addChildJobFn(run_fastqc, r1_id=r1_id, r2_id=r2_id)

    # 添加依赖关系，确保QC完成后再运行MetaPhlAn
    metaphlan_job = fastp_job.addFollowOnJobFn(metaphlan_fn, fastp_job.rv())

    return fastp_job.rv(), metaphlan_job.rv()


def main() -> None:
    init_logging()

    jobstore: str = mkdtemp('tutorial_docker')
    os.rmdir(jobstore)
    options = Job.Runner.getDefaultOptions(jobstore)
    options.logLevel = 'INFO'
    options.clean = 'always'

    # 确保输出目录存在
    output_dir = 'test/clean'
    os.makedirs(output_dir, exist_ok=True)

    with Toil(options) as toil:
        # 导入原始FASTQ文件
        logger.info('导入原始FASTQ文件...')
        r1_id = toil.importFile(f'file://{os.path.abspath("test/raw/M3_1.fq.gz")}')
        r2_id = toil.importFile(f'file://{os.path.abspath("test/raw/M3_2.fq.gz")}')

        # 启动工作流
        logger.info('启动工作流...')
        root_job = Job.wrapJobFn(workflow, r1_id=r1_id, r2_id=r2_id)
        results = toil.start(root_job)

        # 解析结果
        fastp_results, metaphlan_results = results

        # 导出QC结果文件
        logger.info('导出QC结果文件...')
        for file_name, file_id in fastp_results.items():
            output_path = os.path.join(output_dir, file_name)
            toil.exportFile(file_id, f'file://{os.path.abspath(output_path)}')
            logger.info(f'已导出文件: {output_path}')

        # 导出MetaPhlAn结果文件
        logger.info('导出MetaPhlAn结果文件...')
        for file_name, file_id in metaphlan_results.items():
            output_path = os.path.join(output_dir, file_name)
            toil.exportFile(file_id, f'file://{os.path.abspath(output_path)}')
            logger.info(f'已导出文件: {output_path}')


if __name__ == '__main__':
    main()
