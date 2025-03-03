import os

from toil.job import JobFunctionWrappingJob
from toil.lib.docker import apiDockerCall

from app.bioinfo import FASTP_IMAGE


def run_fastqc(job: JobFunctionWrappingJob, r1_id: str, r2_id: str):
    """
    运行fastp进行质量控制和过滤

    Args:
        job: Toil作业对象
        r1_id: 第一个FASTQ文件的文件ID
        r2_id: 第二个FASTQ文件的文件ID (如果是双端测序)

    Returns:
        处理后文件的文件ID列表
    """
    # Read in files and set parameters

    job.fileStore.readGlobalFile(r1_id, os.path.join(job.tempDir, 'R1.fq.gz'))
    parameters = ['fastp', '-i', '/data/R1.fq.gz', '-o', '/data/R1_clean.fq.gz']
    output_files = ['R1_clean.fq.gz']
    if r2_id:
        job.fileStore.readGlobalFile(r2_id, os.path.join(job.tempDir, 'R2.fq.gz'))
        parameters.extend(['-I', '/data/R2.fq.gz', '-O', '/data/R2_clean.fq.gz'])
        output_files.append('R2_clean.fq.gz')

    parameters.extend(['-h', '/data/fastp.html', '-j', '/data/fastp.json', '-w', '8'])
    output_files.append('fastp.html')

    # Call fastQC
    apiDockerCall(
        job=job,
        image=FASTP_IMAGE,
        working_dir=job.tempDir,
        parameters=parameters,
        stdout=True,
        stderr=True,
    )

    output_ids = {}
    for output_file in output_files:
        file_path = os.path.join(job.tempDir, output_file)
        if os.path.exists(file_path):
            file_id = job.fileStore.writeGlobalFile(file_path)
            output_ids[output_file] = file_id

    return output_ids
