import os

from toil.job import JobFunctionWrappingJob
from toil.lib.docker import apiDockerCall

from app.bioinfo import METAPHLAN_IMAGE


def run_metaphlan(job: JobFunctionWrappingJob, r1_id: str, r2_id: str, bowtie_db: str):
    """
    使用MetaPhlAn进行宏基因组分析

    Args:
        job: Toil作业对象
        r1_id: 第一个清洗后FASTQ文件的文件ID
        r2_id: 第二个清洗后FASTQ文件的文件ID (如果是双端测序)
        bowtie_db: Bowtie2数据库路径

    Returns:
        处理后文件的文件ID字典
    """
    # 读取输入文件
    r1_path = os.path.join(job.tempDir, 'R1_clean.fq.gz')
    job.fileStore.readGlobalFile(r1_id, r1_path)
    
    input_files = ['/data/R1_clean.fq.gz']
    if r2_id:
        r2_path = os.path.join(job.tempDir, 'R2_clean.fq.gz')
        job.fileStore.readGlobalFile(r2_id, r2_path)
        input_files.append('/data/R2_clean.fq.gz')
    
    # 设置输出文件
    output_files = ['profiled_metagenome.txt', 'sample.bowtie2.bz2']
    
    # 构建MetaPhlAn命令参数
    parameters = [
        'metaphlan',
        ','.join(input_files),
        '--input_type', 'fastq',
        '--bowtie2out', '/data/sample.bowtie2.bz2',
        '-o', '/data/profiled_metagenome.txt',
        '--bowtie2db', bowtie_db
    ]
    
    # 创建卷映射
    volumes = {
        job.tempDir: {'bind': '/data', 'mode': 'rw'},
        os.path.dirname(bowtie_db): {'bind': os.path.dirname(bowtie_db), 'mode': 'ro'}
    }
    
    # 运行MetaPhlAn
    apiDockerCall(
        job=job,
        image=METAPHLAN_IMAGE,
        working_dir=job.tempDir,
        parameters=parameters,
        volumes=volumes,
        stdout=True,
        stderr=True,
    )

    # 收集输出文件ID
    output_ids = {}
    for output_file in output_files:
        file_path = os.path.join(job.tempDir, output_file)
        if os.path.exists(file_path):
            file_id = job.fileStore.writeGlobalFile(file_path)
            output_ids[output_file] = file_id

    return output_ids
