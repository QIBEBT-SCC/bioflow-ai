from pydantic import BaseModel

from app.utils.config import get_settings

tool_cfg = get_settings().tool


class DefaultArgs(BaseModel):
    fastp_arg: str = f'-w {tool_cfg.THREADS}'
    kraken2_arg: str = f'--db {tool_cfg.KRAKEN_DB}  --threads {tool_cfg.THREADS} --use-names --gzip-compressed --paired'
    bowtie2_arg: str = f'--threads {tool_cfg.THREADS}'
    spades_arg: str = f'--sc --t {tool_cfg.THREADS}'
    checkm2_arg: str = (
        f'--database_path {tool_cfg.CHECKM2_DB} --threads {tool_cfg.THREADS} -x fasta'
    )
