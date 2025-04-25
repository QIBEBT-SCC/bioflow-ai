from pydantic import BaseModel

from app.core.config import get_settings

tool_cfg = get_settings().tool


class DefaultArgs(BaseModel):
    fastp: str = f'--thread {min(8, get_settings().tool.THREADS)}'
    metaphlan: str = f'--input_type fastq --nproc {tool_cfg.THREADS}'
    diamond: str = (
        f'--fast --outfmt 100 --evalue 1e-5 --top 1 --threads {get_settings().tool.THREADS}'
    )
    daa_meganizer: str = f'--threads {tool_cfg.THREADS}'
    daa2rma: str = f'--threads {tool_cfg.THREADS}'
    rma2info_taxon: str = (
        '--class2count Taxonomy --paths --ranks --majorRanksOnly --list --listMore'
    )
    rma2info_go: str = (
        '--class2count INTERPRO2GO --names --list --listMore'
    )
    rma2info_eggnog: str = (
        '--class2count EGGNOG --names --list --listMore'
    )
