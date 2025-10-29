from typing import Optional

from pydantic import BaseModel


class BioDbCreate(BaseModel):
    name: str
    description: Optional[str] = None
    path: str
    last_update: Optional[str] = None


class BioDbSimple(BaseModel):
    id: int
    name: str


class GenomeUpdate(BaseModel):
    id: int
    aliases: Optional[str] = None

    bowtie2_index: Optional[str] = None
    bwa_index: Optional[str] = None
    hisat2_index: Optional[str] = None
    star_index: Optional[str] = None
    minimap2_index: Optional[str] = None
