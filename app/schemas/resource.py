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
