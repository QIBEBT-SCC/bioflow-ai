from pydantic import BaseModel, Field


class ImageMap(BaseModel):
    old_name: str
    file_path: str


class PDFParseResult(BaseModel):
    """Paths to temporary parser artifacts, ready for the shared S3 upload flow."""

    markdown: str
    images: list[ImageMap] = Field(default_factory=list)
