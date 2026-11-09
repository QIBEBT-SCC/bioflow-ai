from enum import IntEnum
from typing import Literal, Self

from pydantic import BaseModel, Field


class ToolType(IntEnum):
    """Supported tool execution backends."""

    DOCKER = 0
    DOCKER_PASSTHROUGH = 1


class ParamDefine(BaseModel):
    """A configurable command-line parameter."""

    description: str
    command: str
    is_position: bool = Field(default=False)
    index: int | None = Field(default=None)


class FileMount(BaseModel):
    """A logical input or output mounted into a tool container."""

    name: str
    description: str
    file_type: Literal['INPUT', 'OUTPUT']
    file_path: str
    mount_path: str
    is_report: bool = Field(default=False)
    is_log: bool = Field(default=False)
    is_required: bool = Field(
        default=True,
        description='For OUTPUT mounts, require the path to exist after execution. Ignored for INPUT mounts.',
    )

    @property
    def is_passthrough(self) -> bool:
        """Whether this handle preserves a writable mount across nodes."""
        return self.file_path.startswith('passthrough:')


class ImageConfig(BaseModel):
    """A normalized container image reference."""

    registry: str = 'docker.io'
    namespace: str = 'library'
    repository: str
    tag: str = 'latest'

    def get_image(self) -> str:
        """Return the fully qualified container image reference."""
        return f'{self.registry}/{self.namespace}/{self.repository}:{self.tag}'

    @classmethod
    def from_str(cls, image_str: str) -> Self:
        """Parse a container image reference."""
        image_group = image_str.split(':')
        if len(image_group) > 2:  # noqa: PLR2004
            err_msg = 'Incorrect image format. It MUST be like `[registry]/[namespace]/<repository>:[tag]`'
            raise ValueError(err_msg)

        tag = 'latest' if len(image_group) == 1 else image_group[1]

        url_group = image_group[0].split('/')
        match len(url_group):
            case 3:
                return cls(registry=url_group[0], namespace=url_group[1], repository=url_group[2], tag=tag)
            case 2:
                return cls(namespace=url_group[0], repository=url_group[1], tag=tag)
            case 1:
                return cls(repository=url_group[0], tag=tag)
            case _:
                err_msg = 'Incorrect image format. It MUST be like `[registry]/[namespace]/<repository>:[tag]`'
                raise ValueError(err_msg)
