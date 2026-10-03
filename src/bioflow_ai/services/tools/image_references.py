"""Generated short references for persisted tool images."""

from collections import Counter
from collections.abc import Sequence

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from bioflow_ai.models.tools import ToolImage
from bioflow_ai.schemas.tool_config import ImageConfig


class DuplicateImageUrlError(ValueError):
    """Two image records use the same fully qualified container image URL."""


def build_image_refs(images: Sequence[ImageConfig]) -> list[str]:
    """Return the shortest path-based unique reference for each image."""
    candidate_groups = [_reference_candidates(image) for image in images]
    full_urls = [candidates[-1] for candidates in candidate_groups]
    duplicate_urls = sorted(url for url, count in Counter(full_urls).items() if count > 1)
    if duplicate_urls:
        duplicates = ', '.join(duplicate_urls)
        msg = f'Duplicate container image URL(s): {duplicates}'
        raise DuplicateImageUrlError(msg)

    counts_by_level = [Counter(candidates[level] for candidates in candidate_groups) for level in range(4)]
    return [
        next(candidates[level] for level, counts in enumerate(counts_by_level) if counts[candidates[level]] == 1)
        for candidates in candidate_groups
    ]


def assign_image_refs(images: Sequence[ToolImage]) -> None:
    """Recalculate generated references for a complete locked image collection."""
    refs = build_image_refs([image.image for image in images])
    for image, image_ref in zip(images, refs, strict=True):
        image.ref = image_ref


def lock_and_get_all(session: Session) -> list[ToolImage]:
    """Serialize image mutations and return all images participating in ref generation."""
    session.execute(text('LOCK TABLE images IN SHARE ROW EXCLUSIVE MODE'))
    return list(session.scalars(select(ToolImage).order_by(ToolImage.uid)).all())


def _reference_candidates(image: ImageConfig) -> tuple[str, str, str, str]:
    repository = image.repository
    namespaced = f'{image.namespace}/{repository}'
    registered = f'{image.registry}/{namespaced}'
    return repository, namespaced, registered, image.get_image()
