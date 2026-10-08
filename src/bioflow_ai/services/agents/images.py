# ruff: noqa: TRY301 - validation errors are converted to recoverable image/tool failures.
"""Prepare bounded visual input and readable image previews for agent messages."""

import base64
import binascii
import io
import math
import re
import warnings
from pathlib import PurePosixPath
from typing import Any, Literal, cast
from xml.etree.ElementTree import Element, tostring

from defusedxml import ElementTree
from PIL import Image, ImageOps, UnidentifiedImageError
from resvg_py import svg_to_bytes
from tinycss2 import parse_component_value_list, parse_declaration_list, serialize

from bioflow_ai.schemas.agent import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_EDGE,
    ImageMessagePart,
)

MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {'PNG', 'JPEG', 'WEBP', 'TIFF'}
SVG_SIZE_DEFAULT = 300
SVG_LENGTH = re.compile(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(px|pt|pc|in|cm|mm|q|em|ex|%)?')
SVG_UNITS = {
    'px': 1,
    'pt': 96 / 72,
    'pc': 16,
    'in': 96,
    'cm': 96 / 2.54,
    'mm': 96 / 25.4,
    'q': 96 / 101.6,
    'em': 12,
    'ex': 6,
}
DATA_URL = re.compile(r'^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$')
INLINE_IMAGE = re.compile(r'data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+')


class ImagePreparationError(ValueError):
    """A user-visible image validation or conversion failure."""


def _check_dimensions(width: int, height: int) -> None:
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        message = 'Image exceeds the 40,000,000 pixel limit or has invalid dimensions.'
        raise ImagePreparationError(message)


def _validate_svg_reference(url: str) -> None:
    """Permit internal fragments and bounded raster data, never external resources."""
    if url.startswith('#'):
        return
    match = DATA_URL.fullmatch(url)
    if match is None:
        message = 'SVG external network and filesystem resources are not supported.'
        raise ImagePreparationError(message)
    try:
        data = base64.b64decode(match[2], validate=True)
        if len(data) > MAX_SOURCE_BYTES:
            message = 'Embedded SVG image exceeds the 20 MiB limit.'
            raise ImagePreparationError(message)
        with Image.open(io.BytesIO(data)) as image:
            _check_dimensions(*image.size)
            actual_type = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}.get(image.format or '')
            if actual_type != match[1]:
                message = 'Unsupported embedded SVG image format.'
                raise ImagePreparationError(message)
            image.load()
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        message = 'Invalid embedded SVG image.'
        raise ImagePreparationError(message) from error


def _validate_svg_css(tokens: list[Any]) -> None:
    """Check parsed CSS, including escaped URLs and nested blocks."""
    for token in tokens:
        if token.type == 'at-keyword' and token.lower_value == 'import':
            message = 'SVG external stylesheets are not supported.'
            raise ImagePreparationError(message)
        if token.type == 'url':
            _validate_svg_reference(token.value)
        elif token.type == 'function':
            if token.lower_name == 'url':
                arguments = [part for part in token.arguments if part.type not in {'whitespace', 'comment'}]
                if len(arguments) != 1 or arguments[0].type != 'string':
                    message = 'Invalid SVG CSS resource reference.'
                    raise ImagePreparationError(message)
                _validate_svg_reference(arguments[0].value)
            else:
                _validate_svg_css(token.arguments)
        elif hasattr(token, 'content'):
            _validate_svg_css(token.content)
        elif token.type == 'error':
            message = 'Invalid SVG CSS resource reference.'
            raise ImagePreparationError(message)


def _svg_length(value: str | None, reference: float) -> float:
    """Resolve root viewport lengths at 96 DPI before allocating a render surface."""
    if value is None:
        return reference
    match = SVG_LENGTH.fullmatch(value.strip().lower())
    if match is None:
        message = 'Unsupported SVG viewport dimension.'
        raise ImagePreparationError(message)
    number = float(match[1])
    unit = match[2] or 'px'
    length = number * (reference / 100 if unit == '%' else SVG_UNITS[unit])
    if not math.isfinite(length) or length <= 0:
        message = 'SVG viewport dimensions must be positive and finite.'
        raise ImagePreparationError(message)
    return length


def _svg_dimensions(root: Element) -> tuple[float, float]:
    reference_width = reference_height = float(SVG_SIZE_DEFAULT)
    if view_box := root.get('viewBox'):
        try:
            x, y, reference_width, reference_height = (float(value) for value in re.split(r'[\s,]+', view_box.strip()))
        except ValueError as error:
            message = 'Invalid SVG viewBox.'
            raise ImagePreparationError(message) from error
        if (
            not all(math.isfinite(value) for value in (x, y, reference_width, reference_height))
            or min(reference_width, reference_height) <= 0
        ):
            message = 'Invalid SVG viewBox.'
            raise ImagePreparationError(message)
    dimensions = {key: root.get(key) for key in ('width', 'height')}
    for declaration in parse_declaration_list(root.get('style', '')):
        if declaration.type == 'declaration' and declaration.lower_name in dimensions:
            dimensions[declaration.lower_name] = serialize(declaration.value).strip()
    width = _svg_length(dimensions['width'], reference_width)
    height = _svg_length(dimensions['height'], reference_height)
    _check_dimensions(math.ceil(width), math.ceil(height))
    return width, height


def _render_svg(data: bytes) -> bytes:
    """Validate SVG resources, then render a bounded PNG with the bundled Rust engine."""
    root = ElementTree.fromstring(data, forbid_dtd=True)
    if root.tag != '{http://www.w3.org/2000/svg}svg':
        message = 'The input is not an SVG image.'
        raise ImagePreparationError(message)
    ids = {element.get('id') for element in root.iter() if element.get('id')}
    for element in root.iter():
        for key, value in element.attrib.items():
            if key == '{http://www.w3.org/XML/1998/namespace}base' and value:
                message = 'SVG external network and filesystem resources are not supported.'
                raise ImagePreparationError(message)
            if key.rsplit('}', 1)[-1] == 'href' and value:
                _validate_svg_reference(value)
                # resvg treats <image href="#..."> as a filesystem path, not an internal element.
                if value.startswith('#') and (element.tag.rsplit('}', 1)[-1] == 'image' or value[1:] not in ids):
                    message = 'SVG image references must use embedded PNG, JPEG or WebP data.'
                    raise ImagePreparationError(message)
            elif key == 'style' or 'url' in value.lower() or '\\' in value:
                tokens = parse_component_value_list(value)
                _validate_svg_css(tokens)
                element.set(key, serialize(tokens))
        if element.tag == '{http://www.w3.org/2000/svg}style':
            tokens = parse_component_value_list(''.join(element.itertext()))
            _validate_svg_css(tokens)
            element.text = serialize(tokens)
    width, height = _svg_dimensions(root)
    scale = min(1.0, MAX_IMAGE_EDGE / max(width, height))
    root.set('width', str(width))
    root.set('height', str(height))
    return svg_to_bytes(
        svg_string=tostring(root, encoding='unicode'),
        width=max(1, round(width * scale)),
        height=max(1, round(height * scale)),
        dpi=96,
        font_size=12,
    )


def prepare_image(data: bytes, filename: str = 'image', page: int = 1) -> ImageMessagePart:
    """Decode, orient and resize one selected page, then return a self-contained image."""
    if not data or len(data) > MAX_SOURCE_BYTES:
        message = 'Image must be non-empty and no larger than 20 MiB.'
        raise ImagePreparationError(message)
    if page < 1:
        message = 'Image page numbers start at 1.'
        raise ImagePreparationError(message)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            try:
                opened = Image.open(io.BytesIO(data))
                source_format = opened.format
            except UnidentifiedImageError:
                data = _render_svg(data)
                opened = Image.open(io.BytesIO(data))
                source_format = 'SVG'
            with opened:
                image = opened
                if source_format not in {*SUPPORTED_FORMATS, 'SVG'}:
                    message = 'Supported image formats are PNG, JPEG, WebP, TIFF and SVG.'
                    raise ImagePreparationError(message)
                page_count = getattr(image, 'n_frames', 1) if source_format == 'TIFF' else 1
                if page > page_count:
                    message = f'Page {page} does not exist; this image has {page_count} page(s).'
                    raise ImagePreparationError(message)
                image.seek(page - 1)
                _check_dimensions(*image.size)
                note = None
                if image.mode in {'I', 'F'} or image.mode.startswith('I;16'):
                    extrema = cast('tuple[float, float]', image.getextrema())
                    low, high = float(extrema[0]), float(extrema[1])
                    image = (
                        image.convert('F')
                        .point(lambda value: (value - low) * 255 / (high - low) if high > low else value * 0)
                        .convert('L')
                    )
                    note = 'High bit depth image normalized to 8-bit display intensity.'
                image = ImageOps.exif_transpose(image)
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
                image = image.convert('RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB')
                output_format = 'JPEG' if source_format == 'JPEG' else 'PNG'
                buffer = io.BytesIO()
                image.save(buffer, format=output_format, **({'quality': 90} if output_format == 'JPEG' else {}))
                encoded = buffer.getvalue()
                if len(encoded) > MAX_IMAGE_BYTES:
                    message = 'Prepared image exceeds the 5 MiB limit.'
                    raise ImagePreparationError(message)
                media_type = 'image/jpeg' if output_format == 'JPEG' else 'image/png'
                return ImageMessagePart(
                    url=f'data:{media_type};base64,{base64.b64encode(encoded).decode("ascii")}',
                    media_type=media_type,
                    filename=PurePosixPath(filename.replace('\\', '/')).name[:255] or 'image',
                    width=image.width,
                    height=image.height,
                    source_format=cast('Literal["PNG", "JPEG", "WEBP", "TIFF", "SVG"]', source_format),
                    page=page,
                    page_count=page_count,
                    conversion_note=note,
                )
    except ImagePreparationError:
        raise
    except Exception as error:
        message = 'Unable to decode or render the image.'
        raise ImagePreparationError(message) from error


def validate_prepared_image(part: ImageMessagePart) -> ImageMessagePart:
    """Revalidate an inline client snapshot without trusting MIME or dimensions."""
    match = DATA_URL.fullmatch(part.url)
    if match is None or match[1] != part.media_type:
        message = 'Image must use a PNG, JPEG or WebP base64 data URL matching its media type.'
        raise ImagePreparationError(message)
    try:
        data = base64.b64decode(match[2], validate=True)
    except binascii.Error as error:
        message = 'Invalid image base64 data.'
        raise ImagePreparationError(message) from error
    if len(data) > MAX_IMAGE_BYTES:
        message = 'Prepared image exceeds the 5 MiB limit.'
        raise ImagePreparationError(message)
    try:
        with Image.open(io.BytesIO(data)) as image:
            actual = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}.get(image.format or '')
            if actual != part.media_type or image.size != (part.width, part.height):
                message = 'Image format or dimensions do not match the supplied metadata.'
                raise ImagePreparationError(message)
            image.load()
    except (OSError, Image.DecompressionBombError) as error:
        message = 'Invalid prepared image.'
        raise ImagePreparationError(message) from error
    if part.page > part.page_count:
        message = 'Selected image page exceeds the page count.'
        raise ImagePreparationError(message)
    return part.model_copy(update={'filename': PurePosixPath(part.filename.replace('\\', '/')).name or 'image'})


def image_content(part: ImageMessagePart) -> list[dict]:
    """Create text labels and a separate visual block for model adapters."""
    label = f'Image: {part.filename}; {part.width}x{part.height}; page {part.page}/{part.page_count}.'
    if part.conversion_note:
        label += f' {part.conversion_note}'
    return [{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {'url': part.url}}]


def media_preview(value: Any) -> Any:
    """Preserve image labels and metadata while replacing encoded pixel data."""
    if isinstance(value, str):
        return INLINE_IMAGE.sub('[inline image]', value)
    if isinstance(value, dict):
        return {
            key: '[inline image]' if key in {'base64', 'data'} and value.get('type') == 'image' else media_preview(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [media_preview(item) for item in value]
    return value
