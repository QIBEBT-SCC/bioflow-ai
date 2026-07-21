from enum import IntEnum
from typing import Any, Optional

from pydantic import TypeAdapter
from sqlalchemy import Dialect, Integer, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.type_api import _T


class IntEnumType(TypeDecorator[int]):
    """Persist an ``IntEnum`` as an integer and restore it when loading."""

    impl = Integer
    cache_ok = True

    def __init__(self, enum_type: type[IntEnum], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enum_type = enum_type

    def process_bind_param(self, value: IntEnum | int | None, dialect: Dialect) -> int | None:  # noqa: D102
        return None if value is None else int(value)

    def process_result_value(self, value: int | None, dialect: Dialect) -> IntEnum | None:  # noqa: D102
        return None if value is None else self.enum_type(value)


class PydanticJson(TypeDecorator):
    impl = JSONB
    cache_ok = True

    def __init__(self, pt, *args, **kwargs) -> None:  # noqa: ANN001
        super().__init__(*args, **kwargs)
        self.pt = TypeAdapter(pt)

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:  # noqa: D102
        if value is None:
            return None
        return self.pt.dump_python(value, mode='json')

    def process_result_value(self, value: Optional[Any], dialect: Dialect) -> Optional[_T]:  # noqa: D102
        if value is None:
            return None
        return self.pt.validate_python(value)
