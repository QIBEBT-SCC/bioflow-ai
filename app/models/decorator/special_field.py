from typing import Any, Optional

from pydantic import TypeAdapter
from sqlalchemy import Dialect, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.type_api import _T


class PydanticJson(TypeDecorator):
    impl = JSONB
    cache_ok = True

    def __init__(self, pt, *args, **kwargs) -> None:  # noqa: ANN001, ANN002
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
