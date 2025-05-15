from pydantic import TypeAdapter
from sqlalchemy import TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB


class PydanticJson(TypeDecorator):
    impl = JSONB()
    cache_ok = True

    def __init__(self, pt):
        super().__init__()
        self.pt = TypeAdapter(pt)
        self.coerce_compared_value = self.impl.coerce_compared_value

    def bind_processor(self, dialect):
        return lambda value: self.pt.dump_json(value).decode('utf-8') if value is not None else None

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, dict)):
                return self.pt.validate_python(value)
            return self.pt.validate_json(value)

        return process
