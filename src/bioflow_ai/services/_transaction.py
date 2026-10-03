from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session


@contextmanager
def committing(session: Session) -> Generator[None, None, None]:
    """Commit a service operation or roll it back on failure."""
    try:
        yield
        session.commit()
    except Exception:
        session.rollback()
        raise
