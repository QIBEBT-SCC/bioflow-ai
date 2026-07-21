import warnings

from sqlalchemy import create_engine
from sqlmodel import Session, select

from app.models.user import User, UserRole


def test_integer_enum_is_restored_when_model_is_loaded_from_database() -> None:
    engine = create_engine('sqlite://')
    User.__table__.create(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            'INSERT INTO users (id, username, email, role, hashed_password, is_active) '
            "VALUES (1, 'admin', 'admin@example.com', 2, 'hashed', 1)"
        )

    with Session(engine) as session:
        user = session.exec(select(User)).one()

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            serialized = user.model_dump_json()

    assert user.role is UserRole.ADMIN
    assert '"role":2' in serialized
