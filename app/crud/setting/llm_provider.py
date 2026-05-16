from typing import Optional

from sqlmodel import Session, col, select

from app.models.setting import LLMProvider
from app.schemas.setting import LLMProviderUpdate


def insert(session: Session, provider: LLMProvider) -> LLMProvider:
    """Insert a new LLM provider into the database."""
    session.add(provider)
    session.commit()
    session.refresh(provider)

    return provider


def get(session: Session, provider_id: int) -> Optional[LLMProvider]:
    """Retrieve an LLM provider by its ID."""
    statement = select(LLMProvider).where(LLMProvider.id == provider_id)

    return session.exec(statement).one_or_none()


def get_all(session: Session) -> list[LLMProvider]:
    """Get all LLM providers ordered by ID."""
    statement = select(LLMProvider).order_by(col(LLMProvider.id).asc())

    return session.exec(statement).all()


def update(session: Session, provider_id: int, provider_update: LLMProviderUpdate) -> LLMProvider:
    """Update an existing LLM provider in the database."""
    provider = get(session, provider_id)
    update_data = provider_update.model_dump(exclude_unset=True)
    provider.sqlmodel_update(update_data)

    session.add(provider)
    session.commit()
    session.refresh(provider)

    return provider


def delete(session: Session, provider_id: int) -> bool:
    """Delete an LLM provider by ID, returning True if successful."""
    provider = get(session, provider_id)
    if provider is None:
        return False

    session.delete(provider)
    session.commit()

    return True
