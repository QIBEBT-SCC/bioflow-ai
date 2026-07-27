from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.setting import LLMProvider
from app.schemas.setting import LLMProviderUpdate


def insert(session: Session, provider: LLMProvider) -> LLMProvider:
    """Stage a new LLM provider."""
    session.add(provider)
    session.flush()
    session.refresh(provider)
    return provider


def get(session: Session, provider_id: int) -> LLMProvider | None:
    """Return an LLM provider by ID, including its models."""
    statement = select(LLMProvider).options(selectinload(LLMProvider.models)).where(LLMProvider.id == provider_id)
    return session.scalar(statement)


def get_all(session: Session) -> Sequence[LLMProvider]:
    """Get all LLM providers ordered by ID."""
    statement = select(LLMProvider).options(selectinload(LLMProvider.models)).order_by(LLMProvider.id.asc())
    return session.scalars(statement).all()


def update(session: Session, provider_id: int, provider_update: LLMProviderUpdate) -> LLMProvider | None:
    """Stage updates to an LLM provider."""
    provider = get(session, provider_id)
    if provider is None:
        return None
    update_data = provider_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(provider, field, value)
    session.flush()
    session.refresh(provider)
    return provider


def update_all(session: Session, providers: Sequence[LLMProvider]) -> None:
    """Stage updates to a collection of LLM providers."""
    session.add_all(providers)
    session.flush()


def delete(session: Session, provider_id: int) -> bool:
    """Stage deletion of an LLM provider, returning whether it existed."""
    provider = get(session, provider_id)
    if provider is None:
        return False

    session.delete(provider)
    session.flush()
    return True
