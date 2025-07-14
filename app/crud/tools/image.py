from sqlmodel import Session

from app.models.tools import ToolImage


def insert(session: Session, image: ToolImage) -> ToolImage:
    session.add(image)
    session.commit()
    session.refresh(image)

    return image
