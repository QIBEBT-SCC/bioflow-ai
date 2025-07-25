import json

from sqlalchemy import text
from sqlmodel import Session

import app.crud.tools.tool_tag as tag_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.image as image_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool as tool_crud
from app.core.session import engine
from app.models.tools import ToolTag, ToolGroup, ToolImage, ToolHelpDoc, ToolDefine


def init_db():
    with Session(engine) as session:
        # init tags
        with open('../presets/tags.json', 'r') as f:
            tags_data = json.loads(f.read())
            tags = [ToolTag.model_validate(tag) for tag in tags_data]
        tag_crud.insert_all(session, tags)
    
        # init group
        with open('../presets/groups.json', 'r') as f:
            groups_data = json.loads(f.read())
            groups = [ToolGroup.model_validate(group) for group in groups_data]
        group_crud.insert_all(session, groups)
    
        # init image
        with open('../presets/images.json', 'r') as f:
            images_data = json.loads(f.read())
            images = [ToolImage.model_validate(image) for image in images_data]
        image_crud.insert_all(session, images)
    
        # init doc
        with open('../presets/tool_docs.json', 'r') as f:
            docs_data = json.loads(f.read())
            docs = [ToolHelpDoc.model_validate(doc) for doc in docs_data]
        doc_crud.insert_all(session, docs)
    
        # init tool
        with open('../presets/tools.json', 'r') as f:
            tools_data = json.loads(f.read())
        tools = []
        with session.no_autoflush:
            for tool_data in tools_data:
                tool = ToolDefine.model_validate(tool_data)
                tag_ids = [tag['id'] for tag in tool_data['tags']]
                tool.tags = tag_crud.get_all_by_id(session, tag_ids)
                tools.append(tool)
            tool_crud.insert_all(session, tools)

    # reset max_id
    table_seq_pairs = [
        ('tool_tags', 'tool_tags_id_seq'),
        ('tool_groups', 'tool_groups_id_seq'),
        ('images', 'images_id_seq'),
        ('tool_docs', 'tool_docs_id_seq'),
        ('tools', 'tools_id_seq'),
    ]
    with engine.connect() as conn:
        for table, seq in table_seq_pairs:
            max_id = conn.execute(text(f'SELECT MAX(id) FROM {table}')).scalar() or 0
            conn.execute(text(f"SELECT setval('{seq}', :max_id)"), {'max_id': max_id})
        conn.commit()


if __name__ == '__main__':
    init_db()
