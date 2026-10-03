"""One-time schema migration for dynamic child workflow runs.

Run with the dispatcher and workers stopped: uv run python -m bioflow_ai.cli.migrate_foreach
"""

from sqlalchemy import inspect, text

from bioflow_ai.infrastructure.database import engine


def main() -> None:
    """Add nullable child-run metadata and its idempotency index."""
    existing = {column['name'] for column in inspect(engine).get_columns('workflow_runs')}
    columns = {
        'parent_node_uid': 'UUID REFERENCES workflow_node_runs(uid) ON DELETE CASCADE',
        'parent_generation': 'INTEGER',
        'item_key': 'VARCHAR(255)',
        'item_index': 'INTEGER',
        'item_name': 'VARCHAR(1024)',
    }
    with engine.begin() as connection:
        for name, ddl in columns.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE workflow_runs ADD COLUMN {name} {ddl}'))
        connection.execute(
            text(
                'CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_runs_foreach_item '
                'ON workflow_runs (parent_node_uid, parent_generation, item_key)'
            )
        )


if __name__ == '__main__':
    main()
