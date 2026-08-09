-- pg_trgm 索引
CREATE INDEX IF NOT EXISTS idx_images_name_trgm ON images USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_tools_name_trgm ON tools USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_bio_dbs_name_trgm ON bio_dbs USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_reference_name_trgm ON reference_genomes USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_reference_aliases_trgm ON reference_genomes USING gin (aliases gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_sample_files_tag ON sample_files (tag);
CREATE INDEX IF NOT EXISTS ix_sample_files_is_dynamic ON sample_files (is_dynamic);
CREATE INDEX IF NOT EXISTS ix_sample_files_producer_workflow_uid ON sample_files (producer_workflow_uid);
CREATE INDEX IF NOT EXISTS ix_project_file_mappings_keyword ON project_file_mappings (keyword);
CREATE INDEX IF NOT EXISTS ix_project_file_mappings_is_dynamic ON project_file_mappings (is_dynamic);
CREATE INDEX IF NOT EXISTS ix_project_file_mappings_producer_workflow_uid
    ON project_file_mappings (producer_workflow_uid);
CREATE INDEX IF NOT EXISTS idx_parent_documents_doc_id ON parent_documents (doc_id);

-- Web 智能体会话和可重放事件
CREATE INDEX IF NOT EXISTS ix_agent_sessions_owner_id ON agent_sessions (owner_id);
CREATE INDEX IF NOT EXISTS ix_agent_sessions_project_id ON agent_sessions (project_id);
CREATE INDEX IF NOT EXISTS ix_agent_sessions_update_time ON agent_sessions (update_time);
CREATE INDEX IF NOT EXISTS ix_agent_runs_session_uid ON agent_runs (session_uid);
CREATE INDEX IF NOT EXISTS ix_agent_runs_project_id ON agent_runs (project_id);
CREATE INDEX IF NOT EXISTS ix_agent_runs_status ON agent_runs (status);
CREATE INDEX IF NOT EXISTS ix_agent_runs_celery_task_id ON agent_runs (celery_task_id);
CREATE INDEX IF NOT EXISTS ix_agent_runs_update_time ON agent_runs (update_time);
CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_runs_one_active_per_session
    ON agent_runs (session_uid)
    WHERE status IN ('queued', 'running', 'waiting_input', 'cancel_requested');
CREATE INDEX IF NOT EXISTS ix_agent_messages_session_uid ON agent_messages (session_uid);
CREATE INDEX IF NOT EXISTS ix_agent_messages_run_uid ON agent_messages (run_uid);
CREATE INDEX IF NOT EXISTS ix_agent_messages_create_time ON agent_messages (create_time);
CREATE INDEX IF NOT EXISTS ix_agent_events_run_uid ON agent_events (run_uid);
CREATE INDEX IF NOT EXISTS ix_agent_events_event_type ON agent_events (event_type);
CREATE INDEX IF NOT EXISTS ix_agent_events_create_time ON agent_events (create_time);
