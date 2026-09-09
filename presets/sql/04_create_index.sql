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
CREATE INDEX IF NOT EXISTS ix_parent_documents_doc_id ON parent_documents (doc_id);
CREATE INDEX IF NOT EXISTS ix_codes_name ON codes (name);
CREATE INDEX IF NOT EXISTS ix_codes_node_type ON codes (node_type);

-- 持久化 DAG 运行与 Outbox 调度
CREATE INDEX IF NOT EXISTS ix_workflow_runs_owner_created ON workflow_runs (owner_id, create_time);
CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_runs_sample
    ON workflow_runs (project_id, workflow_uid, sample_uid)
    WHERE sample_uid IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_runs_project
    ON workflow_runs (project_id, workflow_uid)
    WHERE sample_uid IS NULL AND project_id IS NOT NULL AND workflow_uid IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_node_runs_run_generation_status
    ON workflow_node_runs (run_uid, generation, status);
CREATE INDEX IF NOT EXISTS ix_node_runs_heartbeat ON workflow_node_runs (status, heartbeat_at);
CREATE INDEX IF NOT EXISTS ix_node_dependencies_target ON workflow_node_dependencies (target_node_uid);
CREATE INDEX IF NOT EXISTS ix_node_dependencies_source ON workflow_node_dependencies (source_node_uid);
CREATE INDEX IF NOT EXISTS ix_workflow_outbox_status_created ON workflow_outbox (status, create_time);

-- Web 智能体会话和可重放事件
CREATE INDEX IF NOT EXISTS ix_agent_sessions_owner_id ON agent_sessions (owner_id);
CREATE INDEX IF NOT EXISTS ix_agent_sessions_project_id ON agent_sessions (project_id);
CREATE INDEX IF NOT EXISTS ix_agent_sessions_update_time ON agent_sessions (update_time);
CREATE INDEX IF NOT EXISTS ix_agent_runs_session_uid ON agent_runs (session_uid);
CREATE INDEX IF NOT EXISTS ix_agent_runs_project_id ON agent_runs (project_id);
CREATE INDEX IF NOT EXISTS ix_agent_runs_agent_name ON agent_runs (agent_name);
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
CREATE INDEX IF NOT EXISTS ix_agent_run_targets_source_run_uid ON agent_run_targets (source_run_uid);
