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
