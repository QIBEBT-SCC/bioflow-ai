CREATE INDEX idx_images_name_trgm ON images USING gin (name gin_trgm_ops);
CREATE INDEX idx_tools_name_trgm ON tools USING gin (name gin_trgm_ops);
CREATE INDEX idx_bio_dbs_name_trgm ON bio_dbs USING gin (name gin_trgm_ops);