-- pg_trgm 索引
CREATE INDEX idx_images_name_trgm ON images USING gin (name gin_trgm_ops);
CREATE INDEX idx_tools_name_trgm ON tools USING gin (name gin_trgm_ops);
CREATE INDEX idx_bio_dbs_name_trgm ON bio_dbs USING gin (name gin_trgm_ops);
CREATE INDEX idx_reference_name_trgm ON reference_genomes USING gin (name gin_trgm_ops);

-- pg_vector 索引
CREATE INDEX ON tools USING hnsw (des_vector vector_l2_ops) WITH (m=16,ef_construction = 64);