-- init llm provider
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (1, 'OpenAI', 'OPENAI', NULL, NULL, FALSE, FALSE),
       (2, 'Anthropic', 'ANTHROPIC', NULL, NULL, FALSE, FALSE),
       (3, 'Google', 'GOOGLE', NULL, NULL, FALSE, FALSE),
       (4, 'Deepseek', 'OPENAI', 'https://api.deepseek.com/v1', NULL, FALSE, TRUE),
       (5, 'OpenRouter', 'OPENAI', 'https://openrouter.ai/api/v1', NULL, FALSE, TRUE);
SELECT SETVAL('llm_providers_id_seq', (SELECT MAX(id) FROM llm_providers));


-- init llm model
INSERT INTO llm_models (id, provider_id, display_name, name, input_price, output_price, cache_read_price, is_active)
VALUES (1, 1, 'GPT-5.6 Sol', 'gpt-5.6-sol', 5, 30, 0.5, TRUE),
       (2, 1, 'GPT-5.6 Terra', 'gpt-5.6-terra', 2.5, 15, 0.25, TRUE),
       (3, 1, 'GPT-5.6 Luna', 'gpt-5.6-luna', 1, 6, 0.1, TRUE),
       (4, 1, 'GPT-5 Nano', 'gpt-5-nano', 0.05, 0.4, 0.005, TRUE),
       (5, 2, 'Claude Opus 4.6', 'claude-opus-4-6', 5, 25, 0.5, TRUE),
       (6, 2, 'Claude Sonnet 4.6', 'claude-sonnet-4-6', 3, 15, 0.3, TRUE),
       (7, 2, 'Claude Haiku 4.5', 'claude-haiku-4-5', 1, 5, 0.1, TRUE),
       (8, 3, 'Gemini 3.5 Flash', 'gemini-3.5-flash', 1.5, 9, 0.15, TRUE),
       (9, 3, 'Gemini 3.5 Flash-Lite', 'gemini-3.5-flash-lite', 0.3, 2.5, 0.03, TRUE),
       (11, 4, 'DeepSeek V4 Pro', 'deepseek-v4-pro', 0.435, 0.87, 0.03625, TRUE),
       (12, 4, 'DeepSeek V4 Flash', 'deepseek-v4-flash', 0.14, 0.28, 0.028, TRUE),
       (13, 4, 'DeepSeek V4 Flash (Non-thinking)', 'deepseek-v4-flash', 0.14, 0.28, 0.028, TRUE),
       (14, 5, 'GPT-5.6 Sol', 'openai/gpt-5.6-sol', 5, 30, 0.5, TRUE),
       (15, 5, 'GPT-5.6 Terra', 'openai/gpt-5.6-terra', 2.5, 15, 0.25, TRUE),
       (16, 5, 'GPT-5.6 Luna', 'openai/gpt-5.6-luna', 1, 6, 0.1, TRUE),
       (17, 5, 'GPT-5 Nano', 'openai/gpt-5-nano', 0.05, 0.4, 0.005, TRUE),
       (18, 5, 'Claude Opus 4.6', 'anthropic/claude-opus-4.6', 5, 25, 0.5, TRUE),
       (19, 5, 'Claude Sonnet 4.6', 'anthropic/claude-sonnet-4.6', 3, 15, 0.3, TRUE),
       (20, 5, 'Claude Haiku 4.5', 'anthropic/claude-haiku-4.5', 1, 5, 0.1, TRUE),
       (21, 5, 'Gemini 3.5 Flash', 'google/gemini-3.5-flash', 1.5, 9, 0.15, TRUE),
       (22, 5, 'Gemini 3.5 Flash-Lite', 'google/gemini-3.5-flash-lite', 0.3, 2.5, 0.03, TRUE);
UPDATE llm_models SET reasoning_effort = 'low' WHERE id IN (1, 14);
UPDATE llm_models SET reasoning_effort = 'medium' WHERE id IN (2, 15);
UPDATE llm_models SET reasoning_effort = 'none' WHERE id = 13;
SELECT SETVAL('llm_models_id_seq', (SELECT MAX(id) FROM llm_models));

-- init model setting
INSERT INTO llm_settings (id, chat_model_id, vision_model_id, agent_model_id, coding_model_id, simple_model_id, long_context_model_id, high_performance_model_id)
VALUES (1, 12, 8, 2, 6, 13, 12, 1);

-- init review-status and classification tags
-- Function and Domain are independent dimensions; each tool needs both.
INSERT INTO tool_tags (id, name)
VALUES (1, 'Default'),
       (2, 'AI Checked'),
       (3, 'AI Unchecked'),
       -- Function tags
       (4, 'Data Processing'),
       (5, 'Reference Setup'),
       (6, 'Quality Control'),
       (7, 'Alignment'),
       (8, 'Assembly'),
       (9, 'Variant Analysis'),
       (10, 'Annotation'),
       (11, 'Quantification'),
       (12, 'Feature Detection'),
       (13, 'Statistical Analysis'),
       (14, 'Clustering'),
       (15, 'Taxonomic Profiling'),
       (16, 'Phylogenetics'),
       (17, 'Structure Analysis'),
       (18, 'Visualization'),
       (19, 'Reporting'),
       (20, 'Utilities'),
       -- Domain tags
       (21, 'General'),
       (22, 'Genomics'),
       (23, 'Transcriptomics'),
       (24, 'Epigenomics'),
       (25, 'Genome Architecture'),
       (26, 'Metagenomics'),
       (27, 'Population Genetics'),
       (28, 'Evolution'),
       (29, 'Proteomics'),
       (30, 'Metabolomics'),
       (31, 'Structural Biology'),
       (32, 'Immunology'),
       (33, 'Single Cell'),
       (34, 'Cell Imaging'),
       (35, 'Pharmacology');
SELECT SETVAL('tool_tags_id_seq', (SELECT MAX(id) FROM tool_tags));

-- Flat primary Function groups. Domains and additional functions are tags only.
INSERT INTO tool_groups (id, name, parent_id, tag_id)
VALUES (1, 'Data Processing', NULL, 4),
       (2, 'Reference Setup', NULL, 5),
       (3, 'Quality Control', NULL, 6),
       (4, 'Alignment', NULL, 7),
       (5, 'Assembly', NULL, 8),
       (6, 'Variant Analysis', NULL, 9),
       (7, 'Annotation', NULL, 10),
       (8, 'Quantification', NULL, 11),
       (9, 'Feature Detection', NULL, 12),
       (10, 'Statistical Analysis', NULL, 13),
       (11, 'Clustering', NULL, 14),
       (12, 'Taxonomic Profiling', NULL, 15),
       (13, 'Phylogenetics', NULL, 16),
       (14, 'Structure Analysis', NULL, 17),
       (15, 'Visualization', NULL, 18),
       (16, 'Reporting', NULL, 19),
       (17, 'Utilities', NULL, 20);
SELECT SETVAL('tool_groups_id_seq', (SELECT MAX(id) FROM tool_groups));
