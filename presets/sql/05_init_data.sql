-- init llm provider
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (1, 'OpenAI', 'OPENAI', NULL, NULL, FALSE, FALSE),
       (2, 'Anthropic', 'ANTHROPIC', NULL, NULL, FALSE, FALSE),
       (3, 'Google', 'GOOGLE', NULL, NULL, FALSE, FALSE),
       (4, 'DeepSeek', 'OPENAI', 'https://api.deepseek.com/v1', NULL, FALSE, FALSE),
       (5, 'OpenRouter', 'OPENAI', 'https://openrouter.ai/api/v1', NULL, FALSE, FALSE);
SELECT SETVAL('llm_providers_id_seq', (SELECT MAX(id) FROM llm_providers));


-- init current model families (verified 2026-11-24)
-- Prices are USD per million tokens at the standard text/base context tier.
-- Direct-provider prices and context windows follow src/bioflow_ai/llm/pricing.py.
-- Gemini 4 Carbon introductory rates end 2026-11-30; Flash rates end 2026-12-31.
-- Haiku 5.5 base rates apply up to 100K input tokens; DeepSeek uses peak rates.
-- OpenRouter prices follow https://openrouter.ai/api/v1/models and can differ from direct-provider rates.
INSERT INTO llm_models (
    id, provider_id, display_name, name, input_price, output_price, cache_read_price,
    context_window, reasoning_effort, is_active
)
VALUES (1, 1, 'GPT-6.2 Astra', 'gpt-6.2-astra', 8, 40, 0.4, 1050000, NULL, TRUE),
       (2, 1, 'GPT-6.2 Sol', 'gpt-6.2-sol', 2, 10, 0.1, 1050000, NULL, TRUE),
       (3, 1, 'GPT-6.1 Terra', 'gpt-6.1-terra', 1, 5, 0.05, 1050000, NULL, TRUE),
       (4, 1, 'GPT-6 Luna', 'gpt-6-luna', 0.1, 0.5, 0.01, 1050000, NULL, TRUE),
       (5, 2, 'Claude Fable 5.2', 'claude-fable-5-2', 8, 40, 0.2, 1000000, NULL, TRUE),
       (6, 2, 'Claude Opus 5.5', 'claude-opus-5-5', 4, 20, 0.2, 1000000, NULL, TRUE),
       (7, 2, 'Claude Sonnet 5.5', 'claude-sonnet-5-5', 2, 10, 0.1, 1000000, NULL, TRUE),
       (8, 2, 'Claude Haiku 5.5', 'claude-haiku-5-5', 0.1, 0.5, 0.01, 1000000, NULL, TRUE),
       (9, 3, 'Gemini 4 Carbon', 'gemini-4-carbon', 2, 10, 0.1, 1048576, NULL, TRUE),
       (10, 3, 'Gemini 4 Neon', 'gemini-4-neon', 3, 15, 0.15, 1048576, NULL, TRUE),
       (11, 3, 'Gemini 4 Hydrogen', 'gemini-4-hydrogen', 1.5, 7.5, 0.075, 1048576, NULL, TRUE),
       (12, 3, 'Gemini 3.8 Flash', 'gemini-3.8-flash', 0.75, 3.75, 0.075, 1048576, NULL, TRUE),
       (13, 3, 'Gemini 3.5 Flash-Lite', 'gemini-3.5-flash-lite', 0.3, 2.5, 0.03, 1048576, NULL, TRUE),
       (14, 4, 'DeepSeek V4 Pro', 'deepseek-v4-pro', 1.32, 3.96, 0.044, 1048576, NULL, TRUE),
       (15, 4, 'DeepSeek V4.1 Flash', 'deepseek-flash', 0.3, 1.2, 0.006, 1048576, NULL, TRUE),
       (16, 5, 'GPT-6.2 Astra', 'openai/gpt-6.2-astra', 8, 40, 0.4, 1050000, NULL, TRUE),
       (17, 5, 'GPT-6.2 Sol', 'openai/gpt-6.2-sol', 2, 10, 0.1, 1050000, NULL, TRUE),
       (18, 5, 'GPT-6.1 Terra', 'openai/gpt-6.1-terra', 1, 5, 0.05, 1050000, NULL, TRUE),
       (19, 5, 'GPT-6 Luna', 'openai/gpt-6-luna', 0.1, 0.5, 0.01, 1050000, NULL, TRUE),
       (20, 5, 'Claude Fable 5.2', 'anthropic/claude-fable-5.2', 8, 40, 0.2, 1000000, NULL, TRUE),
       (21, 5, 'Claude Opus 5.5', 'anthropic/claude-opus-5.5', 4, 20, 0.2, 1000000, NULL, TRUE),
       (22, 5, 'Claude Sonnet 5.5', 'anthropic/claude-sonnet-5.5', 2, 10, 0.1, 1000000, NULL, TRUE),
       (23, 5, 'Claude Haiku 5.5', 'anthropic/claude-haiku-5.5', 0.1, 0.5, 0.01, 1000000, NULL, TRUE),
       (24, 5, 'Gemini 4 Carbon', 'google/gemini-4-carbon', 2, 10, 0.1, 1048576, NULL, TRUE),
       (25, 5, 'Gemini 4 Neon', 'google/gemini-4-neon', 3, 15, 0.15, 1048576, NULL, TRUE),
       (26, 5, 'Gemini 4 Hydrogen', 'google/gemini-4-hydrogen', 1.5, 7.5, 0.075, 1048576, NULL, TRUE),
       (27, 5, 'Gemini 3.8 Flash', 'google/gemini-3.8-flash', 0.825, 4.125, 0.0825, 1048576, NULL, TRUE),
       (28, 5, 'Gemini 3.5 Flash-Lite', 'google/gemini-3.5-flash-lite', 0.3, 2.5, 0.03, 1048576, NULL, TRUE);
SELECT SETVAL('llm_models_id_seq', (SELECT MAX(id) FROM llm_models));

-- init model setting
INSERT INTO llm_settings (id, chat_model_id, vision_model_id, agent_model_id, coding_model_id, simple_model_id, long_context_model_id, high_performance_model_id)
VALUES (1, 15, 12, 2, 7, 15, 15, 1);

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
