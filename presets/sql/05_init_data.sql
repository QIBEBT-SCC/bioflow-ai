-- init llm provider
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (1, 'OpenAI', 'OPENAI', NULL, NULL, FALSE, FALSE),
       (2, 'Anthropic', 'ANTHROPIC', NULL, NULL, FALSE, FALSE),
       (3, 'Google', 'GOOGLE', NULL, NULL, FALSE, FALSE),
       (4, 'Deepseek', 'OPENAI', 'https://api.deepseek.com/v1', NULL, FALSE, TRUE),
       (5, 'OpenRouter', 'OPENAI', 'https://openrouter.ai/api/v1', NULL, FALSE, TRUE);
SELECT SETVAL('llm_providers_id_seq', (SELECT MAX(id) FROM llm_providers));


-- init llm model
INSERT INTO llm_models (id, provider_id, display_name, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (1, 1, 'GPT-5.4', 'gpt-5.4', 2.5, 15, 0.25, '{}', TRUE),
       (2, 1, 'GPT-5.4 Mini', 'gpt-5.4-mini', 0.75, 4.5, 0.075, '{}', TRUE),
       (3, 1, 'GPT-5.4 Nano', 'gpt-5.4-nano', 0.2, 1.25, 0.02, '{}', TRUE),
       (4, 1, 'GPT-5 Nano', 'gpt-5-nano', 0.05, 0.4, 0.005, '{}', TRUE),
       (5, 2, 'Claude Opus 4.6', 'claude-opus-4-6', 5, 25, 0.5, '{}', TRUE),
       (6, 2, 'Claude Sonnet 4.6', 'claude-sonnet-4-6', 3, 15, 0.3, '{}', TRUE),
       (7, 2, 'Claude Haiku 4.5', 'claude-haiku-4-5', 1, 5, 0.1, '{}', TRUE),
       (8, 3, 'Gemini 3.1 Pro Preview', 'gemini-3.1-pro-preview', 2, 12, 0.2, '{}', TRUE),
       (9, 3, 'Gemini 3 Flash Preview', 'gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE),
       (10, 3, 'Gemini 3.1 Flash-Lite Preview', 'gemini-3.1-flash-lite-preview', 0.25, 1.5, 0.025, '{}', TRUE),
       (11, 4, 'DeepSeek V4 Pro', 'deepseek-v4-pro', 0.435, 0.87, 0.03625, '{}', TRUE),
       (12, 4, 'DeepSeek V4 Flash', 'deepseek-v4-flash', 0.14, 0.28, 0.028, '{}', TRUE),
       (13, 4, 'DeepSeek V4 Flash (Non-thinking)', 'deepseek-v4-flash', 0.14, 0.28, 0.028, '{
         "thinking": {
           "type": "disabled"
         }
       }', TRUE),
       (14, 5, 'GPT-5.4', 'openai/gpt-5.4', 2.5, 15, 0.25, '{}', TRUE),
       (15, 5, 'GPT-5.4 Mini', 'openai/gpt-5.4-mini', 0.75, 4.5, 0.075, '{}', TRUE),
       (16, 5, 'GPT-5.4 Nano', 'openai/gpt-5.4-nano', 0.2, 1.25, 0.02, '{}', TRUE),
       (17, 5, 'GPT-5 Nano', 'openai/gpt-5-nano', 0.05, 0.4, 0.005, '{}', TRUE),
       (18, 5, 'Claude Opus 4.6', 'anthropic/claude-opus-4.6', 5, 25, 0.5, '{}', TRUE),
       (19, 5, 'Claude Sonnet 4.6', 'anthropic/claude-sonnet-4.6', 3, 15, 0.3, '{}', TRUE),
       (20, 5, 'Claude Haiku 4.5', 'anthropic/claude-haiku-4.5', 1, 5, 0.1, '{}', TRUE),
       (21, 5, 'Gemini 3.1 Pro Preview', 'google/gemini-3.1-pro-preview', 2, 12, 0.2, '{}', TRUE),
       (22, 5, 'Gemini 3 Flash Preview', 'google/gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE),
       (23, 5, 'Gemini 3.1 Flash-Lite Preview', 'google/gemini-3.1-flash-lite-preview', 0.25, 1.5, 0.025, '{}', TRUE);
SELECT SETVAL('llm_models_id_seq', (SELECT MAX(id) FROM llm_models));

-- init model setting
INSERT INTO llm_settings (id, chat_model_id, vision_model_id, agent_model_id, coding_model_id, simple_model_id, long_context_model_id, high_performance_model_id)
VALUES (1, 12, 9, 11, 6, 13, 12, 1);

-- init tool tags
INSERT INTO tool_tags (id, name)
VALUES (1, 'Default'),
       (2, 'AI Checked'),
       (3, 'AI Unchecked'),
       (4, 'Data Handling & QC'),
       (5, 'Core Sequence Analysis'),
       (6, 'Downstream Interpretation'),
       (7, 'Specialized Analyses'),
       (8, 'Visualization'),
       (9, 'Reporting'),
       (10, 'Data Manipulation'),
       (11, 'Quality Control'),
       (12, 'Alignment & Mapping'),
       (13, 'Assembly & Polishing'),
       (14, 'Variant Calling'),
       (15, 'Annotation'),
       (16, 'Quantification'),
       (17, 'Differential Expression'),
       (18, 'Functional Enrichment'),
       (19, 'Single-cell'),
       (20, 'Metagenomics'),
       (21, 'Phylogenetics'),
       (22, 'Structural Biology');

-- init tool group
INSERT INTO tool_groups (id, name, parent_id, tag_id)
VALUES (1, 'Data Handling & QC', NULL, 4),
       (2, 'Core Sequence Analysis', NULL, 5),
       (3, 'Downstream Interpretation', NULL, 6),
       (4, 'Specialized Analyses', NULL, 7),
       (5, 'Visualization', NULL, 8),
       (6, 'Reporting', NULL, 9),
       (7, 'Data Manipulation', 1, 10),
       (8, 'Quality Control', 1, 11),
       (9, 'Alignment & Mapping', 2, 12),
       (10, 'Assembly & Polishing', 2, 13),
       (11, 'Variant Calling', 2, 14),
       (12, 'Annotation', 2, 15),
       (13, 'Quantification', 3, 16),
       (14, 'Differential Expression', 3, 17),
       (15, 'Functional Enrichment', 3, 18),
       (16, 'Single-cell', 4, 19),
       (17, 'Metagenomics', 4, 20),
       (18, 'Phylogenetics', 4, 21),
       (19, 'Structural Biology', 4, 22);
