-- init llm provider
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (1, 'OpenAI', 'OPENAI', NULL, NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (2, 'Anthropic', 'ANTHROPIC', NULL, NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (3, 'Google', 'GOOGLE', NULL, NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (4, 'Deepseek', 'OPENAI', 'https://api.deepseek.com/v1', NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (5, 'Zhipu', 'OPENAI', 'https://open.bigmodel.cn/api/paas/v4', NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (6, 'OpenRouter', 'OPENAI', 'https://openrouter.ai/api/v1', NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (7, 'AIHubMix', 'OPENAI', 'https://aihubmix.com/v1', NULL, FALSE, TRUE);
SELECT SETVAL('llm_providers_id_seq', (SELECT MAX(id) FROM llm_providers));


-- init llm model
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (1, 1, 'gpt-5.2', 1.75, 14, 0.175, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (2, 1, 'gpt-5-mini', 0.25, 2, 0.025, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (3, 1, 'gpt-5-nano', 0.05, 0.4, 0.005, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (4, 1, 'gpt-4.1', 2, 8, 0.5, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (5, 2, 'claude-opus-4-5', 5, 25, 0.5, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (6, 2, 'claude-sonnet-4-5', 3, 15, 0.3, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (7, 2, 'claude-haiku-4-5', 1, 5, 0.1, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (8, 3, 'gemini-3-pro-preview', 2, 12, 0.2, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (9, 3, 'gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (10, 3, 'gemini-2.5-pro', 1.25, 10, 0.125, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (11, 3, 'gemini-2.5-flash', 0.3, 2.5, 0.03, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (12, 4, 'deepseek-chat', 0.28, 0.42, 0.028, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (13, 4, 'deepseek-reasoner', 0.28, 0.42, 0.028, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (14, 5, 'glm-4.6', 0.28, 3.98, 0.06, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (15, 5, 'glm-4.6v', 0.14, 0.43, 0.028, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (16, 5, 'glm-4.5', 0.28, 3.98, 0.06, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (17, 5, 'glm-4.5-air', 0.11, 0.28, 0.02, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (18, 5, 'glm-4.5-flash', 0, 0, 0, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (19, 6, 'openai/gpt-5.2', 1.75, 14, 0.175, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (20, 6, 'openai/gpt-5.1-codex-max', 1.25, 10, 0.125, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (21, 6, 'openai/gpt-5-mini', 0.25, 2, 0.025, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (22, 6, 'anthropic/claude-opus-4.5', 5, 25, 0.5, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (23, 6, 'anthropic/claude-sonnet-4.5', 3, 15, 0.3, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (24, 6, 'anthropic/claude-haiku-4.5', 1, 5, 0.1, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (25, 6, 'google/gemini-3-pro-preview', 2, 12, 0.2, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (26, 6, 'google/gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (27, 6, 'google/gemini-2.5-pro', 1.25, 10, 0.125, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (28, 6, 'deepseek/deepseek-v3.2', 0.24, 0.38, 0.02, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (29, 6, 'z-ai/glm-4.6', 0.39, 1.9, 1.9, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (30, 7, 'gpt-5.2', 1.75, 14, 0.175, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (31, 7, 'gpt-5.1-codex-max', 1.25, 10, 0.125, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (32, 7, 'gpt-5-mini', 0.25, 2, 0.025, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (33, 7, 'claude-opus-4-5', 5, 25, 0.5, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (34, 7, 'claude-sonnet-4-5', 3.3, 16.5, 0.33, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (35, 7, 'claude-haiku-4-5', 1.1, 5.5, 0.11, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (36, 7, 'gemini-3-pro-preview', 2, 12, 0.2, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (37, 7, 'gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (38, 7, 'gemini-2.5-pro', 1.25, 10, 0.125, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (39, 7, 'deepseek-v3.2', 0.3, 0.45, 0.03, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (40, 7, 'deepseek-v3.2-think', 0.3, 0.45, 0.03, '{}', TRUE);
SELECT SETVAL('llm_models_id_seq', (SELECT MAX(id) FROM llm_models));

-- init model setting
INSERT INTO llm_settings (key, model_id)
VALUES ('chat_model', 13);
INSERT INTO llm_settings (key, model_id)
VALUES ('vision_model', 27);
INSERT INTO llm_settings (key, model_id)
VALUES ('agent_model', 26);
INSERT INTO llm_settings (key, model_id)
VALUES ('coding_model', 20);
INSERT INTO llm_settings (key, model_id)
VALUES ('simple_model', 26);
INSERT INTO llm_settings (key, model_id)
VALUES ('long_context_model', 13);
INSERT INTO llm_settings (key, model_id)
VALUES ('high_performance_model', 25);