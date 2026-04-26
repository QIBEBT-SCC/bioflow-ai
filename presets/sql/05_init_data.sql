-- init llm provider
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (1, 'OpenAI', 'OPENAI', NULL, NULL, FALSE, FALSE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (2, 'Anthropic', 'ANTHROPIC', NULL, NULL, FALSE, FALSE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (3, 'Google', 'GOOGLE', NULL, NULL, FALSE, FALSE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (4, 'Deepseek', 'OPENAI', 'https://api.deepseek.com/v1', NULL, FALSE, TRUE);
INSERT INTO llm_providers (id, name, provider_type, base_url, api_key, use_proxy, is_active)
VALUES (5, 'OpenRouter', 'OPENAI', 'https://openrouter.ai/api/v1', NULL, FALSE, TRUE);
SELECT SETVAL('llm_providers_id_seq', (SELECT MAX(id) FROM llm_providers));


-- init llm model
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (1, 1, 'gpt-5.4', 2.5, 15, 0.25, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (2, 1, 'gpt-5.4-mini', 0.75, 4.5, 0.075, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (3, 1, 'gpt-5.4-nano', 0.2, 1.25, 0.02, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (4, 1, 'gpt-5-nano', 0.05, 0.4, 0.005, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (5, 2, 'claude-opus-4-6', 5, 25, 0.5, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (6, 2, 'claude-sonnet-4-6', 3, 15, 0.3, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (7, 2, 'claude-haiku-4-5', 1, 5, 0.1, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (8, 3, 'gemini-3.1-pro-preview', 2, 12, 0.2, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (9, 3, 'gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (10, 3, 'gemini-3.1-flash-lite-preview', 0.25, 1.5, 0.025, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (11, 4, 'deepseek-v4-pro', 0.435, 0.87, 0.03625, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (12, 4, 'deepseek-v4-flash', 0.14, 0.28, 0.028, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (13, 4, 'deepseek-v4-flash(no-think)', 0.14, 0.28, 0.028, '{"thinking": {"type": "disabled"}}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (14, 5, 'openai/gpt-5.4', 2.5, 15, 0.25, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (15, 5, 'openai/gpt-5.4-mini', 0.75, 4.5, 0.075, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (16, 5, 'openai/gpt-5.4-nano', 0.2, 1.25, 0.02, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (17, 5, 'openai/gpt-5-nano', 0.05, 0.4, 0.005, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (18, 5, 'anthropic/claude-opus-4.6', 5, 25, 0.5, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (19, 5, 'anthropic/claude-sonnet-4.6', 3, 15, 0.3, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (20, 5, 'anthropic/claude-haiku-4.5', 1, 5, 0.1, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (21, 5, 'google/gemini-3.1-pro-preview', 2, 12, 0.2, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (22, 5, 'google/gemini-3-flash-preview', 0.5, 3, 0.05, '{}', TRUE);
INSERT INTO llm_models (id, provider_id, name, input_price, output_price, cache_read_price, extra_body, is_active)
VALUES (23, 5, 'google/gemini-3.1-flash-lite-preview', 0.25, 1.5, 0.025, '{}', TRUE);
SELECT SETVAL('llm_models_id_seq', (SELECT MAX(id) FROM llm_models));

-- init model setting
INSERT INTO llm_settings (id, chat_model_id, vision_model_id, agent_model_id, coding_model_id, simple_model_id, long_context_model_id, high_performance_model_id)
VALUES (1, 8, 9, 11, 6, 13, 12, 1);