from app.models.setting import LLMProvider, ProviderType
from app.utils.init.providers import ProviderConfiguration, apply_provider_configurations, validate_provider_selection


def test_provider_selection_requires_at_least_one_provider() -> None:
    assert validate_provider_selection([]) == '请至少选择一个模型 Provider。'
    assert validate_provider_selection(['OpenAI']) is True


def test_apply_provider_configurations_enables_selected_and_disables_others() -> None:
    openai = LLMProvider(
        id=1,
        name='OpenAI',
        provider_type=ProviderType.OPENAI,
        api_key=None,
        is_active=False,
    )
    deepseek = LLMProvider(
        id=2,
        name='Deepseek',
        provider_type=ProviderType.OPENAI,
        api_key='old-key',
        is_active=True,
    )

    apply_provider_configurations(
        [openai, deepseek],
        [
            ProviderConfiguration(
                name='OpenAI',
                api_key='new-key',
                base_url=None,
                use_proxy=True,
            )
        ],
    )

    assert openai.is_active is True
    assert openai.api_key == 'new-key'
    assert openai.base_url is None
    assert openai.use_proxy is True
    assert deepseek.is_active is False
    assert deepseek.api_key == 'old-key'
