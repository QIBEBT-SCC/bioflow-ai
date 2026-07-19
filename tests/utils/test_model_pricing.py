from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.setting import LLMModelCreate
from app.utils.model_pricing import ModelPrice, find_model_price


def test_find_model_price_matches_model_name_independently_of_prefix() -> None:
    assert find_model_price('proxy/openai/gpt-5.4-mini') == ModelPrice(
        input=Decimal('0.75'), output=Decimal('4.5'), cache_read=Decimal('0.075')
    )
    assert find_model_price('ANTHROPIC/CLAUDE-SONNET-4.6') == ModelPrice(
        input=Decimal(3), output=Decimal(15), cache_read=Decimal('0.3')
    )


def test_model_create_populates_known_model_prices() -> None:
    model = LLMModelCreate(provider_id=1, display_name='GPT-5.4 Nano', name='gpt-5.4-nano')

    assert model.display_name == 'GPT-5.4 Nano'
    assert (model.input_price, model.output_price, model.cache_read_price) == (0.2, 1.25, 0.02)


def test_model_create_requires_display_name() -> None:
    with pytest.raises(ValidationError, match='display_name'):
        LLMModelCreate(provider_id=1, name='gpt-5.4-nano')


def test_model_create_preserves_explicit_price_override() -> None:
    model = LLMModelCreate(
        provider_id=1,
        display_name='Gemini 3 Flash Preview',
        name='google/gemini-3-flash-preview',
        input_price=9,
    )

    assert (model.input_price, model.output_price, model.cache_read_price) == (9, 3, 0.05)


def test_unknown_model_requires_all_prices() -> None:
    with pytest.raises(ValidationError, match='No default price found'):
        LLMModelCreate(provider_id=1, display_name='Custom Model', name='custom-model')


def test_unknown_model_accepts_user_defined_prices() -> None:
    model = LLMModelCreate(
        provider_id=1,
        display_name='Custom Model',
        name='custom-model',
        input_price=1,
        output_price=2,
        cache_read_price=0.1,
    )

    assert model.input_price == 1
