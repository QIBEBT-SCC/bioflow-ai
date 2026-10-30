import re
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class ModelPrice(BaseModel):
    """LLM token prices in micro-units per token (equivalent to USD per million tokens)."""

    model_config = ConfigDict(frozen=True)

    input: Decimal
    output: Decimal
    cache_read: Decimal


class ModelDefaultRule(BaseModel):
    """A case-insensitive model-name pattern and its default prices and context window."""

    model_config = ConfigDict(frozen=True)

    pattern: str
    price: ModelPrice
    context_window: int

    def matches(self, model_name: str) -> bool:
        """Return whether the complete model name matches this rule."""
        return re.fullmatch(self.pattern, model_name, flags=re.IGNORECASE) is not None


def _price(input_price: str, output_price: str, cache_read_price: str) -> ModelPrice:
    return ModelPrice(input=Decimal(input_price), output=Decimal(output_price), cache_read=Decimal(cache_read_price))


# Verified against official catalogs on 2026-10-30. Prices are standard text rates
# for the lowest context tier; service tiers, cache writes and tool fees are separate.
# https://developers.openai.com/api/docs/pricing
# https://platform.claude.com/docs/en/about-claude/pricing
# https://ai.google.dev/gemini-api/docs/pricing
# https://api-docs.deepseek.com/quick_start/pricing/
# Keep more specific model variants before their parent families.
MODEL_DEFAULT_RULES: tuple[ModelDefaultRule, ...] = (
    ModelDefaultRule(
        pattern=r'gpt-6-astra(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('10', '50', '1'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-6\.1-sol(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('2', '10', '0.1'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-6\.1-terra(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('1', '5', '0.05'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-6-sol(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('2', '10', '0.2'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-6-luna(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('0.1', '0.5', '0.01'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.4-mini(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('0.75', '4.5', '0.075'),
        context_window=400_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.4-nano(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('0.2', '1.25', '0.02'),
        context_window=400_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('2.5', '15', '0.25'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5-nano(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('0.05', '0.4', '0.005'),
        context_window=400_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.5(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('5', '30', '0.5'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.6-sol(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('4', '20', '0.4'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.6-terra(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('2', '12', '0.2'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'gpt-5\.6-luna(?:-\d{4}-\d{2}-\d{2})?',
        price=_price('0.2', '1.2', '0.02'),
        context_window=1_050_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-fable-5[.-]1',
        price=_price('10', '50', '0.25'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-mythos-5[.-]1',
        price=_price('10', '50', '0.25'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-opus-5[.-]5',
        price=_price('4', '20', '0.2'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-sonnet-5[.-]5',
        price=_price('2', '10', '0.1'),
        context_window=1_000_000,
    ),
    # Haiku 5.5's base rates apply to prompts up to 100K tokens.
    ModelDefaultRule(
        pattern=r'claude-haiku-5[.-]5',
        price=_price('0.1', '0.5', '0.01'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-fable-5',
        price=_price('10', '50', '1'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-mythos-5',
        price=_price('10', '50', '1'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-opus-5',
        price=_price('5', '25', '0.5'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-sonnet-5',
        price=_price('2', '10', '0.2'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-opus-4[.-][678]',
        price=_price('5', '25', '0.5'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-sonnet-4[.-]6',
        price=_price('3', '15', '0.3'),
        context_window=1_000_000,
    ),
    ModelDefaultRule(
        pattern=r'claude-haiku-4[.-]5(?:-20251001)?',
        price=_price('1', '5', '0.1'),
        context_window=200_000,
    ),
    # Argon introductory rates expire after 2026-11-30 (then 4 / 20 / 0.2).
    ModelDefaultRule(
        pattern=r'gemini-4-argon',
        price=_price('2', '10', '0.1'),
        context_window=1_048_576,
    ),
    # Flash 3.8/3.6 introductory rates expire after 2026-12-31 (then 1.5 / 7.5 / 0.15).
    # Gemini 3.7 Flash routes to 3.8; Gemini 3.5 Flash routes to 3.6.
    ModelDefaultRule(
        pattern=r'gemini-3\.[678]-flash',
        price=_price('0.75', '3.75', '0.075'),
        context_window=1_048_576,
    ),
    ModelDefaultRule(
        pattern=r'gemini-3\.1-pro-preview',
        price=_price('2', '12', '0.2'),
        context_window=1_048_576,
    ),
    ModelDefaultRule(
        pattern=r'gemini-3\.1-flash-lite',
        price=_price('0.25', '1.5', '0.025'),
        context_window=1_048_576,
    ),
    ModelDefaultRule(
        pattern=r'gemini-3-flash-preview',
        price=_price('0.5', '3', '0.05'),
        context_window=1_048_576,
    ),
    ModelDefaultRule(
        pattern=r'gemini-3\.5-flash-lite',
        price=_price('0.3', '2.5', '0.03'),
        context_window=1_048_576,
    ),
    ModelDefaultRule(
        pattern=r'gemini-3\.5-flash',
        price=_price('0.75', '3.75', '0.075'),
        context_window=1_048_576,
    ),
    # DeepSeek's peak rates; off-peak requests cost half as much.
    ModelDefaultRule(
        pattern=r'deepseek-v4-pro',
        price=_price('1.32', '3.96', '0.044'),
        context_window=1_048_576,
    ),
    ModelDefaultRule(
        pattern=r'deepseek-flash',
        price=_price('0.3', '1.2', '0.006'),
        context_window=1_048_576,
    ),
)


def find_model_defaults(model_name: str) -> ModelDefaultRule | None:
    """Return the catalog entry for a known model name."""
    normalized_name = model_name.strip().rsplit('/', maxsplit=1)[-1]
    return next((rule for rule in MODEL_DEFAULT_RULES if rule.matches(normalized_name)), None)
