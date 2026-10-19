import re
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class ModelPrice(BaseModel):
    """LLM token prices in micro-units per token (equivalent to USD per million tokens)."""

    model_config = ConfigDict(frozen=True)

    input: Decimal
    output: Decimal
    cache_read: Decimal


class ModelPriceRule(BaseModel):
    """A case-insensitive model-name pattern and its default prices."""

    model_config = ConfigDict(frozen=True)

    pattern: str
    price: ModelPrice

    def matches(self, model_name: str) -> bool:
        """Return whether the complete model name matches this rule."""
        return re.fullmatch(self.pattern, model_name, flags=re.IGNORECASE) is not None


def _price(input_price: str, output_price: str, cache_read_price: str) -> ModelPrice:
    return ModelPrice(input=Decimal(input_price), output=Decimal(output_price), cache_read=Decimal(cache_read_price))


# Keep more specific model variants before their parent families.
MODEL_PRICE_RULES: tuple[ModelPriceRule, ...] = (
    ModelPriceRule(pattern=r'gpt-5\.4-mini(?:-\d{4}-\d{2}-\d{2})?', price=_price('0.75', '4.5', '0.075')),
    ModelPriceRule(pattern=r'gpt-5\.4-nano(?:-\d{4}-\d{2}-\d{2})?', price=_price('0.2', '1.25', '0.02')),
    ModelPriceRule(pattern=r'gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?', price=_price('2.5', '15', '0.25')),
    ModelPriceRule(pattern=r'gpt-5-nano(?:-\d{4}-\d{2}-\d{2})?', price=_price('0.05', '0.4', '0.005')),
    ModelPriceRule(pattern=r'gpt-5\.5(?:-\d{4}-\d{2}-\d{2})?', price=_price('5', '30', '0.5')),
    ModelPriceRule(pattern=r'gpt-5\.6-sol(?:-\d{4}-\d{2}-\d{2})?', price=_price('4', '20', '0.4')),
    ModelPriceRule(pattern=r'gpt-5\.6-terra(?:-\d{4}-\d{2}-\d{2})?', price=_price('2', '12', '0.2')),
    ModelPriceRule(pattern=r'gpt-5\.6-luna(?:-\d{4}-\d{2}-\d{2})?', price=_price('0.2', '1.2', '0.02')),
    ModelPriceRule(pattern=r'claude-opus-4[.-]6', price=_price('5', '25', '0.5')),
    ModelPriceRule(pattern=r'claude-sonnet-4[.-]6', price=_price('3', '15', '0.3')),
    ModelPriceRule(pattern=r'claude-haiku-4[.-]5', price=_price('1', '5', '0.1')),
    ModelPriceRule(pattern=r'gemini-3\.1-pro-preview', price=_price('2', '12', '0.2')),
    ModelPriceRule(pattern=r'gemini-3\.1-flash-lite-preview', price=_price('0.25', '1.5', '0.025')),
    ModelPriceRule(pattern=r'gemini-3-flash-preview', price=_price('0.5', '3', '0.05')),
    ModelPriceRule(pattern=r'gemini-3\.5-flash-lite', price=_price('0.3', '2.5', '0.03')),
    ModelPriceRule(pattern=r'gemini-3\.5-flash', price=_price('1.5', '9', '0.15')),
    ModelPriceRule(pattern=r'gemini-3\.6-flash', price=_price('1.5', '7.5', '0.15')),
    ModelPriceRule(pattern=r'deepseek-v4-pro', price=_price('1.32', '3.96', '0.044')),
    ModelPriceRule(pattern=r'deepseek-v4-flash', price=_price('0.44', '1.32', '0.014')),
)


def find_model_price(model_name: str) -> ModelPrice | None:
    """Return the default price for a known model name."""
    normalized_name = model_name.strip().rsplit('/', maxsplit=1)[-1]
    return next((rule.price for rule in MODEL_PRICE_RULES if rule.matches(normalized_name)), None)
