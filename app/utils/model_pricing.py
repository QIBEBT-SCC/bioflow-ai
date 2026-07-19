import re
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """LLM token prices in micro-units per token (equivalent to USD per million tokens)."""

    input: Decimal
    output: Decimal
    cache_read: Decimal


@dataclass(frozen=True, slots=True)
class ModelPriceRule:
    """A case-insensitive model-name pattern and its default prices."""

    pattern: str
    price: ModelPrice

    def matches(self, model_name: str) -> bool:
        """Return whether the complete model name matches this rule."""
        return re.fullmatch(self.pattern, model_name, flags=re.IGNORECASE) is not None


def _price(input_price: str, output_price: str, cache_read_price: str) -> ModelPrice:
    return ModelPrice(Decimal(input_price), Decimal(output_price), Decimal(cache_read_price))


# Keep more specific model variants before their parent families.
MODEL_PRICE_RULES: tuple[ModelPriceRule, ...] = (
    ModelPriceRule(r'gpt-5\.4-mini(?:-\d{4}-\d{2}-\d{2})?', _price('0.75', '4.5', '0.075')),
    ModelPriceRule(r'gpt-5\.4-nano(?:-\d{4}-\d{2}-\d{2})?', _price('0.2', '1.25', '0.02')),
    ModelPriceRule(r'gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?', _price('2.5', '15', '0.25')),
    ModelPriceRule(r'gpt-5-nano(?:-\d{4}-\d{2}-\d{2})?', _price('0.05', '0.4', '0.005')),
    ModelPriceRule(r'claude-opus-4[.-]6', _price('5', '25', '0.5')),
    ModelPriceRule(r'claude-sonnet-4[.-]6', _price('3', '15', '0.3')),
    ModelPriceRule(r'claude-haiku-4[.-]5', _price('1', '5', '0.1')),
    ModelPriceRule(r'gemini-3\.1-pro-preview', _price('2', '12', '0.2')),
    ModelPriceRule(r'gemini-3\.1-flash-lite-preview', _price('0.25', '1.5', '0.025')),
    ModelPriceRule(r'gemini-3-flash-preview', _price('0.5', '3', '0.05')),
    ModelPriceRule(r'deepseek-v4-pro', _price('0.435', '0.87', '0.03625')),
    ModelPriceRule(r'deepseek-v4-flash', _price('0.14', '0.28', '0.028')),
)


def find_model_price(model_name: str) -> ModelPrice | None:
    """Return the default price for a known model name."""
    normalized_name = model_name.strip().rsplit('/', maxsplit=1)[-1]
    return next((rule.price for rule in MODEL_PRICE_RULES if rule.matches(normalized_name)), None)
