from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qwen3 import Qwen3Embeddings
from pydantic import SecretStr
from sqlmodel import Session

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_setting as setting_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.setting import ProviderType
from llm.llm_client import ChatDeepseekReasoning, ChatOpenAIReasoning
from llm.utils.cache import cache_model


def _get_model_from_db(model_id: int) -> BaseChatModel:
    with Session(engine) as session:
        model_config = model_crud.get(session, model_id)
        assert model_config.provider.api_key is not None, 'No available api_key has been set yet!'
        proxy = get_settings().network.get_proxy_url() if model_config.provider.use_proxy else None

        match model_config.provider.provider_type:
            case ProviderType.ANTHROPIC:
                model = ChatAnthropic(
                    model=model_config.name,
                    api_key=SecretStr(model_config.provider.api_key),
                    anthropic_proxy=proxy,
                )
            case ProviderType.GOOGLE:
                client_args = {'proxy': proxy} if proxy else None
                model = ChatGoogleGenerativeAI(
                    model=model_config.name,
                    api_key=SecretStr(model_config.provider.api_key),
                    client_args=client_args,
                )
            case ProviderType.OPENAI:
                if model_config.provider.base_url is None:
                    model = ChatOpenAI(
                        model=model_config.name,
                        api_key=SecretStr(model_config.provider.api_key),
                        openai_proxy=proxy,
                    )
                elif 'deepseek' in model_config.name.lower():
                    model = ChatDeepseekReasoning(
                        model=model_config.name,
                        base_url=model_config.provider.base_url,
                        api_key=SecretStr(model_config.provider.api_key),
                        openai_proxy=proxy,
                    )
                else:
                    model = ChatOpenAIReasoning(
                        model=model_config.name,
                        base_url=model_config.provider.base_url,
                        api_key=SecretStr(model_config.provider.api_key),
                        openai_proxy=proxy,
                    )
            case _:
                raise ValueError('invalid provider type')

        model.metadata = {
            'model_id': model_id,
            'input_price': model_config.input_price,
            'output_price': model_config.output_price,
            'cache_read_price': model_config.cache_read_price,
        }
        return model


def get_agent_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'agent_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'agent_model'
    return model


def get_chat_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'chat_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'chat_model'
    return model


def get_vision_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'vision_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'vision_model'
    return model


def get_coding_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'coding_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'coding_model'
    return model


def get_simple_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'simple_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'simple_model'
    return model


def get_long_context_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'long_context_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'long_context_model'
    return model


def get_high_performance_model() -> BaseChatModel:
    with Session(engine) as session:
        setting = setting_crud.get(session, 'high_performance_model')
        model_id = setting.model_id

    model = _get_model_from_db(model_id)
    model.metadata['setting_key'] = 'high_performance_model'
    return model


@cache_model()
def _local_embedding_model() -> Qwen3Embeddings:
    """有缓存的本地向量模型."""
    return Qwen3Embeddings(
        model=get_settings().knowledge_base.local.MODEL,
        use_modelscope=get_settings().knowledge_base.local.USE_MODELSCOPE,
        device=get_settings().knowledge_base.local.DEVICE,
    )


def qwen3_embedding() -> Embeddings:
    """加载向量模型."""
    if get_settings().knowledge_base.USE_LOCAL_MODEL:
        embedding = _local_embedding_model()
    else:
        embedding = OpenAIEmbeddings(
            model=get_settings().knowledge_base.remote.MODEL,
            base_url=get_settings().knowledge_base.remote.BASE_URL,
            api_key=get_settings().knowledge_base.remote.API_KEY,
        )
    return embedding
