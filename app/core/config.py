import os
import secrets
from functools import lru_cache
from typing import Literal, Type

import tomli_w
from loguru import logger
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class SecurityConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    SECRET_KEY: SecretStr = secrets.token_urlsafe(32)
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24


class LogConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    FASTAPI_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    CELERY_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    ERROR_LEVEL: Literal['ERROR'] = 'ERROR'
    LOG_RETENTION: str = '1 week'
    LOG_ROTATION: str = '10 MB'


class NetworkConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    HOST: str = '0.0.0.0'
    PORT: int = 8000
    PROXY_TYPE: Literal['http', 'socks5'] = 'http'
    PROXY_HOST: str = 'localhost'
    PROXY_PORT: int = 7890
    USE_PROXY: bool = False

    def get_proxy_url(self) -> str:
        return f'{self.PROXY_TYPE}://{self.PROXY_HOST}:{self.PROXY_PORT}'


class StorageConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    UPLOAD_DIR: str = './storage/uploads'
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_DOCUMENT_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_TYPES: list[str] = [
        'image/jpeg',
        'image/jpg',
        'image/png',
        'image/gif',
        'image/bmp',
        'image/webp',
        'image/tiff',
    ]
    ALLOWED_DOCUMENT_TYPES: list[str] = ['application/pdf', 'text/plain', 'text/markdown']


class PersistenceConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    DB_URL: str = 'postgresql://bioflow_ai:123456@127.0.0.1:5432/bioflow_ai'
    CHECKPOINT_URL: str = 'postgresql://bioflow_ai:123456@127.0.0.1:5432/llm_history'
    MONITOR_CACHE: str = 'redis://localhost:6379/1'
    API_CACHE: str = 'redis://localhost:6379/2'
    STORAGE_PATH: str = './storage'
    GENOME_PATH: str = './storage/genome'
    TEST_DATA_PATH: str = './testdata'
    MAX_THREAD: int = 8
    MAX_MEM: str = '4G'

    api_storage: StorageConfig = StorageConfig()


class KnowledgeBaseConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    MILVUS_URL: str = 'http://localhost:19530'
    EMBEDDING_MODEL: str = 'Qwen/Qwen3-Embedding-4B'
    USE_MODELSCOPE: bool = False
    EMBEDDING_DEVICE: str = 'cuda:0'


class LLMApiConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    PROVIDER: str = ''
    BASE_URL: str = ''
    API_KEY: SecretStr = ''


class LLMToolConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    JINA_API: SecretStr = ''


class LLMConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    GEMINI: LLMApiConfig = LLMApiConfig()
    GPT: LLMApiConfig = LLMApiConfig()
    CLAUDE: LLMApiConfig = LLMApiConfig()
    GROK: LLMApiConfig = LLMApiConfig()
    DEEPSEEK: LLMApiConfig = LLMApiConfig()
    GLM: LLMApiConfig = LLMApiConfig()
    VLLM: LLMApiConfig = LLMApiConfig()

    tools: LLMToolConfig = LLMToolConfig()


class CeleryConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    BROKER: str = 'amqp://bioflow_ai:123456@localhost:5672//'
    BACKEND: str = 'redis://localhost:6379/0'


class DockerConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    CONTAINER: Literal['docker', 'podman'] = 'docker'
    USERNAME: str = ''
    PASSWORD: SecretStr = ''


class Settings(BaseSettings):
    model_config = SettingsConfigDict(toml_file='config.toml', alias_generator=lambda field_name: field_name.lower())

    PROJECT_NAME: str = 'BioFlow AI'
    VERSION: str = '0.0.1'

    security: SecurityConfig = SecurityConfig()
    logger: LogConfig = LogConfig()
    network: NetworkConfig = NetworkConfig()
    persistent: PersistenceConfig = PersistenceConfig()
    docker: DockerConfig = DockerConfig()
    celery: CeleryConfig = CeleryConfig()
    llm: LLMConfig = LLMConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)

    def save_to_file(self, file_path: str):
        """将当前配置保存到TOML文件"""
        config_dict = self.model_dump(by_alias=True)

        for key, value in config_dict.items():
            if isinstance(value, dict) and hasattr(self, key) and isinstance(getattr(self, key), BaseModel):
                config_dict[key] = getattr(self, key).model_dump(by_alias=True)

        with open(file_path, 'wb') as f:
            tomli_w.dump(config_dict, f)

        logger.info(f'配置已保存到 {file_path}')


@lru_cache
def get_settings() -> Settings:
    config_path = 'config.toml'
    if not os.path.exists(config_path):
        settings = Settings()
        settings.save_to_file('config.example.toml')
        logger.warning(f'配置文件"{config_path}"不存在，已自动初始化默认配置，请完善相关设置')
        exit(-1)

    settings = Settings()
    return settings


def main() -> None:
    setting = get_settings()
    print(setting)


if __name__ == '__main__':
    main()
