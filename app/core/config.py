import os
import secrets
from functools import lru_cache
from typing import Literal

import tomli_w
from loguru import logger
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class BaseSectionConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower(), extra='ignore')


class SecurityConfig(BaseSectionConfig):
    SECRET_KEY: SecretStr = SecretStr(secrets.token_urlsafe(32))
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24


class LogConfig(BaseSectionConfig):
    LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    FASTAPI_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    CELERY_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    ERROR_LEVEL: Literal['ERROR'] = 'ERROR'
    LOG_RETENTION: str = '1 week'
    LOG_ROTATION: str = '10 MB'


class NetworkConfig(BaseSectionConfig):
    HOST: str = '0.0.0.0'
    PORT: int = 8000
    PROXY_TYPE: Literal['http', 'socks5'] = 'http'
    PROXY_HOST: str = 'localhost'
    PROXY_PORT: int = 7890
    USE_PROXY: bool = False

    def get_proxy_url(self) -> str:
        return f'{self.PROXY_TYPE}://{self.PROXY_HOST}:{self.PROXY_PORT}'


class DatabaseConfig(BaseSectionConfig):
    DB_URL: str = 'postgresql://bioflow_ai:123456@127.0.0.1:5432/bioflow_ai'


class CacheConfig(BaseSectionConfig):
    LLM_CACHE: str = 'redis://localhost:6379/0'
    MONITOR_CACHE: str = 'redis://localhost:6379/2'
    API_CACHE: str = 'redis://localhost:6379/2'


class S3StorageConfig(BaseSectionConfig):
    ENDPOINT: str = 'http://localhost:9000'
    REGION: str = 'us-east-1'
    BUCKET: str = 'bioflow-ai'
    ACCESS_KEY: SecretStr = SecretStr('minioadmin')
    SECRET_KEY: SecretStr = SecretStr('minioadmin')
    USE_SSL: bool = True
    PRESIGN_EXPIRE: int = 3600  # 秒


class StorageConfig(BaseSectionConfig):
    STORAGE_PATH: str = './storage'
    GENOME_PATH: str = './storage/genome'
    TEST_DATA_PATH: str = './testdata'
    s3: S3StorageConfig = S3StorageConfig()


class ResourceConfig(BaseSectionConfig):
    MAX_THREAD: int = 8
    MAX_MEM: str = '4G'


class PersistenceConfig(BaseSectionConfig):
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    storage: StorageConfig = StorageConfig()
    resource: ResourceConfig = ResourceConfig()

    # 兼容旧访问方式，便于逐步迁移
    @property
    def DB_URL(self) -> str:
        return self.database.DB_URL

    @property
    def LLM_CACHE(self) -> str:
        return self.cache.LLM_CACHE

    @property
    def MONITOR_CACHE(self) -> str:
        return self.cache.MONITOR_CACHE

    @property
    def API_CACHE(self) -> str:
        return self.cache.API_CACHE

    @property
    def STORAGE_PATH(self) -> str:
        return self.storage.STORAGE_PATH

    @property
    def GENOME_PATH(self) -> str:
        return self.storage.GENOME_PATH

    @property
    def TEST_DATA_PATH(self) -> str:
        return self.storage.TEST_DATA_PATH

    @property
    def MAX_THREAD(self) -> int:
        return self.resource.MAX_THREAD

    @property
    def MAX_MEM(self) -> str:
        return self.resource.MAX_MEM


class DockerConfig(BaseSectionConfig):
    CONTAINER: Literal['docker', 'podman'] = 'docker'
    USERNAME: str = ''
    PASSWORD: SecretStr = SecretStr('')
    ENABLE_GPU: bool = False
    GPU_DEVICES: str = 'all'  # 'all' 或指定设备如 '0,1' 或 '0'


class CeleryConfig(BaseSectionConfig):
    BROKER: str = 'amqp://bioflow_ai:123456@localhost:5672//'
    BACKEND: str = 'redis://localhost:6379/1'


class MinerUConfig(BaseSectionConfig):
    API_URL: str = 'https://api.mineru.app'
    API_KEY: SecretStr = SecretStr('')
    TIMEOUT: int = 600  # 秒


class OpenMinerUConfig(BaseSectionConfig):
    API_URL: str = 'http://localhost:8000'
    SERVER_URL: str = 'http://localhost:30000'
    TIMEOUT: int = 600  # 秒


class PDFParserConfig(BaseSectionConfig):
    ENGINE: Literal['mineru', 'doc2x', 'grobid', 'open_mineru', 'local'] = 'local'
    BATCH_SIZE: int = 5

    mineru: MinerUConfig = MinerUConfig()
    open_mineru: OpenMinerUConfig = OpenMinerUConfig()


class LocalEmbeddingModel(BaseSectionConfig):
    MODEL: str = 'Qwen/Qwen3-Embedding-4B'
    USE_MODELSCOPE: bool = False
    DEVICE: str = 'cuda:0'


class RemoteEmbeddingModel(BaseSectionConfig):
    MODEL: str = 'Qwen/Qwen3-Embedding-4B'
    BASE_URL: str = 'https://api.siliconflow.cn/v1/'
    API_KEY: SecretStr = SecretStr('')


class KnowledgeBaseConfig(BaseSectionConfig):
    MILVUS_URL: str = 'http://localhost:19530'
    USE_LOCAL_MODEL: bool = False

    local: LocalEmbeddingModel = LocalEmbeddingModel()
    remote: RemoteEmbeddingModel = RemoteEmbeddingModel()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml',
        alias_generator=lambda field_name: field_name.lower(),
        env_nested_delimiter='__',
        extra='ignore',
    )

    PROJECT_NAME: str = 'BioFlow AI'
    VERSION: str = '0.0.1'

    security: SecurityConfig = SecurityConfig()
    logger: LogConfig = LogConfig()
    network: NetworkConfig = NetworkConfig()
    persistent: PersistenceConfig = PersistenceConfig()
    docker: DockerConfig = DockerConfig()
    celery: CeleryConfig = CeleryConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    pdf_parser: PDFParserConfig = PDFParserConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 环境变量最高优先，其次 toml，最后初始化参数
        return env_settings, TomlConfigSettingsSource(settings_cls), init_settings

    def save_to_file(self, file_path: str):
        """将当前配置保存到TOML文件"""
        config_dict = self.model_dump(by_alias=True, mode='json', exclude_none=True)

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
