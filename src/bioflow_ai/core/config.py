import secrets
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import tomli_w
from loguru import logger
from pydantic import (
    AmqpDsn,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

INSECURE_NONE_COOKIE_ERROR = 'SameSite=None cookies require cookie_secure=true'
DEFAULT_CRAN_REPOSITORY = 'https://cloud.r-project.org'
DEFAULT_PYTHON_INDEX_URL = 'https://pypi.org/simple'


class BaseSectionConfig(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.lower(),
        populate_by_name=True,
        extra='ignore',
    )


class SecurityConfig(BaseSectionConfig):
    SECRET_KEY: SecretStr = SecretStr(secrets.token_urlsafe(32))
    SESSION_IDLE_TIMEOUT_HOURS: int = Field(default=24, gt=0)
    SESSION_ABSOLUTE_TIMEOUT_DAYS: int = Field(default=7, gt=0)
    LOGIN_MAX_FAILURES: int = Field(default=10, gt=0)
    LOGIN_FAILURE_WINDOW_MINUTES: int = Field(default=15, gt=0)
    USER_CACHE_TTL_SECONDS: int = Field(default=300, gt=0)
    COOKIE_SECURE: bool = False
    COOKIE_DOMAIN: str | None = None
    COOKIE_SAMESITE: Literal['lax', 'strict', 'none'] = 'lax'

    @model_validator(mode='after')
    def validate_cookie_transport(self) -> 'SecurityConfig':
        """Require encrypted transport for cross-site cookies."""
        if self.COOKIE_SAMESITE == 'none' and not self.COOKIE_SECURE:
            raise ValueError(INSECURE_NONE_COOKIE_ERROR)
        return self


class LogConfig(BaseSectionConfig):
    LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    FASTAPI_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    CELERY_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    BOTO3_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    ERROR_LEVEL: Literal['ERROR'] = 'ERROR'
    LOG_RETENTION: str = '1 week'
    LOG_ROTATION: str = '10 MB'


class NetworkConfig(BaseSectionConfig):
    HOST: str = '0.0.0.0'  # noqa: S104 - the containerized Web UI reaches the host API through host-gateway.
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ['http://localhost:3000']
    PROXY_TYPE: Literal['http', 'socks5'] = 'http'
    PROXY_HOST: str = 'localhost'
    PROXY_PORT: int = 7890
    USE_PROXY: bool = False

    def get_proxy_url(self) -> str:
        """返回代理链接."""
        return f'{self.PROXY_TYPE}://{self.PROXY_HOST}:{self.PROXY_PORT}'


class DatabaseConfig(BaseSectionConfig):
    DB_URL: PostgresDsn = PostgresDsn('postgresql+psycopg://bioflow_ai:123456@127.0.0.1:5432/bioflow_ai')


class CacheConfig(BaseSectionConfig):
    LLM_CACHE: RedisDsn = RedisDsn('redis://localhost:6379/0')
    MONITOR_CACHE: RedisDsn = RedisDsn('redis://localhost:6379/2')
    API_CACHE: RedisDsn = RedisDsn('redis://localhost:6379/2')
    NOTIFICATION_CACHE: RedisDsn = RedisDsn('redis://localhost:6379/2')
    CATALOG_GENERATION_KEY_PREFIX: str = 'bioflow:catalog-generation'
    CATALOG_GENERATION_LOCK_TIMEOUT_SECONDS: int = 900
    CATALOG_GENERATION_LOCK_RENEW_INTERVAL_SECONDS: int = 300
    CATALOG_GENERATION_SEARCH_EPOCH_TTL_SECONDS: int = 24 * 60 * 60


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
    BIO_DB_PATH: str = './storage/bio_dbs'
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


class DockerConfig(BaseSectionConfig):
    CONTAINER: Literal['docker', 'podman'] = 'docker'
    USERNAME: str = ''
    PASSWORD: SecretStr = SecretStr('')
    ENABLE_GPU: bool = False
    GPU_DEVICES: str = 'all'  # 'all' 或指定设备如 '0,1' 或 '0'


class CodeConfig(BaseSectionConfig):
    CRAN_REPOSITORY: AnyHttpUrl = AnyHttpUrl(DEFAULT_CRAN_REPOSITORY)
    PYTHON_INDEX_URL: AnyHttpUrl = AnyHttpUrl(DEFAULT_PYTHON_INDEX_URL)


class CodeEditingConfig(BaseSectionConfig):
    SESSION_TTL_SECONDS: int = Field(default=1800, ge=60)
    TURN_TIMEOUT_SECONDS: int = Field(default=900, ge=10)
    WORKER_STALE_SECONDS: int = Field(default=60, ge=10)


class CeleryConfig(BaseSectionConfig):
    BROKER: AmqpDsn = AmqpDsn('amqp://bioflow_ai:123456@localhost:5672//')
    BACKEND: RedisDsn = RedisDsn('redis://localhost:6379/1')


class WorkflowConfig(BaseSectionConfig):
    """System-wide execution limits for workflow coordination."""

    MAX_PARALLEL_ITEMS: int = Field(default=4, ge=1)


class MinerUConfig(BaseSectionConfig):
    API_URL: str = 'https://api.mineru.app'
    API_KEY: SecretStr = SecretStr('')
    TIMEOUT: int = 600  # 秒


class OpenMinerUConfig(BaseSectionConfig):
    API_URL: str = 'http://localhost:8000'
    TIMEOUT: int = 600  # 秒


class PDFParserConfig(BaseSectionConfig):
    ENGINE: Literal['mineru', 'doc2x', 'grobid', 'open_mineru', 'local'] = 'local'
    BATCH_SIZE: int = 5
    SIMILARITY_THRESHOLD: float = Field(default=0.5, ge=-1, le=1)

    mineru: MinerUConfig = MinerUConfig()
    open_mineru: OpenMinerUConfig = OpenMinerUConfig()


class KnowledgeBaseConfig(BaseSectionConfig):
    MILVUS_URL: str = 'http://localhost:19530'

    EMBEDDING_MODEL: str = 'Qwen/Qwen3-Embedding-4B'
    BASE_URL: str = 'https://api.siliconflow.cn/v1/'
    API_KEY: SecretStr = SecretStr('')
    TIMEOUT: float = 60


class DevelopConfig(BaseSectionConfig):
    CHECKPOINT_URL: RedisDsn = RedisDsn('redis://localhost:6379/0')


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml',
        alias_generator=lambda field_name: field_name.lower(),
        populate_by_name=True,
        env_nested_delimiter='__',
        extra='ignore',
    )

    security: SecurityConfig = SecurityConfig()
    logger: LogConfig = LogConfig()
    network: NetworkConfig = NetworkConfig()
    persistent: PersistenceConfig = PersistenceConfig()
    docker: DockerConfig = DockerConfig()
    code: CodeConfig = CodeConfig()
    code_editing: CodeEditingConfig = CodeEditingConfig()
    celery: CeleryConfig = CeleryConfig()
    workflow: WorkflowConfig = WorkflowConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    pdf_parser: PDFParserConfig = PDFParserConfig()
    develop: DevelopConfig = DevelopConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """环境变量最高优先, 其次 toml, 最后初始化参数."""
        return (
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
            init_settings,
        )

    def save_to_file(self, file_path: str) -> None:
        """将当前配置保存到TOML文件."""
        config_dict = self.model_dump(by_alias=True, mode='json', exclude_none=True)

        with Path(file_path).open('wb') as f:
            tomli_w.dump(config_dict, f)

        logger.info(f'配置已保存到 {file_path}')


@lru_cache
def get_settings() -> Settings:
    """全局配置对象."""
    config_path = Path('config.toml')
    if not config_path.exists():
        logger.warning(
            f'The configuration file "{config_path}" does not exist. Run `uv run bioflow-ai auto-init` to generate it.'
        )
        sys.exit(-1)

    return Settings()
