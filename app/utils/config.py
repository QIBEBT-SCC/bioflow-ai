import os
from functools import lru_cache
from typing import Type, Literal
import tomli_w

from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)


class ImageConfig(BaseModel):
    REPOSITORY: str
    TAG: str = 'latest'

    def get_image(self) -> str:
        return f'{self.REPOSITORY}:{self.TAG}'


class DockerConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    fast_qc: ImageConfig = ImageConfig(
        REPOSITORY='imagehub.cstcloud.cn/biocontainers/fastqc', TAG='0.11.5--4'
    )
    fastp: ImageConfig = ImageConfig(
        REPOSITORY='imagehub.cstcloud.cn/biocontainers/fastp', TAG='0.23.4--h125f33a_5'
    )
    fastx: ImageConfig = ImageConfig(
        REPOSITORY='imagehub.cstcloud.cn/biocontainers/fastx_toolkit', TAG='0.0.14--he1b5a44_8'
    )
    bbmap: ImageConfig = ImageConfig(
        REPOSITORY='imagehub.cstcloud.cn/biocontainers/bbmap', TAG='39.10--h92535d8_0'
    )
    spades: ImageConfig = ImageConfig(
        REPOSITORY='imagehub.cstcloud.cn/biocontainers/spades', TAG='4.0.0--h6dccd9a_3'
    )


class LogConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    FASTAPI_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    PREFECT_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    ERROR_LEVEL: Literal['ERROR'] = 'ERROR'
    LOG_RETENTION: str = '1 week'
    LOG_ROTATION: str = '10 MB'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml', alias_generator=lambda field_name: field_name.lower()
    )

    PROJECT_NAME: str = 'Smart SCHOOL'
    VERSION: str = '2.0.0'

    logger: LogConfig = LogConfig()
    docker: DockerConfig = DockerConfig()

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

    def save_to_file(self, file_path: str = 'config.toml'):
        """将当前配置保存到TOML文件"""
        # 将配置转换为字典
        config_dict = self.model_dump(by_alias=True)

        for key, value in config_dict.items():
            if (
                isinstance(value, dict)
                and hasattr(self, key)
                and isinstance(getattr(self, key), BaseModel)
            ):
                config_dict[key] = getattr(self, key).model_dump(by_alias=True)

        with open(file_path, 'wb') as f:
            tomli_w.dump(config_dict, f)

        logger.info(f'配置已保存到 {file_path}')


@lru_cache
def get_settings() -> Settings:
    config_path = 'config.toml'
    if not os.path.exists(config_path):
        settings = Settings()
        settings.save_to_file(config_path)
        logger.warning(f'配置文件"{config_path}"不存在，已自动初始化默认配置，请完善相关设置')

    settings = Settings()
    return settings


def main() -> None:
    setting = get_settings()
    print(setting)


if __name__ == '__main__':
    main()
