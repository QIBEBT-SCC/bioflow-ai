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


class DockerConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    FAST_QC_IMAGE: str = 'imagehub.cstcloud.cn/biocontainers/fastqc:0.11.5--4'
    FASTP_IMAGE: str = 'imagehub.cstcloud.cn/biocontainers/fastp:0.23.4--h125f33a_5'
    METAPHLAN_IMAGE: str = 'imagehub.cstcloud.cn/biocontainers/metaphlan:4.1.1--pyhdfd78af_0'
    PGGB_IMAGE: str = 'imagehub.cstcloud.cn/biocontainers/pggb:0.7.2--h9ee0642_0'


class LogConfig(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml', alias_generator=lambda field_name: field_name.lower()
    )

    PROJECT_NAME: str = 'Smart SCHOOL'
    VERSION: str = '2.0.0'

    docker: DockerConfig = DockerConfig()
    logger: LogConfig = LogConfig()

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
