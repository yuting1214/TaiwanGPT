import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "My App"
    APP_VERSION: str = "0.1.0"

    @property
    def DB_URL(self):
        if self.ENV_MODE == "dev":
            return self.DEV_DB_URL
        else:
            if self.DATABASE_URL:
                return self.DATABASE_URL
            else:
                return '{}://{}:{}@{}:{}/{}'.format(
                    self.DB_ENGINE,
                    self.DB_USERNAME,
                    self.DB_PASS,
                    self.DB_HOST,
                    self.DB_PORT,
                    self.DB_NAME
                )

    @property
    def ASYNC_DB_URL(self):
        if self.ENV_MODE == "dev":
            return "sqlite+aiosqlite:///./dev.db"
        else:
            if self.DATABASE_URL:
                URL_split = self.DATABASE_URL.split("://")
                return f"{URL_split[0]}+asyncpg://{URL_split[1]}"
            else:
                return '{}+asyncpg://{}:{}@{}:{}/{}'.format(
                    self.DB_ENGINE,
                    self.DB_USERNAME,
                    self.DB_PASS,
                    self.DB_HOST,
                    self.DB_PORT,
                    self.DB_NAME
                )

class DevSettings(Settings):
    # Environment mode: 'dev' or 'prod'
    ENV_MODE: str = 'dev'

    # Database settings for development
    DEV_DB_URL: str = "sqlite:///./dev.db"

    model_config = SettingsConfigDict(env_file=".env", extra='allow')

class ProdSettings(Settings):
    # Environment mode: 'dev' or 'prod'
    ENV_MODE: str = 'prod'

    # Database settings for production
    DB_ENGINE: str = os.getenv('DB_ENGINE', '')
    DB_USERNAME: str = os.getenv('DB_USERNAME', '')
    DB_PASS: str = os.getenv('DB_PASS', '')
    DB_HOST: str = os.getenv('DB_HOST', '')
    DB_PORT: str = os.getenv('DB_PORT', '')
    DB_NAME: str = os.getenv('DB_NAME', '')

    # Extra Database settings for deploying on Railway; if you provide DATABASE_URL, the above settings will be ignored
    DATABASE_URL: str = os.getenv('DATABASE_URL', '')

    # Database settings for production
    model_config = SettingsConfigDict(env_file=".env", extra='allow')

def get_settings(env_mode: str = "dev"):
    if env_mode == "dev":
        return DevSettings()
    return ProdSettings()