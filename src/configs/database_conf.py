from typing import Any

from pydantic import Field, NonNegativeInt, PositiveInt, SecretStr, computed_field
from pydantic_settings import BaseSettings
from sqlalchemy import URL
from sqlalchemy.util import immutabledict


class DatabaseConfig(BaseSettings):
    DB_HOST: str = Field(
        description="Hostname or IP address of the database server.",
        default="localhost",
    )

    DB_PORT: PositiveInt = Field(
        description="Port number for database connection.",
        default=5432,
    )

    DB_USERNAME: str = Field(
        description="Username for database authentication.",
        default="postgres",
    )

    DB_PASSWORD: SecretStr = Field(
        description="Password for database authentication.",
        default=SecretStr("password"),
    )

    DB_DATABASE: str = Field(
        description="Name of the database to connect to.",
        default="pdf_injection.db",
    )

    DB_TYPE: str = Field(
        description="Database URI scheme for SQLAlchemy connection.",
        default="sqlite",
    )

    DB_CHARSET: str = Field(
        description="Character set for database connection.",
        default="",
    )

    SQLALCHEMY_POOL_SIZE: NonNegativeInt = Field(
        description="Maximum number of database connections in the pool.",
        default=30,
    )

    SQLALCHEMY_MAX_OVERFLOW: NonNegativeInt = Field(
        description="Maximum number of connections that can be created beyond the pool_size.",
        default=10,
    )

    SQLALCHEMY_POOL_RECYCLE: NonNegativeInt = Field(
        description="Number of seconds after which a connection is automatically recycled.",
        default=3600,
    )

    SQLALCHEMY_POOL_PRE_PING: bool = Field(
        description="If True, enables connection pool pre-ping feature to check connections.",
        default=False,
    )

    SQLALCHEMY_ECHO: bool | str = Field(
        description="If True, SQLAlchemy will log all SQL statements.",
        default=False,
    )

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.DB_TYPE.lower()

    @property
    def is_postgres(self) -> bool:
        return "postgresql" in self.DB_TYPE.lower()

    @property
    def driver(self) -> str:
        if "postgresql" in self.DB_TYPE:
            return "postgresql+pg8000"
        if "sqlite" in self.DB_TYPE:
            return "sqlite+pysqlite"
        if "mysql" in self.DB_TYPE:
            return "mysql+aiomysql"
        if "oracle" in self.DB_TYPE:
            return "oracle+oracledb"
        raise ValueError("DataBase not Supported")

    @property
    def query(self) -> immutabledict[str, str]:
        if self.is_sqlite:
            return immutabledict({"check_same_thread": "false"})
        return immutabledict({"client_encoding": self.DB_CHARSET}) if self.DB_CHARSET else immutabledict({})

    @computed_field
    def SQLALCHEMY_ENGINE_OPTIONS(self) -> dict[str, Any]:  # noqa: N802
        if self.is_sqlite:
            return {}
        return {
            "pool_size": self.SQLALCHEMY_POOL_SIZE,
            "max_overflow": self.SQLALCHEMY_MAX_OVERFLOW,
            "pool_recycle": self.SQLALCHEMY_POOL_RECYCLE,
            "pool_pre_ping": self.SQLALCHEMY_POOL_PRE_PING,
            # "connect_args": {"options": "-c timezone=UTC"},
        }

    @computed_field
    def SQLALCHEMY_DATABASE_URI(self) -> str:  # noqa: N802
        drivername_ = self.driver
        username_ = None if self.is_sqlite else self.DB_USERNAME
        password_ = None if self.is_sqlite else self.DB_PASSWORD
        host_ = None if self.is_sqlite else self.DB_HOST
        port_ = None if self.is_sqlite else self.DB_PORT
        return URL(
            drivername=drivername_,
            username=username_,
            password=password_.get_secret_value() if password_ else None,
            host=host_,
            port=port_,
            database=self.DB_DATABASE,
            query=self.query,  # type: ignore
        ).render_as_string(hide_password=False)
