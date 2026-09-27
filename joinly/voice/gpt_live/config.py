from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class GptLiveConfig(BaseSettings):
    """Configuration for GPT-Live voice mode, read from `JOINLY_*` env variables."""

    openai_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="OPENAI_API_KEY"
    )
    gpt_live_url: str = "wss://api.openai.com/v1/live/sessions"
    gpt_live_model: str = "gpt-live-1"
    gpt_live_voice: str = "marin"
    gpt_live_sample_rate: Literal[16000, 24000] = 24000

    gpt_live_name_aliases: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "helius",
            "helio's",
            "helio",
            "elios",
            "helias",
            "hey leos",
            "hilios",
        ]
    )
    gpt_live_chat_trigger: str | None = None
    gpt_live_followup_seconds: float = 20.0
    gpt_live_hold_seconds: float = 1.5
    gpt_live_bargein_ms: int = 800
    gpt_live_chat_poll_seconds: float = 4.0
    gpt_live_disclosure: bool = True

    backend_model: str = "gpt-5.6-terra"
    backend_reasoning_effort: str = "low"
    backend_service_tier: str = "auto"
    backend_web_search: bool = False
    backend_tool_timeout: float = 10.0

    tools_url: str | None = None
    backend_token: SecretStr = Field(default=SecretStr(""))

    model_config = SettingsConfigDict(
        env_prefix="JOINLY_", extra="ignore", populate_by_name=True
    )

    @field_validator("gpt_live_name_aliases", mode="before")
    @classmethod
    def _split_aliases(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("tools_url", mode="before")
    @classmethod
    def _empty_url_is_none(cls, value: object) -> object:
        return value or None
