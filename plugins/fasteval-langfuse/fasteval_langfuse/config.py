"""Configuration for fasteval-langfuse."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class LangfuseConfig(BaseModel):
    """
    Configuration for Langfuse integration.

    Attributes:
        public_key: Langfuse public API key (or from LANGFUSE_PUBLIC_KEY env)
        secret_key: Langfuse secret API key (or from LANGFUSE_SECRET_KEY env)
        host: Langfuse host URL (or from LANGFUSE_HOST env)
        default_project: Default project name for traces
        auto_push_scores: Automatically push evaluation scores back to Langfuse
        batch_size: Batch size for fetching traces
        max_parallel_evals: Maximum parallel evaluations
        retry_on_failure: Retry failed score pushes
        score_name_prefix: Prefix for score names in Langfuse
    """

    public_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY")
    )
    secret_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY")
    )
    host: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    default_project: Optional[str] = None
    auto_push_scores: bool = True
    batch_size: int = 50
    max_parallel_evals: int = 5
    retry_on_failure: bool = True
    score_name_prefix: str = "fasteval_"


# Global configuration instance
_config: Optional[LangfuseConfig] = None


def configure_langfuse(config: LangfuseConfig) -> None:
    """
    Configure the Langfuse integration.

    Args:
        config: LangfuseConfig instance

    Example:
        from fasteval_langfuse import configure_langfuse, LangfuseConfig

        configure_langfuse(LangfuseConfig(
            public_key="pk-...",
            secret_key="sk-...",
            default_project="production"
        ))
    """
    global _config
    _config = config


def get_config() -> LangfuseConfig:
    """
    Get the current Langfuse configuration.

    Returns:
        Current LangfuseConfig, or default if not configured

    Example:
        config = get_config()
        print(config.default_project)
    """
    global _config
    if _config is None:
        _config = LangfuseConfig()
    return _config
