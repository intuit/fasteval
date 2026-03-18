"""Tests for fasteval_langfuse.config."""

import os
from unittest.mock import patch

import pytest

import fasteval_langfuse.config as config_module
from fasteval_langfuse.config import LangfuseConfig, configure_langfuse, get_config


class TestLangfuseConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = LangfuseConfig()
            assert config.public_key is None
            assert config.secret_key is None
            assert config.host == "https://cloud.langfuse.com"
            assert config.auto_push_scores is True
            assert config.batch_size == 50
            assert config.score_name_prefix == "fasteval_"

    def test_from_env(self):
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
                "LANGFUSE_HOST": "https://custom.host.com",
            },
        ):
            config = LangfuseConfig()
            assert config.public_key == "pk-test"
            assert config.secret_key == "sk-test"
            assert config.host == "https://custom.host.com"

    def test_custom_values(self):
        config = LangfuseConfig(
            public_key="pk",
            secret_key="sk",
            default_project="prod",
            auto_push_scores=False,
            batch_size=100,
            score_name_prefix="custom_",
        )
        assert config.default_project == "prod"
        assert config.auto_push_scores is False
        assert config.batch_size == 100
        assert config.score_name_prefix == "custom_"


class TestConfigureLangfuse:
    def setup_method(self):
        config_module._config = None

    def test_configure_and_get(self):
        custom = LangfuseConfig(public_key="pk", secret_key="sk")
        configure_langfuse(custom)
        assert get_config() is custom

    def test_get_config_default(self):
        config = get_config()
        assert isinstance(config, LangfuseConfig)

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
