from dataclasses import replace

import pytest

from src.config import load_config, validate_config


def test_default_config_is_valid():
    config = load_config("config.yaml")
    assert config.environment.id == "ALE/SpaceInvaders-v5"
    assert config.evaluation.episodes == 5
    assert config.algorithm.n_steps == 3


def test_invalid_epsilon_is_rejected():
    config = load_config("config.yaml")
    invalid_algorithm = replace(config.algorithm, exploration_final_eps=1.1)
    with pytest.raises(ValueError, match="epsilon"):
        validate_config(replace(config, algorithm=invalid_algorithm))


def test_double_frameskip_is_avoided():
    config = load_config("config.yaml")
    assert config.environment.frame_skip == 4
    source = open("src/envs.py", encoding="utf-8").read()
    assert '"frameskip": 1' in source

