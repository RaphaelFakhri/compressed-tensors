# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from compressed_tensors.config import (
    CompressionFormat,
    ExpertSparsityConfig,
    MoEMethod,
    SparsityCompressionConfig,
)
from pydantic import ValidationError


def test_moe_method_values():
    assert MoEMethod("prune") == MoEMethod.PRUNE
    assert MoEMethod("novice") == MoEMethod.NOVICE


def test_moe_method_invalid():
    with pytest.raises(ValueError):
        MoEMethod("invalid")


@pytest.mark.parametrize("method", ["prune", "novice"])
def test_expert_sparsity_basic(method):
    cfg = ExpertSparsityConfig(
        moe_method=method,
        moe_num_experts=[None, None, 129, 253, 128, 128],
    )
    assert cfg.format == CompressionFormat.expert_sparsity.value
    assert cfg.moe_method == MoEMethod(method)
    assert cfg.moe_num_experts == [None, None, 129, 253, 128, 128]
    assert cfg.sparsity_structure == "unstructured"


def test_expert_sparsity_accepts_enum_method():
    cfg = ExpertSparsityConfig(
        moe_method=MoEMethod.PRUNE,
        moe_num_experts=[None, 128],
    )
    assert cfg.moe_method == MoEMethod.PRUNE


def test_expert_sparsity_registry_lookup():
    cfg = SparsityCompressionConfig.load_from_registry(
        CompressionFormat.expert_sparsity.value,
        moe_method="prune",
        moe_num_experts=[None, 129],
    )
    assert isinstance(cfg, ExpertSparsityConfig)


def test_expert_sparsity_json_roundtrip():
    cfg = ExpertSparsityConfig(
        moe_method="novice",
        moe_num_experts=[None, 129, 253, 128],
    )
    dumped = cfg.model_dump(mode="json")
    assert dumped["moe_method"] == "novice"
    assert dumped["format"] == CompressionFormat.expert_sparsity.value
    assert dumped["moe_num_experts"] == [None, 129, 253, 128]

    restored = ExpertSparsityConfig.model_validate(dumped)
    assert restored.moe_method == MoEMethod.NOVICE
    assert restored.moe_num_experts == [None, 129, 253, 128]


def test_expert_sparsity_invalid_method():
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(moe_method="bad", moe_num_experts=[None, 1])


def test_expert_sparsity_empty_list():
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(moe_method="prune", moe_num_experts=[])


def test_expert_sparsity_all_dense_rejected():
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(moe_method="prune", moe_num_experts=[None, None])


@pytest.mark.parametrize("bad", [0, -1])
def test_expert_sparsity_non_positive_count(bad):
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(moe_method="prune", moe_num_experts=[None, bad])


def test_expert_sparsity_bool_rejected():
    # bool is a subclass of int but should not be a valid expert count
    with pytest.raises(ValidationError):
        ExpertSparsityConfig(moe_method="prune", moe_num_experts=[None, True])
