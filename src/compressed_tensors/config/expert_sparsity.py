# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional

from compressed_tensors.config import (
    CompressionFormat,
    MoEMethod,
    SparsityCompressionConfig,
)
from pydantic import StrictInt, field_validator, model_validator


__all__ = ["ExpertSparsityConfig"]


@SparsityCompressionConfig.register(name=CompressionFormat.expert_sparsity.value)
class ExpertSparsityConfig(SparsityCompressionConfig):
    """
    Configuration for storing a Mixture-of-Experts (MoE) model with expert
    sparsity, where the number of experts is reduced on a per-layer basis.

    Two methods of inducing expert sparsity are supported via ``moe_method``:

    * ``"prune"`` (e.g. REAP): low-value experts are pruned entirely. Retained
      experts are left-shifted in the checkpoint.
    * ``"novice"`` (e.g. MoNe): experts are collapsed into fewer "novice"
      experts. Retained experts are left-shifted and novices remain as a single
      tensor.

    The ``moe_num_experts`` field encodes the per-layer expert count. It is a
    list whose length equals the total number of decoder layers in the model.
    Each entry is either:

    * ``None`` for a dense (non-MoE) layer, or
    * an ``int`` giving the number of experts retained in that MoE layer.

    Example (REAP / ``prune`` or MoNe / ``novice``)::

        {
            "moe_method": "prune",
            "moe_num_experts": [
                null,  # dense layers
                ...
                129,   # per-layer sparsity
                253,
                128,
                128,
                ...
            ]
        }

    :param moe_method: method used to induce expert sparsity, one of
        :class:`MoEMethod` ("prune" or "novice")
    :param moe_num_experts: per-layer retained expert counts. ``None`` entries
        denote dense layers; integer entries denote the number of experts in
        that MoE layer
    :param global_sparsity: average sparsity of the entire model
    :param sparsity_structure: structure of the sparsity, always "unstructured"
        for expert sparsity
    """

    format: str = CompressionFormat.expert_sparsity.value
    moe_method: MoEMethod
    moe_num_experts: List[Optional[StrictInt]]
    global_sparsity: Optional[float] = 0.0
    sparsity_structure: Optional[str] = "unstructured"

    @field_validator("moe_method", mode="before")
    @classmethod
    def _validate_moe_method(cls, value):
        # allow passing the enum directly or its string value
        if isinstance(value, MoEMethod):
            return value
        return MoEMethod(value)

    @field_validator("moe_num_experts")
    @classmethod
    def _validate_moe_num_experts(cls, value):
        if not value:
            raise ValueError("moe_num_experts must be a non-empty list")
        for i, num in enumerate(value):
            if num is None:
                continue
            if num < 1:
                raise ValueError(
                    f"moe_num_experts[{i}] must be >= 1 for an MoE layer, " f"got {num}"
                )
        return value

    @model_validator(mode="after")
    def _validate_has_moe_layer(self):
        if all(num is None for num in self.moe_num_experts):
            raise ValueError(
                "moe_num_experts must contain at least one MoE layer "
                "(a non-None entry)"
            )
        return self
