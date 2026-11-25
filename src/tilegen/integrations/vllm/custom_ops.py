# src/tilegen/integrations/vllm/custom_ops.py
#
# vLLM OOT custom-operator overrides (CustomOp.register_oot)

from __future__ import annotations

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm


@CustomOp.register_oot(name="RMSNorm")
class TileGenRMSNorm(RMSNorm):
    """TileGen PyTorch-native RMSNorm (matches vLLM's ``RMSNorm`` interface)."""

    def forward_native(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        orig = x.dtype
        xf = x.to(torch.float32)
        if residual is not None:
            xf = xf + residual.to(torch.float32)
            residual = xf.to(orig)
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.variance_epsilon)
        xf = xf.to(orig)
        if self.has_weight:
            xf = xf * self.weight.to(orig)
        return xf if residual is None else (xf, residual)

    def forward_cuda(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        return self.forward_native(x, residual)


__all__ = ["TileGenRMSNorm"]
