# tests/test_parallel.py
#
# Tests for the generic CP/SP parallelism flow (tilegen.parallel)

import pytest

torch = pytest.importorskip("torch")


def test_parallelism_spec_validation():
    from tilegen.parallel import CommPattern, ParallelismSpec

    s = ParallelismSpec(cp_degree=4, sp_degree=1, tp_degree=2)
    assert s.world_size == 8
    assert s.is_sharded is True
    assert s.to_dict()["cp_degree"] == 4

    with pytest.raises(ValueError):
        ParallelismSpec(cp_degree=0)
    with pytest.raises(ValueError):
        ParallelismSpec(cp_degree=2, comm_pattern=CommPattern.RING, devices=("cuda",))
    assert ParallelismSpec().is_sharded is False


def test_parallelism_from_cp_serving_mapper():
    from tilegen.parallel import parallelism_from_cp

    s = parallelism_from_cp(cp_size=4, tp_size=2)
    assert s.cp_degree == 4 and s.tp_degree == 2 and s.world_size == 8


def test_collective_ops_build_serialize_and_verify():
    """The collective ops build into a VTM kernel and serialize."""
    from tilegen.ir.builder import VTMBuilder
    from tilegen.ir.topology import TopologySpec
    from tilegen.ir.types import MemoryTier

    b = VTMBuilder("cp_attention", TopologySpec())
    b.input("Q", (1, 8, 128, 64), "bf16")
    b.input("K", (1, 8, 128, 64), "bf16")
    b.output("O", (1, 8, 128, 64), "bf16")
    b.alloc("s_acc", (128, 128), "fp32", MemoryTier.ACCUMULATOR)
    b.alloc("p_reg", (128, 128), "bf16", MemoryTier.COMPUTE_INPUT)
    b.all_gather("K", dim=2, degree=4)
    b.online_softmax("s_acc", "p_reg", global_reduce=True, is_final=True)
    b.reduce_scatter("O", dim=2, degree=4)
    kernel = b.build()
    s = kernel.serialize()
    assert "vtm.all_gather" in s
    assert "global_reduce = true" in s
    assert "vtm.reduce_scatter" in s


def test_online_softmax_global_reduce_flag():
    from tilegen.ir.ops import OnlineSoftmaxOp

    op = OnlineSoftmaxOp("s_acc", "p_reg", global_reduce=True)
    assert op.global_reduce is True
    assert OnlineSoftmaxOp("s_acc", "p_reg").global_reduce is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_parallel_executor_unsharded_falls_through():
    """cp_degree=1: the executor returns the local kernel's output unchanged."""
    from tilegen.parallel import ParallelExecutor, ParallelismSpec

    torch.manual_seed(0)
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ex = ParallelExecutor(ParallelismSpec()) # unsharded
    out = ex.attention(q, k, v, is_causal=True)
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16


def test_parallel_executor_all_gather_orchestration(monkeypatch):
    """The executor gathers K/V along the sequence dim before the local kernel."""
    from tilegen.parallel import ParallelExecutor, ParallelismSpec
    import tilegen.parallel.executor as ex_mod

    spec = ParallelismSpec(cp_degree=2)
    calls = {}

    def fake_all_gather(t, dim=0, group=None):
        return torch.cat([t, t], dim=dim) # simulate 2 ranks

    monkeypatch.setattr(ex_mod.collectives, "all_gather", fake_all_gather)

    def local(q, k, v, is_causal=False, scale=None):
        calls["k_shape"] = tuple(k.shape)
        calls["v_shape"] = tuple(v.shape)
        return torch.zeros_like(q)

    ex = ParallelExecutor(spec, local_forward=local)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    ex.attention(q, k, v, is_causal=True)
    assert calls["k_shape"] == (1, 4, 16, 16) # seq dim doubled by the gather
    assert calls["v_shape"] == (1, 4, 16, 16)


def test_build_cp_attention_vtm_all_gather():
    """build_cp_attention_vtm emits all_gather collectives around the local attention."""
    from tilegen.ops.attention import AttentionSpec, build_cp_attention_vtm
    from tilegen.parallel import CommPattern, ParallelismSpec
    from tilegen.target.spec import get_target

    spec = AttentionSpec(seq_len=2048, head_dim=64)
    par = ParallelismSpec(cp_degree=4, comm_pattern=CommPattern.ALL_GATHER)
    kernel = build_cp_attention_vtm(spec, get_target("sm_80"), par)
    s = kernel.serialize()
    assert "vtm.all_gather %K {dim = 2, degree = 4}" in s
    assert "vtm.all_gather %V {dim = 2, degree = 4}" in s
    assert "global_reduce = false" in s


def test_build_cp_attention_vtm_ring_uses_global_reduce():
    """The RING comm pattern marks the online softmax for global reduction."""
    from tilegen.ops.attention import AttentionSpec, build_cp_attention_vtm
    from tilegen.parallel import CommPattern, ParallelismSpec
    from tilegen.target.spec import get_target

    spec = AttentionSpec(seq_len=2048, head_dim=64)
    par = ParallelismSpec(cp_degree=4, comm_pattern=CommPattern.RING)
    kernel = build_cp_attention_vtm(spec, get_target("sm_80"), par)
    assert "global_reduce = true" in kernel.serialize()
