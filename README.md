# TileGen

A lightweight, layout-decoupled tile kernel generation, self-evolution, and execution framework for LLM inference across heterogeneous GPUs and Domain-Specific Accelerators (DSAs).

## 📖 Overview

Modern LLM serving workloads demand extreme kernel efficiency across rapidly evolving hardware architectures while handling complex memory layouts—such as non-contiguous **Paged KV Caches** and multi-precision sub-byte/8-bit quantization (**INT4**, **INT8**, **FP4** [`NVFP4`/`MXFP4`], and **FP8** [`E4M3`/`E5M2`/`MXFP8`]). Monolithic operator libraries often suffer from JIT compilation state-space explosion, cold-start latency, and tight coupling between compute math and pointer arithmetic.

**TileGen** (`tilegen`) solves this through four core architectural pillars:

1. **Orthogonal Compute, Layout & Quantization Descriptors**: Separates tile-level compute semantics (`tg.matmul`, `tg.online_softmax`) from physical storage mappings (`ContiguousLayout`, `PagedKVLayout`, and unified `QuantSpec` for `INT4`/`INT8`/`FP4`/`FP8`).
2. **Evolvable Virtual Tile Machine (VTM) IR**: Expresses multi-tier on-chip memory movement (`Global` $\to$ `Scratchpad` $\to$ `ComputeInput` $\to$ `Accumulator`) alongside explicit structural topology decisions ($M_0$: `GridTopology`, `ExecutionRoles`, `RecomputeBoundary`) and tactical levers ($M_1 \dots M_k$), lowering cleanly to **TileLang** or **Triton (`TLE-Lite` / `TLE-Struct` / `TLE-Raw` tiered)** emitters.
3. **Pluggable Kernel Synthesis & Self-Evolution (`src/tilegen/evolve/`)**: Exposes a unified `SynthesisEngine` / `KernelEvaluator` / `KnowledgeStore` interface supporting **ForgeKernel-style reference-guided self-evolution** alongside **evolutionary search**, **world-model tree search**, **RL rollout loops**, and **single-episode coding agents**.
4. **Strict Compile-Time / Runtime Split**: Heavy compiler and synthesis dependencies (TileLang, Triton-TLE, TVM TIR, LLVM, LLM agents) are isolated strictly to offline/build-time AOT compilation. Production serving runs through a zero-compiler-dependency C++ dispatcher (`src/runtime/`) with full **PyTorch SDPA**, **vLLM**, and **CUDA Graph** compatibility.

Serving integrations ship out of the box for **PyTorch SDPA**, **vLLM** (Out-of-Tree plugin), and **[vLLM-PSP-KV](https://github.com/virthub/vllm-pspkv)** (`TILEGEN_PSPKV`, a Triton decode backend for the fork's multi-precision byte cache).

## 🏗️ Supported Architectures, Quantization & Attention Schedules

TileGen synthesizes generation-specific Attention and GEMM kernels from hardware-agnostic specifications (`AttentionSpec`, `QuantSpec`) and automatically dispatches based on target capabilities (`TargetSpec`):

| **Variant / Backend** | **Target Hardware** | **Key Architectural & Quantization Mechanisms** |
| :--- | :--- | :--- |
| **PyTorch SDPA** | Universal | Native `TORCH_LIBRARY` & `tilegen.sdpa_kernel()` dispatch with automatic fallback to standard PyTorch SDPA for dense unquantized inputs. |
| **TileGen-FA1** | Reference / Low-SRAM | Outer-KV / inner-Q tiled online softmax baseline for low-scratchpad targets and schedule ablation. |
| **TileGen-FA2** | NVIDIA Ampere (`sm_80`), Ada (`sm_89`) | Outer-Q / inner-KV warp-partitioned (`Split-Q`) prefill, parallel `Split-KV` decoding with `lse` reduction, native `INT8`/`FP8`, and fused `INT4`/`FP4` register dequantization. |
| **TileGen-FA3** | NVIDIA Hopper (`sm_90a`) | Producer-consumer **Warp Specialization**, `T.tma_copy` asynchronous bulk transfers, ping-pong warpgroup scheduling, native `FP8`/`INT8` `WGMMA`, and fused `INT4`/`FP4` register unpack. |
| **TileGen-FA4** | NVIDIA Blackwell (`sm_100a` / `sm_120`) | Asynchronous `tcgen05.mma` targeting dedicated **Tensor Memory (`TMEM`)** with 2-CTA cooperative MMA, hybrid FMA/SFU `exp2` emulation, conditional softmax rescaling, and native micro-scaled **`NVFP4`**, **`MXFP4`**, and **`MXFP8`** block-scaled instructions. |
| **TileGen-DSA / Triton-TLE** | Multi-Chip Accelerators | Lowers VTM tiers and role tags via `TileLang` or `Triton-TLE` (`TLE-Lite` / `TLE-Struct` / `TLE-Raw`), supporting native `INT8`/`MXFP8`/`MXFP4` matrix ops and Vector-pipelined `INT4`/`FP4` dequant. |


## 🚀 Installation

TileGen installs and runs inside a project-local **`.venv/`** managed by [`uv`](https://docs.astral.sh/uv/). `install.sh` bootstraps `uv` if absent, provisions the venv (CPython 3.12), installs build tooling and runtime dependencies, and compiles the zero-compiler-dependency C++ runtime (`libtilegen_runtime`) via scikit-build-core + CMake:

```bash
./install.sh --backend cuda   # base developer installation
```

| **Option** | **What it does** |
| :--- | :--- |
| `--backend cuda` | Base developer install: bootstraps uv, provisions `.venv` (CPython 3.12), build tooling + runtime deps, compiles the C++ runtime |
| `--backend cuda --runtime-only` | Lightweight production install (zero TileLang/Triton/TVM/agent dependencies) |
| `--with-vllm-pspkv` | [vLLM-PSP-KV](https://github.com/virthub/vllm-pspkv) serving integration (clone, or point at a local checkout via `PSPKV_SOURCE_DIR`) |
| `--backend rocm` | Alternative backend for other GPU targets |

> ⚠️ GPU note: the base install does not include PyTorch, Triton, or TileLang. Provision
> those on the GPU node (e.g. the vLLM environment) and run GPU commands there with
> `PYTHONPATH=src python -m tilegen.cli …`.

## 🛠️ Usage & Unified Workflow

All commands execute directly through `run.sh`, which enforces `uv run --no-sync` inside `.venv/`—no manual `source .venv/bin/activate` step is required:

```bash
./run.sh --bench --causal --seq-len 4096 --head-dim 64
```

| **Command** | **What it does** |
| :--- | :--- |
| `./run.sh --compile --conf <yaml>` | AOT-compile Attention & QuantGEMM kernels (`--backend-ir tilelang` or `triton_tle`) |
| `./run.sh --autotune --conf <yaml>` | Constraint-pruned autotuning of tile schedules |
| `./run.sh --evolve --strategy <name> --conf <yaml>` | Pluggable kernel synthesis & self-evolution (ForgeKernel, Evolutionary, World-Model, RL, Goal-Loop) |
| `./run.sh --test` | Unit tests, numerical parity checks (vs. FP64), CUDA Graph capture tests |
| `./run.sh --verify --conf <yaml>` | End-to-end utilization, dispatch interception & regression gates |
| `./run.sh --bench --causal --seq-len N --head-dim N` | End-to-end latency and roofline benchmarks |
| `./run.sh --benchmark <script> …` | Run a harness from `benchmarks/` (flags are forwarded) |
| `./run.sh --benchmark pspkv_tilegen_bench …` | Fork-dependent vLLM-PSP-KV harnesses (set `TILEGEN_BENCH_PYTHON` / `TILEGEN_BENCH_PYTHONPATH` overrides) |
| `./run.sh --clean` (add `--clean-all` for `.venv/`) | Clean dynamically generated files |

### ⚙️ Codegen tiers

The Triton and TileLang backends dispatch four **runnable** VTM patterns to
specialized kernels (`dense_fa2`, `fa3` warp-spec, paged `decode`, paged
`prefill`); any other topology (e.g. `ring_step`) lowers to an explicitly
marked **sketch tier** — the generated source carries
`# SKETCH TIER - NOT RUNNABLE` and `TILEGEN_STRICT_CODEGEN=1` turns the
sketch path into an error so skeleton code is never consumed as a kernel.

### 🧱 VTM IR

The Virtual Tile Machine IR is the single source of truth: declare a kernel, lower
it to runnable source, and inspect the serialized IR text. Its defining
characteristics:

- **Memory-tiered virtual machine**: on-chip movement (`Global` → `Scratchpad` → `ComputeInput` → `Accumulator`) is expressed as explicit IR stages, not hidden compiler internals.
- **Layout- and quantization-decoupled**: compute ops (`tg.matmul`, `tg.online_softmax`) are independent of storage (`ContiguousLayout`, `PagedKVLayout`) and encoding (`QuantSpec`).
- **Two-level decision space**: structural topology (M₀: `GridTopology`, `ExecutionRoles`, `RecomputeBoundary`) plus tactical levers (M₁…Mₖ) — the evolver mutates tactics while structure stays fixed per variant.
- **Backend-agnostic lowering**: one IR lowers to TileLang or the Triton TLE tiers.
- **Stable identity**: the serialized IR hashes to a semantic SHA-256 used for AOT dispatch and cache keys in the C++ runtime.

```python
from tilegen.backend import ensure_backends_registered, get_backend
from tilegen.ops.attention import AttentionSpec, build_dense_attention_vtm
from tilegen.target import get_target

# 1. Build: a dense FA2 prefill (topology + tactical knobs).
kernel = build_dense_attention_vtm(
    AttentionSpec(batch=1, num_heads=8, head_dim=64, seq_len=512, causal=True),
    get_target("sm_120"),
    variant="fa2_prefill",
)

# 2. Lower: `_variant_of` picks the backend automatically
#    (FA1–FA3 → Triton, FA4 tcgen05 → TileLang).
ensure_backends_registered()
src = get_backend("tilelang").lower(kernel, get_target("sm_120"))

# 3. Serialize: inspect the IR as text.
print(kernel.serialize())
```

Entry points: `build_attention_vtm` (paged, quantized), `build_dense_attention_vtm`
(dense prefill), `build_decode_vtm` (batched paged decode). The same IR feeds the
self-evolution engine — `GeneratedKernelEvaluator` lowers and times candidates, so
the evolver consumes the VTM IR end-to-end.

## 📝 License

This project is licensed under the [MIT License](LICENSE).
