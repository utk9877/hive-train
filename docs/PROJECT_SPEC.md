# Project specification

Status: 2026-10-11. M1 model/reference, M2 transport/state primitives and M3 real serial coordinator/workers implemented with supplemental CPU/socketpair validation. Target CUDA/MPS/LAN acceptance and M4 checkpoint recovery remain pending. See [M1](M1.md), [M2](M2.md), and [M3](M3.md).

## Objective and evidence

Train **one neural network jointly**, with forward computation, backward gradients, and full-parameter optimizer updates distributed between two computers. Independent jobs, inference-only partitioning, and adapter-only training do not meet this objective.

Three separate outcomes:

1. **Correctness:** a distributed step implements the same objective and update as a deterministic unsplit reference, within numerical tolerances.
2. **Capacity:** demonstrate a distributed configuration that trains beyond *each accelerator's* feasible model size under explicitly matched settings. Assess CPU/offload alternatives separately before any claim about an entire machine.
3. **Performance:** measure whether distribution improves throughput or time to comparable quality for models that fit locally. A slowdown is an acceptable research result, particularly for tiny models.

Evidence vocabulary throughout these documents: **D** documented capability; **A** author-reported experimental result; **T** personally tested, with environment and result; **I** engineering inference/design; **U** unverified possibility or target check. A design is not a benchmark. Sources and reuse decisions are in [RESEARCH](RESEARCH.md); platform evidence is in [PLATFORM](research/PLATFORM.md).

## Hardware and scope

| Role | Initial hardware | Constraint |
|---|---|---|
| A: coordinator + CUDA stage 0 | Windows, RTX 4050 6 GB, Ryzen 7000H, 16 GB RAM | GPU memory separate from host RAM; driver, OS, power mode unknown |
| B: MPS stage 1 | Apple M4 MacBook Air, 16 GB unified memory | CPU, GPU, OS, and other apps share memory; sustained thermal behaviour unknown |
| Link | College Wi-Fi or private hotspot | Client isolation, bidirectional throughput, jitter, and disconnect frequency unmeasured |

No Ethernet, cloud GPU, extra accelerator, or fast interconnect is assumed. Both computers execute native Python and the common PyTorch version selected in [SETUP](SETUP.md). Windows hosts the coordinator beside a separate CUDA worker; Mac connects as the MPS worker. Model orientation is initially fixed, later profiled in both directions.

Version 1 covers a single decoder-only transformer, two fixed workers, contiguous block partitions, FP32, AdamW, one microbatch in flight, manual split, untied embeddings, and restart from complete coordinated checkpoints. Membership changes, AMP, compressed activations, tied embeddings, custom accelerator kernels, optimizer offload, and pipelining are not v1 requirements. Offload is still evaluated as a baseline. No distributed collective backend is required.

## Reference workload contract

Authoritative workload definition; architecture owns its execution.

| Setting | M1–M4 deterministic fixture |
|---|---|
| Vocabulary / positions | 256 token IDs; learned token and learned position embeddings; maximum sequence length 128 |
| Blocks | 4 pre-LayerNorm decoder blocks; width 128; 4 heads of width 32; MLP width 512 |
| Block computation | LN → biased Q/K/V linear projections → scaled causal attention → biased output projection → residual; LN → biased linear → GELU (`approximate='none'`) → biased linear → residual |
| Attention | Explicit matmul/scale/mask/softmax/matmul, FP32; strictly upper-triangular `-inf` mask before softmax; `decoder_v2` cancels the softmax-invariant key bias before score arithmetic, as specified below; no FlashAttention/custom kernel requirement |
| Normalization/output | LayerNorm eps 1e-5, affine; final LayerNorm and separate bias-free output projection; all parameters trainable |
| Dropout | 0 throughout initial equivalence tests |
| Batch | Initially b=2 sequences, T=32, K=1 microbatch per optimizer step; accumulation tests K=2 with unequal valid-token counts |
| Objective | Next-token cross-entropy. Dataset forms x=tokens[:, :-1], y=tokens[:, 1:] once. Ignore index -100 for target padding; loss-stage sum divided by total valid targets in the whole step |
| Optimizer | AdamW lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01; one group including all parameters; foreach=False, fused=False, amsgrad=False; no clipping, scheduler, AMP, or loss scaling |
| Initial split | Stage 0: embeddings + blocks 0–1; stage 1: blocks 2–3 + final norm + output head |
| Initialization | CPU FP32; normal mean 0/std .02 for embedding and linear weights; zero linear biases; LN scale 1/bias 0. Stable per-parameter RNG seed from SHA-256 of canonical global parameter name and base seed 2026, specified in implementation; canonical names identical regardless of partition |

M1 freezes the parameter-seed encoding as UTF-8 `hive-train:init:v1:<decimal_seed>:<global_name>`, digest first 8 bytes big-endian masked to 63 bits; canonical names and golden seed tests are documented in [M1](M1.md). A single small CPU state fixture supplies all initial equivalence tests; independently seeding each accelerator is insufficient. Large-model loading must initialize or load only owned parameters, at most one bounded tensor/chunk on CPU, without constructing a full model on any coordinator. See [ARCHITECTURE](ARCHITECTURE.md).

Numerical evaluation revision, 2026-10-20: `decoder_v2` computes the key projection as `linear(x, key.weight, key.bias * 0.0)`. In the conventional expression, adding key bias contributes the same `q @ key.bias` to every allowed key score of a query; softmax cancels this row constant. The revised expression performs that cancellation algebraically before FP32 arithmetic, avoiding a rounding-only gradient amplified by AdamW. The named key-bias parameter is retained with `requires_grad=True`, a present exact-zero gradient, ordinary Adam moments/counter, and decoupled weight decay. This is not a gradient threshold or a post-backward clamp. Parameter names, shapes, ownership, initialization values and model size are unchanged; the other projections retain their usual biases. This equivalence relies on the current dot-product attention with no position-dependent transformation of keys. Both reference and workers use the same revision. Runtime configuration hashes include `architecture="decoder_v2"`, and old `decoder_v1` runs/snapshots are incompatible. Numerical thresholds are unchanged; fresh MPS/CUDA acceptance is still required.

Explicit optimizer experiments: `verify --adam-eps <positive finite value>` and `train_cli coordinator --adam-eps <value>` select a recorded, hashed epsilon on both compared instances or both workers. The default/reference workload above remains eps=1e-8. An eps=1e-6 run is a separate numerical-stability experiment, not acceptance of the default workload; every matched baseline must use the same epsilon. There is no automatic retry or epsilon adjustment after a failure. See M1 and decision D16 for measured motivation.

Synthetic integer fixtures make initial tests offline and deterministic; they are not language-quality evidence. A later small local text corpus uses a fixed byte-level tokenizer (UTF-8 bytes 0–255), train/validation file hashes, split boundaries, and recorded token sampling order. Corpus rights and preprocessing must be recorded before use. Freeze corpus and quality target before comparing speed.

## Required invariants

- Every global parameter has exactly one worker owner and one optimizer owner. Every parameter receives a gradient on the fixture; no accidental detached trainable region.
- A step contains K microbatches. All forwards/backwards in it use the same parameter version. Exactly one optimizer update is applied per worker for that step, after all backward passes finish and both workers report ready.
- Global valid-token normalization is applied once at the loss stage. Boundary gradients already include this factor; upstream workers do not divide again. Empty-token steps are rejected before forward.
- Workers reject stale, duplicate, malformed, oversized, out-of-state, and cross-run messages before mutating training state. No unsafe retry after an uncertain update.
- End-to-end transport is bounded CPU memory, with binary tensor payloads and no untrusted pickle. Protocol identifiers, limits, errors, and transitions are authoritative in [PROTOCOL](PROTOCOL.md).
- A complete checkpoint denotes a single agreed model/optimizer version plus matching data/RNG/config state. A surviving partial shard is not a checkpoint. Resume uses fresh connections and a new run ID.
- GPU claims require explicit device placement, synchronized timing, unsupported-operation failures with MPS fallback disabled, and actual device diagnostics.
- Downloaded dependencies and data suffice for local operation; no external coordinator, public port, or internet access is required for training.

## Acceptance and authority map

| Gate | Objective evidence | Authoritative detail |
|---|---|---|
| M0 readiness | Both environments resolve the selected release; local GPU forward/backward/AdamW and peer TCP echo pass | [SETUP](SETUP.md) |
| M1 portable model | Same initialized model, loss, gradients, and AdamW updates pass unsplit versus local split tests including K=2 masking | [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md), [VALIDATION](VALIDATION.md) |
| M2 transport | Tensor bit-preserving round trips; fragmentation, length limits, disconnects, timeouts, duplicate controls tested | [PROTOCOL](PROTOCOL.md) |
| M3 joint training | Windows CUDA + Mac MPS run actual forward/backward/updates; reference equivalence and short-learning checks pass | [VALIDATION](VALIDATION.md) |
| M4 recovery | Interrupted executions restore the last complete checkpoint with matched state and no repeated update | [ARCHITECTURE](ARCHITECTURE.md), [VALIDATION](VALIDATION.md) |
| M5 capacity + static policy | Reproducible fit/OOM evidence for matched accelerator baselines; offload assessment; measured split enumeration | [FEASIBILITY](FEASIBILITY.md), [VALIDATION](VALIDATION.md) |
| M6 pipeline | Multiple in-flight microbatches preserve equivalence and measured memory limits; speed measured rather than required | [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) |
| M7 adaptation | Safe-boundary migration costs included; policy beats or abstains versus best fixed configuration on predeclared traces | [FEASIBILITY](FEASIBILITY.md) |
| M8 extension | Third worker/backend obeys same ownership, versioning, transport, and checkpoint invariants | [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) |

M5 may conclude that these machines cannot meet the capacity objective under a particular workload. Record that finding and refine model dimensions within the same baseline class; do not quietly change precision, optimizer, sequence length, or offload assumptions. Hardware/setup unknowns are nonblocking planning assumptions, tracked in [RISKS](RISKS.md).
