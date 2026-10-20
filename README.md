# Hive Train

Building a distributed application that trains **one neural network jointly** on a Windows RTX 4050 laptop (6 GB VRAM, Ryzen 7000H, 16 GB RAM) and an Apple M4 MacBook Air (16 GB unified memory), over local Wi-Fi or a private hotspot.

Status, **2026-10-20**: M1 model/reference, M2 transport/state primitives and M3 serial coordinator/workers are implemented. Separate CPU worker processes execute real forward/backward/AdamW steps and match the reference. User-run Mac MPS numerical suites pass with Python 3.13/PyTorch 2.11.0, `decoder_v2` and explicit AdamW epsilon 1e-6; Windows CUDA, peer TCP/Wi-Fi and actual joint-training acceptance remain pending. Checkpoint recovery is next at M4. Correctness, increased trainable capacity, and speed are separate goals; no capacity gain or speedup has been demonstrated.

## Recommended design

Native Python/PyTorch on both systems; native CUDA and MPS computation. Windows runs a CPU coordinator and a separate CUDA prefix worker; Mac runs the suffix/loss worker. A single Windows TCP listener on port **29550** relays bounded binary CPU tensor buffers. Workers run local autograd and exchange activations forward and boundary gradients backward. FP32 AdamW, untied embeddings, contiguous blocks, one microbatch in flight, fixed membership, and synchronous update barriers establish correctness first. Failures stop the run and restore a complete sharded checkpoint.

No NCCL-on-Windows, MPS collective, CUDA translation layer, Kubernetes, cloud GPU, or standalone CUDA Toolkit is required. Local training does not require internet after dependencies/data are downloaded. Initial plaintext transport assumes a trusted private network; no public ports.

## First steps

1. Read [project instructions](AGENTS.md) and the [specification](docs/PROJECT_SPEC.md).
2. Follow [SETUP](docs/SETUP.md) on each machine: create isolated environments, run GPU forward/backward/update diagnostics, then verify local TCP connectivity. Exact OS, driver and link performance are setup checks.
3. Run the implemented [M1 local diagnostics](docs/M1.md) on each accelerator. Acceptance requires that split loss, all gradients, and AdamW updates agree with the unsplit reference under [declared tolerances](docs/VALIDATION.md#numerical-thresholds).
4. Run the [M2 authenticated transport probe](docs/M2.md) on the actual link, then follow [M3 training launches and comparison](docs/M3.md). M3 supports fresh serial training; checkpoint resume is still future work.

The selected package pair is CPython 3.13 with PyTorch 2.11.0, platform-appropriate wheels. [SETUP](docs/SETUP.md) owns installation and compatibility qualifications. Earlier blocked downloads required cached Python 3.11.15/PyTorch 2.13.0 for supplemental CPU tests; the selected pair now passes Mac local numerical suites at explicit epsilon 1e-6. Local `doctor`, `verify`, `fixture` commands and the `train_cli`, `local_run`, `snapshot_check` modules exist; exact commands are in M1/M3.

Update, 2026-10-20: user Terminal evidence confirms MPS execution, but the ten-step verifier exposed rounding-only key-bias update sensitivity. The current `decoder_v2` evaluation cancels the softmax-invariant key bias before score arithmetic while retaining its parameter and optimizer state. Numerical thresholds are unchanged. Use matching updated source on both machines; old `decoder_v1` runs are incompatible. Revised target GPU acceptance remains pending; see the [M1 investigation](docs/M1.md#mps-investigation--2026-10-20).

Follow-up: at the explicit experimental eps=1e-6 setting, all nine ten-step `decoder_v2` same-device MPS cases and all nine MPS-versus-CPU cases pass (90 compared steps per suite) in the user's supplemental Python 3.11.15/PyTorch 2.13.0 environment, with fallback disabled. Local Mac numerical checks are complete for that configuration. Same-device MPS also passes at default eps=1e-8, but its CPU-reference check still fails one ordinary weight element at the first masked step. Defaults and numerical thresholds remain unchanged. See [M1 cross-device findings](docs/M1.md#cross-device-optimizer-sensitivity). The selected deployment package pair, Windows CUDA, LAN and joint training acceptance remain pending.

Selected-pair follow-up: both nine-case ten-step suites also pass on Mac with Python 3.13.14/PyTorch 2.11.0 at epsilon 1e-6, fallback disabled. Local Mac numerical checks are complete for that configuration; Windows CUDA, LAN and joint training remain pending. See [M1 selected-pair evidence](docs/M1.md#selected-package-pair-on-mac).

## Documentation map

| Document | Authority |
|---|---|
| [PROJECT_SPEC](docs/PROJECT_SPEC.md) | Objectives, scope, exact tiny workload, invariants and acceptance map |
| [RESEARCH](docs/RESEARCH.md) | Primary-source prior art, reuse/licensing, capability distinctions and novelty hypotheses |
| [FEASIBILITY](docs/FEASIBILITY.md) | Complete memory budgets, illustrative Wi-Fi costs, static search and migration break-even rule |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Processes, ownership, backward boundary, graph lifetime, checkpoints and recovery |
| [PROTOCOL](docs/PROTOCOL.md) | Normative version-1 frames, schemas, messages, state transitions and failure handling |
| [SETUP](docs/SETUP.md) | Machine/shell-specific installation, GPU diagnostics, scoped firewall, IP/hotspot and local network checks |
| [PLATFORM evidence](docs/research/PLATFORM.md) | Version/wheel/driver evidence and source qualifications |
| [IMPLEMENTATION_PLAN](docs/IMPLEMENTATION_PLAN.md) | M0–M8 deliverables, dependency order, parallel work and review log |
| [VALIDATION](docs/VALIDATION.md) | Numerical thresholds, fault injection, capacity/offload baselines and fair performance comparisons |
| [M1 implementation and commands](docs/M1.md) | Runnable local diagnostics, frozen interfaces, tests and actual environment limitations |
| [M2 implementation and commands](docs/M2.md) | Authenticated network probe, transport/state integration contract and local-test limitations |
| [M3 implementation and commands](docs/M3.md) | Real serial training, separate-process CPU verification, target launch commands and snapshot comparison |
| [RISKS](docs/RISKS.md) | Detection, mitigations, fallback decisions and unresolved setup facts |
| [DECISIONS](docs/DECISIONS.md) | Selected architecture, alternatives, rationale and reconsideration triggers |

## Research and verification status

Multiple specialists drafted platform/setup, prior art and runtime documents. Service usage limits interrupted some agents; the orchestrator completed evaluation/feasibility and integrated the surviving drafts. An independent specialist reviewed the integrated contracts and identified corrections recorded in the plan.

Research includes official PyTorch/CUDA/MPS documentation, SWARM, Hivemind, DiLoCo/OpenDiLoCo, heterogeneous/cross-vendor training, DeepSpeed, MLX, and inference-only precedents. Each source is linked with an access date and capability/evidence label. None establishes this exact pair's training behaviour without testing.

The planning run passed document-link, snippet-syntax, framing-arithmetic and model-budget checks. M1 passed 19 tests and nine CPU split/reference scenarios for ten steps each. M2 adds schema/authentication, framing, timeout, fail-stop state, codec and fake-step integration tests, including bidirectional 64 MiB socketpair payloads. Independent reviews produced interruption-safety and update-barrier fixes with regression tests. [M1](docs/M1.md) and [M2](docs/M2.md) record the offline environment and exact limitations. No target GPU step, Windows shell execution, peer Wi-Fi benchmark, OOM/capacity test, checkpoint recovery, or migration was performed.

M3 adds real CPU training over authenticated streams, tested across every legal tiny-model cut, serial masked accumulation, interrupted updates, and separate worker processes. A 100-step synthetic-pattern run improves fixed evaluation loss and matches its reference. [M3](docs/M3.md) records exact results and limitations. Snapshot files are diagnostic evidence, not resumable checkpoints.

Remaining assumptions: exact target OS/driver versions, available memory/disk, permitted peer connectivity/hotspot, power/thermal behaviour, and measured device/link performance. These do not block CPU prototype development. A capacity claim must state whether each single accelerator fails under matched training settings and what CPU/offload alternatives can do; adding devices does not guarantee faster training.
