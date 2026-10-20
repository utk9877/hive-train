# Architectural decisions

Status: accepted for first implementation unless marked deferred; evidence reviewed 2026-10-07. Design choices are engineering inferences, not tested hardware claims. Sources live in [RESEARCH](RESEARCH.md) and [PLATFORM](research/PLATFORM.md).

| ID | Decision and reason | Alternatives / reconsideration trigger |
|---|---|---|
| D01 | Native Python and one compatible PyTorch release, native CUDA/MPS math. Backend-specific wheels share application semantics. | WSL2 would add setup complexity and does not establish MPS collectives. Revisit only if native Windows blocks a necessary feature. |
| D02 | Explicit stage-boundary autograd over CPU-buffered TCP. Backend portability and transparent correctness outweigh early transport tuning. | Gloo CPU staging is a later measured alternative, contingent on both wheel builds and functional tests. PyTorch pipeline/RPC integrations are not assumed to support this exact pair. No NCCL on Windows or MPS requirement. |
| D03 | Windows coordinator relays one connection from each worker on TCP 29550; CUDA connects locally, Mac outbound. Coordinator owns no model/optimizer. | Direct peer transport may remove host copies later, but adds endpoint/firewall complexity. A relay still crosses Wi-Fi once per activation or gradient. |
| D04 | Full-parameter contiguous model partitioning, no replication. A owns prefix, B suffix/loss initially. | DDP/local SGD/DiLoCo alone require a complete replica per participant and do not meet capacity goal. Tensor parallelism demands more frequent communication. Evaluate reversed placement at M5. |
| D05 | FP32, untied embeddings, portable attention, AdamW single-tensor implementation. Establish a reference with transparent memory/state. | Tied weights require shared ownership/gradient synchronization. AMP, fused kernels and compression add numerical/backend variables; defer until equivalence. |
| D06 | One microbatch in flight first; accumulate K serial microbatches at fixed parameter version. | GPipe-style flush schedule first at M6; synchronous 1F1B later only if activation retention warrants it. No stale-weight/asynchronous updates. |
| D07 | All-ready then STEP_APPLY then all-ack before next step. Abort on uncertainty; recover only a complete checkpoint. | This is a synchronization barrier, not an atomic distributed transaction. Exactly-once application holds within a healthy run; crash recovery rolls the entire run back. Durable per-step consensus/replay is unnecessary initially. |
| D08 | Worker-local sharded checkpoints plus small coordinator completion manifest; bounded stagewise initialization/loading. | Gathering model/Adam state onto the coordinator violates the capacity objective. Peer-local shard loss requires an older available checkpoint or separately configured backup. |
| D09 | Protocol version 1, explicit binary framing, validated metadata and no pickle; authenticated admission on a trusted private LAN. | Plain TCP does not encrypt data. Use a trusted hotspot initially; TLS can be added before untrusted-LAN use without opening public ports. No internet rendezvous. |
| D10 | Enumerate legal split points and a small configuration grid using measured stage/network/memory costs. | Reinforcement learning is unjustified for two devices and a small discrete space. Larger-device optimization only after search cost becomes a measured problem. |
| D11 | Change partition only at drained, committed safe boundaries when predicted savings exceed all migration costs plus uncertainty. | Threshold/hysteresis/cost-aware placement already have prior art; contribution is a hypothesis about this constrained mixed-backend setting, not a novelty assertion. |
| D12 | Publish correctness, capacity, and performance results separately with matched baselines. | An accelerator OOM is not proof a machine cannot train with CPU/offload, checkpointing, reduced precision, or more time. Baseline variants must be explicit. |
| D13 | M2 uses deterministic run/step UUIDv5 apply IDs plus active phase/version gates; no historical UUID set. | Independent review found unbounded history growth. UUID identifies an operation, not a security token; HMAC remains the authentication mechanism. Exact derivation is in PROTOCOL. |
| D14 | Update-barrier receive-only PING/PONG tolerance for adjacent versions; outgoing heartbeat versions reflect observed/local state. | Independent review reproduced crossed-heartbeat false aborts. Training/STATUS/checkpoint version checks remain strict; FIFO dispatch and unchanged compute deadlines are required. |

## Evidence-driven alternatives

DeepSpeed, SWARM/Hivemind, MLX, and pipeline frameworks are useful prior art, not mandatory dependencies. A framework may save code if a small prototype proves this exact Windows CUDA–macOS MPS path, memory ownership, and recovery contract without introducing unsupported collectives. Separate vendor/backend listings are insufficient evidence. [RESEARCH](RESEARCH.md) records current capabilities and license checks, including recently documented capabilities rather than relying on historic limitations.

Initial dependencies are limited to the setup-selected PyTorch build and its resolved dependencies; Python's standard library covers transport/control/metadata. Test-only tools may be added at M1. No standalone CUDA Toolkit is assumed for prebuilt wheels and ordinary operations. Checkpoint binary tensor storage should reuse the protocol's validated tensor encoding plus JSON manifest; adopting safetensors later requires a version/license review but must not alter ownership.

## Deferred decisions and triggers

- **Precision/compression:** compare wire FP16 or lossless compression only after loss/gradient/error budgets and bandwidth profiles exist. Compute FP32 remains the baseline; changing wire precision creates a new experiment.
- **Activation checkpointing:** add as a matched baseline and optional optimization once activation memory is dominant; measure recompute and dropout RNG correctness. Never enable only on the distributed run while claiming equal settings.
- **Checkpoint frequency:** correctness tests checkpoint each step; later choose based on measured duration, loss-of-work target, and failure rate. No checkpoint policy excuses mixed-version recovery.
- **Encryption:** initial private-LAN assumption must be documented. Shared college Wi-Fi with untrusted peers requires transport protection before carrying real data, or use a private hotspot.
- **Third device:** introduce explicit membership/topology manifest and multi-boundary routing only at M8. Preserve per-parameter ownership and no full-state coordinator requirement.

Change these decisions alongside the authoritative protocol/spec and regression tests. Temporary agent assignments belong only in [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md).
