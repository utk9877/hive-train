# Feasibility, profiling, and optimization

Status: engineering estimates and experiment design, 2026-10-07. **No target memory, throughput, latency, or training result has been measured.** D/A/T/I/U meanings are in [PROJECT_SPEC](PROJECT_SPEC.md). Arithmetic below is I and was checked locally with standard-library Python; it is not a hardware benchmark. [SETUP](SETUP.md) owns connectivity instructions and [VALIDATION](VALIDATION.md) owns evidence requirements.

## What is feasible in principle

Local autograd with explicit boundary derivatives is sufficient to train a split differentiable model (I; see [ARCHITECTURE](ARCHITECTURE.md)). No common accelerator collective is necessary. A slow link affects efficiency, while bounded staging and stage ownership permit combining *usable* memory. Aggregate advertised memory is not a single allocation pool: each indivisible layer, its local state, and its workspaces must fit its owner. The 6 GB RTX and 16 GB Mac cannot be described as a 22 GB GPU.

Correct two-device training is the first gate. Increased capacity depends on actual free memory, partition balance and model shape; a model beyond both individual accelerators is plausible, not established. A tiny model will likely be slower distributed (I) because the first serial schedule exposes every transfer and does not overlap device computation.

## Memory accounting and candidate budgets

Use bytes in logs; report GiB as bytes/2^30 and distinguish vendor-labelled GB. For P local FP32 trainable parameters with ordinary FP32 AdamW:

`M_state ≈ 4P parameters + 4P gradients + 4P first moments + 4P second moments = 16P bytes`.

There is no separate FP32 master-weight copy in this FP32 baseline. Add per-parameter counters, optimizer temporaries, and allocator overhead. Adam moments are lazily allocated, so inference or forward-only peaks do not predict training fit. `foreach=False, fused=False` avoids foreach tensor-list overhead, not all workspace. D: [AdamW documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html), accessed 2026-10-07.

`M_peak_device = M_state + saved_activations + live_outputs/grads + operator_workspace + device_staging + allocator/framework reserve`.

`M_peak_host = process/runtime + dataset window + CPU tensor serialization/receive/relay buffers + initialization/checkpoint staging + any explicitly enabled offloaded state`.

On Mac, these are overlapping views into the same physical DRAM. Do not add RSS and all MPS-reported allocations as if disjoint; record driver/process/OS observations and use OS memory pressure plus conservative physical-residency estimates. On Windows, host and GPU allocations occupy separate budgets: host RAM is not additional fast VRAM. Pinning host memory is not initially required.

| Resource | Initial planning envelope (I, not measured or guaranteed) | Measurement needed |
|---|---|---|
| RTX 4050 | Plan at most ~5 GiB total training allocations pending actual 6 GB device/free figures; retain ~1 GiB for display/framework/variation | CUDA allocated and reserved peaks, driver free/total, cold init, Adam first step, longest forward/backward |
| Windows 16 GB host | Start with ≤8 GiB total application RSS across coordinator+worker; leave several GiB for OS/other tasks; less if idle usage high | Combined processes, available physical RAM, page faults/paging, communication/save peaks |
| Mac 16 GB unified | Start with ~9–10 GiB total application residency; preserve ~4–6 GiB for OS/other use, then tune only from observed pressure | MPS current/driver allocation, recommended working set, RSS, OS pressure and swap growth, cold/save peaks |
| Coordinator relay | At hard 64 MiB frames, budget up to 256 MiB for two connections' one inbound/one outbound payload each, plus conversion/socket buffers and data window | Real allocator peak; aliasing may reduce this, extra conversions may increase it |
| Each worker staging | Up to 128 MiB at frame cap plus any dtype/serialization copy and bounded checkpoint chunk | Exact in-flight buffer ownership and peak RSS |

These are conservative starting envelopes, not artificial proof of OOM. Capacity claims must report hardware/framework limits and safety policy, attempt the matched full run within that policy, and explain whether failure was actual OOM, safety-limit rejection, or time limit. Apply the same policy to singles and distributed workers.

For manual attention, a materialized score/probability tensor alone is `4*b*heads*T*T` bytes; backward may keep several such intermediates per block. Hidden vectors scale with `b*T*d`; MLP activations with `b*T*4d`; final logits with `b*T*V`, often with another loss/gradient buffer. Do not assign a universal activation multiplier: profile the selected PyTorch version and operation graph. Endpoint costs matter: a large vocabulary can make the untied output head and logits dominate the suffix. Activation checkpointing trades retained activations for recompute and is a separately matched variant, not part of the first baseline.

### Parameter arithmetic and capacity search

For the specified biased pre-LN block with MLP width 4d, `P_block=12*d*d+13*d`. Total parameters with untied token/output embeddings and learned positions are `P=L*P_block+2*V*d+Tmax*d+2*d`. Stage 0 at split k owns `k*P_block+V*d+Tmax*d`; stage 1 owns `(L-k)*P_block+V*d+2*d`.

| Model family example | Parameters | State alone (GiB) | Interpretation |
|---|---:|---:|---|
| Fixture L4,d128,V256,Tmax128 | 875,264 | 0.013 | Correctness only; distribution has no capacity rationale |
| L48,d1024,V256,Tmax512 | 605,669,376 | 9.025 | Exceeds RTX state capacity; may still fit Mac, so not a beyond-both witness |
| L64,d1024,V256,Tmax512 | 807,208,960 | 12.028 | Candidate for beyond-both accelerator track; Mac fit/OOM must be measured |

For L64 and k=14, state alone is 2.639 GiB on RTX and 9.389 GiB on Mac, before activations/overhead. More RTX blocks may balance memory better; enumerate k rather than accepting this example. Start capacity runs at b=1,T=128,K=1 (Tmax512) and increment depth, comparing the **identical** model/batch/sequence/optimizer to both singles. Use 16 heads for d1024. Explore depths 32,40,48,56,64,72 with legal partitions; near a threshold refine depth. Do not promise that L64 fits distributed or fails on Mac. The tiny vocabulary is an explicit byte-level objective, not evidence that an equally sized large-vocabulary pretrained model fits.

Checkpointing at a boundary moves ~12P bytes for FP32 parameters and two moments (gradients discarded) plus counters/RNG/metadata; retained previous checkpoints increase disk use. Budget ≥2 complete checkpoint generations plus one in-progress generation locally, and test free disk before starting. Stream tensor chunks (e.g. 16 MiB), including parameters larger than a chunk, without gathering state. Stage creation must avoid even a temporary full CPU model: construct owned modules or meta modules materialized per parameter, load moments directly to their final owner, and test cold initialization and restore. Normal `state_dict()` references may be cheap, but cloning all tensors to CPU is not.

## Wi-Fi communication model

Let `S=4*b*T*d` be one FP32 boundary payload, `r_AB/r_BA` measured directional **application** bytes/s, and `lat_AB/lat_BA` fixed per-message/control costs. Approximate one serial microbatch:

`t_micro ≈ F_A+B_A+F_B+B_B + S/r_AB+S/r_BA + lat_AB+lat_BA + D2H_A+H2D_B+D2H_B+H2D_A + host_pack/relay/control`.

Then `t_step ≈ K*t_micro + optimizer_A+optimizer_B + barriers/data`, with actual overlap measured (optimizer work can overlap, so summing it is conservative). Do not double-count allocation/staging if measured effective transfer costs already include them. The relay adds loopback and host copies, but a boundary activation crosses Wi-Fi once, and its gradient once. Targets add `8*b*T` bytes on A→B, typically small; inputs travel loopback. TCP acknowledgments, authenticated headers, Wi-Fi contention and retransmits consume additional time.

Illustrative **payload-only** transfer time for activation plus gradient, equal rates each direction; excludes latency, copies, targets, control, contention, optimizer and compute. Rates are assumptions in decimal Mb/s, not measured claims:

| b,T,d | One tensor (MiB) | 20 Mb/s | 100 Mb/s | 300 Mb/s |
|---|---:|---:|---:|---:|
| 2,32,128 | 0.03125 | 0.026 s | 0.0052 s | 0.0017 s |
| 1,128,512 | 0.25 | 0.210 s | 0.042 s | 0.014 s |
| 1,512,1024 | 2 | 1.678 s | 0.336 s | 0.112 s |
| 4,512,1024 | 8 | 6.711 s | 1.342 s | 0.447 s |
| 1,1024,1024 | 4 | 3.355 s | 0.671 s | 0.224 s |

Two peers on one access point may consume two wireless hops per delivered byte; use measured end-to-end rates, not advertised Wi-Fi link speed. A hotspot may improve or worsen this. College client isolation is a binary connectivity gate; throughput tuning cannot fix it.

### Measurement procedure

1. Follow SETUP to find actual interface IPs, bind the Windows listener, scope firewall access, pass loopback and peer echo, and compare a permitted private hotspot if needed. Keep devices stationary and record network topology, band, signal, VPN state and background traffic.
2. Run 4-byte/1 KiB request/response latency with warmup then ≥1,000 observations. Report RTT p50/p95/p99, max and timeout count; do not call RTT/2 measured one-way latency. Repeated tests at different times capture variability.
3. Sweep 1 KiB, 256 KiB, 1,4,16,64 MiB and all actual boundary shapes, ≥30 timed transfers after ≥5 warmups each direction. Use random finite FP32 bytes and int64 fixtures for correctness, then protocol-realistic authenticated frames. Record effective bytes/s, allocations, tail latency and retries at TCP level if available. Do not reconnect per message.
4. Separately synchronize and measure accelerator→CPU, CPU serialization/contiguity, local loopback relay, LAN transfer, CPU→accelerator; report blocking allocation and reused-buffer variants separately. MPS shared memory does not make these free.
5. Record full training step wall time from STEP_BEGIN through both STEP_COMMITTED plus per-stage intervals. Cross-host clocks need not be synchronized: measure durations locally and round trips at the coordinator. Compare decomposed prediction to actual time.
6. Repeat the chosen configuration after ≥20 minutes sustained training on AC power; record temperatures/power where accessible, Mac pressure/swap, and drift. Do not confuse cold peak performance with steady behaviour. Do not change clocks or safety limits.

## Static split selection and pipelining

For each model and fixed benchmark objective enumerate k=1..L−1 and both physical device orientations. Protocol 1 binds stage0/worker0 to CUDA and stage1/worker1 to MPS; reverse-orientation profiling is offline initially. Running the reversed topology requires an explicit later role-to-device map in RUN_CONFIG and capability checks, before selecting it for deployment. Enumerate b in {1,2,4} and K so b*K equals the declared effective batch. Sequence lengths {128,256,512} are separate workload tracks, never substituted mid-comparison. Exclude b/T values exceeding tensor caps, device/host budgets or legal attention dimensions.

Profile each block/end point forward+backward on each device, plus stage-level full graphs and optimizer. Layer sums are a coarse predictor; retained activations, cache behaviour and allocators make stage measurements authoritative. For M3's serial schedule minimize the *sum* of stage compute and transfer costs subject to memory. Balancing stages primarily matters once work overlaps; equal layer counts are not a reliable proxy for either compute or memory. Validate finalists and report prediction error and regret versus best measured feasible candidate.

At constant effective batch, smaller b can reduce graph memory but increases K, per-message costs and pipeline opportunities. Larger b amortizes latency and may improve device utilization but enlarges activations/logits. Larger T grows transfer linearly and ordinary attention workspace quadratically; it changes the task and must be held fixed for a fair speed comparison. Changing k between same-width blocks does not reduce boundary S; it changes ownership, memory and device compute. Include endpoint asymmetry.

M6 begins with a bounded synchronous flush schedule, then considers 1F1B if flush retention is prohibitive. All microbatches use one weight version. The ideal balanced p-stage pipeline efficiency `K/(K+p−1)` is only a simplified bound ignoring backward asymmetry, transfers and resource contention, not a forecast for two laptops. Measure graph count/peaks and actual overlap; reject schedules that exceed memory even if the compute model predicts speedup.

## Migration policy and research hypothesis

At M7, use rolling robust stage/link timings (initial proposal: three consecutive 20-step windows), uncertainty intervals, minimum residency 100 steps and no-change candidate. Thresholds are tunable experiment parameters, not established optimal values. Estimate savings over `H=min(remaining_steps, credible_duration_of_changed_conditions)`:

`H * lower_confidence_bound(t_current − t_candidate) > upper_confidence_bound(C_migration) + margin`.

Use paired/block-resampled traces to reflect autocorrelation; where uncertainty is too large, abstain. Include profiling overhead and compare against fixed measured splits. A proposed margin is max(10% of migration estimate, one current step), tested by sensitivity analysis. Short spikes should not cause reassignment.

`C_migration = drain + checkpoint/verification + state_staging/transfer + deallocate/reallocate + load + warmup + resume/barriers`.

At a safe boundary, moving Q FP32 parameters means at least `12Q` bytes for weights and two Adam moments, plus counters/RNG/config; not just 4Q parameter bytes. A 100M-parameter move is ≥1.2 GB, ≥96 s at assumed 100 Mb/s before other costs. If it saves 0.2 s/step, payload alone needs >480 future steps to repay. Windows relay/local copies and checkpoint work can raise the break-even horizon substantially.

Drain all work, commit/checkpoint old assignment, free only after recoverable ownership is established, then stream into a memory-feasible new plan with a fresh run ID. Avoid simultaneously holding old and new full stages; sequential unload/load may require restart to release allocator caches. Verify name/state hashes and both workers' readiness before resuming the same data cursor and optimizer counters. Interrupted migration restores the old complete checkpoint. No partial live-graph migration, weight staleness or optimizer resets.

Hypothesis: conservative adaptation can improve net time under sustained drift, while declining moves on noisy Wi-Fi. Existing systems already adapt scheduling/placement; [RESEARCH](RESEARCH.md) constrains novelty. Test constant, sustained compute slowdown, sustained bandwidth drop, and transient-jitter traces against fixed/manual/profiled and hindsight baselines. Include a negative control: at fixed b/T/d in the serial schedule, a bandwidth-only change leaves boundary traffic unchanged across k, so k-only migration should normally abstain. Benefits would require batch/schedule/orientation changes or coupled compute effects. Extra devices/backend search is M8, after the two-worker policy is understood.

## Go/no-go gates

| Gate | Pass evidence | Fallback if it fails |
|---|---|---|
| Connectivity | Trusted peer TCP link works | Permitted private hotspot; otherwise local development only |
| Backend coverage | Local complete decoder backward/update succeeds with fallback off | Portable replacement for unsupported operation; CPU diagnostic does not pass GPU gate |
| Numerical semantics | Split gradients/updates pass declared tolerances | Fix boundary/init/normalization before optimization |
| Memory feasibility | Full step, cold init, checkpoint and restore fit with measured headroom | Different valid split/workload dimensions; document if capacity objective is unmet |
| Capacity witness | Same distributed workload fits and both matched single-accelerator runs fail, with offload assessment | Report RTX-only gain or no gain precisely |
| Useful performance | Repeated matched throughput/time-to-quality improvement | Retain correct capacity-focused runtime without speed claim |
| Adaptation benefit | Net improvement including profiling/migration and uncertainty | Remain on best static assignment |

D measurement API sources, accessed 2026-10-07: [CUDA peak allocations](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html), [MPS allocation/driver/working-set and synchronize APIs](https://docs.pytorch.org/docs/2.11/mps.html). CUDA peak allocation is not total device use; MPS current/driver observations are not automatically a peak tracker. VALIDATION specifies sampling and reporting limits.
