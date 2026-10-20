# Validation and benchmark contract

Status: test specifications with supplemental CPU results in [M1](M1.md), transport/state results in [M2](M2.md), and real CPU process training results in [M3](M3.md). Target GPU/LAN acceptance remains pending. Updated 2026-10-11; research access date 2026-10-07. [PROJECT_SPEC](PROJECT_SPEC.md) fixes the workload; [PROTOCOL](PROTOCOL.md) fixes messages; [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) fixes milestone order. Never turn an expected result below into a claimed observation.

## Reproducibility and evidence record

Each run records source revision (or source tree hash before Git exists), run/attempt IDs, config/hash, environment, model/partition/global parameter names, init algorithm/fixture hash, data/tokenizer/split hashes, sampler order/seed, batch b/T/K, valid token counts, optimizer and checkpoint settings, fallback/TF32 flags, start/end reason, raw step logs, memory, and timeouts. Record Windows and Mac separately; hardware clocks, power mode, background load, and Wi-Fi topology matter. Keep secrets and dataset payloads out of logs.

Before comparisons, create the small full CPU FP32 fixture once and load its corresponding tensors on each device. Equal seeds on different device RNGs are not sufficient. Verify every named tensor loaded exactly and every parameter has exactly one stage owner. Use dropout 0, TF32 disabled, fixed causal mask, fixed next-token shift, same loss and AdamW parameters. No AMP, custom kernels, activation compression or stochastic sampling differences. Deterministic algorithms may be requested diagnostically; unsupported deterministic operations fail visibly rather than being hidden by CPU fallback.

## M0–M1 local and numerical acceptance

1. Run SETUP's diagnostics on actual CUDA and MPS. Then exercise the **complete** fixture: embedding, positional addition, matmul, masking, softmax, LayerNorm, GELU, logits, summed cross-entropy, backward, and AdamW. Assert declared device for all parameters/gradients and moment tensors (scalar optimizer counters may be CPU). Synchronize before pass reporting. Capture unsupported-operation errors with `PYTORCH_ENABLE_MPS_FALLBACK=0` set before import.
2. Build unsplit and same-process split instances from identical CPU state. The split test must detach/copy activation, construct a new device leaf, call suffix backward, and feed `leaf.grad` into the retained prefix output. Verify `.is_leaf`, nonmissing boundary gradient, matching shape/dtype, and no accidental cross-stage autograd graph.
3. Compare loss, logits, boundary derivative, every named parameter gradient, updated parameters, and both Adam moments/counters. Compare the *update delta* as well as weights: a small relative error on a large unchanged weight can conceal a missing update. Do 1 and 10 steps with identical deterministic batches. Integer counters, ownership/name sets and token counts must match exactly.
4. Test K=2 against one effective batch: b=2,T=32 per microbatch; mask targets to yield different positive valid counts, then make one microbatch entirely ignored. Loss is the sum over valid targets divided once by total step tokens. Gradients accumulate without an extra K/worker average. Reject an all-empty step before compute. Constant-width left-aligned target masking suffices; initial attention inputs remain valid fixed-length sequences.
5. Exercise every legal tiny-model cut k=1,2,3. Force a wrong boundary shape, duplicate backward request, missing gradient and stale identifier in the future boundary wrapper; each must fail clearly rather than continue.

### Numerical thresholds

Initial acceptance thresholds for the tiny FP32 fixture (I, must be recorded before hardware runs). Elementwise comparison means `abs(actual-reference) <= atol + rtol*abs(reference)` for **every** value, with finite-value checks; report worst name/index, max absolute error and normalized L2 error per tensor. Compare near-zero tensors using the absolute threshold, not unstable relative percentages.

| Comparison | Loss / logits `(atol,rtol)` | Gradients, boundary gradient, moments, parameters and update deltas `(atol,rtol)` |
|---|---|---|
| CPU unsplit vs CPU split | `(1e-6,1e-5)` | `(1e-6,1e-4)` |
| Same accelerator unsplit vs split | `(1e-5,1e-4)` | `(1e-5,1e-3)` |
| Mixed CUDA–MPS vs CPU FP32 reference | `(1e-4,1e-3)` | `(1e-4,1e-2)` |

Additionally require per-tensor normalized L2 gradient/update error ≤1e-3 for CPU, ≤5e-3 for same accelerator, ≤2e-2 for mixed devices when reference norm >1e-6; denominator is `max(norm(reference),1e-6)`. For tiny reference norms report absolute errors and ensure zero/nonzero behaviour is explained. These conservative mixed-backend tolerances are not permission to miss parameters: assert parameter coverage, nonmissing gradients and nonzero total update norm. Permit reference-consistent structural zeros (for example key-projection bias under softmax shift invariance); do not require every gradient tensor to have nonzero norm. If a threshold fails, localize the first divergent operation against CPU; investigate normalization/init/detach/TF32/device math before any tolerance change. A documented threshold revision needs measured error evidence and independent review, never an automatic loosening.

Transport round trips of finite FP32/int64 fixtures are byte-exact before accelerator computation; accelerator losses/gradients are **not** required bitwise equal. Cross-device same-seed equality is not assumed.

M1 numerical failures retain the completed cases and the incomplete case in the JSON report. The `failure` object identifies the zero-based step, split and scenario, failed criterion, absolute error, actual/reference norms and the flattened index/values with the largest absolute error. Parameter failures also report gradient, update and Adam moment/counter norms for both instances. These are diagnostics only: the command still exits nonzero, and the thresholds and training computation are unchanged. Keep the original failing report and use a new output filename for the diagnostic rerun. Small key-bias gradients may be structural-zero rounding residuals; this is a hypothesis to investigate against CPU/high precision, not an automatic exemption for updates.

As of 2026-10-20, the separate `decoder_v2` evaluation revision performs the mathematical key-bias cancellation before score arithmetic (PROJECT_SPEC; decision D15). All thresholds above still apply, including key-bias gradients and updates; there is no tensor-name exception. CPU regression tests require matching conventional biased attention forward/input/parameter derivatives in FP64 with nonzero key biases, exact-zero FP32 key-bias gradients/moments, preserved Adam step counters/weight decay, and identical initialization fingerprints. New reference, GPU and process reports must identify `decoder_v2`; historical `decoder_v1` pass/fail evidence cannot establish acceptance of the revision.

The `--adam-eps` option records a separate optimizer experiment. Epsilon must be identical across split/unsplit comparison, both workers, reference replay and matched baselines; it changes config hashes. Default eps=1e-8 results remain distinct from an eps=1e-6 experiment. All acceptance thresholds still apply without exceptions. Failure reports now include the actual gradient, before/after parameter and both moments at the comparison's worst flattened index; M1 also retains the config hash on numerical failure. A CPU FP64 probe can identify sensitivity but is diagnostic only, not the FP32 production/reference workload or an assumption of MPS FP64 support.

## M2 transport and state-machine acceptance

Test both local sockets and actual Windows↔Mac LAN path. Cover every header length boundary, every legal dtype, little-endian fixture decoding, descriptor/byte-size agreement, one-byte fragmentation, many coalesced frames, small writes, early EOF at each frame segment, and noncontiguous source tensors made contiguous deliberately. Hash payloads before/after round trip. Include 1 KiB, 256 KiB, 1/4/16/64 MiB and actual model boundary shapes.

Negative tests: wrong magic/version/flags, excessive metadata/depth, JSON duplicate keys/nonfinite values/bool-as-int, oversized or overflowing shapes, unknown message/field/dtype, wrong authentication tag, repeated nonce/session/run/sequence, stale step/version/microbatch, duplicate apply, unexpected control during compute, bad target indices and nonfinite gradients. No malformed message may cause allocation beyond configured caps, GPU indexing, or optimizer execution. Assert a fake optimizer invocation counter remains zero for rejected commands and at most one for a valid step.

Throttle/stop the receiver, stall halfway through header/payload/tag, and exhaust bounded queues. Verify configured deadlines terminate the session and that peak buffered bytes remain within the implementation's declared bound. Heartbeats may not extend an absolute compute deadline; large frames may delay heartbeats within the transfer deadline. Collect socket backlog/buffer settings with measurements. Authentication tests do not establish a formal security audit.

## M3 real distributed training acceptance

Run coordinator + CUDA worker on Windows and MPS suffix on Mac. Require diagnostic assertions and trace evidence of actual local forward/backward/updates; two CPU workers or inference do not pass. Run the full numerical suite against the small CPU unsplit reference and compare CUDA-only/MPS-only controls. Coordinator logs reconcile all activation/backward/apply/commit IDs, versions, valid counts and tensor hashes where enabled. Every global parameter must have one gradient and one update per step.

Then execute at least 100 steps over the same deterministic learnable byte pattern (for example repeated `0..255`, precomputed on CPU) and over a small fixed local text train/validation split. Evaluate initial and every 10 steps on held-out fixed batches, `eval()` and no gradients. Require finite losses, lower final pattern validation loss than initial, and distributed final validation loss within max(0.02 nats/token, 2% of reference loss) of each healthy unsplit reference; report complete curves, not just endpoints. For text data, comparable behaviour is the gate: do not assume arbitrary 100-step text training must improve. If all references fail to learn the pattern, debug the workload before blaming distribution. Longer quality/performance comparisons use predeclared targets and ≥3 seeds.

## M4 recovery acceptance

Produce initial version-0 checkpoint and per-step checkpoints in fault tests. Compare restored parameters, moments, counters, config, ownership, CPU/device RNG and next data cursor with the last complete checkpoint. Same-device replay should pass same-accelerator tolerances; stochastic modules are off for initial recovery tests.

| Inject failure | Required outcome |
|---|---|
| Mid-forward / partial activation / partial gradient | Both workers stop; no next-step progress; last complete checkpoint remains usable |
| After STEP_READY but before APPLY | Abort or controlled stop; no update retries |
| Between workers' updates; lost STEP_APPLIED; during optimizer | Treat version as uncertain; discard both live states; new run restores last complete checkpoint |
| Coordinator dies after both applied but before commit/manifest | Workers stop; uncheckpointed work is replayed from complete version, never inferred from surviving memory |
| Worker dies during shard write; only one shard ready | No new complete manifest; orphan files ignored |
| Manifest committed but CHECKPOINT_COMMIT/ACK lost | Existing valid manifest is still complete; load it only after all shard verification |
| Hash mismatch, missing shard, truncated index | Reject checkpoint and use newest older complete generation; if none, report unavailable |
| Delayed old-run frame or attempted reconnect | Reject; fixed membership cannot rejoin a live run |

After each fault restore under a **new run_id**, run 10 further steps and compare to uninterrupted execution starting from the same checkpoint/data. Validate save/load/first-update peaks, disk-full errors and bounded staging for a large synthetic shard without instantiating a full training model on coordinator. Process-kill tests establish process-crash behaviour; power-loss/storage durability claims need separately performed tests. Report the distinction.

## M5 capacity track

Freeze one experiment tuple: architecture/vocabulary/positions/depth/width, FP32, full-parameter AdamW and optimizer options, b/T/K/effective batch, loss, activation-checkpointing policy, data, memory safety policy, optimizer state placement and no compression. Change model depth across separate candidate tuples. For a candidate, run:

| Baseline | Definition / interpretation |
|---|---|
| CUDA-only | Whole model+grads+Adam state on RTX; host data/staging only; no automatic offload |
| MPS-only | Whole model+grads+Adam state on MPS; default allocator guards; same physical DRAM as host; fallback off |
| Distributed fixed split | Same whole logical model/settings, local stage ownership; include both device and host peaks |
| CPU-only per machine | Same FP32 model/AdamW/data, CPU computation; OS RAM limit/headroom recorded; may be slow but can invalidate an entire-machine impossibility claim |
| Practical single-machine offload | Separately tested CPU optimizer/state or supported offload path with identical mathematical optimizer/objective; record framework support, precision, transferred state, RAM and time |

If no practical offload implementation is available, record **not tested / setup unsupported**, not OOM or impossible. DeepSpeed backend documentation is a lead, not proof that its Windows/offload combination installs. A diagnostic baseline may be implemented later, or state arithmetic may rule out a specific fully resident arrangement; neither proves every offload strategy impossible. Mac CPU offload shares physical memory and may add copies. Windows CPU offload can use separate host RAM and PCIe, trading speed for capacity. Disk/NVMe offload and arbitrary recomputation are outside the initial baseline, explicitly excluded from claims.

A fit requires cold init, complete forward/backward/first AdamW step, ≥100 finite steps, ≥20-minute sustained run, and complete checkpoint save/restore without exceeding documented headroom or sustained swap growth. OOM probes run in isolated processes; record phase, stack/error, last memory values, shape, and actual failure versus preflight estimate. Do not intentionally push Mac into OS termination: conservative safety rejection must be labelled as such, not called allocator OOM. Repeated runs establish the usable threshold range. A distributed success beyond both matched single-accelerator thresholds passes the accelerator-capacity objective; qualify whether CPU/offload succeeds, fails, times out, or is untested. State that exact baseline in any claim.

## M5–M7 performance track

Use models that fit on **both** individual devices. Compare CUDA-only, MPS-only, manual static distributed split, best measured static split, later pipeline, later adaptive policy. Keep effective batch, sequence length, precision, AdamW, full objective, tokenizer/data and validation target identical. If b changes, adjust K to preserve effective batch/token semantics; change T only as a new workload track. Keep evaluation/checkpoint frequency equal or report amortized costs separately. An equal-layer split alone is an insufficient optimization baseline.

Measure valid tokens/s, samples/s, step wall time p50/p95/p99, full-run elapsed time, accelerator utilization where available, peak host/device memory, wire bytes, transfer/copy time and idle fraction. Report time to the same predeclared held-out loss (and total trained tokens), with censoring when a run never reaches it. One-step speed is not a convergence result. Use ≥20 warmup steps and ≥100 timed steps, plus a ≥20-minute sustained sample; repeat ≥3 times and use ≥3 seeds for learning comparisons. Show individual runs and confidence intervals via run-level or block bootstrap, accounting for correlated Wi-Fi/thermal drift. Do not count warmup as zero cost in time-to-quality.

CUDA: synchronize timing boundaries, reset/read peak allocated and reserved stats around the trial, record driver free/total and other-process use. MPS: synchronize and sample `current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory` at stage boundaries and periodically alongside process/OS metrics. Sampling can miss short peaks; report observed maxima as sampled, not exact allocator high-water marks. Record process RSS and available RAM/pressure/swap without summing shared pages twice. No claim of GPU performance if fallback was enabled or meaningful compute ran on CPU unnoticed; CPU data/control/copy work is intentional and reported separately.

For static search report all valid split candidates, rejected candidates and reason, measured-vs-predicted time and chosen plan. For adaptation use the cost rule in FEASIBILITY; compare against best initial static, best hindsight static, an aggressive reactive baseline and an offline oracle clearly labelled as diagnostic. Include state transfer, checkpoint, restart, warmup and profiling overhead in elapsed time. Test jitter-only traces for no unnecessary moves, sustained drift for break-even behaviour, and failed migration for correct rollback.

## Evidence from this planning run

- Primary-source documentation/paper/repository research and official wheel-index inspection were performed; no external runtime was installed or reproduced.
- Local environment inspection observed Python 3.14.6, macOS 15.5 ARM64, and no discoverable torch in that interpreter. This is the orchestration environment, not proof that either target machine is configured as specified.
- Seven setup Python snippets and the autograd pseudocode passed syntax parsing; memory/communication examples were recomputed with standard-library Python. All 85 local Markdown links and referenced anchors across 13 documents resolved; the proposed `!4sHHIIQ` prefix passed 24-byte packing/unpacking checks. Schema/state consistency was reviewed manually, not exercised by an implemented codec.
- No CUDA/MPS model step, Windows shell execution, peer Wi-Fi transfer, capacity OOM probe, migration, or checkpoint recovery experiment has been performed. M0–M8 hardware gates remain pending.

## Subsequent implementation evidence — 2026-10-08

The combined M1/M2 unittest suite passes **97 tests** in the supplemental macOS ARM64 Python 3.11.15/PyTorch 2.13.0 CPU environment. M2 uses real authenticated byte streams over socketpairs, not TCP/Wi-Fi: malformed schemas, fragmentation/coalescing, EOF at every frame byte, corrupt authentication, stale identifiers, bounded transfer deadlines/backpressure, sealed admission, bidirectional payload sizes through 64 MiB, codec bit preservation, and fake-step relay/update barriers are exercised. Source-hashed receipt: `artifacts/validation/m2-local.json` (local ignored artifact).

Independent integration review found and fixed barrier heartbeat-version races and growing apply-ID history; reviewed regressions pass. Current tests establish declared single-frame bounds and absence of internal queues, **not** sustained RSS limits or target network throughput. Checkpoint state messages are tested with fake events, not disk durability or restoration. Runnable target probe commands and precise limitations are in [M2](M2.md). No additional GPU/capacity/performance claim follows from these tests.

## M3 CPU implementation evidence — 2026-10-11

122 tests pass in the supplemental environment above. Real worker tests exercise autograd and AdamW over authenticated socketpairs across all legal cuts and ten steps each, with K2 masking and entirely ignored microbatches inside nonempty steps. Per-name losses/logits/boundary gradients/parameter gradients/updates/moments/counters pass M1's numerical thresholds. Invalid target values/counts, NaN activation/boundary payloads, missing gradients, optimizer corruption, interrupted updates, and disconnect after a real update stop without replay; cleanup and stage ownership are checked.

Separate-process verification passed ten K1 and ten masked K2 steps plus 100 synthetic-pattern steps. Pattern evaluation start157 is excluded from training; transitions overlap by design in the repeating byte pattern. Loss at that fixed start fell 5.587015→4.377883 and matched the CPU reference. This is a bounded synthetic learning check, not text-corpus quality or accelerator acceptance. New snapshot comparison verifies final-step evidence only and rejects mismatched run/config/stage identities. Source-hashed receipt and process reports are `artifacts/validation/m3-*.json`; commands and target limitations are in [M3](M3.md).
