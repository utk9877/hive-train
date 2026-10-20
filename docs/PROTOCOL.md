# Training protocol 1

Normative application protocol. M2 implements schemas, framing, authentication and pure state gates; [M3](M3.md) integrates real serial training. M4 checkpoint storage remains future work. See [M2](M2.md) for transport APIs and tested limitations. [ARCHITECTURE.md](ARCHITECTURE.md) defines model semantics, ownership and checkpoint contents. The Windows coordinator binds one IPv4 listener to **0.0.0.0:29550**, so it accepts both CUDA's loopback connection and MPS's connection to the Windows local IP. Scope external firewall permission to the verified Python executable, port, local interface address and Mac peer; never expose public ports. No separate data/control/heartbeat ports. TLS is a future extension for untrusted networks, not implemented; initial deployment uses a trusted private LAN/hotspot and a per-run secret. Authentication alone does not encrypt tensors or metadata.

## Framing and validation

Every frame is `24-byte prefix | metadata | payload | 32-byte authentication tag`. The prefix is Python `struct` layout `!4sHHIIQ` (network/big-endian integers): magic `b"HTRN"`, protocol major `1` (uint16), flags `0` (uint16), metadata bytes (uint32), reserved `0` (uint32), payload bytes (uint64). Reject unknown versions, nonzero flags/reserved fields, empty metadata, metadata over 65,536 bytes, or payload over negotiated `max_tensor_bytes` before allocating payload memory. Default and protocol-1 hard maximum: 64 MiB (`67,108,864` bytes). Control frames have zero payload. One tensor per frame; tensors are not JSON arrays. Initial training refuses configurations whose largest tensor exceeds the limit; chunked training transport requires a future protocol revision.

Metadata is strict UTF-8 JSON, one object, maximum nesting depth 8. Reject duplicate keys, unknown fields per message schema, NaN/Infinity, booleans where integers are required, and integers outside `[0,2**63-1]` except dimension-independent `ignore_index=-100` in run configuration. Common fields, always present:

| Field | Definition |
|---|---|
| `type` | Exact uppercase name from the message table |
| `run_id` | Coordinator-generated UUID, lowercase canonical string; new on every restart |
| `sender`, `receiver` | `coordinator`, `worker-0`, `worker-1`; connected identity must match sender |
| `seq` | Independent monotonically increasing frame sequence per connection direction, starts at 0; no gaps or duplicates |
| `step_id` | Optimizer step index, or null outside step scope |
| `microbatch_id` | Zero-based index in the current step, or null |
| `parameter_version` | Number of committed/applied updates associated with this message; null for handshake and pre-configuration CAPABILITIES. RUN_CONFIG carries starting_version; RUN_READY echoes loaded_version. Early ERROR/ABORT may use null until configuration is accepted |
| `body` | Per-message object containing exactly its declared fields |
| `tensor` | Null for control; descriptor for tensor messages |

Tensor descriptor fields are exactly `name`, `dtype`, `shape`, `byte_order`, `nbytes`. `dtype` is `float32`, `int64` or `uint8`; item sizes 4, 8, 1. `shape` is a list of 1–4 strictly positive integers; every dimension at most `2**31-1`. `byte_order="little"`, C-contiguous layout implicit, no strides or object types. Use checked multiplication: `product(shape)*itemsize == nbytes == payload_length <= max_tensor_bytes`. Check the descriptor against the configured message shape/dtype **before** reading payload. On a hypothetical big-endian host, byteswap explicitly. Reject nonfinite activations/gradients on device before sending and after receipt; token bounds and target ignored values are checked before device indexing/loss.

`recv_exact(n)` loops until exactly `n` bytes are available, handles partial reads, and treats `recv()==b""` as terminal EOF even at a frame boundary. Read and validate prefix, metadata, then payload and tag; authenticate before constructing tensors or dispatching commands. Prefix plus metadata must finish within 10 seconds of the first prefix byte, without extending that deadline on each read. A frame may arrive fragmented or several frames may arrive together. A single writer serializes complete frames using a complete-write loop or `sendall`; no interleaving from heartbeat/compute threads. Do not serialize GPU storage, use pickle, execute network-supplied code, or accept arbitrary file paths.

Owned writable CPU buffers back received tensors. Their lifetime extends through any host-to-device copy. Sending owns its immutable CPU buffer until complete write returns. The receiver keeps at most one inbound tensor, the writer one outbound tensor per connection, plus at most 16 small control frames. The coordinator admits only the expected worker/frame for the current phase; it may hold one complete tensor while relaying and pauses reading that source when destination capacity is unavailable. Count this memory and unavoidable conversion copies in profiling. TCP flow control plus bounded queues provide backpressure; never launch the next microbatch to relieve queue pressure. Local socket buffer limits are measured/configured, not assumed zero.

## Admission and integrity

Distribute a fresh random 32-byte run secret through a private out-of-band channel and supply it through a protected local file or interactive prompt. Never put it in command-line arguments, checkpoints, logs or Git. Do not expose 29550 publicly. Fixed worker IDs are configured locally; admission permits one connection per expected ID, at most two pending handshakes and two admitted sockets; reject excess connections and close admission once membership is sealed. Before `WELCOME`, only the three handshake frames below are accepted, with a 10-second total handshake deadline and 16 KiB metadata cap. Their trailing tag is 32 zero bytes. Hex strings use lowercase hex of the specified byte count.

1. Coordinator sends `CHALLENGE`, `seq=0`, with `body={server_nonce:<32-byte hex>}`, sender `coordinator` and receiver `pending` (a CHALLENGE-only exception). The slot is identified by the following HELLO, not by remote address or connection order.
2. Worker sends `HELLO`, `seq=0`, with `body={worker_nonce:<32-byte hex>, proof:<32-byte hex>}`. Its sender is its configured worker ID and receiver is `coordinator`. Define `auth_bytes = b"HTRN1\0" + UUID(run_id).bytes + server_nonce_bytes + worker_nonce_bytes + worker_id.encode("ascii")`. `proof = HMAC-SHA256(secret, auth_bytes)`. Validate in constant time. Nonces are cryptographically random and never reused.
3. Coordinator sends `WELCOME`, `seq=1`, correct receiver worker ID, with `body={proof:<32-byte hex>, max_tensor_bytes:67108864}` where proof is `HMAC-SHA256(secret, b"server\0" + auth_bytes)`. A smaller server-selected positive cap is allowed; it must fit the client's locally configured cap or admission fails (no symmetric negotiation). The worker validates before disclosing diagnostics/model/data. Derive `session_key = HMAC-SHA256(secret, b"session\0" + auth_bytes)`.

The literal byte strings above contain NUL bytes (`\0`), not backslash characters. Every subsequent frame's tag is `HMAC-SHA256(session_key, prefix_bytes + exact_metadata_bytes + payload_bytes)`. Authenticate exact bytes, not reserialized JSON. Sequence numbers continue from handshake (worker next 1, coordinator next 2); authenticated common sender/receiver fields prevent reflection. JSON must be parsed with bounded limits before authentication to find frame structure; never act on its contents until the tag verifies. A forged/invalid admission closes that connection; after membership is sealed any authenticated worker fault aborts the run. This modest integrity layer is an engineering design, not a formally audited security protocol; trusted local participants remain a requirement.

## Message definitions

`C` means coordinator, `W` either worker, `0/1` worker IDs. Every control has `tensor=null`; tensor messages have the exact descriptor given. Relay frames get a new coordinator sender/sequence and destination while retaining run/step/microbatch/version. No worker opens a connection to the other worker.

| Type and direction | Body / tensor; legal effect |
|---|---|
| `CHALLENGE`, `HELLO`, `WELCOME` | As above; step/microbatch/version null |
| `CAPABILITIES` W→C | `platform` bounded object (OS, architecture, Python, torch build, backend, device name, CUDA driver/runtime or macOS version), `diagnostics_passed` bool, `fallback_enabled` bool, `memory` object (total/available/allocated/reserved bytes, unknown values null); after authenticated admission |
| `RUN_CONFIG` C→W | `config_hash`, `config` (model/optimizer/data/init specs from architecture), `stage_id`, `block_start`, `block_end`, `checkpoint_id` UUID or null, `starting_step`, `starting_version`, `max_tensor_bytes`, `timeouts` object; no tensor/code/path payload |
| `RUN_READY` W→C | `config_hash`, `stage_id`, `loaded_version`, `parameter_name_hash`; confirms successful local init/load and validated ownership |
| `STEP_BEGIN` C→W | `microbatch_count`, `total_valid_tokens`, `batch_plan_hash`; idle→accumulating, clear parameter gradients once |
| `STEP_STARTED` W→C | Empty; accepts immutable step/version and has cleared gradients |
| `TARGET` C→1 | `valid_tokens`; tensor name `targets`, int64 `[b,T]`; stage 1 holds it until activation/backward |
| `TARGET_READY` 1→C | Empty; corresponding targets validated and retained |
| `INPUT` C→0 | Empty; `input_ids`, int64 `[b,T]`; starts forward only after TARGET_READY |
| `ACTIVATION` 0→C→1 | Empty; `hidden`, float32 `[b,T,d]`; original output/graph retained on 0 |
| `BOUNDARY_GRAD` 1→C→0 | `loss_sum` finite number, `valid_tokens`; `hidden_grad`, float32 same `[b,T,d]`; stage 1 has finished backward and released its graph |
| `MICRO_DONE` 0→C | Empty; original graph backward complete, graph released; coordinator may schedule next m |
| `STEP_PREPARE` C→W | Empty; only after every microbatch completes, seals accumulation |
| `STEP_READY` W→C | `gradient_finite` (must true), `microbatches_completed`; prepared, no graph retained |
| `STEP_APPLY` C→W | `apply_id` UUID, `next_version`; requires ready, current `v`, next `v+1` |
| `STEP_APPLIED` W→C | `apply_id`, `next_version`; exactly one local update completed and synchronized; common parameter_version is `v+1` |
| `STEP_COMMIT` C→W | `apply_id`, `next_version`; sent after both APPLIED, marks committed boundary |
| `STEP_COMMITTED` W→C | `apply_id`; common version `v+1`; idle and safe to checkpoint/start next step |
| `CHECKPOINT_BEGIN` C→W | `checkpoint_id` UUID; idle committed boundary only |
| `CHECKPOINT_READY` W→C | `checkpoint_id`, `shard_index_hash` SHA-256 hex, `shard_bytes`; worker derives local path from configured root + safe UUID/stage, never network path |
| `CHECKPOINT_COMMIT` C→W | `checkpoint_id`, `manifest_hash`; coordinator published complete manifest |
| `CHECKPOINT_ACK` W→C | `checkpoint_id`; paused→idle; next step waits for both |
| `STATUS` W→C | `phase` fixed state enum, `last_completed_microbatch` nullable integer, `last_progress_age_ms` integer; heartbeat and progress, no state mutation |
| `PING` C→W, `PONG` W→C | `nonce` uint63; liveness, never authorizes a state transition |
| `ERROR` either direction | `code` enum, `detail` sanitized string max 1,024 chars; terminal, no retryable application errors |
| `ABORT` C→W | `code`, `detail`; terminal; release resources and exit, no checkpoint of uncertain state |
| `STOP` C→W, `STOPPED` W→C | Empty; graceful termination only at idle committed boundary; coordinator ordinarily checkpoints first |

Microbatch messages require current step, version and `0<=m<K`; all other operational messages use `microbatch_id=null`. STEP_APPLY common version is the input version; STEP_APPLIED/COMMIT/COMMITTED use the new version. CHECKPOINT messages have step ID equal to the next step to execute and committed version; other non-step operational messages have step null. Initial starting step/version are 0/0; after applying step 0, version is 1 and next step is 1. Restore preserves these numbers but gets a new run ID.

Nested object schemas are fixed as follows (all keys required; no extensions). Strings are at most 256 UTF-8 bytes unless a tighter rule is given. Hashes are lowercase 64-character SHA-256 hex. Nonnegative integers use the common uint63 restriction; positive quantities reject zero.

- `CAPABILITIES.platform`: string fields `os`, `architecture`, `python`, `torch`, `backend` (`cuda`, `mps`, or diagnostic-only `cpu`), `device_name`; nullable strings `cuda_driver`, `cuda_runtime`, `macos_version`. `memory`: nullable integer fields `total_bytes`, `available_bytes`, `allocated_bytes`, `reserved_bytes` (backend-specific meanings documented in results).
- `RUN_CONFIG.config`: objects `model`, `optimizer`, `data`, `init`, plus string `parameter_spec_hash`. `model`: positive integers `layers`, `width`, `heads`, `mlp_width`, `vocab_size`, `max_sequence_length`; enums `architecture="decoder_v1"`, `dtype="float32"`; all operation semantics are fixed by [PROJECT_SPEC](PROJECT_SPEC.md), including zero dropout. Require `width % heads == 0` and `layers>=2`. `optimizer`: `name="adamw"`, finite floats `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, with `lr,eps>0`, betas in `[0,1)`, weight decay nonnegative; `foreach=false`, `fused=false`, `amsgrad=false`. `data`: `dataset_hash`, `tokenizer="byte_v1"` or `"synthetic_v1"`, positive integer `max_microbatch_size`, positive integer `sequence_length`, `ignore_index=-100`; sequence length cannot exceed model maximum. `init`: `algorithm="named_cpu_v1"`, integer `master_seed`.
- `RUN_CONFIG.timeouts`: positive integer milliseconds `connect_ms`, `handshake_ms`, `idle_ms`, `status_interval_ms`, `compute_ms`, `checkpoint_ms`, `transfer_no_progress_ms`; positive integer `min_directional_bytes_per_second`; finite positive `transfer_multiplier`. The fixed transfer floor is 30,000 ms. Stage IDs are 0 or 1; block endpoints are integers satisfying the complete, nonoverlapping partition described in architecture.
- A configuration hash is SHA-256 of UTF-8 `json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)` produced by the common Python release. Workers reserialize and verify it. The parameter specification hash identifies the canonical global name/shape/dtype list frozen in M1; `parameter_name_hash` hashes that stage's ordered UTF-8 names joined by newline with no final newline. Runtime counters, partition assignment and checkpoint ID are outside `config`, so a validated future repartition can preserve the logical workload hash.

In tensor messages `b` comes from the checked descriptor, is in `1..max_microbatch_size`, and must match the corresponding target/input/boundary frame; `T` equals configured sequence length. `valid_tokens<=b*T`; summed valid counts must equal STEP_BEGIN total. Configuration and checkpoint files are selected from preconfigured local roots/IDs, never downloaded executable objects.

## State and update safety

Coordinator states: `ADMITTING → CONFIGURING → IDLE → ACCUMULATING → PREPARING → APPLYING → COMMITTING → IDLE`. Checkpoint adds `IDLE → CHECKPOINTING → IDLE`. `ABORTED` and `STOPPED` are terminal. Worker states match, with `PREPARING` ending in `READY`; only compute thread may transition to `APPLYING`.

The coordinator waits for both RUN_READY; both STEP_STARTED; for each microbatch TARGET_READY, boundary gradient and MICRO_DONE; both STEP_READY; both STEP_APPLIED; then both STEP_COMMITTED. It compares actual valid-token counts against the step plan and rejects missing, repeated or out-of-order microbatches. One activation must yield exactly one gradient and one stage-0 backward. No next step before the final commit barrier. Fixed membership is sealed before RUN_CONFIG; there is no in-run rejoin.

Workers store `(run_id, step_id, parameter_version, apply_id, phase)` in process memory. STEP_APPLY is valid only in READY for the expected current version; mark APPLYING **before** invoking the optimizer. A duplicate sequence, apply ID, step or wrong version is terminal; never call `optimizer.step()` a second time or resend a step in response to a missing acknowledgment. This provides at-most-once application inside a live run, not durable exactly-once execution. After any process loss, discard all live state and restore a complete checkpoint; the fresh run ID fences delayed old frames. Coordinator loss likewise stops workers and requires restart.

`apply_id` is deterministic: `str(uuid.uuid5(uuid.UUID(run_id), f"step:{step_id}"))`, with the step's canonical decimal nonnegative index. Both gates require this ID before application. Keep only the active ID and monotonic step/version, not an ever-growing history of UUIDs. Reuse of a prior step's ID fails derivation; duplicate current-step commands fail phase/counter checks. UUIDv5 here identifies an operation; HMAC, not UUID hashing, provides authentication.

All failures after admission are fail-stop. Error codes: `BAD_FRAME`, `AUTH_FAILED`, `VERSION_MISMATCH`, `WRONG_STATE`, `DUPLICATE`, `SHAPE_MISMATCH`, `NONFINITE`, `DEVICE_ERROR`, `OOM`, `TIMEOUT`, `PEER_LOST`, `CHECKPOINT_ERROR`, `CONFIG_MISMATCH`. Log code, worker, run/step/microbatch/version and recent phase, never secrets or raw dataset payloads. Best-effort ABORT is not an acknowledgment barrier: locally close immediately if a peer is gone, then terminate compute processes safely. A timeout during APPLYING is explicitly uncertain and cannot be repaired by replay.

## Deadlines and progress

Use monotonic time; no synchronized wall clocks required. Initial configurable defaults: connect/handshake 10 s; control idle liveness 30 s; STATUS every 5 s; compute phase 300 s; checkpoint 600 s. For tensor transfer, absolute deadline from prefix receipt is `max(30 s, 5*payload_bytes/min_measured_directional_bytes_per_second)`; reject a run lacking measured bandwidth or use a conservative configured 1 MiB/s bootstrap rate. Also impose 30 s without byte progress. The absolute deadline cannot be extended forever by a trickle of bytes. Tune phase budgets using measured p99 plus margin and record them in RUN_CONFIG; no silent timeout auto-extension.

A STATUS heartbeat proves the I/O loop is alive, not that the GPU made progress. Only a phase-completion event resets the compute deadline. Because TCP is one ordered stream, a heartbeat cannot overtake a large in-progress tensor; while a bounded transfer is progressing, its transfer deadline governs liveness. The single writer may prioritize a control frame only between whole frames. If device compute blocks the Python interpreter and heartbeat emission stalls, use the compute phase deadline during that declared phase rather than treating 30 s as a second hidden compute timeout. No ping or heartbeat can extend the absolute compute/checkpoint deadline. These rules permit slow valid operations and still detect hangs.

Update-barrier heartbeat versions account for crossed packets: during APPLYING/COMMITTING only, a worker may **receive PING**, and the coordinator may **receive PONG**, with either active step input version `v` or adjacent applied version `v+1`. Newly sent PING uses the coordinator's observed version of that peer (advance after its APPLIED); newly sent PONG uses the worker's current local version. In coordinator COMMITTING, both observed versions are already `v+1`. All other phases, STATUS, training and checkpoint messages retain exact expected versions. Heartbeats never change training state or renew compute deadlines. Serialized per-direction processing ensures old heartbeat frames precede the corresponding commit barrier; do not dispatch heartbeat replies later from an unordered queue.

M2 implements stricter bounds than the maxima above: one pending handshake at a time, no application frame queues, one active reader/writer per stream, serial one-payload relay. `accept_members` owns and closes its listener on success/failure and seals after two distinct IDs; an admission error terminates that attempt. M3 converts transport/compute failures into run-wide connection closure and passes absolute phase deadlines to blocking receives. Its synchronous loop emits no periodic STATUS during blocked device work, so compute deadlines govern those waits. A hung device call may require process termination; these library primitives cannot preempt kernels or restore checkpoints. Current M3 limits K to four and supports only fresh initialization.

## Required codec and recovery tests

Test arbitrary fragmentation/coalescing, zero-byte EOF at every frame position, invalid UTF-8/JSON/duplicate keys, overflowing shape products, wrong length/dtype/endian, oversized declarations, bad tags, stale run/step/version, repeated sequence/apply, partial writes, bounded queue saturation and control starvation. Assert no optimizer update on malformed input. Inject a lost APPLIED, failure between stage updates, and coordinator crash before/after checkpoint manifest publication. All uncertain-step cases must restart from a validated complete checkpoint without accepting late frames. See [VALIDATION.md](VALIDATION.md) for numerical and benchmark tests.
