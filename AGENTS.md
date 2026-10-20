# Project instructions

This project researches and builds joint neural-network training on a Windows RTX 4050 6 GB laptop and an Apple M4 MacBook Air with 16 GB unified memory over local Wi-Fi. Read [README.md](README.md), [the specification](docs/PROJECT_SPEC.md), and [the implementation plan](docs/IMPLEMENTATION_PLAN.md) before implementation; consult the authoritative architecture, protocol, setup, and validation documents linked there.

- Preserve existing work. Keep changes scoped; do not install globally, expose public ports, weaken firewalls, or change drivers without a concrete need.
- Separate correctness, capacity, and measured speed. Never claim hardware tests, mixed CUDA–MPS support, or novelty without evidence. Record source URLs, access dates, assumptions, and test environments.
- Initial implementation: native Python/PyTorch, CUDA and MPS local autograd, CPU-buffered binary TCP, FP32 AdamW, contiguous partitions, untied embeddings, fixed membership, one microbatch in flight. Do not add NCCL-on-Windows, MPS collectives, cloud services, custom drivers, or CUDA translation as prerequisites.
- Keep identifiers, ownership, normalization, step barriers, and restart semantics consistent with docs/PROTOCOL.md. Never retry an uncertain optimizer update; abort and restore a complete checkpoint.
- Use bounded memory and validated binary payloads; never deserialize untrusted pickle. Do not assemble the entire large model or optimizer on the coordinator.
- Use virtual environments, explicit device placement, ordinary portable operations, and small modules. Label commands for future code as proposed until that code exists.
- Validate local forward/backward, framing, gradients/updates against a deterministic reference, recovery, and measured peak memory before performance claims. Use docs/VALIDATION.md for tolerances and matched baselines; record CPU fallback and synchronization.
- When multiple agents are requested, establish a shared contract first, assign disjoint file ownership, and leave shared document integration to the orchestrator. Require an independent review of integrated designs. Keep temporary assignments and progress in docs/IMPLEMENTATION_PLAN.md, not here.
- Research/planning runs must not silently expand into a full training implementation. Architectural changes require updating the relevant authoritative documents and decisions together.
