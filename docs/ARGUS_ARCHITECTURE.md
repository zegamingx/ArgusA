# Argus Runtime and Project Architecture Contract

**Status:** Architecture baseline for implementation  
**Repository:** `zegamingx/ArgusA`  
**Branch audited:** `master`  
**Audited revision:** `63b94a589b60a7739796903e7591c41e9efb74e0`  
**Roadmap authority:** Existing Argus M1–M10 roadmap

This document fixes the project boundaries and runtime contracts used by subsequent implementation work. It does not add milestones, reorder M1–M10, or authorize merging `master` into `main`.

## 1. Non-negotiable architecture decisions

1. Argus has one model architecture and one backend-independent training/state contract. GPU vendors do not receive separate model or training implementations.
2. Vulkan is the primary compute backend. CPU is a recovery/fallback backend when Vulkan is unavailable, incompatible, out of memory, or fails a validated operation. A performance-calibration policy must not silently route healthy supported workloads to CPU.
3. Device adaptation may select execution parameters such as dispatch shape, precision path, micro-batch size, temporary workspace, and memory residency budget. It must not silently rewrite an existing model architecture or invalidate a checkpoint.
4. Model weights, optimizer state, tokenizer state, and training transaction state must remain recoverable independently of the currently selected compute backend.
5. Device selection is capability-driven. Vendor IDs and GPU names are diagnostic metadata, not proof that a feature or kernel can execute.
6. Keep `:app` as the sole Gradle module during this architecture foundation. Do not introduce module boundaries until they provide independently testable ownership and a measurable build or maintenance benefit.
7. Use released, stable Android, Kotlin, Gradle, NDK, CMake, C++, and Vulkan APIs for the 2026/2027 development window. Do not add preview APIs or deprecated interfaces.

## 2. Audited repository baseline

At the audited revision, the repository has one Android application module and these Kotlin package groups under `com.argus.agi`:

- Application entry points: `ArgusApp.kt`, `MainActivity.kt`
- `data/`: Room database, DAOs, entities, and repository
- `model/`: architecture configuration, Genesis, tokenizer, neural engine, training manager, adaptive policies, Vulkan backend, and residency registries
- `sensor/`, `service/`, `skill/`, and `ui/`: application features and presentation

There are no separate `device/`, `runtime/`, `training/`, `backend/`, `memory/`, or `persistence/` Kotlin packages yet. Most engine, orchestration, runtime policy, and Vulkan bridge responsibilities currently sit in `model/`.

The native source is also currently flat under `app/src/main/cpp/`: the JNI Vulkan implementation, Tensor Arena header, and compute shaders are siblings. CMake explicitly compiles the existing shader files from those paths. Moving them is therefore a coordinated CMake/JNI change, not a cosmetic folder rename.

Relevant existing components:

| Concern | Existing implementation |
|---|---|
| Model topology and identity | `ArgusArchitectureConfig.kt` |
| Device-selected trainable model tier | `ArgusDeviceModelPolicy.kt` |
| RAM-based capacity estimate | `DeviceModelCapacityPolicy.kt` |
| Runtime memory, thermal, batch, and context limits | `DeviceAdaptiveRuntimePolicy.kt` |
| Training state memory estimate | `TrainingMemoryPlanner.kt` |
| Neural model and checkpoint implementation | `ArgusNeuralEngine.kt` |
| Genesis lifecycle | `GenesisModelManager.kt` |
| Training queue and transaction orchestration | `TrainingManager.kt`, `TrainingCommitJournal.kt` |
| Vulkan capability report and native bridge | `VulkanTrainingBackend.kt`, `vulkan_training_jni.cpp` |
| Model tensor residency | `VulkanTensorResidencyRegistry.kt` |
| FP32 AdamW state residency | `VulkanOptimizerStateRegistry.kt` |
| Activation tensor residency | `VulkanActivationTensorRegistry.kt` |
| Native storage allocation | `tensor_arena_core.h` |

The existing source tree also contains focused tests for architecture configuration, device model policy, adaptive runtime policy, checkpoint compatibility, transaction journaling, Vulkan capability parsing, tensor residency, optimizer residency, and native Tensor Arena behavior.

## 3. Target Kotlin package boundaries

The following is the target ownership map. Existing files are not to be bulk-moved. Migration must be incremental and preserve the current public/internal API until call sites and tests are updated.

```text
com.argus.agi/
  device/
    DeviceProbe
    DeviceProfile
    DeviceCapabilities
    DeviceMemoryBudget
    DeviceCompatibility
    DeviceQuirkRegistry

  runtime/
    AdaptiveRuntimePlanner
    AdaptiveComputePerformanceMonitor
    RuntimeConfiguration
    RuntimeReadiness
    ComputeBackendSelector
    PrecisionPolicy

  model/
    ArgusArchitectureConfig
    ModelCompatibility
    GenesisModelManager
    ArgusNeuralEngine
    Tokenizer
    ModelState

  training/
    TrainingManager
    TrainingScheduler
    AdaptiveBatchPlanner
    TrainingMemoryPlanner
    OptimizerPolicy
    TrainingState
    TrainingCommitJournal

  backend/
    ComputeBackend
    BackendCapabilities
    vulkan/
      VulkanRuntime
      VulkanTrainingBackend
      VulkanKernelRegistry
      VulkanPerformanceMonitor
    cpu/
      CpuFallbackBackend

  memory/
    TensorResidencyManager
    OptimizerStateManager
    MemoryRecoveryManager

  persistence/
    CheckpointManager
    CheckpointFormat
    ModelStateRepository
    TrainingRecoveryManager

  data/
  sensor/
  service/
  skill/
  ui/
```

Do not create empty packages or placeholder classes merely to match this tree. Each package is introduced when a real class is moved or a real contract is consumed by production code and tests.

## 4. Target native source boundaries

Native source migration is later than the Kotlin contract work. Preserve existing CMake shader compilation inputs until each move is updated and validated in the same change.

```text
app/src/main/cpp/
  core/
    tensor_arena_core.h
  vulkan/
    device/
    memory/
    jni/
    profiling/
    kernels/
      portable/
      subgroup/
      fp16/
      int8/
      resident/
  cmake/
  CMakeLists.txt
```

Kernel families describe implementation capabilities, not vendor names. A Mali-specific or Adreno-specific workaround belongs in a narrowly scoped quirk/capability rule only when a reproducible driver issue requires it. Do not fork the model or optimizer by vendor.

The passive `AdaptiveComputePerformanceMonitor` now lives in the `runtime/` package. Its CPU/GPU timing samples are diagnostic only and are not allowed to participate in backend selection.

## 5. Component contracts

### 5.1 Device profile contract

A device profile must distinguish:

- physical-device identity: vendor/device IDs, device type, device UUID when available, driver UUID, pipeline-cache UUID, driver name/version, and effective Vulkan API version;
- limits: compute workgroup counts/sizes, invocation count, shared memory, storage-buffer range, descriptor limits, and relevant alignment constraints;
- memory: heap/type flags, budget and usage when the extension is available, and conservative fallback estimates when it is not;
- feature state: queried support, logical-device enabled state, and kernel availability as three separate facts;
- execution capabilities: subgroup stages/operations and size, timestamp support, FP16/storage16 support, integer-dot-product variants, and other extensions only when actually queried.

A capability being supported does not mean it is enabled. A feature being enabled does not mean Argus has a matching kernel. A kernel being compiled does not mean the selected device can execute it safely. The planner must only select a path when all required conditions are true.

### 5.2 Runtime plan contract

The runtime planner consumes an immutable device profile and current resource/thermal observations. It produces one explicit runtime plan containing:

- selected compute backend and a human-readable selection reason;
- selected precision and accumulation policy;
- dispatch/workgroup parameters;
- micro-batch and training batch limits;
- training context/workspace limits;
- model, activation, optimizer, and temporary-memory budgets;
- selected kernel variants and the required capabilities;
- safe fallback/recovery policy.

Replanning is allowed when memory, thermal, foreground/background state, or device readiness changes. Replanning must not modify the persisted model architecture.

### 5.3 Model architecture identity contract

The persisted model identity includes the architecture schema/version and ID, layer count, hidden size, attention heads, feed-forward size, context length, adapter topology, vocabulary capacity, tokenizer schema/algorithm, and any other dimension that changes tensor shape or token-ID semantics.

Once a model has a checkpoint, these values are immutable for that model identity. A different architecture requires a new model identity or an explicit, versioned migration that preserves the source checkpoint. Runtime batch size, micro-batch size, dispatch shape, memory residency, and a numerically compatible execution precision are runtime choices, not permission to mutate architecture.

### 5.4 Compute backend contract

A backend owns execution resources, not authoritative learned state. The contract must cover:

- initialize only after the model and restored state are ready;
- expose queried/enabled/usable capabilities and readiness/failure status;
- execute supported model/training operations with validated tensor shapes and numerical results;
- report dispatch timing and recoverable errors;
- synchronize mutated device state back to authoritative host/checkpoint state before checkpoint, backend switch, CPU recovery, or teardown;
- release/rebuild device allocations without changing model identity or optimizer step;
- fail closed when an operation is unsupported or device state cannot be recovered.

The Vulkan implementation is the primary backend. The CPU implementation is an explicit fallback that must preserve the same model and optimizer semantics.

### 5.5 Tensor residency contract

Host model/optimizer arrays remain the recoverable mirror. Each resident tensor tracks a monotonically changing host revision and GPU revision. A GPU mutation makes the host mirror stale until a successful readback/reconciliation updates the host revision. A failed dispatch must not be treated as a committed mutation.

Before serialization, CPU fallback, backend restart, or recovery, all authoritative device mutations must be reconciled. Residency registries may defer routine readback for performance, but they may not lose the ability to recover the last committed state.

### 5.6 Checkpoint and training-recovery contract

Checkpoints must be backend- and GPU-vendor-independent. The canonical format must be versioned and validate magic, header size, architecture identity, tokenizer identity, named tensor manifest, tensor types/shapes, declared payload length, checksum, and end-of-file. Writes must use a temporary file, durable flush, atomic replacement where supported, and independent read-back verification.

The current ARG4 implementation already:
- stores architecture and tokenizer metadata, named model tensors, Adam first/second moments, and optimizer step;
- serializes tensor values through a canonical float32 representation, even where in-memory model parameters use FP16;
- validates architecture dimensions/ID, tokenizer schema/algorithm/capacity, tensor manifest, payload length, and CRC32;
- synchronizes Vulkan model/optimizer mirrors before checkpoint serialization;
- verifies the saved file by reading its header and streaming the payload checksum.

The training queue transaction is separate: `TrainingCommitJournal` and Room queue state coordinate prepared/committed optimizer updates and recovery. Do not conflate the model checkpoint with queue persistence.

Before claiming full training-session continuity, separately verify whether optimizer hyperparameters, learning-rate schedule position, random-number-generator/sampler state, dataset cursor, and all pending queue metadata are persisted or deterministically reconstructed. Missing state must be reported as a gap, not assumed to be in ARG4.

A checkpoint load must reject incompatible architecture/tokenizer state without modifying the currently active model. Any future migration requires a new format/schema version and explicit compatibility tests.

## 6. Important audited gaps and risks

1. **CPU dispatch selection gap — corrected in the current implementation.** The former `AdaptiveComputeDispatchPolicy` scheduled CPU timing probes and selected CPU when measured throughput was competitive. That selection logic and its workload-size thresholds have been removed. CPU/GPU timings are now passive telemetry only; they cannot choose a backend. Vulkan remains gated by runtime readiness, enabled capabilities, kernel availability, validated tensor shapes, and safe resource checks. Any CPU execution must come from an unavailable, unsupported, or failed Vulkan path; continue verifying every call site and runtime trace.
2. **Package ownership is not yet enforced.** Device/runtime/training/backend/memory/checkpoint classes remain concentrated in `model/`. Introduce the boundaries incrementally; do not duplicate the existing policies under new names.
3. **The device capability report is broad, but kernel coverage is narrower.** Report feature support separately from actual compiled and selected kernels. In particular, capability reporting for subgroup, integer dot product, cooperative matrix, ARM tensor, or shader-core features must not be marketed as working Argus kernels until the corresponding shader/dispatch/test path exists.
4. **Capacity recommendations are not the training model selector.** `ArgusDeviceModelPolicy` chooses the current locally executable training family. `DeviceModelCapacityPolicy` contains much larger RAM-based parameter recommendations and explicitly is not proof that the current Argus trainer can build or train those model sizes. Keep these concepts separate until the capacity estimator is reconciled with the actual backend memory/compute implementation.
5. **Checkpoint continuity needs a complete state inventory.** ARG4 model/optimizer tensor persistence and the queue transaction journal are substantial foundations, but they do not by themselves prove that all training-session state (schedule, RNG/sampler, dataset cursor, and pending work) is durable.

## 7. Safe migration sequence

1. Audit the current tree, call sites, native build inputs, and tests before each move.
2. Establish and test the contracts in this document against the existing implementation.
3. Extract one responsibility at a time: device profile, runtime planner, backend interface, residency/recovery, and checkpoint persistence. Update imports, JNI bindings, shader paths, CMake dependencies, and tests in the same change.
4. Preserve `:app` as one Gradle module until package-level boundaries are stable and independently testable.
5. Do not run duplicate CI workflows. Build and test the complete change once the implementation batch is coherent and the user has enabled the intended validation workflow.
6. Work on `master`. Do not merge into `main` until the implementation is complete, reviewed, validated, and explicitly ready.

## 8. Completion criteria

The architecture foundation is complete only when the production code—not merely this document—enforces the contracts above; no existing feature has been lost during migration; model/optimizer state survives backend release/reinitialization; checkpoint compatibility and transaction recovery are tested; GPU dispatch does not deliberately use CPU for calibration; and the relevant build, unit, native, and device-level validation evidence is recorded.
