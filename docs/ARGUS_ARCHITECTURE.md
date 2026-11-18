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

The current ARG6 implementation already:
- stores architecture and tokenizer metadata, named model tensors, Adam first/second moments, and optimizer step;
- serializes tensor values through a canonical float32 representation, even where in-memory model parameters use FP16;
- validates architecture dimensions/ID, tokenizer schema/algorithm/capacity, tensor manifest, payload length, and CRC32;
- synchronizes Vulkan model/optimizer mirrors before checkpoint serialization;
- verifies the saved file by reading its header and streaming the payload checksum.

The active writer emits ARG6 with the same 96-byte fixed header and existing ARG5 payload representation, plus a 32-byte optimizer-configuration SHA-256 fingerprint immediately after the tokenizer fingerprint. The loader remains backward-compatible with ARG5 and validates the optimizer fingerprint before mutating the active tokenizer or tensor state. The current clean-break loader explicitly rejects pre-release ARG3 and ARG4 files; this is covered by `preReleaseArg3AndArg4CheckpointsAreRejectedWithoutMigration`. Therefore this branch does **not** provide ARG3/ARG4 restore compatibility. Any future legacy restore must be an explicit versioned reader with preservation tests, not an implicit change to ARG6.

The training queue transaction is separate: `TrainingCommitJournal` and Room queue state coordinate prepared/committed optimizer updates and recovery. Do not conflate the model checkpoint with queue persistence.

Before claiming full training-session continuity, separately verify whether optimizer hyperparameters, learning-rate schedule position, random-number-generator/sampler state, dataset cursor, and all pending queue metadata are persisted or deterministically reconstructed. Missing state must be reported as a gap, not assumed to be in ARG6.

A checkpoint load must reject incompatible architecture/tokenizer state without modifying the currently active model. Any future migration requires a new format/schema version and explicit compatibility tests.

## 6. Important audited gaps and risks

1. **CPU dispatch selection gap — corrected in the current implementation.** The former `AdaptiveComputeDispatchPolicy` scheduled CPU timing probes and selected CPU when measured throughput was competitive. That selection logic and its workload-size thresholds have been removed. CPU/GPU timings are now passive telemetry only; they cannot choose a backend. Vulkan remains gated by runtime readiness, enabled capabilities, kernel availability, validated tensor shapes, and safe resource checks. Any CPU execution must come from an unavailable, unsupported, or failed Vulkan path; continue verifying every call site and runtime trace.
2. **Package ownership is not yet enforced.** Device/runtime/training/backend/memory/checkpoint classes remain concentrated in `model/`. Introduce the boundaries incrementally; do not duplicate the existing policies under new names.
3. **The device capability report is broad, but kernel coverage is narrower.** Report feature support separately from actual compiled and selected kernels. In particular, capability reporting for subgroup, integer dot product, cooperative matrix, ARM tensor, or shader-core features must not be marketed as working Argus kernels until the corresponding shader/dispatch/test path exists.
4. **Capacity recommendations are not the training model selector.** `ArgusDeviceModelPolicy` chooses the current locally executable training family. `DeviceModelCapacityPolicy` contains much larger RAM-based parameter recommendations and explicitly is not proof that the current Argus trainer can build or train those model sizes. Keep these concepts separate until the capacity estimator is reconciled with the actual backend memory/compute implementation.
5. **Checkpoint continuity needs a complete state inventory.** ARG5 model/optimizer tensor persistence and the queue transaction journal are substantial foundations, but they do not by themselves prove that all training-session state (schedule, RNG/sampler, dataset cursor, and pending work) is durable.

## 7. Safe migration sequence

1. Audit the current tree, call sites, native build inputs, and tests before each move.
2. Establish and test the contracts in this document against the existing implementation.
3. Extract one responsibility at a time: device profile, runtime planner, backend interface, residency/recovery, and checkpoint persistence. Update imports, JNI bindings, shader paths, CMake dependencies, and tests in the same change.
4. Preserve `:app` as one Gradle module until package-level boundaries are stable and independently testable.
5. Do not run duplicate CI workflows. Build and test the complete change once the implementation batch is coherent and the user has enabled the intended validation workflow.
6. Work on `master`. Do not merge into `main` until the implementation is complete, reviewed, validated, and explicitly ready.

## 8. Completion criteria

The architecture foundation is complete only when the production code—not merely this document—enforces the contracts above; no existing feature has been lost during migration; model/optimizer state survives backend release/reinitialization; checkpoint compatibility and transaction recovery are tested; GPU dispatch does not deliberately use CPU for calibration; and the relevant build, unit, native, and device-level validation evidence is recorded.


## 9. Implementation progress after the baseline audit

The audited source tree and package inventory in section 2 describe the baseline revision recorded at the top of this document. They are historical audit evidence, not a claim that the current `master` tree is unchanged.

The following architecture increments have since been committed to `master`:

- **Device profile:** immutable `com.argus.agi.device.DeviceProfile` now represents physical-device identity, Vulkan limits, feature support versus enablement, and memory-heap budgets. The existing JNI report remains the parsing boundary through `VulkanDeviceCapabilities.toDeviceProfile()`.
- **Runtime configuration:** `DeviceAdaptiveRuntimePolicy` now belongs to `com.argus.agi.runtime`. `RuntimeConfigurationPlanner` combines a device profile with live heap, system-memory, thermal, battery, and active-model constraints. It calculates execution batch/context limits without changing persisted architecture dimensions.
- **Compute backend contract:** `com.argus.agi.backend.ComputeBackend` is implemented by the Vulkan backend for readiness, FP16 self-test readiness, device profile, state synchronization, and device-resource recovery. The Kotlin CPU path remains the engine's existing fallback; it has not been replaced with an empty backend stub.
- **Model state:** `ModelState` captures architecture identity, tokenizer configuration and snapshot fingerprint, optimizer step, active optimizer-configuration identity, checkpoint format, read-back verification state, and the optimizer-configuration fingerprint actually persisted in the checkpoint. ARG6 persists and verifies that fingerprint; ARG5 remains readable but reports no persisted optimizer identity. ARG6 keeps the 96-byte fixed header and extends the payload with the 32-byte optimizer fingerprint. `ModelCheckpointFormat.ARG6` is the active writer version.

These are implementation increments, not completion evidence. The current changes have not yet passed a complete Android build, lint, JVM unit-test suite, native validation, or device-level validation. No merge to `main` is authorized by this status note.

Remaining architecture work includes routing model operations through the backend contract where appropriate, auditing end-to-end checkpoint and training-session recovery (including schedule, RNG/sampler, dataset cursor, and pending queue state), and completing the existing M1–M10 validation sequence. The cross-GPU evidence audit remains deferred until implementation and milestone validation are complete.

## 10. Training-session persistence audit

This audit describes the current `master` implementation, not a claim that every item below has passed CI or device validation.

### 10.1 Durable state already present

- **Model and optimizer tensors:** ARG6 stores named model tensors, Adam first/second moments, optimizer step, and optimizer-configuration fingerprint. Tensor payload values use the checkpoint's canonical float32 representation. The tokenizer manifest and snapshot fingerprint are stored with the architecture metadata. ARG5 remains readable through the legacy-format loader without a persisted optimizer identity.
- **Checkpoint integrity:** the writer uses a temporary file, flushes and syncs it, atomically replaces the destination where supported, then independently verifies header, declared length, and CRC32. The loader validates architecture dimensions/ID, tokenizer schema/algorithm/configuration/fingerprint, tensor manifest and shapes, payload length, checksum, and end-of-file before accepting the restored state.
- **Queue and dataset position:** each `TrainingItemEntity` stores item ID, source/title/content, document ID, chunk index, total chunk count, timestamp, status, final loss, and step duration. The durable queue order and remaining rows therefore represent the materialized replay cursor; there is no separate persisted integer cursor.
- **Batch transaction:** new `TrainingCommitJournal` v3 records PREPARED/COMMITTED state, baseline and committed optimizer steps, item IDs, per-item loss/token/context/optimizer metrics, CRC32, a SHA-256 `ModelStateIdentity.fingerprint`, and a separate SHA-256 optimizer-configuration fingerprint. The reader remains backward-compatible with v1 and v2 journals; v1 has neither identity, v2 has model identity only, and both preserve their prior recovery path with explicit warnings that optimizer configuration is unverified. `TrainingManager` verifies both v3 identities before checkpoint replacement, Room status mutation, rollback completion, or committed-item finalization.
- **Startup recovery:** the worker reconciles the journal against the restored optimizer step before converting interrupted Room `TRAINING` rows back to `QUEUED`. It restores the remaining committed and pending rows in persisted timestamp order. A committed row is finalized without applying the optimizer update again.
- **Validation promotion:** `ConversationalValidationPersistence` separately stores baseline/latest reports, decision/status, and an accepted checkpoint filename plus SHA-256. This is validation-promotion state, not a replacement for the training batch journal.

### 10.2 RNG, replay scheduling, and cursor semantics

- The active training update path does not currently use a stochastic sample selector. The engine's Kotlin `Random` calls are used for initial/new tensor weight initialization and generation sampling; generation sampling is not training sampler state. Restored weights make the initialization RNG unnecessary after a valid checkpoint exists.
- `CorpusReplayScheduler` uses `java.util.Random(seed)` to create the complete deterministic order for the requested source snapshot, epoch count, seed, and policy. The scheduler materializes every selected chunk into queue items before worker execution. Room then persists those rows and their timestamps; restart continues the already-materialized order without needing to restore the RNG stream.
- The replay seed, selected policy, and a distinct replay-session ID are **not** persisted as separate metadata. Epoch/chunk labels and aggregate chunk indexes are encoded in queue item metadata/title, and the ordered queue preserves execution order. This is sufficient to continue an intact materialized queue, but it is not sufficient to independently audit or regenerate the exact original replay plan after its source corpus changes.

### 10.3 Remaining gaps

1. **Optimizer configuration identity — CHECKPOINTED FOR ARG6.** The optimizer hyperparameters and semantic policy IDs are centralized in immutable, validated `OptimizerConfig`; the stable SHA-256 fingerprint uses raw IEEE-754 float bits and is persisted in ARG6 after the tokenizer fingerprint. The loader verifies it against the active optimizer configuration before restoring mutable model state. ARG5 remains readable but its optimizer compatibility is unverified; no fingerprint is invented for legacy files. Future optimizer changes require an explicit version/migration decision instead of silently resuming with changed semantics.
2. **Journal-to-model identity — PARTIAL.** Journal v2 now stores and verifies `ModelStateIdentity.fingerprint` (architecture/tokenizer identity); v1 remains readable but unverified and retains its prior step-only behavior. This fingerprint is not a digest of learned tensor contents, so it cannot independently prove that two checkpoints with the same architecture/tokenizer contain identical weights. The ARG5 checksum protects checkpoint payload integrity, but a future transaction format may still bind the journal to a specific checkpoint-file digest.
3. **Replay provenance — PARTIAL.** The full queue order and chunk cursor are durable once scheduling completes, but seed, policy, epoch request, and source-snapshot fingerprint are not stored as a first-class replay-session record.
4. **Training-session telemetry — SESSION-LOCAL.** `TrainingSessionMetrics`, loss history, active-document UI state, and transient retry counters are in-memory runtime state. They must not be presented as lifetime or crash-resumable counters unless separately reconstructed from durable queue/corpus records.
5. **Legacy checkpoint compatibility — NOT SUPPORTED BY CURRENT CLEAN-BREAK POLICY.** ARG4 is explicitly rejected by the current ARG5 loader. Preserve this observed behavior until an explicit, tested compatibility migration is approved; do not describe ARG4 restore as implemented.

### 10.4 Safe next implementation order

1. **COMPLETED:** immutable optimizer configuration identity is persisted and verified in ARG6, while ARG5 remains backward-readable. The exact checkpoint-file digest is still not attached to the training transaction journal.
2. **COMPLETED in journal v3:** new writes record both `ModelStateIdentity.fingerprint` and the optimizer-configuration fingerprint; the reader accepts v1, v2, and v3. Legacy v1/v2 optimizer compatibility remains explicitly unverified. A later transaction-format increment may add the exact checkpoint-file digest if crash-safe ordering is designed and tested.
3. Add a durable replay-session descriptor (session ID, source snapshot fingerprint, seed, policy, epochs, and schedule format) only with a versioned Room migration and transactionally consistent enqueue path. Keep existing queue rows as the authoritative per-item cursor.
4. Add fault-injection tests for crashes between PREPARED, optimizer mutation, COMMITTED, checkpoint replacement, Room commit, and item finalization.
5. Only after those contracts and migrations are complete should the full build, lint, JVM/native validation, and relevant device checks run in the authorized validation sequence.

Do not retrofit RNG persistence for a training sampler that does not exist. Do not break the existing queue/journal recovery path merely to add metadata.
