AGENTS.md — Argus Project Engineering Rules

«Purpose: Define the mandatory engineering standards, architecture invariants, technology constraints, security requirements, and verification procedures for all AI agents working on Argus.

Priority: This document is authoritative for automated agents working in this repository. Existing architecture, approved technical decisions, and explicit project-owner instructions must be respected.

Primary Objective: Build Argus as a fully local Android AI system capable of initializing, training, evolving, and preserving its own model using on-device compute, with Vulkan as the primary training backend and CPU as a compatibility or recovery fallback.»

---

1. Core Development Principles

All agents MUST follow these principles:

1. Preserve the existing architecture unless an architectural change is explicitly required.
2. Never downgrade pinned dependencies or toolchain versions to make a build pass.
3. Never introduce deprecated Android, Kotlin, Java, C++, Vulkan, or third-party APIs.
4. Prefer current stable APIs and features officially supported by the pinned toolchain.
5. Never replace a genuine implementation with a mock, placeholder, simulation, or hardcoded success result.
6. Never silently remove functionality to resolve compilation errors.
7. Never sacrifice model integrity or user-owned learning data. Apply checkpoint compatibility to published production formats; follow the explicit pre-release clean-break policy in Section 8 when the project owner authorizes a foundational reset.
8. Fix root causes instead of hiding symptoms.
9. Implement changes completely before initiating the final verification pipeline.
10. Never claim that an implementation works without appropriate test evidence.
11. Do not introduce unnecessary dependencies or duplicate existing functionality.
12. Prioritize correctness, memory safety, determinism, performance, maintainability, and long-term compatibility.

All implementation decisions must be consistent with the long-term objective of Argus.

---

2. Pinned Technology Stack

AI agents MUST strictly target, compile, and generate code adhering to the following pinned stack.

Do not downgrade versions or silently substitute another dependency.

Component| Pinned Version| Purpose
Android Gradle Plugin| "9.5.1"| Android build integration
Kotlin| "2.4.21"| Kotlin compiler
Kotlin Compose Compiler Plugin| "2.4.21"| Jetpack Compose compiler integration
Kotlin Symbol Processing (KSP)| "2.3.12"| Annotation processing
Gradle Wrapper| "9.8.1"| Build orchestration
JDK (Gradle Runtime)| "25"| JVM runtime executing Gradle
Java Source Compatibility| "21"| Java language compatibility
Java Target Compatibility| "21"| JVM bytecode target
Kotlin JVM Target| "21"| Kotlin bytecode target
Android SDK Platform| "37"| Compile SDK and target SDK
Android Compile SDK| "37"| Compilation API level
Android Target SDK| "37"| Runtime compatibility target
Android Min SDK| "34"| Minimum supported API level
Android SDK Build Tools| "37.0.0"| APK packaging
Android NDK| "30.0.16248370"| Native C/C++ toolchain
C++ Standard| "C++23"| Native Vulkan implementation
CMake| "4.5.0"| Native build configuration
Minimum CMake Requirement| "3.28"| Native build compatibility floor
JUnit| "6.1.3"| Host-side unit testing

2.1 Version Management Rules

- Keep Gradle Wrapper at "9.8.1".
- Keep AGP at "9.5.1".
- Keep Kotlin at "2.4.21".
- Keep KSP at "2.3.12".
- Keep Android API level at "37".
- Keep minimum supported Android API level at "34".
- Keep JDK runtime at "25".
- Keep JVM bytecode compatibility at Java "21".
- Keep native compilation on C++23.
- Keep CMake at "4.5.0".
- Keep Android NDK at "30.0.16248370".

Do not:

- Downgrade Gradle to accommodate outdated plugins.
- Downgrade Kotlin to accommodate outdated libraries.
- Change Java target to JDK runtime version "25".
- Replace C++23 with C++17 or C++20.
- Replace Vulkan with another compute framework.
- Change minimum SDK without explicit architectural approval.
- Change target SDK because of unrelated build problems.

When a dependency is incompatible with the pinned stack:

1. Identify the incompatibility.
2. Verify the official compatibility requirements.
3. Find a compatible stable release.
4. Update only the necessary dependency.
5. Preserve existing functionality.
6. Document any unavoidable limitations.

Never use arbitrary version changes as a debugging shortcut.

---

3. Modern Android Engineering

3.1 Android Gradle Plugin

Use the modern Android Gradle Plugin architecture.

AGP 9.x includes built-in Kotlin support. Follow the supported AGP 9.5.1 Kotlin configuration.

Requirements:

- Prefer AGP built-in Kotlin for Android modules.
- Do not redundantly apply the legacy "org.jetbrains.kotlin.android" plugin when built-in Kotlin is enabled.
- Use the matching Kotlin Compose Compiler Gradle plugin.
- Use supported Gradle configuration APIs.
- Avoid deprecated Android DSL properties.
- Avoid obsolete Gradle configurations.
- Avoid dynamic dependency versions.

Gradle configuration must remain reproducible.

3.2 Kotlin Development

Use idiomatic Kotlin 2.4+.

Preferred technologies:

- Kotlin Coroutines.
- Structured concurrency.
- StateFlow.
- SharedFlow.
- Immutable state models.
- Suspend functions.
- Explicit lifecycle management.
- Type-safe configuration.
- Modern collection APIs.
- Proper exception handling.

Use "collectAsStateWithLifecycle()" for lifecycle-aware Compose state collection.

Do not use:

- "GlobalScope".
- Unmanaged application threads.
- "AsyncTask".
- "runBlocking" in production application flows.
- Deprecated coroutine APIs.
- Unnecessary manual callback chains.
- Unnecessary mutable global state.
- Forced non-null assertions ("!!").
- Safe-call operators on statically non-null receivers.
- Unsafe unchecked type assumptions.

When a type is uncertain, declare the correct type explicitly and validate the input.

3.3 Jetpack Compose

Use modern Jetpack Compose and Material 3 Expressive.

Design requirements:

- Calm UI principles.
- Strong typography hierarchy.
- Consistent spacing.
- Comfortable whitespace.
- Clear content grouping.
- Minimal decorative elements.
- Material 3 surface containers for visual depth.
- Responsive layouts.
- Accessibility semantics.
- Proper loading, empty, error, and success states.

Avoid:

- Deprecated Compose APIs.
- Legacy Material components.
- Unnecessary nested layouts.
- Excessive animations.
- Heavy card shadows.
- Hardcoded screen dimensions.
- Unnecessary recomposition.
- Blocking operations inside composables.

Keep business logic outside UI composables.

Use ViewModels and lifecycle-aware observable state.

---

4. Native C++23 Engineering

All native code MUST use C++23.

4.1 General Rules

Use:

- RAII.
- Smart pointers.
- Standard library containers.
- Strong typing.
- Explicit ownership.
- Move semantics.
- "std::span" where appropriate.
- "std::expected" where supported and appropriate.
- "std::optional" for optional values.
- Scoped resource management.
- Thread-safe synchronization.

Avoid:

- Manual owning raw pointers.
- "malloc()" and "free()".
- C-style casts.
- Unnecessary global mutable variables.
- Unchecked buffer access.
- Undefined behavior.
- Memory leaks.
- Uncontrolled native threads.
- Deprecated compiler extensions.

Non-owning pointers and native handles may be used where required by Vulkan and JNI interfaces, provided ownership and lifetime are explicitly managed.

4.2 Exception Safety

- Never allow an uncaught C++ exception to cross a JNI boundary.
- Catch and translate native exceptions at JNI entry points.
- Return explicit failure results.
- Preserve useful error information without exposing sensitive data.
- Never convert native failures into unconditional success responses.

4.3 Memory Safety

All native operations must validate:

- Buffer sizes.
- Tensor dimensions.
- Element counts.
- Memory allocation results.
- Arithmetic overflow.
- JNI array lengths.
- Array offsets.
- Input data types.
- Pointer lifetimes.
- Vulkan resource ownership.

Never assume an input buffer has the expected size without validation.

Use overflow-safe size calculations before allocating memory.

---

5. Argus Architecture Invariants

These rules protect the fundamental Argus architecture.

An agent MUST NOT alter these architectural decisions without explicit authorization.

5.1 Local AI Architecture

Argus is designed as a fully local Android AI system.

Core requirements:

- Model initialization occurs locally.
- Model training occurs locally.
- Model weights are stored locally.
- Tokenizer configuration is stored locally.
- Optimizer states are preserved locally.
- Training progress is preserved locally.
- Checkpoints are stored locally.
- Personal learning data remains local.
- Model restoration must work without a remote training server.

Do not introduce mandatory cloud training.

Do not make external server access a requirement for local model execution or training.

Optional online capabilities must never become mandatory dependencies for the core local AI system.

5.2 Compute Backend Policy

Vulkan is the primary compute backend.

CPU is permitted only as:

1. A compatibility fallback.
2. A Vulkan initialization fallback.
3. A device capability fallback.
4. A controlled recovery mechanism when GPU execution cannot safely continue.

CPU must not silently become the permanent primary backend on a compatible device.

Do not introduce:

- NNAPI.
- OpenCL.
- clvk.
- Another compute abstraction as a replacement for Vulkan.

Do not execute unnecessary CPU workloads when the corresponding operation is already supported and successfully running on Vulkan.

When CPU fallback is required:

- Record the actual fallback reason.
- Preserve model parameters.
- Preserve optimizer state.
- Preserve training progress.
- Maintain the same numerical and semantic operation contract.
- Ensure backend transitions do not reset the model.

Never report GPU execution when the operation was actually executed on CPU.

5.3 Vulkan Initialization Lifecycle

Vulkan must NOT initialize eagerly during application startup.

The preferred initialization sequence is:

1. Application startup.
2. Device capability discovery.
3. RAM and hardware assessment.
4. SoC and GPU discovery.
5. Vulkan availability assessment.
6. Model configuration selection.
7. Model initialization or checkpoint restoration.
8. Background Vulkan initialization.
9. Vulkan feature discovery.
10. GPU resource allocation.
11. Compute pipeline preparation.
12. GPU capability self-check.
13. Backend readiness confirmation.
14. GPU training availability.

Vulkan initialization must not block the main thread.

Use Kotlin Coroutines for application-level asynchronous operations and explicit safe ownership in native code.

Initialization failures must not cause an application crash.

A failed Vulkan initialization must leave the model and its persistent data intact.

5.4 Device Capability Discovery

Never assume every Android device supports the same Vulkan capabilities.

Inspect:

- Vulkan instance version.
- Physical device properties.
- Vulkan API version.
- Device type.
- GPU vendor.
- GPU name.
- Driver version.
- Queue family properties.
- Compute queue support.
- Memory heap sizes.
- Device-local memory availability.
- Supported buffer usage.
- Storage buffer limits.
- Descriptor limits.
- Workgroup dimensions.
- Workgroup invocation limits.
- Supported floating-point features.
- Supported shader features.
- Required extensions.
- Required optional features.

Enable optional features only when the physical device reports support.

Never force-enable unsupported Vulkan features.

Use actual physical device limits when calculating workgroup dimensions and allocation sizes.

Do not hardcode a specific Mali, Adreno, PowerVR, or other GPU as the only supported device.

5.5 Adaptive Model Architecture

The model configuration must be based on available device resources and supported architecture limits.

Consider:

- Available RAM.
- Memory pressure.
- GPU memory.
- GPU architecture.
- Vulkan capabilities.
- Storage availability.
- Model parameter count.
- Context length.
- Vocabulary size.
- Batch size.
- Training workspace.
- Optimizer memory.
- Activation memory.

Do not assume that theoretical maximum RAM equals safely available training memory.

Architecture changes must be evidence-driven.

Do not silently change model dimensions, vocabulary, tokenizer, context length, or training semantics to make an operation fit.

For published production checkpoint formats, preserve compatibility or implement an explicit migration/new-model strategy. During pre-release development, the project owner may explicitly authorize a clean break: obsolete internal checkpoint/tokenizer formats then require no compatibility reader or migration. Recreate the model, vocabulary, embeddings, and optimizer state as one consistent new model lineage. Preserve imported datasets, corpora, user-created content, and unrelated local data. Once a model/checkpoint format is declared production-supported, its compatibility guarantees apply.

5.6 Training Correctness

A working Vulkan backend is not sufficient proof of a working training engine.

The complete training path must be validated:

1. Dataset loading.
2. Input validation.
3. Tokenization.
4. Token ID validation.
5. Batch preparation.
6. Tensor allocation.
7. Forward propagation.
8. Loss computation.
9. Backward propagation.
10. Gradient calculation.
11. Gradient accumulation, if configured.
12. Gradient validation.
13. AdamW optimizer update.
14. Parameter update verification.
15. Training state update.
16. Checkpoint persistence.
17. Model restoration.
18. Training continuation.

Every operation must use correct tensor shapes, layouts, data types, and memory boundaries.

Do not use hardcoded training outputs or simulated gradients as replacements for real computation.

---

6. Vulkan Engineering

6.1 Vulkan Resource Management

All Vulkan resources must have deterministic lifetimes.

Manage:

- "VkInstance".
- "VkPhysicalDevice".
- "VkDevice".
- "VkQueue".
- "VkCommandPool".
- "VkCommandBuffer".
- "VkBuffer".
- "VkDeviceMemory".
- "VkDescriptorPool".
- "VkDescriptorSet".
- "VkPipelineLayout".
- "VkPipeline".
- "VkShaderModule".
- "VkFence".
- "VkSemaphore".

Use RAII wrappers or equivalent deterministic ownership mechanisms.

Never destroy resources that are still being used by submitted GPU operations.

Ensure synchronization before releasing buffers, memory, pipelines, or other referenced resources.

6.2 Synchronization

Validate:

- Queue submission order.
- Command buffer lifecycle.
- Fence completion.
- Semaphore signaling.
- Semaphore waiting.
- Pipeline barriers.
- Buffer memory visibility.
- Host/device synchronization.
- Resource access dependencies.

Do not assume submission completion means GPU execution has finished.

Do not use unnecessary global device-idle synchronization in performance-critical training loops.

Prefer appropriate synchronization primitives and explicit dependency management.

6.3 GPU Memory

Implement proper memory management.

Requirements:

- Query memory properties.
- Select compatible memory types.
- Respect memory allocation limits.
- Validate alignment.
- Respect buffer usage flags.
- Avoid unnecessary allocation churn.
- Prevent out-of-bounds buffer access.
- Prevent overlapping writes without synchronization.
- Handle memory exhaustion safely.

Do not assume all GPU memory is host-visible.

Handle coherent and non-coherent memory correctly.

Flush and invalidate mapped memory when required by the selected memory properties.

6.4 Shader Development

Use valid Vulkan shader code and SPIR-V binaries.

Requirements:

- Validate descriptor bindings.
- Validate push constant sizes.
- Validate specialization constants.
- Validate workgroup dimensions.
- Validate buffer layouts.
- Validate shader data types.
- Validate shader execution assumptions.
- Validate supported Vulkan target environment.

Use SPIR-V validation tools when available.

Do not commit invalid or unverified shader binaries as production-ready code.

Do not hide shader compilation or validation failures.

6.5 Numerical Correctness

Numerical correctness must be verified against independent reference implementations.

Check:

- Floating-point precision.
- Overflow.
- Underflow.
- NaN propagation.
- Infinity propagation.
- Accumulation accuracy.
- Reduction accuracy.
- Matrix multiplication correctness.
- Gradient correctness.
- Optimizer update correctness.

FP16 may be used only when the device supports the required features and the numerical behavior has been validated.

Do not silently change tensor precision to resolve incorrect GPU results.

6.6 Vulkan Failure Recovery

Handle:

- Device initialization failures.
- Device loss.
- Out-of-memory errors.
- Unsupported capabilities.
- Pipeline creation failures.
- Shader failures.
- Synchronization failures.
- Submission failures.
- Invalid resource states.

On failure:

1. Capture the failure category.
2. Stop unsafe GPU submissions.
3. Synchronize or safely abandon affected resources as required.
4. Release resources in the correct order.
5. Preserve model and optimizer state.
6. Restore a consistent training state.
7. Switch to an available fallback only when necessary.
8. Allow Vulkan reinitialization where safe.

Do not automatically delete checkpoints or recreate the model following Vulkan errors.

---

7. JNI Engineering and Safety

All JNI interfaces must be designed with explicit ownership and error handling.

7.1 JNI Input Validation

Validate:

- Null references.
- Array lengths.
- Array element types.
- Numeric ranges.
- Tensor dimensions.
- Buffer capacities.
- String inputs.
- Object array contents.

Never assume Java arrays have compatible lengths.

Never assume JNI objects remain valid after their permitted lifetime.

7.2 JNI Reference Management

- Use local references appropriately.
- Delete local references when they are no longer required in long-running loops.
- Use global references only when necessary.
- Delete global references deterministically.
- Avoid excessive JNI object creation.
- Prevent local reference table exhaustion.

For loops processing large arrays of Java objects, release temporary references during iteration.

7.3 JNI Array Access

Choose the correct JNI array access mechanism.

Respect whether returned array elements are copied or pinned.

Release arrays with the appropriate release mode.

Do not hold pinned Java array memory across long-running GPU operations.

Avoid unnecessary JNI data copies.

7.4 JNI Error Handling

- Check pending Java exceptions.
- Never overwrite an existing pending exception carelessly.
- Translate native errors into appropriate Java/Kotlin failures.
- Prevent native crashes from invalid Java inputs.
- Keep JNI contracts consistent across Kotlin and C++.

A successful JNI function must represent a genuinely completed operation, not merely successful function entry.

---

8. Model, Optimizer, and Checkpoint Integrity

This section is mandatory. Distinguish disposable pre-release development artifacts from published production models and user-owned learning data.

8.0 Release Maturity and Clean-Break Policy

Argus is currently pre-release and has no production users or production model/checkpoint format. When the project owner explicitly authorizes a foundational clean break:

- Do not retain obsolete tokenizer algorithms or legacy checkpoint readers solely for backward compatibility.
- Use one canonical tokenizer/runtime contract and one new checkpoint/snapshot format for the new model lineage.
- Initialize model weights, token embeddings/output head, optimizer moments, and all dependent state consistently for the new tokenizer. Do not reuse tensors whose token-ID semantics have changed.
- Old pre-release model checkpoints are disposable and need not be migrated or restored. Detect unsupported old formats safely; never parse them using the new layout.
- Keep the new checkpoint writes atomic, integrity-checked, and fingerprint-bound even though old-format compatibility is not required.
- Preserve imported datasets, corpora, user-created content, settings, and unrelated local files. A model/checkpoint reset does not authorize deleting training data.
- Restrict any cleanup to explicitly identified Argus model-state files; never perform a broad storage wipe.

This clean-break exception applies only to owner-authorized pre-release development. Once a checkpoint/tokenizer format is published or designated production-supported, preserve that format's guarantees or use an explicit migration/new-model strategy. Never silently discard production or user-owned learning state.

8.1 Checkpoint Preservation

Never automatically:

- Delete a supported production checkpoint.
- Reset a production model's weights, optimizer moments, or training progress.
- Replace an existing production model with random initialization.
- Delete tokenizer state or training metadata belonging to a supported model lineage.
- Erase imported datasets, corpora, user-created content, or local learning data.
- Overwrite the active checkpoint with an unverified or partially written checkpoint.

The prohibitions above do not block an explicitly authorized pre-release clean break under Section 8.0. Any cleanup operation affecting model state or training data must be narrowly scoped; deleting datasets or user-owned content always requires separate explicit authorization.

8.2 Backend Independence

Checkpoint formats must be independent of the compute backend.

A model trained using Vulkan must remain restorable without requiring the original Vulkan resources.

The following must remain portable between supported backends:

- Model parameters.
- Optimizer state.
- Training step.
- Epoch progress.
- Random number generator state.
- Model configuration.
- Tokenizer configuration.
- Vocabulary metadata.
- Training metadata.

Backend switching must never implicitly reconstruct the model.

8.3 Checkpoint Contents

Preserve all state required to resume training correctly, including:

- Model weights.
- Optimizer first moments.
- Optimizer second moments.
- Optimizer step counters.
- Learning rate scheduler state.
- Gradient accumulation state, when applicable.
- Random number generator state.
- Training step.
- Epoch state.
- Dataset and tokenizer fingerprints.
- Model architecture metadata.
- Checkpoint format version.

If an item is intentionally not persisted, document the resulting limitation and its effect on training reproducibility.

8.4 Atomic Checkpoint Writing

Checkpoint writes must be crash-safe.

Preferred approach:

1. Write to a temporary file in the same storage location.
2. Validate the serialized data.
3. Flush data appropriately.
4. Verify integrity.
5. Atomically replace the target checkpoint where supported.
6. Preserve the previous valid checkpoint until the new checkpoint is verified.

Never delete the last valid checkpoint before a replacement is safely available.

8.5 Checkpoint Versioning

Every checkpoint format must include:

- Format version.
- Model architecture version.
- Tensor metadata.
- Data type information.
- Shape information.
- Tokenizer fingerprint.
- Required training state.
- Integrity information.

Unsupported checkpoint formats must fail safely.

Do not silently load incompatible tensors.

Provide explicit migration logic for published, production-supported formats when migration is required. A project-owner-authorized pre-release clean break may introduce a new canonical format without a legacy reader or migration path.

8.6 Recovery

When a checkpoint is corrupted:

- Detect the corruption.
- Preserve the corrupted file for diagnosis where appropriate.
- Identify the latest valid checkpoint.
- Recover from a verified valid state.
- Report the amount of training progress lost, if any.

Never silently reset a supported production model because a checkpoint cannot be loaded; use the defined recovery path. An unsupported pre-release checkpoint may be rejected and replaced by a fresh model only under the explicit clean-break policy in Section 8.0.

---

9. Training Data and Tokenizer Engineering

9.1 Dataset Processing

All imported datasets are untrusted input.

Validate:

- File size.
- Encoding.
- Format.
- Message structure.
- Token limits.
- Invalid records.
- Empty records.
- Malformed conversations.
- Path safety.
- Storage availability.

Support the approved training conversation format consistently.

Do not silently discard invalid records without reporting their count and reason.

9.2 Tokenizer Integrity

The tokenizer is part of the model's identity. Argus's new pre-release baseline is one canonical byte-level BPE implementation: BYTE_LEVEL_BPE_V1. Do not carry GREEDY_LONGEST_MATCH_V1 or LEARNED_BPE_V1 forward as active training/inference algorithms.

The new token-ID layout reserves:

- IDs 0–31 for canonical special/control tokens, including conversation, FIM, tool-call/result, web, image, and audio markers.
- IDs 32–287 for the 256 byte-alphabet entries.
- IDs 288 and above for learned BPE merge vocabulary entries.

Define the exact special-token-to-ID manifest once and test it as a stable contract. Never allow training merges to create a reserved control token. Existing IDs must remain stable within a model/tokenizer lineage.

The canonical runtime contract must also require:

- One shared left-to-right special-token parser for training, encoding, and decoding, with symmetric literal escaping/restoration (including U+E000 and literal control-token text).
- Inference encoding of malformed UTF-16 to replace orphan surrogates with U+FFFD; strict rejection may remain at dataset-ingestion/validation boundaries.
- Deterministic hard-coded ASCII pre-token classes, not runtime-dependent Character.isLetter classification.
- Prefix whitespace attached to the following word where possible, separate punctuation, and per-digit numeric pre-tokens; preserve leading/trailing/repeated whitespace losslessly.
- A bounded pre-token cache and identical pre-tokenization rules in training and inference.
- An immutable inference snapshot after training, with atomically published state rather than a global synchronized encode/train bottleneck.
- IntArray-oriented hot-path APIs to avoid boxed List<Int> allocation.

For the authorized pre-release clean break in Section 8.0, no compatibility with previous internal tokenizer IDs, embeddings, snapshots, or checkpoints is required. Build new model tensors against the new ID manifest. Once this format is production-supported, preserve IDs within that model lineage and do not silently regenerate its tokenizer.

9.3 Dataset Provenance

Maintain provenance information where applicable:

- Dataset source.
- Dataset version.
- Import timestamp.
- License information.
- Dataset fingerprint.
- Processing configuration.

Do not incorporate datasets with unknown or prohibited redistribution rights into bundled production assets without verification.

Never represent synthetic data as real-world sourced data.

---

10. Persistence and Database Engineering

Use modern persistence APIs.

For Room:

- Use KSP.
- Avoid KAPT.
- Use typed entities.
- Use explicit migrations.
- Use parameterized queries.
- Use transactions for related state changes.
- Validate database constraints.
- Preserve existing data.

Never perform destructive database migrations without authorization.

Avoid automatic destructive migration fallbacks in production.

Database migrations must be tested with existing database snapshots where applicable.

Model checkpoints and large tensor files should use suitable filesystem or storage abstractions rather than inefficiently storing entire model binaries as ordinary database records.

Database metadata and filesystem persistence must remain consistent.

---

11. Security and Data Protection

11.1 Secret Management

Never hardcode:

- API keys.
- Access tokens.
- Credentials.
- Signing passwords.
- Private certificates.
- Service account credentials.
- Sensitive environment variables.

Do not commit secrets to Git.

Do not include secrets in logs or crash reports.

11.2 Filesystem Safety

Validate every external filesystem input.

Prevent:

- Path traversal.
- Unsafe archive extraction.
- Directory escape.
- Unauthorized file replacement.
- Symlink-based escapes where applicable.
- Accidental deletion of unrelated files.
- Uncontrolled storage growth.

Use canonical path validation and verify that destinations remain inside their approved storage directories.

11.3 SQL Security

All database queries must use parameterized bindings.

Never construct SQL queries using untrusted input concatenation.

11.4 Logging

Production code must not contain uncontrolled debugging output.

Remove temporary statements such as:

- "println()"
- "Log.d()"
- "std::cout"
- "__android_log_print()"

Use a centralized, controlled logging strategy.

Diagnostic logging must:

- Avoid sensitive information.
- Avoid model content dumps.
- Avoid private dataset contents.
- Avoid credentials.
- Avoid excessive output.
- Provide actionable error categories.

Do not suppress important errors merely to keep logs clean.

---

12. Concurrency and Lifecycle Management

All application-level asynchronous work must use structured concurrency.

Requirements:

- Correct CoroutineScope ownership.
- Lifecycle-aware cancellation.
- Proper exception propagation.
- Thread-safe shared state.
- No main-thread GPU operations.
- No blocking disk operations on the main thread.
- Safe model access during training.
- Safe backend switching.
- Safe checkpoint creation during training.

Native operations must have clear ownership and synchronization boundaries.

Do not destroy a native engine while native operations are still executing.

Do not allow application shutdown to leave partially written checkpoints.

Do not use cancellation as an excuse to abandon native resource cleanup.

---

13. Performance and Resource Management

Argus is an on-device AI application.

Performance and resource utilization matter.

Monitor:

- RAM availability.
- GPU memory.
- Tensor allocation.
- Native heap usage.
- Java heap usage.
- Storage usage.
- Training throughput.
- GPU execution time.
- CPU utilization.
- Memory pressure.
- Thermal limitations.

Avoid:

- Unnecessary tensor copies.
- Repeated shader compilation.
- Repeated native allocation in hot loops.
- Excessive JNI transitions.
- Excessive synchronization.
- Unbounded coroutine launches.
- Unbounded queues.
- Unbounded caches.
- Unnecessary CPU preprocessing.

Use profiling data before introducing performance optimizations.

Do not optimize by sacrificing numerical correctness.

Implement controlled memory pressure and recovery behavior.

---

14. User Interface and Training Transparency

The UI must reflect actual engine state.

Never display a successful initialization or training status before the underlying operation has succeeded.

Display distinct states for:

- Device assessment.
- Model initialization.
- Checkpoint restoration.
- Vulkan initialization.
- GPU capability verification.
- Resource allocation.
- Pipeline preparation.
- Backend readiness.
- Training initialization.
- Training execution.
- Checkpoint saving.
- Recovery.
- Failure.

When CPU fallback is active, clearly identify it.

When Vulkan is ready, identify the actual selected GPU.

Never label CPU execution as GPU execution.

Training progress must come from actual training state rather than fabricated percentages.

The UI must remain responsive during initialization, training, checkpoint operations, and model loading.

---

15. Testing and Quality Assurance

Testing must verify actual behavior.

A compilation success is not sufficient proof of correctness.

15.1 Kotlin Unit Tests

Test:

- Business logic.
- State transitions.
- Input validation.
- Model configuration.
- Checkpoint metadata.
- Serialization.
- Deserialization.
- Migration behavior.
- Error handling.
- Coroutine cancellation.
- Lifecycle-related behavior.

15.2 Native Unit Tests

Test:

- Tensor shape validation.
- Tensor allocation.
- Matrix operations.
- Numerical accuracy.
- Gradient calculations.
- AdamW updates.
- Memory boundaries.
- Resource lifecycle.
- Error handling.

15.3 Vulkan Tests

Where supported, test:

- Instance initialization.
- Physical device selection.
- Logical device creation.
- Queue selection.
- GPU buffer allocation.
- Memory mapping.
- Descriptor configuration.
- Shader and SPIR-V validation.
- Pipeline creation.
- Command submission.
- Synchronization.
- Tensor upload.
- Tensor download.
- Compute execution.

15.4 Numerical Tests

Compare native GPU results with an independent CPU reference implementation.

Validate:

- Absolute error.
- Relative error.
- Numerical stability.
- Tensor dimensions.
- Data types.
- Special values.
- Gradient correctness.
- Optimizer correctness.

Use documented tolerances appropriate to the selected precision.

Do not use overly permissive tolerances to hide numerical defects.

15.5 End-to-End Training Tests

Where the hardware or test environment supports it, verify:

1. Model initialization from random weights.
2. Dataset processing.
3. Tokenization.
4. Forward propagation.
5. Loss calculation.
6. Backward propagation.
7. Gradient calculation.
8. Optimizer update.
9. Training progress advancement.
10. Checkpoint saving.
11. Checkpoint restoration.
12. Training continuation after restoration.
13. Backend switching.
14. State preservation after application restart.

15.6 Hardware Validation

A host-side unit test cannot prove physical GPU compatibility.

A successful native build cannot prove GPU compute correctness.

A successful Vulkan instance creation cannot prove training correctness.

Report each validation level independently.

Never claim physical-device validation unless it has actually been performed on the relevant device.

---

16. Mandatory Build and Verification Pipeline

Before declaring implementation complete, verify the environment and execute the required build pipeline.

16.1 Environment Verification

Run:

./gradlew --version

Verify:

- Gradle version.
- JDK runtime version.
- JVM runtime information.

Also verify the available Android SDK, NDK, CMake, and Build Tools versions.

Do not silently install or select alternative versions.

16.2 Non-Packaging Verification

Routine implementation and CI validation must not produce application APKs.

Run the release-variant checks without assembling an APK:

```bash
./gradlew :app:lintRelease :app:testReleaseUnitTest
```

Do not run `assembleDebug`, `assembleRelease`, `exportDebugApk`, or
`exportReleaseApk` during ordinary development or non-release CI validation.
Do not upload intermediate, debug, unsigned, or additional APK artifacts.

The only application APK assembly is `:app:assembleRelease` inside the single
`Argus Production Release` workflow, triggered by pushes to `main`. That workflow
must build, sign, and verify one production APK, then move the `Release` tag to
the exact source commit and publish or replace exactly one signed `ARGUS.apk`.
The `Release` tag is the publication target, not the workflow trigger. Do not add
tag-only triggers, separate tag-promotion workflows, or a second APK-producing
workflow.

Do not claim that any command passed unless its actual exit status confirms success.

16.3 Native Verification

The single main-triggered production release workflow is responsible for
compiling and packaging the production native libraries for every configured
ABI, along with the Vulkan and shader build inputs. Do not start a separate
APK-producing native build.

Host-side native unit tests may run independently because they do not package an
Android application. Verify C++23 configuration, allocator and tensor behavior,
numerical correctness, JNI boundaries, Vulkan linkage, and shader/SPIR-V validity
through the configured tests and the single production build.

Use the project's actual test task names and CMake configuration.

Do not invent task names or report nonexistent tests as executed.

16.4 Warning and Deprecation Verification

Inspect compiler, Gradle, Kotlin, Android lint, and native build output.

Investigate:

- Deprecated API warnings.
- Gradle deprecation warnings.
- Kotlin warnings.
- Java warnings.
- C++ warnings.
- Vulkan validation warnings.
- Resource ownership warnings.
- Unused code.
- Unchecked operations.

Do not disable warnings simply to achieve a clean build.

Do not use blanket warning suppression.

Resolve the underlying issue or document a justified external limitation.

16.5 Test Environment Limitations

If an emulator, physical device, Vulkan driver, or required SDK component is unavailable:

1. Execute all tests that are possible.
2. Identify exactly what could not be tested.
3. Explain why it could not be tested.
4. Do not report the unavailable validation as successful.
5. Provide the required next validation step.

---

17. Git and Repository Management

Git operations must preserve repository integrity.

17.1 Branch Policy

Before modifying files, inspect:

git status --short
git branch --show-current
git log -1 --oneline

For ArgusA:

- Inspect the actual available branches; do not assume "master" exists.
- Use the owner-designated development branch for ongoing work.
- Treat "main" as the canonical production source; a push to "main" triggers the
  single production build and updates the `Release` tag only after verification.
- Never merge or push into "main" without explicit authorization.

Always inspect the actual repository state rather than assuming branches are synchronized.

17.2 Change Protection

Never:

- Discard unrelated user changes.
- Run destructive Git commands without authorization.
- Force-push without explicit authorization.
- Rewrite public Git history without authorization.
- Commit secrets.
- Commit generated temporary build artifacts.
- Delete existing project functionality to simplify implementation.

Never use destructive reset or clean operations on user-modified data.

Preserve all uncommitted changes.

17.3 Commit Quality

Commits must:

- Represent coherent implementation changes.
- Use descriptive messages.
- Exclude temporary debugging code.
- Exclude unrelated modifications.
- Preserve required project documentation.

Do not commit code that has not undergone the required verification for its scope.

---

18. GitHub Actions and CI Safety

GitHub Actions execution must be controlled.

The only production workflow is `.github/workflows/argus-installable-release.yml`.
Every push to `main` automatically triggers it. It runs release lint, unit tests,
native host tests, one production APK build, signing and verification, then moves
the `Release` tag to that exact main commit and publishes or replaces the single
`ARGUS.apk` asset plus its verification metadata.

The `Release` tag is the publication destination, not a workflow trigger.
Do not add a tag-only build trigger, manual dispatch input, separate tag-promotion
workflow, or separate native-validation workflow unless the project owner
explicitly requests a different pipeline.

Do not manually:
- Dispatch or rerun production workflows.
- Start duplicate or redundant release builds.
- Publish APK assets outside the canonical production workflow.
- Create or retarget the `Release` tag outside the canonical production workflow.
- Create additional APK-producing workflows.
- Merge pull requests without authorization.
- Change repository secrets.
- Modify repository security settings.

Before changing the canonical production workflow:

1. Complete the implementation.
2. Review the full diff.
3. Verify that the workflow still produces exactly one signed `ARGUS.apk`.
4. Confirm that only pushes to `main` trigger production packaging.
5. Confirm the `Release` tag is moved only after the APK is built and verified.

Never report a successful CI build using an earlier run when the current commit
has not been tested. Do not create repeated commits merely to trigger CI without
meaningful changes.

---

19. Agent Execution Workflow

Every significant engineering task must follow this workflow.

Phase A — Repository Inspection

Before making changes:

1. Inspect the repository structure.
2. Inspect the active branch.
3. Inspect Git status.
4. Read relevant architecture documentation.
5. Inspect affected source files.
6. Identify existing implementations.
7. Review related tests.
8. Identify dependencies and call sites.

Never modify a subsystem without understanding its existing responsibilities.

Phase B — Root Cause Analysis

Before fixing a bug:

1. Identify the actual failure.
2. Trace the execution path.
3. Identify the responsible component.
4. Check resource ownership.
5. Check concurrency behavior.
6. Check input validation.
7. Check related persistence operations.
8. Determine potential regressions.

Do not apply speculative changes without identifying the underlying problem.

Phase C — Implementation Planning

Define:

- Objective.
- Affected modules.
- Dependencies.
- Required API changes.
- Data compatibility considerations.
- Failure scenarios.
- Test coverage.
- Regression risks.

Prefer the smallest complete change that correctly addresses the problem.

Phase D — Implementation

Implement the required changes completely.

Requirements:

- Maintain architecture boundaries.
- Preserve existing behavior unless intentionally changed.
- Preserve compatibility for published production checkpoint formats; apply Section 8.0 for explicitly authorized pre-release clean breaks.
- Add regression tests.
- Update documentation when architecture changes.
- Avoid unrelated refactoring.
- Maintain code readability.

Phase E — Code Review

Review the complete diff.

Check:

- Incorrect assumptions.
- Deprecated APIs.
- Type safety.
- Memory safety.
- Race conditions.
- Lifecycle errors.
- JNI reference leaks.
- Vulkan resource leaks.
- Persistence regressions.
- Checkpoint incompatibility.
- Unnecessary dependencies.
- Unrelated file modifications.

Phase F — Verification

Execute the mandatory verification pipeline.

Run additional native and hardware tests when applicable.

Investigate actual failures before declaring completion.

Phase G — Completion Report

Provide a factual report containing:

1. Engineering objective.
2. Root cause.
3. Implemented solution.
4. Files modified.
5. Tests executed.
6. Actual build results.
7. Actual CI results, if applicable.
8. Remaining problems.
9. Hardware validation requirements.
10. Checkpoint and data preservation status.

Do not claim success based on assumptions.

---

20. Forbidden Shortcuts

The following actions are prohibited:

- Downgrading pinned dependencies to bypass compilation errors.
- Replacing Vulkan with CPU as a permanent workaround.
- Introducing NNAPI, OpenCL, or clvk.
- Returning hardcoded success values.
- Using fake GPU execution results.
- Using simulated training results as production functionality.
- Removing tests to make CI pass.
- Disabling lint checks to hide defects.
- Suppressing compiler warnings indiscriminately.
- Deleting or resetting supported production checkpoints/models to hide restoration errors. Explicitly authorized pre-release clean breaks under Section 8.0 are not prohibited.
- Resetting trained model weights during backend migration.
- Silently changing tokenizer vocabulary.
- Removing optimizer state.
- Ignoring JNI reference management.
- Ignoring Vulkan synchronization.
- Ignoring GPU memory limits.
- Using deprecated APIs.
- Committing temporary debug code.
- Triggering unauthorized CI workflows.
- Merging into "main" without authorization.
- Claiming hardware validation that was not performed.

If a proposed fix requires one of these actions, stop and find a valid engineering solution.

---

21. Definition of Done

A task is complete only when all applicable requirements are satisfied.

Code Quality

- Implementation follows the pinned stack.
- No newly introduced deprecated APIs.
- No unaddressed critical compiler warnings.
- No known resource leaks.
- No unsafe ownership assumptions.
- No unrelated code changes.

Functional Correctness

- The intended feature works.
- Existing features remain functional.
- Error handling is implemented.
- Required tests pass.
- Relevant regression scenarios are covered.

Model Integrity

- Published production checkpoints and user-owned learning state remain intact.
- Explicitly authorized pre-release clean breaks create a fresh, internally consistent model/tokenizer/embedding/optimizer lineage without legacy migration code.
- Imported datasets, corpora, user-created content, and unrelated local data are preserved during model resets.
- Tokenizer IDs and fingerprints remain stable within each supported model lineage.
- Backend switching does not reset learning state.

Vulkan Correctness

Where applicable:

- Device capabilities are verified.
- Required features are supported.
- Memory allocations are valid.
- Synchronization is correct.
- Shader execution is validated.
- Numerical correctness is tested.
- GPU execution is distinguished from CPU fallback.

Build Verification

- The protected `Release`-tag production assembly succeeds after non-packaging validation.
- Host unit tests succeed.
- Android lint succeeds or all existing exceptions are documented.
- Native compilation succeeds for configured targets.
- Remaining hardware limitations are clearly stated.

Repository Integrity

- Correct branch is used.
- Unrelated user changes are preserved.
- No unauthorized CI execution occurs.
- No unauthorized merge or release occurs.

A task that compiles but fails its functional or data-integrity requirements is NOT complete.

---

22. Final Engineering Objective

Argus is not intended to be merely an Android application that can execute a few Vulkan operations.

Its long-term objective is a fully local AI system that can:

1. Initialize its own model from random weights.
2. Build its model according to device capabilities.
3. Process local training datasets.
4. Train using Vulkan as the primary compute backend.
5. Use CPU only as a controlled fallback.
6. Calculate forward and backward propagation correctly.
7. Perform real optimizer updates.
8. Preserve all model and optimizer state.
9. Save and restore checkpoints safely.
10. Continue learning after application restarts.
11. Maintain tokenizer identity within a model lineage and preserve dataset provenance/data; do not require compatibility with superseded pre-release tokenizer or checkpoint formats.
12. Develop its knowledge through local learning.
13. Operate without mandatory external training infrastructure.
14. Preserve user-owned learning data across software updates.

Every engineering decision must contribute to this objective.

The highest priority is correctness and preservation of learning state, not merely obtaining a successful APK build.

Build success is a requirement. Correct execution is the objective. Preserving learned knowledge is mandatory.
