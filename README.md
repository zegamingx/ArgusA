# Argus

Argus is a native Android application built with Kotlin and Jetpack Compose.

## Build toolchain

| Component | Pinned version | Purpose |
| --- | ---: | --- |
| Android Gradle Plugin (AGP) | 9.5.1 | Android build integration |
| Kotlin | 2.4.21 | Kotlin compiler and Compose compiler plugin |
| Kotlin Symbol Processing (KSP) | 2.3.12 | Room code generation |
| Gradle Wrapper | 9.8.1 | Build orchestration |
| JDK | 25 | JVM runtime used to execute Gradle |
| Java source/bytecode target | 21 | Language and bytecode compatibility for app sources |
| Android SDK Platform | 37 | Compile SDK and target SDK |
| Android minimum SDK | 34 | Minimum supported Android API |
| Android SDK Build Tools | 37.0.0 | APK packaging and signing tools |
| Android NDK | 30.0.16248370 | Native C/C++ toolchain |
| C++ language standard | C++23 | Vulkan JNI native library |
| JUnit | 6.1.3 | Host-side unit-test platform and Jupiter API |
| CMake | 4.5.0 | Native build configuration; minimum project requirement is 3.28 |

JDK 25 is the runtime for Gradle; it does not set the application's Java
bytecode target. Argus compiles Java-compatible application bytecode for
Java 21.

KSP is pinned to 2.3.12 rather than 2.3.13 because the latter has a reported
regression affecting concurrent symbol-processing executions. Keep the KSP
version explicit until a fixed release is verified.

**Compatibility gate:** Kotlin 2.4.21's published fully supported range ends
at Gradle 9.7.0 and AGP 9.3.1. Argus intentionally retains Gradle 9.8.1 and
AGP 9.5.1, which are newer than those documented maxima. Kotlin documents
newer combinations as usable but not fully verified; CI must validate this
project's actual combination before it is treated as build-verified.

The wrapper distribution is checksum-pinned in
`gradle/wrapper/gradle-wrapper.properties`. Do not replace it with an
unversioned system Gradle installation.

Local JVM unit tests use JUnit 6.1.3 on the JDK 25 host runtime. Existing
JUnit 4 test classes, including Robolectric suites that require
`RobolectricTestRunner`, run through the JUnit Vintage 6.1.3 engine while
new tests can use JUnit Jupiter. AndroidX instrumentation dependencies remain
on their AndroidX/JUnit 4 runner path; JUnit 6 is not placed on the
instrumentation runtime for the API 34 minimum-device tier.

## Local builds

Build a debug APK:

```bash
./gradlew :app:assembleDebug
```

Run release checks and build:

```bash
./gradlew :app:lintRelease :app:testReleaseUnitTest :app:assembleRelease
```

A release build requires the Argus release signing key at `signing/argus-release.p12`, or a different path supplied through `KEYSTORE_PATH`. The alias is `upload`; the keystore format is PKCS#12.

## GitHub releases

Every push to `main` runs release lint, JUnit/Robolectric unit tests, a signed release build, and APK signature/native-ABI verification. It does not run an Android emulator smoke test. Publishing the signed APK to the `Release` tag is a separate, explicit `workflow_dispatch` operation with `publish_release=true`; ordinary pushes only validate and build.

The private repository keeps the CI signing-key backup in the private GitHub Release named `argus-signing-key`. Repository access therefore also grants access to the app-signing identity. Keep this repository private and restrict collaborator access. Never redistribute the signing-key asset. The asset-pruning step only targets old `ARGUS-v*.apk` files in the `Release` tag and does not touch this signing-key backup.

## Runtime crash diagnostics

On startup, Argus reads Android's historical process-exit record and keeps a bounded
local diagnostic report containing the exit reason, status, description, PSS, and RSS.
Uncaught Java/Kotlin exceptions and foreground-service promotion failures are also
recorded locally. When a report exists, the app displays a **Copy diagnostics**
control so the user can share the exact startup failure for investigation. Reports
are not uploaded automatically.

## Vulkan training backend

The native training module is built with Android NDK r30, CMake 4.5.0
(project minimum 3.28), and C++23 for arm64-v8a, armeabi-v7a, and x86_64.
It negotiates up to the highest Vulkan API version supported by the pinned
headers, Android loader, and physical device (including 1.4 when available).
GPU execution uses Vulkan Synchronization 2 (Vulkan 1.3 core or
`VK_KHR_synchronization2`); devices without that feature remain on the Kotlin
CPU fallback. The backend does not issue legacy Synchronization 1 queue-submit,
pipeline-barrier, or timestamp commands. Buffer copies use the Synchronization 2
`COPY` stage and storage-buffer-specific access masks; timestamp placement does
not use deprecated top-of-pipe or bottom-of-pipe pseudo-stages.
It uses the NDK-bundled glslc compiler to compile the AdamW (`adamw.comp`),
batched-linear (`linear.comp`), resident attention (`attention_resident.comp`),
resident SwiGLU (`swiglu_resident.comp`), and Vulkan 1.2 FP16-weight linear
(`linear_fp16.comp`) compute shaders to SPIR-V during the native build.

When a compatible Vulkan GPU and compute queue are available, sufficiently large
matrix products in batched Q/K/V projections, attention output projection, SwiGLU,
the output head, output-head backward, attention output-projection backward,
SwiGLU backward, batched attention backward, and Q/K/V weight-gradient accumulation
can run through Vulkan Compute (using `linear_fp16.comp` with FP16 weights and FP32
accumulation when GPU FP16 arithmetic passes post-model runtime self-checks, or `linear.comp` otherwise). The three Q/K/V input-gradient products used by
embedding backpropagation are also batched through Vulkan. The fused AdamW update
runs on Vulkan for workloads with at least 49,152 optimizer elements. Attention
causal score generation, stable softmax, weighted V reduction, W_O plus residual,
and the post-attention LayerNorm, plus the element-wise SwiGLU activation, now run
in the batched training forward graph on Vulkan after independent native self-checks.
The post-SwiGLU LayerNorm and autoregressive generation LayerNorm remain in Kotlin,
as do the softmax-Jacobian reduction, LayerNorm backward, low-rank LoRA gradient
accumulation, token-ID embedding scatter, and global gradient-norm reduction. This is an incremental GPU training
path, not a fully GPU-resident training graph.

Vulkan is prepared asynchronously only after the local model has been constructed
or restored; application cold-start does not create the Vulkan runtime. Devices
without a compatible driver, compute queue, shader pipeline, suitable storage
memory, or sufficiently large optimizer workload use the existing Kotlin CPU
optimizer. Runtime verification failures also preserve the CPU fallback.
NPU/APU detection is informational only; no vendor-specific NPU training
runtime is currently integrated.

The CI pull-request workflow builds the native library and verifies that the
APK contains the configured ABI libraries. Main, pull-request, and release
workflows execute Gradle on JDK 25; the Android app's Java source and bytecode
compatibility target remains Java 21. All workflows install the checksum-pinned
CMake 4.5.0 distribution and NDK r30 for native builds.

## Background training service

The foreground service uses the `specialUse` type on Android 14 (API 34) and later (the app's supported `minSdk = 34` target) for the ongoing user-visible local compute workload. The manifest and service helper retain the legacy `dataSync` fallback only for Android 13 (API 33) compatibility tests, and supply the required `specialUse` subtype explanation. Google Play still reviews the declared `specialUse` use case; declaring it does not guarantee policy approval.

## Local knowledge ingestion

### Web retrieval

- Web Search queries the general web first through DuckDuckGo's HTML result endpoint. Wikipedia OpenSearch is consulted only as a supplementary/fallback provider and does not outrank general web results.
- Search results return titles, snippets, provider names, and actual destination URLs. Explicit requests such as `/web <query>`, `cari web <query>`, and `carikan <query> di internet` invoke the registered Web Search skill independently of the local model's confidence score.
- Google Custom Search JSON API is closed to new customers, and existing customers must migrate by January 1, 2027. Argus does not scrape Google's consumer result pages or rely on undocumented Google endpoints. A future Google-backed provider requires an eligible legacy API configuration or a supported Google replacement.
- Article extraction fetches a selected page and records its canonical URL, publisher, domain, publication date (when present), language, extracted character count, and SHA-256 digest.
- Search snippets and unresolved user queries are not inserted into training.
- A web article is shown in a review preview and is queued only after the user acknowledges that they have the rights or permission to use it. Both `TrainingManager` and the repository persistence boundary enforce this rule for text, stream, and validation imports: web sources with `UNVERIFIED` rights status are rejected before corpus storage.
- Article extraction is limited to readable HTML pages. Login walls, blocked pages, non-HTML content, and short landing pages are rejected.

### Local document formats

Supported imports include plain text, Markdown, JSON/JSONL, CSV/TSV, logs, Kotlin, Java, Python, HTML/XML, PDF, DOCX, EPUB, and ZIP archives containing supported formats.

- PDF extraction supports selectable text. Scanned/image-only PDFs are rejected; OCR is not included.
- DOCX reads the Word document XML body. EPUB follows its package spine order.
- Archive entry counts, per-entry expansion, total expansion, source file size, and extracted text length are bounded.
- Import budgets are reduced using the device's live app-heap headroom and a bounded queue-capacity estimate.
- `Reader` ingestion stages UTF-16BE code units in a temporary cache file using fixed-size buffers. The file is removed after success, rejection, or coroutine cancellation.
- JSONL is validated and processed one complete object per line, with per-record size and total generated-record limits. JSONL records are persisted only after the entire source has passed validation, so a malformed tail cannot leave a partially imported document.
- Single-object ChatML/ShareGPT, reasoning, and legacy plain-text inputs retain document-level parsing semantics. Their in-memory parser path is still bounded by the live-heap limit; this is not a constant-memory parser for every format.
- The active training queue still holds generated chunk records in app memory; the queue itself is not fully disk-backed.

### Persistent corpus

- Imported documents are retained in Room as corpus-document metadata and normalized text chunks, independently from the active training queue.
- A SHA-256/UTF-16BE fingerprint identifies the exact imported character sequence. Identical content is not scheduled for training twice, while additional source and rights-provenance records are retained.
- Clearing the pending training queue does not delete retained corpus documents or their chunks, and does not delete a queue row once the worker has marked it active. The clear transaction marks documents with removed unfinished work as `PARTIAL`; a sole active final chunk is allowed to finish and mark its document `TRAINED`.
- Active queue rows are marked `TRAINING`; startup recovery returns interrupted rows to `QUEUED`. This is at-least-once recovery, so a chunk may be optimized again if the model checkpoint was saved but its completion status was not persisted before process death.
- Retained documents can be replayed into the training queue without reopening the original file or fetching the article again.
- Corpus status records whether training is ready, active, complete, or partial after a failed chunk.
- Room schema version 3 adds corpus tables and an optional training-item document reference through additive migrations 1 to 2 and 2 to 3. Existing chat, memory, queue rows, corpus documents, chunks, and source identities are preserved; destructive migration fallback is not enabled.
- Version 3 adds nullable source-level license identity/name/URL, attribution, rights-evidence URL, user declaration text/timestamp, and reviewed timestamp. Historical rights claims without sufficient source evidence are conservatively migrated to `UNVERIFIED`; user-declared provenance is retained only when a prior explicit approval marker exists.
- Aggregate document rights are recalculated from every retained source. A document is not treated as license-reviewed merely because a different source row declared permission.
- Legacy queue rows remain valid but are not retroactively grouped into corpus documents because their original document boundaries and fingerprints were not stored.

## Argus-native architecture contract

- `ArgusArchitectureConfig` describes the model with explicit layer count, hidden size, attention heads, feed-forward size, vocabulary size, context length, adapter rank, and Argus-owned operator metadata.
- `ArgusArchitectureParameterEstimator` calculates parameters across every declared block. Token embeddings and the output projection are counted once; attention, SwiGLU, and optional Q/V adapter matrices scale with layer count. Parameter arithmetic is overflow-checked.
- The estimator distinguishes trainable parameters from resident model tensors. The current fixed sinusoidal position table occupies memory but is not a trainable parameter. The legacy `countTotalParameters()` method retains its historical resident-tensor count; `countTrainableParameters()` exposes the trainable-weight count.
- The executable engine runs a configurable stack of causal multi-head attention + SwiGLU blocks with separate token embeddings/output projection and Q/V LoRA. Forward execution, reverse backpropagation, optimizer state, KV cache, and ARG4 checkpoint tensors use the configured layer count. ARG3 remains a legacy one-block compatibility format.
- The tokenizer configuration is versioned and persisted with each ARG4 checkpoint. Its default is `BYTE_LEVEL_BPE_V2` (learned byte-level BPE with a `TokenizerConfig` default of 2,048 vocabulary entries—or tier-specific `1,024 / 2,048 / 4,096 / 8,192` capacities selected by `EngineConfig.forDevice`—and a 16 UTF-16-code-unit token limit), alongside `LEARNED_BPE_V1` and `GREEDY_LONGEST_MATCH_V1` compatibility modes; larger capacities and token lengths can be selected explicitly. ARG3 checkpoints retain the legacy greedy-longest-match algorithm and deterministic Unicode fallback.
- The architecture contract is Argus-native metadata, not a third-party model format or runtime. It does not add a pretrained model, change random initialization, or alter existing ARG3 checkpoint bytes.
- The inference memory estimator can consume the same validated architecture descriptor, deriving trainable weight count, KV-cache shape, and the fixed sinusoidal buffer instead of accepting unrelated parameter and shape values. Its result remains a planning estimate; it does not load or quantize tensors.

## Device-adaptive training model profiles

Argus selects one of four native model families from installed RAM, the current available-system-memory snapshot, the app heap limit, and CPU core count:

| Model family | Family parameter band | Default training batch | Tier vocabulary capacity |
| --- | ---: | ---: | ---: |
| Argus Compact | 1M–1.4M | 1 | 1,024 |
| Argus Balanced | 1.4M–1.9M | 2 | 2,048 |
| Argus Performance | 1.9M–4.8M | 4 | 4,096 |
| Argus Flagship | 4.8M–7M | 8 | 8,192 |

The family band is an architecture target capped at the 7M executable parameter ceiling (`MINIMUM_EXECUTABLE_PARAMETER_BUDGET = 1,122,304`), not a guarantee that every device reaches the tier's upper boundary; Neural Core reports the family ceiling and actual active parameter count separately. The active model shape is selected from tier-specific dimensions (`d_model` 128–256, `d_ff` 384–1,024, 4–8 attention heads, 1–32 layers, context 64–192, and LoRA rank 4 or 8), then capped by a conservative memory budget. The budget uses at most 45% of the app heap and 25% of currently available system RAM; only 55% of that smaller budget is assigned to the 18-byte-per-parameter accumulated training-state floor (2-byte FP16 parameters + 4-byte FP32 gradients + 8-byte FP32 AdamW moments + 4-byte FP32 accumulation buffer). Remaining memory is reserved for activations, attention scratch, tokenizer expansion, checkpoint buffers, and allocator overhead.

`ArgusArchitectureConfig` and `InferenceMemoryEstimator` still support planning descriptions for much larger architectures independently of active training. The active Android training runtime rejects shapes that exceed JVM array limits or the measured app-heap budget.

The training queue uses a true serial micro-batch of 1–8 independent samples. Each sample's context windows are processed sequentially, their gradients are target-count weighted and accumulated, and one AdamW update is applied after the batch. Activations are not retained for all samples at once. The training worker assigns one optimizer update to the first trainable sample in the batch so session metrics do not count the same update more than once.

Existing ARG3/ARG4 checkpoints retain their saved dimensions. When the current device cannot safely reserve the additional accumulation buffer, Argus keeps the saved model on its single-sample compatibility path (14 bytes/parameter baseline) and forces batch size 1; it does not resize or silently replace the user's model. A fresh installation requires enough memory headroom before Genesis allocates its model.

**Note:** Multi-billion-parameter figures in the separate inference-planning policy do not represent active on-device training capacity; the current FP16-weight / FP32-state trainer caps executable training models at 7M parameters. The running model's actual parameter count is reported separately from its family ceiling and inference planning range.

## Device-adaptive inference capacity

- The initial RAM-to-parameter policy uses conservative currently available system memory (Android available memory minus the low-memory threshold), not installed RAM.
- The planning anchors are 1 GB: 0.5B–1B; 2 GB: 1.5B–2.5B; 3 GB: 2.5B–3.5B; 4 GB: 3B–5B; 5 GB: 4B–6B; 6 GB: 5B–7B; 7 GB: 6B–8B; 8 GB: 7B–9B; 10 GB: 9B–12B; 12 GB: 12B–14B; 16 GB: 18B–21B; 24 GB: 28B–32B; and 32 GB: 38B–42B.
- Values between anchors are linearly interpolated. Above 32 GB, the estimate grows by 1.3B–1.5B parameters per additional GB.
- These values are RAM-only planning targets, not guaranteed model-load limits. A future inference loader must inspect the actual model quantization and architecture metadata, estimate KV cache and runtime workspace, and enforce a memory safety reserve before loading.
- The custom FP16-weight / FP32-state training engine remains independent from inference sizing. Its 14-byte-per-parameter base floor accounts for FP16 parameters (2B), FP32 gradients (4B), and two FP32 Adam moments (8B). The serial gradient-accumulation path adds an FP32 gradient buffer, bringing its state floor to 18 bytes per parameter before activations and temporary buffers.
- An Argus-native multi-billion-parameter inference runtime is not yet integrated. Argus must load its own checkpoint, architecture configuration, tokenizer state, and tensor data without requiring an external model format or runtime. Quantized on-disk checkpoint storage is a future Argus-owned feature.

## Argus checkpoint format

- `ARG3` remains a supported legacy read format. Its fixed tensor order is retained in the compatibility loader.
- New checkpoints are written as `ARG4`. The fixed header records the architecture schema and dimensions, tokenizer schema/algorithm/capacity, optimizer step, payload size, and checksum.
- The payload stores the Argus architecture ID, vocabulary entries, learned BPE merge rules (when the tokenizer uses BPE merges), a named tensor directory, and `FLOAT32_TENSOR_TYPE` serialized tensor data. Active in-memory model weights use FP16 (`Float16ParameterArray`) while optimizer moments remain FP32; tensors quantize/dequantize cleanly across checkpoint load/save, and tensor names and element counts are validated against the active engine before restore.
- Tensor payloads are written and read in bounded chunks. The writer uses a temporary file and atomic replacement; the payload CRC32 is verified before model or tokenizer state is changed.
- Restoring an `ARG3` checkpoint does not discard it. The next successful checkpoint save writes the current state in `ARG4`.
- The current engine supports configurable multi-layer execution with FP16 in-memory weights, FP32 optimizer state, and `FLOAT32_TENSOR_TYPE` ARG4 checkpoint serialization. The named manifest stores each layer's tensors; it is not an INT4/INT8 quantized checkpoint format.

## On-device training scope

Fresh model initialization selects tensor dimensions using total RAM, app heap, and current available RAM. Existing ARG3 checkpoints remain readable and retain their saved tensor dimensions. New saves use ARG4, an Argus-native checkpoint with architecture/tokenizer metadata, learned BPE merge rules, a named tensor manifest, streamed tensor I/O, and a CRC32 payload check.

Genesis currently seeds three short foundation notes as retained TRAIN corpus documents. Their scheduling is idempotent: already queued, active, or trained notes are not duplicated, while a cleared or interrupted note can be replayed from Room. These notes are a small bootstrap seed, not a general-purpose language dataset; no C4 or other broad offline language corpus is bundled.

Training currently uses causal next-token cross-entropy, full-parameter backpropagation, gradient clipping, and Adam. Imported documents can be scheduled for up to three passes on devices that meet the memory thresholds; smaller or larger documents use fewer passes. Loss-only validation skips backpropagation gradient-buffer allocation and does not update model weights or optimizer state. An internal central finite-difference diagnostic and test cover representative output-head, embedding, MLP, attention, and LoRA parameter groups; execution remains part of final verification.

For each training micro-batch, gradients are accumulated across all scored next-token positions in every sample and context window, then one AdamW update is applied. Longer inputs are split into sliding windows with up to 24 tokens of overlap; all windows belonging to the micro-batch contribute to the same update. Each sample's loss is reported as a target-count-weighted aggregate measured before that shared update. It is a training-progress statistic, not a single post-update evaluation of the full document.

Current limitations:

- Learned BPE (`BYTE_LEVEL_BPE_V2` by default, plus `LEARNED_BPE_V1`) trains and persists merge rules in `ARG4` checkpoints up to the configured vocabulary capacity, while Unigram tokenization is not implemented and `ARG3` retains legacy `GREEDY_LONGEST_MATCH_V1` compatibility.
- Corpus documents use a document-level TRAIN or VALIDATION split. Validation documents retain one normalized source-text payload, create no training chunks, never enter the training queue, and cannot be replayed into training.
- Held-out evaluation uses evaluateValidationCorpus, aggregates cross-entropy by the exact number of scored next-token targets, and computes perplexity without updating model weights or optimizer state.
- A content fingerprint cannot exist in both splits. Importing identical source text into the opposite split is rejected to prevent train/validation leakage.
- Split selection is explicit through TrainingManager.importValidationDocument; the app does not randomly split chunks or automatically choose held-out documents. The caller must select representative validation documents before training.
- Held-out metrics are calculated over the normalized document source stream, not a reconstruction of each importer-specific training chunk boundary. Compare runs using the same fixed validation documents and model configuration; do not interpret validation loss as an exact reproduction of the per-chunk training loss.
- Corpus replay supports three deterministic policies: `SEQUENTIAL_DOCUMENTS` preserves stable import order; `SHUFFLED_DOCUMENTS` shuffles document order once per epoch; `INTERLEAVED_DOCUMENTS` randomly selects the next available document chunk while preserving each document's chunk order. Chunks are never shuffled within a source document, so overlapping windows and source sequence remain intact.
- Corpus-wide replay requires an explicit seed. The same seed, policy, epoch count, and unchanged corpus snapshot reproduce the same queue order. The returned `CorpusReplayResult` records the seed and policy used. The single-document compatibility API defaults to seed 0 and sequential ordering unless the caller overrides them.
- Corpus replay rejects documents that already have queued or active training work, preventing accidental duplicate scheduling. A completed or cleared corpus can be replayed again using the same seed.
- Replay is bounded to 500 documents per scheduling request. Its default chunk limit is estimated from the current app heap (using a conservative 4 KiB per queue item) and capped at 4,096; callers may configure a larger limit up to the 50,000-item hard ceiling. Requests exceeding the configured limit are rejected before queue insertion.
- `replayCorpusDocument` remains available for one-document replay and defaults to sequential chunk order.
- Vulkan executes eligible batched forward/backward matrix products (via `linear_fp16.comp` or `linear.comp`) and AdamW updates (`adamw.comp`) after native numerical self-checks. The batched training forward graph uses Vulkan for causal attention, W_O plus residual, and post-attention LayerNorm after native verification. QKV/LoRA gradient accumulation, LayerNorm backward, and global gradient-norm reduction remain Kotlin operations. NPU/APU hardware detection is informational only; no vendor-specific NPU/APU training runtime is integrated.
- This pipeline is not yet validated as scalable from-random-initialization pretraining or general intelligence training.

## Compatibility note

The standalone signing identity is separate from the legacy upload key. APKs signed with the standalone key cannot update installations signed by a different certificate. Keep the existing installation until its data has been safely exported or backed up before switching.
