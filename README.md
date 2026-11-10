# Argus

Argus is a native Android application built with Kotlin and Jetpack Compose.

## Requirements

- JDK 17
- Android SDK Platform 37
- Android SDK Build Tools 36.0.0
- Gradle Wrapper (repository-pinned)

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

Publishing a versioned GitHub Release (for example, `v1.0.7`) runs lint, unit tests, a signed release build, certificate verification, and APK publication.

The private repository keeps the CI signing-key backup in the private GitHub Release named `argus-signing-key`. Repository access therefore also grants access to the app-signing identity. Keep this repository private and restrict collaborator access. Never redistribute the signing-key asset.

## Local knowledge ingestion

### Web retrieval

- Web Search returns result titles, snippets, provider names, and actual destination URLs.
- Article extraction fetches a selected page and records its canonical URL, publisher, domain, publication date (when present), language, extracted character count, and SHA-256 digest.
- Search snippets and unresolved user queries are not inserted into training.
- A web article is shown in a review preview and is queued only after the user acknowledges that they have the rights or permission to use it.
- Article extraction is limited to readable HTML pages. Login walls, blocked pages, non-HTML content, and short landing pages are rejected.

### Local document formats

Supported imports include plain text, Markdown, JSON/JSONL, CSV/TSV, logs, Kotlin, Java, Python, HTML/XML, PDF, DOCX, EPUB, and ZIP archives containing supported formats.

- PDF extraction supports selectable text. Scanned/image-only PDFs are rejected; OCR is not included.
- DOCX reads the Word document XML body. EPUB follows its package spine order.
- Archive entry counts, per-entry expansion, total expansion, source file size, and extracted text length are bounded.
- Import budgets are reduced using the device's available RAM and app heap. The active training queue still holds generated chunk records in app memory; ingestion is buffered, not a fully disk-backed queue.

### Persistent corpus

- Imported documents are retained in Room as corpus-document metadata and normalized text chunks, independently from the active training queue.
- A SHA-256/UTF-16BE fingerprint identifies the exact imported character sequence. Identical content is not scheduled for training twice, while additional source and rights-provenance records are retained.
- Clearing the pending training queue does not delete retained corpus documents or their chunks, and does not delete a queue row once the worker has marked it active. The clear transaction marks documents with removed unfinished work as `PARTIAL`; a sole active final chunk is allowed to finish and mark its document `TRAINED`.
- Active queue rows are marked `TRAINING`; startup recovery returns interrupted rows to `QUEUED`. This is at-least-once recovery, so a chunk may be optimized again if the model checkpoint was saved but its completion status was not persisted before process death.
- Retained documents can be replayed into the training queue without reopening the original file or fetching the article again.
- Corpus status records whether training is ready, active, complete, or partial after a failed chunk.
- Room schema version 2 adds corpus tables and an optional training-item document reference through an additive version 1 to 2 migration. Existing chat, memory, and training-queue rows are preserved; destructive migration fallback is not enabled. Legacy queue rows remain valid but are not retroactively grouped into corpus documents because their original document boundaries and fingerprints were not stored.

## On-device training scope

Fresh model initialization selects tensor dimensions using total RAM, app heap, and current available RAM. Existing ARG3 checkpoints retain their saved tensor dimensions so an application update does not silently discard the user's model.

Training currently uses causal next-token cross-entropy, full-parameter backpropagation, gradient clipping, and Adam. Imported documents can be scheduled for up to three passes on devices that meet the memory thresholds; smaller or larger documents use fewer passes. Loss-only validation skips backpropagation gradient-buffer allocation and does not update model weights or optimizer state. An internal central finite-difference diagnostic and test cover representative output-head, embedding, MLP, attention, and LoRA parameter groups; execution remains part of final verification.

Within each context window, gradients are accumulated over its scored next-token positions and one Adam update is applied. Longer inputs are split into sliding windows with up to 24 tokens of overlap; each window performs its own optimizer update. The `trainStep` loss is a target-count-weighted aggregate of the window losses measured before each window update. It is a training-progress statistic, not a single post-update evaluation of the full document.

Current limitations:

- The tokenizer uses a fixed seed vocabulary and deterministic Unicode fallback. A learned BPE/Unigram tokenizer is not implemented.
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
- NPU/APU and Vulkan may be detected for device information, but the current training implementation executes through Kotlin CPU operations. Detection is not accelerator dispatch.
- This pipeline is not yet validated as scalable from-random-initialization pretraining or general intelligence training.

## Compatibility note

The standalone signing identity is separate from the legacy upload key. APKs signed with the standalone key cannot update installations signed by a different certificate. Keep the existing installation until its data has been safely exported or backed up before switching.
