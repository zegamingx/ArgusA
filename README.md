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
- Import budgets are reduced using the device's available RAM and app heap. The training queue still holds generated chunk records in app memory; ingestion is buffered, not a fully disk-backed queue.

## On-device training scope

Fresh model initialization selects tensor dimensions using total RAM, app heap, and current available RAM. Existing ARG3 checkpoints retain their saved tensor dimensions so an application update does not silently discard the user's model.

Training currently uses causal next-token cross-entropy, full-parameter backpropagation, gradient clipping, and Adam. Imported documents can be scheduled for up to three passes on devices that meet the memory thresholds; smaller or larger documents use fewer passes. Validation loss and perplexity can be measured without updating model weights.

Current limitations:

- The tokenizer uses a fixed seed vocabulary and deterministic Unicode fallback. A learned BPE/Unigram tokenizer is not implemented.
- There is no corpus-level shuffle/replay scheduler or automatic held-out corpus management yet.
- The validation APIs require the caller to supply held-out text; they do not create a train/validation split automatically.
- NPU/APU and Vulkan may be detected for device information, but the current training implementation executes through Kotlin CPU operations. Detection is not accelerator dispatch.
- This pipeline is not yet validated as scalable from-random-initialization pretraining or general intelligence training.

## Compatibility note

The standalone signing identity is separate from the legacy upload key. APKs signed with the standalone key cannot update installations signed by a different certificate. Keep the existing installation until its data has been safely exported or backed up before switching.
