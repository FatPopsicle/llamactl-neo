# Third-party notices

## HuggingFaceModelDownloader

The Hugging Face download planner, resumable ranged-transfer strategy, retry
behavior, SHA-256 verification flow, and GGUF selection concepts in
`src/huggingface.rs` are adapted from:

- **HuggingFaceModelDownloader**
- https://github.com/bodaay/HuggingFaceModelDownloader
- Inspected revision: `6dd57ee5b872b97e6698d2ec080b045f5dff7d2e`
- Copyright 2025, HuggingFaceModelDownloader contributors
- License: Apache License 2.0

A copy of the Apache License 2.0 is provided at
`LICENSES/Apache-2.0.txt`. The Rust implementation is modified substantially
for llamactl NEO and does not depend on or vendor the upstream Go package.
