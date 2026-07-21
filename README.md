# Bio-KCoT

Code and replay artifacts for knowledge-graph-enhanced long-chain-of-thought
biomolecular reasoning.

## Dataset availability

The manuscript's benchmark is named **BioMolKGQA**. The original generated rows
live under the gitignored `dataset/data/` path and are therefore not included in
a normal clone of this repository.

For a fully shareable pipeline fixture, this repository includes a deterministic
`BioMolKGQA-Synthetic` generator and a small generated dataset. It matches the
four task schemas and supplies ready-to-read SFT, GRPO, and evaluation files, but
contains only fictional mechanisms and cannot reproduce the paper's scientific
scores. See [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md).

For checkpoint replay instructions, see [REPLAY.md](REPLAY.md).
