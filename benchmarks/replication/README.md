# External replication harnesses

Reproductions of *published* results in their original authors' own reference
setups. Nothing here runs inside archex's retrieval pipeline, and nothing here
measures archex. These harnesses exist to answer one question, asked by the S0
replication gate: can this project reproduce a result somebody else published?

Every arm is `replication` class. A result produced here never licenses a claim
about archex's own retrieval.

| Harness | Paper | Status |
| --- | --- | --- |
| `rlcoder/` | RLCoder, arXiv:2407.19487 (ICSE 2025) | primary S0 arm |
| _(none)_ | cAST, arXiv:2506.15655 (Findings of EMNLP 2025) | no harness: the released artifact is the chunker only, see `.docs/spikes/S0-replication-gate.md` |

Upstream code is never vendored. What is checked in is the pin, the patch, and
the commands. Every fetch is by explicit revision -- the git harness by commit,
each HuggingFace dataset and model by SHA -- and a resolved revision that differs
from its pin aborts the run rather than proceeding against different bytes.

Evidence artifacts land in `benchmarks/evidence/` and are validated with
`archex benchmark validate --kind replication`.
