# Invalidated Artifacts Notice

The files in this directory:
- `label_agreement_fabricated.json`
- `qrels_fabricated.json`

were generated in error during an unverified simulation in turn 1664 (commit `9f27253`). They falsely treated programmatic keyword/random annotations as human review.

### Status:
- **INVALID & VOID**: These results are entirely excluded from all official evaluation benchmarks, tables, and documentation.
- Authentic human review will be conducted on `labels_to_review.csv` and imported strictly via `labels_reviewed.csv`.
