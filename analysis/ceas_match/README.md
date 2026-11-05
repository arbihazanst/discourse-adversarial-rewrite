# CEAS-08 download and independent text-matching audit

Downloaded on 3 November 2026 from <https://www.kaggle.com/datasets/doryanay/ceas-08> using Kaggle's public dataset-download API. Exact download time and ZIP/CSV SHA256 hashes are recorded in `download_provenance.json`. The downloaded CSV is preserved unchanged as `CEAS_08.csv`.

## Findings

The downloaded corpus has **39,154 records: 21,842 label-1 spam and 17,312 label-0 ham**.

- **1,484 / 1,484 clean benchmark records match exactly** at their existing CEAS record IDs, including constructed email text, subject, body, and label.
- **742 / 742 primary rewrites pass their lineage checks**: the original email matches the downloaded CEAS record exactly, and the rewrite text and source ID agree with the linked historical LLM result. This checks provenance, not semantic validity of the rewrite.
- The historical audit covers **16,887 original-text rows from 16 GAN/LLM files**. These are repeated experiment rows, not 16,887 unique emails.

| Fresh matching method | Historical rows |
|---|---:|
| Exact constructed subject + body | 3,810 |
| Equal after whitespace and case normalization | 12,677 |
| Body-only equality after whitespace and case normalization | 68 |
| No equality under the above rules; fuzzy/manual review required | 332 |
| Total | 16,887 |

The first three groups recover **16,555 existing source IDs** automatically. Body-only matches remain review-required because they do not verify subject identity. Of those automatically matched rows, **118 have multiple candidate CEAS records**; this is a subset of the table, not an additional category. No arbitrary first candidate is presented as a uniquely identified record.

The remaining **332 rows represent 36 distinct raw original-text strings**. They receive candidate suggestions and explicit comparisons against the previous mapping. Similarity does not automatically confirm identity. Differences include replacement characters, encoding damage, and punctuation. Previous audit categories differ because its normalization/manual matching rules were broader. See the saved evidence instead of interpreting stricter matching as proof that the previous IDs were wrong.

The historical LLM files contain **1,787 distinct normalized original texts and 1,787 historically mapped source IDs**. This cannot establish a single shared 1,500-spam baseline. The historical attack logs do not identify the original 1,500 ham records. **The exact 3,000-email baseline from the paper has therefore not been recovered.** Downloading the full corpus establishes membership and provenance, not historical sample selection.

## Method and interpretation

1. Parse the downloaded CSV with a CSV reader; preserve 1-based record numbers excluding the header. Multiline emails make physical file line numbers unsuitable as record IDs.
2. Construct source text as `Subject: {subject}` + two newlines + `{body}`. Search the entire source corpus, including both labels; do not assume an attack input must match spam.
3. Decode historical mixed UTF-8/Windows-1252 bytes while keeping valid UTF-8. Original files remain unchanged. Match each raw historical original against successively broader rules: exact, normalized line endings, collapsed whitespace plus casefold, Unicode typography normalization, and normalized body-only equality.
4. Preserve every candidate at the first matching level. Only then compare fresh candidates with the saved historical crosswalk. That crosswalk does not determine the initial candidate search.
5. For unresolved texts, retrieve candidates from all 39,154 emails using word TF-IDF cosine (at most 150,000 features, sublinear term frequency, Unicode accent stripping). Rerank the top 20 plus all best-score ties by normalized character similarity within rounded cosine ties; retain the best five. This is candidate retrieval, not exhaustive edit-distance ranking and not a semantic embedding score.
6. Report normalized character similarity with `SequenceMatcher(autojunk=False)` and token-set Jaccard. A TF-IDF score of 1 can coexist with damaged punctuation or non-Latin text; do not treat it as character equality. Direct comparisons against old IDs are supplied separately even when those IDs are outside the top five retrieved candidates.

All fuzzy identities and body-only matches require review. Exact/normalized matches also retain duplicate-ID ambiguity. None of these similarity measures establish that an adversarial rewrite preserves spam intent or qualifies as a valid attack.

## Files to inspect

- `historical_matches.csv`: every historical row, fresh matching category, all exact/normalized candidate IDs, previous source ID, ambiguity, and best fuzzy suggestion where needed.
- `match_summary_by_file.csv`: compact per-experiment counts.
- `previous_mapping_verification.csv`: similarity of every original against its previously mapped ID in the new download.
- `fuzzy_candidates_review.csv`: five suggestions per unresolved raw text, with scores and full original/candidate texts.
- `encoding_differences_review.csv`: grouped comparisons with the old source IDs, full texts, and differing text segments. Each group includes the number of affected historical rows.
- `ceas_source_usage.csv`: all 39,154 source IDs, labels, hashes, and counts of automatic historical candidate matches. A zero count means no match under these rules, not that the email was absent from the historical baseline.
- `benchmark_clean_verification.csv`: all 1,484 clean benchmark checks.
- `primary_rewrite_lineage_verification.csv`: all 742 rewrite provenance checks.
- `summary.json`: machine-readable totals and hashes of analysis inputs.
- `download_provenance.json`: download provenance and corpus hash.
- `ceas-08.zip`, `CEAS_08.csv`: the downloaded archive and extracted corpus.
- `match_ceas.py`: reproducible audit code. Requires Python, NumPy, and scikit-learn for fuzzy retrieval; it performs no model inference and makes no network calls.

From the project root, rerun the matching with:

```powershell
python analysis/ceas_match/match_ceas.py
```

This only replaces audit outputs in this folder. Existing benchmark datasets, historical results, classification pipelines, and their output files are not modified.
