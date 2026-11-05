# GPT-5.2 historical-file audit

These files help identify the GPT-5.2 attack subset, but do not recover the complete 3,000-email baseline. The raw file is byte-for-byte identical to the previously audited archive, and the combined file adds no new original-email keys.

## Evidence

| Check | Finding |
|---|---|
| Raw LLM results | 1,150 rows; semicolon-delimited |
| Combined GAN/LLM analysis | 1,142 rows; pipe-delimited |
| Combined original-email keys absent from raw LLM file | 0 |
| Combined rows reproduced from raw LLM text and labels using the old Latin-1 decoding | 1,142 / 1,142 |
| Combined keys/cardinalities equal the GAN/LLM inner join in the existing merge.py | Yes |
| Raw rows omitted by the inner join | 8 |
| Previously mapped CEAS source IDs in the raw file | 1,143 distinct IDs across 1,150 rows |
| Source labels at those IDs in the downloaded CEAS corpus | All spam (label 1) |
| Original texts affected by encoding differences in the combined file | 21 rows |

The 1,143 IDs retain previous mapping decisions. They must not all be described as independently unambiguous identities. Against the fresh Kaggle download, raw historical rows split into:

- 210 exact full-text matches.
- 908 full-text matches after whitespace/case normalization.
- 7 body-only matches requiring subject review.
- 25 rows without equality under those rules, requiring encoding/fuzzy review.

Among the 1,118 full-text matches, 10 rows have multiple duplicate source candidates. Thus 1,108 rows are independently unambiguous under the full-text matching rules, 10 are ambiguous, and 32 require review. These counts refer to historical rows, not distinct source emails.

## Why the combined file has fewer records

The existing `adversarialattack/merge.py` reads both raw files using Latin-1, joins on stripped original text with `how="inner"`, then exports UTF-8 with `|` delimiters. Reproducing those merge keys yields the exact 1,142 combined keys and row multiplicities. All combined LLM original/rewritten texts and both result labels also reproduce under that decoding.

The eight omitted raw record numbers (1-based CSV records, excluding the header) are **39, 48, 87, 217, 322, 389, 636, 910**. Five have blank Step-1 labels and three have SPAM labels. Their absence is explained by the intersection join; it does not prove they were outside the experiment. Encoding differences in 21 combined original texts are consistent with the script's Latin-1 interpretation of mixed/UTF-8 bytes. Use the raw file with documented decoding for lineage, and preserve original bytes.

## Missing-label evidence

The raw Step-1 results contain **1,088 SPAM + 52 NOT SPAM + 10 blank = 1,150 rows**. This accounts for the ten outcomes missing from the paper's two displayed label counts. The CSV does not explain why labels are blank; they cannot be silently counted as failed attacks, refusals, or API errors. Five Step-1 rewrite texts are blank. There are eight nonempty Step-2 rewrites labeled NOT SPAM after a Step-1 SPAM result; these are observed outcomes, not independently validated semantic successes.

## Limits for recovering the paper dataset

The files contain selected attack examples and their rewritten versions. They provide no complete ham population, baseline prediction per original input, sampling manifest, or seed. The GAN true-label column and CEAS labels establish recorded ground truth, not that GPT-5.2 originally classified every input correctly. Membership in an attack log alone cannot independently verify that baseline prediction.

The combined file's heuristic rubbish/success flags add derived analysis, not source identity or the missing baseline records. Neither file supplies the original 1,500 ham emails or establishes which other spam emails were in the reported baseline. Use the reconstructed source subset for traceability with its confidence flags, not as the full baseline or a proven random sample of CEAS.

## Outputs

- `row_mapping.csv`: each of 1,150 raw rows, preserved old CEAS ID, fresh candidates/confidence, stage labels, and membership in the combined file.
- `combined_to_raw_mapping.csv`: each combined row linked back to raw results, including encoding-difference indicators.
- `dropped_by_inner_join.csv`: eight omitted rows and their mapped source IDs.
- `historically_mapped_spam_sources.csv`: 1,143 distinct historically mapped CEAS source records, with original subject/body, occurrence counts, and identity status. This is an audit export, not a replacement benchmark.
- `summary.json`: counts and SHA256 hashes of all evidence inputs.

Reproduce from the project root with `python analysis/ceas_match/analyze_gpt52_files.py`. This uses the separately downloaded corpus and matching outputs in the parent folder. Original CSVs and classification pipelines are unchanged.
