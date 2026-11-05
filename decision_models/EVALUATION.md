# Paper and source evaluation for decision models

Evaluation reads saved original texts, rewrites, labels, and response artifacts. It does not call the attacker or defender. Baseline and attack metrics are written automatically by the adaptive runner. Similarity and content-retention evaluation is a separate command or terminal-menu action, so it can be rerun without repeating inference.

## Definitions and source mapping

The supplied NLPJ paper (Sections 3.7, 4.7–4.8 and results tables) and IEEE Access paper (Sections III.G, IV.G–H, V.D–F; Table 6) define the measures below. The implementation also consults `OpenAttack-Share/chart.ipynb`, cells 0–1, and the previously supplied `adversarialattack/content_retention_analysis_all_models.py`. Exact source paths and SHA256 hashes are in `analysis/paper_evaluation/sources.json`.

| Measure | Definition / implementation |
|---|---|
| Clean accuracy | `(TP + TN) / valid baseline classifications`; report missing baseline rows separately |
| Spam precision, recall, F1 | `TP/(TP+FP)`, `TP/(TP+FN)`, `2TP/(2TP+FP+FN)` |
| Ham and aggregate classification | Ham precision/recall/F1; explicit macro and support-weighted precision/recall/F1 so aggregate conventions are unambiguous |
| Step-1 ASR (%) | `100 * Step1 flips / true-positive spam pool` |
| Step-2 gain (%) | `100 * additional Step2 flips / rows still SPAM after Step1` |
| Final ASR (%) | `100 * (Step1 flips + additional Step2 flips) / true-positive spam pool` |
| Robust accuracy (%) | `100 - final ASR`; this is robustness on the eligible spam pool, not accuracy across all 3,000 emails |
| Modified ASR (%) | Source notebook: flips divided by eligible rows with a generated first rewrite |
| Effective robust accuracy (%) | Source notebook: `100 - modified ASR` |
| SBERT | Cosine of normalized original/rewrite sentence embeddings |
| BLEU | Sentence BLEU-4, uniform n-gram weights, Chen–Cherry method1 smoothing, epsilon 0.1 |
| ROUGE-L | Token LCS F1, no stemming |
| Jaccard | Unique normalized-token intersection / union |
| Length ratio | Rewrite token count / original token count |
| Keyword retention | Fraction of original top-10 frequency keywords present in rewrite |
| Conditional similarity delta | Mean score of successful flips minus mean score of failures, separately per step |
| Similarity statistics | Group counts, means, sample standard deviation (`ddof=1`); SBERT independent two-sided Welch t-tests |

ASR fields are explicitly labeled paper-style **label-flip** rates. Similarity scores are descriptive proxies; they do not independently prove spam-intent preservation. The `semantic_validity` flag remains unassessed by a human.

Incomplete runs do not receive a final ASR/robust accuracy or effective ASR. Instead, bounds retain the full observed eligible pool, and `source_observed_modified_flip_pct` exposes the notebook-style observed rate with unresolved rows still present. Zero denominators and missing measurements are blank/JSON null. No empty rewrite, skipped baseline false negative, or classification error becomes a successful attack.

## Explicit reproduction choices

The supplied papers describe SBERT without naming an exact checkpoint, and the inspected source folder does not provide the BLEU/ROUGE/SBERT scoring implementation. Therefore the metric families and formulas are reproduced, but exact numeric equivalence to the historical similarity tables is not claimed.

- Default SBERT is `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. This is a documented implementation choice, not an attribution to the paper. Override with `--sbert-model` and its immutable `--sbert-revision` to use a recovered original checkpoint.
- The encoder's own maximum sequence length is retained. Original/rewrite truncation flags are saved; long-email scores may represent only the encoded prefix. Model revision, maximum length, device, and package versions are recorded in `evaluation.json` and the cache identity.
- Lexical/content scoring lowercases and tokenizes using `[a-zA-Z0-9]+`. The keyword top-10 procedure and small stopword set match `content_retention_analysis_all_models.py`. Empty keyword sets are missing rather than the archive's zero; missing data is never silently treated as lost content.
- ROUGE-L uses these same normalized tokens, without stemming. BLEU uses the explicit settings above; these details were unspecified in the paper.
- The paper's independent two-sample test is implemented as Welch's test. P-values are descriptive and unadjusted. Groups smaller than two and degenerate/nonfinite tests receive a status, not a fabricated p-value. Preserved duplicate inputs limit independence; Step 2 is a conditional subset and must not be interpreted as an independent Step-1 comparison.
- SBERT bands follow the paper descriptively: `>0.80` strong, `0.75–0.80` moderate, `<0.75` potential drift. They do not filter the ASR denominator.

The [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html) documents pinned revisions and normalized embeddings. The [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes the selected encoder and its truncation behavior.

## Commands

For the interactive workflow, open the relevant CLI and choose **4 Results and evaluation**, then select the saved run. Direct Python commands remain available below.

From the project root, install optional full-evaluation dependencies in your environment:

```bash
python3 -m pip install -r scripts/requirements-evaluation.txt
python scripts/evaluate_paper.py --model clef
```

The first full evaluation downloads the pinned public embedding model into `.cache/evaluation_models/` and runs it locally on CPU. No email text is uploaded. Use `--device cuda` if desired, or `--local-files-only` once weights are cached. Classification itself still requires only the standard library. The evaluation packages may require a newer Python than the classifier's minimum; use a supported Python environment for those dependencies.

Evaluate all available saved models, or another experiment directory:

```bash
python scripts/evaluate_paper.py --model all
python scripts/evaluate_paper.py --run-dir outputs/paper3000_hybrid --model all
```

Without optional dependencies/model weights:

```bash
python scripts/evaluate_paper.py --model clef --lexical-only
```

Lexical-only explicitly leaves SBERT and its significance tests uncomputed. Use full evaluation later to add them. If the project was moved to another server, pass `--dataset dataset/paper_baseline3000.csv`; the dataset hash must match the saved run.

Evaluation follows the saved run's selected IDs, including its pilot/full scope. Old runs without a recoverable selection manifest use all declared dataset rows, retaining missing predictions. It validates run identities and raw artifact hashes before computing results. A per-pair content/settings cache is saved as work progresses; rerunning the same command reuses completed scores. Outcome groups are always recalculated from current saved labels, so retrying a failed classification does not leave stale group assignments.

## Output

For each model, baseline/attack metrics remain in `metrics.csv` and `metrics.json`. Similarity evaluation writes:

```text
outputs/paper3000_adaptive/<model>/evaluation/
  paper_metrics.csv
  similarity_per_rewrite.csv
  similarity_summary.csv
  evaluation.json
  checkpoint.json
```

`similarity_per_rewrite.csv` contains Step 1, Step 2, and final-candidate rows. Every comparison uses the original email as its reference. Final candidates are the successful Step-1 rewrite or a classified Step-2 rewrite. An unfinished Step 2 is not replaced with a Step-1 failure. Do not pool final rows with step rows as independent examples. Generated but unclassified candidates are kept in an `unresolved` group.

`similarity_summary.csv` provides per-stage, per-metric summaries for all available rewrites, successes, failures, and unresolved candidates, including group counts and conditional deltas. The stage labels identify conditional populations. Changing similarity settings changes the score-cache identity; it never invalidates saved model inference.

The main GAN pipeline is available through main CLI menu 1/2 -> GAN. Use the same evaluator with `--run-dir outputs/paper3000_gan`. GAN evaluation separates all `gan_candidate` proposals from the selected `gan_returned` texts and uses their corresponding probe/post-attack labels. It never substitutes fallback originals as perfect-similarity rewrites. See [GAN setup, protocol, and evaluation](GAN.md). Historical GAN archive values are not imported into new results.
