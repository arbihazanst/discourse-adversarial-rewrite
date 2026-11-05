# Recovery of the supplied 3,000-row baseline

Source: `C:/KERJA DANI/vscode/OPEN ATTACK/OpenAttack-Share/CEAS_08_gpt_oss_latest_transparent_20260220_185010.csv`.

The supplied file contains **3,000 baseline records: 1,500 spam and 1,500 ham**, with original input text, true labels, predicted labels, and latency. Its filename identifies a GPT-OSS run; it does not record an exact checkpoint digest. A byte-identical copy is preserved here, and `summary.json` records source hashes. This is the first supplied file in this investigation that identifies the full ham baseline subset.

## Match to downloaded CEAS

All 3,000 records match downloaded CEAS with agreeing ground-truth labels:

| Matching rule | Rows |
|---|---:|
| Exact constructed subject/body | 579 |
| Whitespace/case-normalized equality | 2,416 |
| Empty source subject represented as `Subject: nan` | 5 |
| Total | 3,000 |

The five placeholder cases are baseline records 24, 389, 522, 702, and 2767. Matching that placeholder is restricted to source emails whose subject is actually empty and whose normalized body matches. There is no arbitrary removal of the word `nan` from other text. No fuzzy matching was needed for these baseline records.

The file has 2,997 distinct whitespace/case-normalized inputs. Three spam texts occur twice: baseline rows 249/1751, 632/2462, and 681/2050. All are predicted spam in this run. Separately, 13 baseline records have multiple candidate CEAS row IDs because the corpus contains duplicates. Preserve all candidate IDs and original baseline row numbers; do not silently deduplicate or replace original input text when reproducing the baseline.

## Historical baseline predictions

| Outcome | Count |
|---|---:|
| True-positive spam | 1,276 |
| True-negative ham | 1,396 |
| False-positive ham | 104 |
| False-negative spam | 224 |

Accuracy is **89.07%**, spam recall **85.07%**, spam precision **92.46%**, and spam F1 **88.61%**, calculated over the supplied rows including duplicates. These are recorded historical results, not new model inference.

## Connection to historical attack files

The following counts use the saved historical CEAS mappings, followed by equality of normalized downloaded source text with baseline source text. They preserve earlier mapping confidence and manual/encoding caveats; they are not all fresh exact matches of the raw historical text.

| Historical LLM | Attack rows | Linked source present in this baseline | Linked source outside this baseline |
|---|---:|---:|---:|
| GPT-OSS:20B | 1,276 | 1,276 | 0 |
| Qwen-2.5:32B | 1,200 | 1,200 | 0 |
| Gemma-3:1B | 1,344 | 1,344 | 0 |
| Qwen-2.5:0.5B | 872 | 872 | 0 |
| DeepSeek-R1:1.5B | 332 | 332 | 0 |
| GPT-5.2 | 1,150 | 878 | 272 |
| DeepSeek-V3.1 | 1,157 | 899 | 258 |
| Llama-4-Maverick | 1,123 | 880 | 243 |

All 1,276 GPT-OSS attack rows link to source texts whose ground truth and prediction are spam in the supplied baseline. Of these, 1,248 historical original texts also match the recorded baseline text directly after whitespace/case normalization. The other links rely on the existing source mappings.

Equal row counts do not establish a perfect one-to-one correspondence: one distinct normalized true-positive source-text variant, appearing at baseline rows 681 and 2050, has no linked GPT-OSS attack row under these matching rules, while the attack log repeats source identities. This variant contains a C1 control character where related IPTV email variants use a dash or replacement character; the apparent gap may be encoding-related, so it is not proof of a missing real-world email. See the duplicate manifest and `gptoss_baseline_tp_without_historical_attack.csv`. Thus the file strongly supports recovery of the GPT-OSS baseline and its attack population, with a remaining text-variant/multiplicity issue to reconcile.

Full attack-source containment for the other four models is compatible with this baseline, but does not independently recover their complete baseline predictions or prove that their ham samples were identical. The three models with outside-source records require their own sampling/baseline records or an explanation of replacement sampling, changed populations, or earlier mapping errors. Do not claim that all historical models used this same 3,000-row dataset.

## Relationship to the current benchmark

All **742 current clean spam sources** are present. Only **70 of the 742 current clean ham sources** are present, consistent with the current ham having been sampled separately. The existing 2,226-input benchmark has not been replaced.

This baseline can provide a fixed 3,000-row input set for future decision-model evaluation. Use the preserved original input text and ground-truth labels; each new defender must generate its own predictions and true-positive attack eligibility. Historical GPT-OSS predictions cannot serve as Clef, d3, or PPLX predictions. Any use of historical rewrites must retain coverage, matching confidence, and attack-stage provenance.

## Outputs and reproducibility

- `baseline_3000_manifest.csv`: original input text, stable baseline row ID, labels, every candidate CEAS ID, and matching rule.
- Original named CSV: byte-identical source copy.
- `duplicate_baseline_inputs.csv`: repeated normalized inputs and their predictions.
- `historical_attack_overlap.csv`: every historical LLM attack row, direct-text and source-linked baseline rows, source-match confidence, and attack-stage availability.
- `overlap_by_model.csv`: aggregate historical source coverage.
- `gptoss_attack_pool_discrepancies.csv`: attack rows linking to missing/non-eligible baseline sources; empty in this audit.
- `gptoss_baseline_tp_without_historical_attack.csv`: the reverse coverage gap, one distinct baseline true-positive source.
- `current_benchmark_overlap.csv`: current clean benchmark overlap with the recovered baseline.
- `summary.json`: counts, metrics, source paths, and hashes.

Reproduce from the project root with `python analysis/ceas_match/recover_baseline.py`. This audit reads the original file and existing source evidence, and writes only to this audit directory. It does not run inference or change classification scripts, benchmark inputs, or historical evidence.
