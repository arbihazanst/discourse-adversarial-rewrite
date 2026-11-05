# GAN rewriting in main

The GAN path follows the default `OpenAttack.GANAttacker()` used by the supplied `OpenAttack-Share/GANSSILAB.py`: **SST**, not SNLI. It uses the source SST autoencoder/resource on CPU and the existing structured decision-model adapters. It does not use a Gemma rewrite attacker or the two-step LLM prompts. When the defender is PPLX Tool, Gemma still orchestrates its classification calls.

## Setup and run

Use a Python environment compatible with PyTorch (Python 3.10+ is recommended):

```bash
python3 -m pip install -r scripts/requirements-gan.txt
python3 scripts/prepare_gan.py
bash run_cli.sh --task gan --dry-run
bash run_cli.sh --task gan
bash run_cli.sh --task gan --retry-errors
```

`prepare_gan.py` downloads the approximately 55 MiB official `SGNAE.zip` resource through OpenAttack's official CDN into `.cache/openattack_gan/AttackAssist.SGAN/`. The original download host had an expired TLS certificate during implementation; the CDN is used with TLS verification enabled. Only the five expected model/vocabulary files are extracted. Their hashes are saved and verified before runs.

To import an existing resource instead:

```bash
python3 scripts/prepare_gan.py --from-dir /path/to/AttackAssist.SGAN --output-dir .cache/gan_from_server
python scripts/gan_pipeline.py --config decision_models/adaptive_main.json --model pplx --gan-resource-dir .cache/gan_from_server --pilot
```

The resource directory must contain `vocab.json`, `a.pkl`, `i.pkl`, `g.pkl`, and `d.pkl`. Resources are loaded as CPU state dictionaries with PyTorch `weights_only=True`. Do not replace weights inside a completed run: resource hashes form part of attack identity.

Use the two main CLI launchers (run_cli.bat / run_cli.sh). Choose **1 Pilot -> 3 GAN -> 1 Run**, or **2 Benchmark -> 3 GAN** for the full dataset with verified revisions. **Menu 3** resumes; **menu 4** evaluates saved results; **menu 5** configures models and prepares GAN weights. Other defenders can be selected for full benchmarks. Docker is not managed automatically.

## Protocol and baseline reuse

1. Classify the fixed 3,000 originals, or reuse matching saved baseline predictions from the LLM experiment.
2. Select true spam whose baseline prediction is SPAM. Ham and baseline false negatives are excluded.
3. Run the source SST GAN generator and record its candidate. The decision-model adapter returns a NumPy array with `SPAM=1`, `NOT_SPAM=0`, as expected by OpenAttack.
4. The source GAN returns the candidate only if its internal untargeted goal succeeds. A rejected candidate is recorded even though the attacker returns `None`.
5. Follow the source's post-attack check: classify the returned rewrite, or classify the original again when no adversarial text is returned. An unchanged-original label change is reported separately as `fallback_label_drift` and never counted as a GAN evasion.
6. Save the candidate, returned/used text, labels, errors, and times. There is no second GAN rewrite stage in this source's default SST path.

By default, GAN attempts to reuse baselines from the profile's LLM `output_dir` (`outputs/paper3000_adaptive`). Reuse requires identical dataset bytes, defender route/model/revision, and classification criteria. Existing target successes are preserved; missing baselines are classified normally. A mismatch fails before new requests, rather than mixing results.

```bash
python scripts/gan_pipeline.py --config decision_models/adaptive_main.json --model pplx --baseline-from outputs/my_matching_llm_run --pilot
# Explicitly start fresh if no compatible baseline should be reused:
python scripts/gan_pipeline.py --config decision_models/adaptive_main.json --model pplx --no-reuse-baseline --pilot
```

The GAN output directory must differ from its baseline source. Source results are read under a lock and copied with artifact provenance; the source run is not modified. Imported baseline timings retain the original measurements. `baseline_reused` reports how many were imported, so timing/request counts are not mistaken for new network calls.

## Checkpoints and output

```text
outputs/paper3000_gan/<model>/
  run.json
  checkpoint.json
  events.csv
  responses/
  gan_rewrites.csv
  rewrites/gan/<sample_id>_<hash>.csv
  sample_results.csv
  metrics.csv
  metrics.json
  evaluation/                  created by evaluation
```

Stages are `baseline`, `gan_probe`, `gan_attack`, and `gan_classify`. Every completed query and attack result is committed separately. Each attempted original gets its own current CSV snapshot; `events.csv` retains all attempts. CSV columns distinguish `candidate_text`, `returned_text`, `used_text`, `is_modified`, probe/final labels, statuses, and times.

If interrupted after the probe but before the attacker return is committed, generation is replayed deterministically with the recorded per-input seed; the identical candidate's saved classification is reused. If the returned rewrite is already saved, generation is skipped. Classification errors remain unresolved until `--retry-errors` is supplied. Changing the seed, weights, source implementation, or PyTorch/NumPy versions rejects attack resume; compatible baselines can still be reused in a new output folder.

`gan_search_elapsed_sec` includes local generation and any internal probe time. It overlaps `probe_time_sec` and must not be added to it as separate work. Baseline, probe, and post-attack classification times are also reported independently. Timings are client-observed request durations, not isolated GPU compute time.

## Evaluation

```bash
python scripts/evaluate_paper.py --run-dir outputs/paper3000_gan --model all --lexical-only
# Full local SBERT evaluation after installing requirements-evaluation.txt:
python scripts/evaluate_paper.py --run-dir outputs/paper3000_gan --model all
```

The evaluator detects GAN runs automatically and requires no defender credentials. It reports baseline metrics, eligible pool, returned modifications, rejected candidates, confirmed GAN flips, ASR, robust accuracy, unresolved bounds, and timing. A confirmed success requires a nonempty changed text returned by GAN and a valid NOT_SPAM post-attack classification. Incomplete runs have no final ASR; missing results are not counted as failures or successes.

Similarity uses the same SBERT, BLEU, ROUGE-L, Jaccard, length-ratio, and keyword-retention definitions as LLM evaluation. Two populations are kept separate:

- `gan_candidate`: every recorded proposed candidate, including rejected candidates; outcome uses the internal probe label.
- `gan_returned`: only returned candidates; outcome uses the post-attack classification.

Empty candidates retain an explicit `EMPTY_TEXT` status and no SBERT score. Fallback originals are not presented as rewritten texts with artificially perfect similarity. Returned-only similarity is selection-biased and should not be interpreted as similarity of all generated candidates. Semantic validity remains unassessed by a human.

## Source fidelity and limits

The minimal GAN algorithm and SGAN neural-network definitions are vendored from the supplied folder under `scripts/openattack_gan/`, with the MIT license and source/modified hashes in `provenance.json`. The default SST generation logic is retained. Changes replace package-wide imports with the direct-call interfaces and load state dictionaries on CPU with `weights_only=True`. The full old experiment scripts and embedded credentials are not copied or executed.

The source default truncates/pads to 100 source tokens, lowercases text, decodes through an SST-trained autoencoder, and cuts output at its first period/end token. It can produce short, content-degraded outputs. This is an important property of the GAN baseline being evaluated, not evidence of a valid meaning-preserving paraphrase. SNLI, SCPN, GAN training, and a multi-candidate search budget are outside this default source path.

The new runner makes replay seeds explicit, preserves failures, uses the recovered true-positive spam pool, returns the correct decision-label array interface, and logs rejected proposals. These fixes support reproducibility but do not establish exact historical numerical equivalence. The archive's original GAN label-mapping uncertainty remains; new results have an explicit mapping.
