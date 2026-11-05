# Proposed decision-model experiment on the recovered 3,000 inputs

Status: source review and experimental design only. Classification pipelines and active benchmark defaults have not been changed; no inference was started.

## Controlling input and objective

Freeze the supplied GPT-OSS baseline's 3,000 original input rows and true labels (1,500 spam and 1,500 ham), using `baseline_3000_manifest.csv` and its original-file SHA256. Preserve recorded text and row IDs, including the three repeated normalized texts. Report both row-level metrics and duplicate-aware sensitivity analysis if making statistical comparisons. The chosen cohort is a recovered historical subset selected through a previous model's successful-response process, not a new random sample of all CEAS.

Evaluate each new defender's clean classification, then its resistance to at most two meaning-preserving rewrite candidates on its own true-positive spam pool. Reuse historical Step-1 texts where source identity and generation compatibility are established. Generate a fresh refinement after the new defender rejects Step 1. This is a historical Step-1 plus adaptive refinement design; disclose reused/new generation separately rather than calling it an exact fresh rerun.

## Findings from the supplied source folder

Source root: `C:/KERJA DANI/vscode/OPEN ATTACK/OpenAttack-Share`.

- `a.ipynb` code cell 18 and `b.ipynb` code cell 2 increment their per-label quota only after obtaining parseable model predictions. Errors/unparseable responses are skipped. Work is parallel and results are appended in completion order. A fixed random seed therefore does not force identical exported 3,000-row sets across models. This supplies a concrete mechanism consistent with the previously observed cohort differences; individual historical run behavior still depends on the notebook version actually executed.
- `LLMattLLM.py`, lines 30-73 and 387-503, provides the neutral-tone Step-1/Step-2 prompts and conditional two-attempt loop. Step 2 includes original text, Step-1 text, and feedback that Step 1 remained spam. `LLMATTACKSILAB.py` uses equivalent prompt text for the Azure-defender route.
- `LLMATTACKSILAB2.py` uses different prompt wording. `LLMattLLM2.py` also overrides the attacker at its entry point. Defaults differ across scripts (for example attacker temperature 0 versus 0.3, and output limits 1,024 versus 700). Do not assume every historical CSV was produced with the same attacker revision, prompt, or settings merely because it has matching columns or a similar name.
- The old attack loop checks the defender's original prediction, but assumes its input file was already filtered to spam. Feeding all 3,000 rows directly to it could attack false-positive ham. The new controller must explicitly require ground truth SPAM AND a successful baseline prediction SPAM.
- An original input skipped as NOT SPAM is written into the old first-result column with no rewrite. Errors also produce empty stage fields. Treat these as distinct states, not successful rewrites or ordinary attack failures.
- Some legacy source files embed credentials. Do not execute those scripts as-is or copy those defaults into the new pipeline. Use environment-based credentials for any authorized endpoint; no credential values are reproduced here.

## Six-model archive availability audit

`rewrite_availability.csv` reports availability for all 1,500 spam rows in the frozen baseline, independent of future new-defender predictions:

| Availability                                                         | Baseline spam rows |
| -------------------------------------------------------------------- | -----------------: |
| At least one nonempty Step-1 candidate with a full-text source match |              1,408 |
| Only source links requiring review                                   |                 29 |
| No nonempty Step-1 candidate in the six-model archive                |                 63 |

This audits the six primary archived defenders, not every newly discovered pilot file. A nonempty candidate is not necessarily semantically valid or generation-compatible. The 1,408 is candidate coverage, not automatic approval to reuse. Required new generation depends on the new defender's true-positive pool and the candidate/provenance review. Existing conditional Step-2 chains cover 1,258 spam rows under the same source-matching rules, but are historical transfer evidence until compatible reuse is established.

## Execution order

1. Freeze a separate 3,000-row experiment manifest with original row ID, text hash, true label, source candidates, and duplicate group. Do not overwrite the current 2,226-input transfer benchmark or append to its prediction files.
2. Classify all 3,000 rows once per defender. Reuse a previous prediction only for exactly the same input, endpoint/inference contract, and model revision. Save failures for retry without replacement sampling. Record predicted label, optional probabilities, latency, raw response, model digest, and status.
3. Build the eligible pool independently for each defender: true label SPAM, valid baseline result SPAM. Ham and baseline false negatives do not enter the evasion denominator.
4. Select at most one historical Step-1 candidate per eligible original using a predeclared rule independent of historical/new attack success. Require source identity and record source file/row, attacker/prompt provenance, and generation settings. Keep the same selected initial candidate across new defenders wherever the protocol allows. Generate once and cache for absent/incompatible candidates using the frozen Step-1 prompt and attacker. Log refusals, blank outputs, and truncation as distinct failures, not fabricated rewrites.
5. Classify Step 1 using the same defender/contract as baseline. If NOT_SPAM, record a label flip and stop for this sample. If SPAM, proceed. If error/unparseable, retry or record incomplete; do not interpret an error as SPAM or consume a new candidate silently.
6. For Step 2, send the original, selected Step-1 text, and actual new-defender SPAM feedback to the frozen attacker contract. Classify the result with that same defender and stop regardless of the label. This is at most two rewrite candidates total.
7. Save separate baseline, Step-1, and Step-2 results, raw evidence, checkpoints, and metrics. Restart at the first unfinished stage; do not regenerate completed outputs. Bind caches to input, selected Step-1 text where relevant, prompt version/hash, model/checkpoint, and generation settings.

## Possible Step-2 cache optimization

A historical Step-2 text is not automatically a response to the new defender. A cache hit can be compatible with a label-only adaptive protocol only if the complete attacker request is demonstrably identical: original text, selected first rewrite, SPAM feedback, prompt, attacker revision, and settings. If victim identity is absent from the request, sharing such an already-generated response can avoid computation. Predeclare the cache policy, handle stochastic sampling consistently, and distinguish logical attack attempts from physical API calls. The supplied CSVs alone do not prove that identity; leave unverified historical Step 2 in a separate transfer analysis. Default to fresh conditional refinement until provenance is established.

## Decision-model integration

Reuse the existing structured classification adapters, not the legacy generative defender parser. The local experimental configuration has PPLX `server:pplxdeciderflash:latest` and Clef `clef:27b` at `http://127.0.0.1:11435/v1/systemone`. The main runner also supports d3 through the configured Open WebUI/native routes. These are code configurations, not a fresh verification of live inference availability.

Choose and freeze one inference contract per model/run. Record System One instructions and criteria because they differ in interface and wording from the old generative classifier prompt. Do not mix routes/revisions within one result file. The generative attacker remains a separate Gemma-3:27B chat endpoint; decision classifiers do not generate rewrites. Its live endpoint, revision, and prompt compatibility still need verification before generation.

Implement a separate stage controller and analysis path: existing runners consume a static input list and do not yet provide this adaptive lifecycle. Extend existing checkpoints/ETA to stage-level state. Keep all local experimental outputs under a new directory such as `experimenting/outputs/decision3000/<run>/<defender>/` so ongoing old experiments remain isolated.

Start with an explicit, fixed pilot containing clean ham, clean spam, and eligible spam that exercises both rewrite branches where observable. Validate structured output, stage eligibility, error handling, lineage, and resume. Then extend the same immutable configuration to all 3,000 inputs. A failed pilot does not authorize changing the dataset to easier emails.

## Reporting

- Clean accuracy, macro-F1, spam recall, and false-positive rate, with missing/error counts and coverage.
- True-positive pool size per defender; distinct original identities and duplicate counts.
- Step-1 flips; additional Step-2 flips among Step-1 failures; cumulative flips, counting an original once. Do not add both successes for the same original.
- Eligible samples lacking completed stages, refused/blank/truncated generations, and classification errors. Do not silently remove these from denominators; provide completed-pair rates separately from full-pool coverage/bounds.
- Reused versus newly generated candidates, attacker/defender revisions, logical two-candidate budget, physical requests including retries, and latency by operation.
- Independent checks that successful rewrites preserve spam intent. Until validated, call outcomes observed label flips rather than validated ASR. Similarity alone is not sufficient validation.

For a complete run with no retry/errors, defender request count is 3,000 + T + F, where T is the new defender's eligible spam count and F is the number remaining SPAM after Step 1. New generation is needed for missing/incompatible Step-1 candidates and uncacheable conditional Step-2 refinements; it need not repeat all historical generation.
