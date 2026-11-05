# CEAS 2008 data lineage audit — 31 October 2026

## Main conclusion

**The uploaded `CEAS_08.csv` is conclusively identified as the source of the spam email text used in the available GAN and LLM attack-result CSVs.** Every one of the **16,887 result rows** has a corresponding spam email in CEAS, allowing reasonable normalization of whitespace/encoding and **26 near-match cases** resolved by manual comparison with the source. This is not a claim of byte-identical text in every row.

## Source file contents

- CEAS_08.csv: **39,154 source records**; **21,842** spam (`label=1`) and **17,312** legitimate (`label=0`).
- The original attack text is built as `Subject: {subject}` followed by two line breaks and `{body}`; attack CSVs typically do not contain sender, receiver, date, or source IDs.
- Source record numbers are **1-based data records excluding the CSV header**. The CEAS CSV has multiline bodies, so the physical line number differs from the record number.

## Matching results, 16 attack CSVs

| Match type | Result rows | Interpretation |
|---|---:|---|
| Exact string | 3,810 | Character-for-character equal to constructed CEAS subject+body |
| Replacement character normalization | 30 | Only Unicode replacement marks/line endings |
| Whitespace normalized | 12,813 | Spacing/newlines differ |
| Punctuation/encoding normalized | 140 | Curly quotes, mojibake, replacement glyphs |
| Body-only match | 68 | Body matches, subject differs/was corrupted (lower confidence) |
| Manually inspected near match | 26 | Four unique emails with encoding loss or `nan` subject |
| **Total** | **16,887** | All linked to CEAS `label=1` rows |

A total of **166** results have multiple candidate CEAS source rows with the same matching text; these establish source-corpus membership but **do not establish a unique original record ID**. The first source row is selected deterministically in the crosswalk.

### Manual exceptions (four unique original texts repeated across 26 result rows)

1. **CEAS record 2105**: empty subject became `Subject: nan`; Chinese-language text was replaced with question marks in the attack result. Strong contextual matching, but substantial text loss.
2. **CEAS record 28070**: a hiring-related email, nearly identical apart from punctuation mojibake.
3. **CEAS record 32845**: preserved body and subject but two bullets became question marks.
4. **CEAS record 17287**: preserved body and subject but two bullets became question marks.

## Manuscript issue: the reported 1,500-spam subset cannot be inferred

Across the eight **LLM** result files, there are **1,787 distinct original-email text strings** after trimming/collapsing whitespace and ignoring case, **more than the 1,500 spam emails claimed for one fixed evaluation dataset**. This means the attack logs cannot all originate from a *single shared* 1,500-spam set unless there is an undocumented change in preprocessing, selection, or experimental populations. Different victim-model attack pools appear to have been drawn from partially different original-email sets. The original sampling manifest or baseline full-dataset logs are needed to explain this.

The 16 attack CSVs store **only spam examples selected for attacks**. They do **not** let us identify which 1,500 legitimate emails (if any) were used for baseline classification, nor verify the complete 3,000-email experimental dataset.

## Files in this folder

- `COMPLETE_result_to_CEAS_row_crosswalk.csv`: one row for every original input in every GAN/LLM result CSV, with source record ID, match type, label, text hash, and exception note.
- `EVERY_CEAS08_source_row_usage.csv`: one row for each of the original 39,154 CEAS records indicating whether it was linked to an attack result.
- `FINAL_match_summary_by_file.csv`: match method and counts by attack CSV.
- `GAN_LLM_source_overlap.csv`: overlap within each victim model (with first-match ID limitations).
- `unmatched_for_manual_check.csv`: initial four unresolved unique texts, now resolved using source evidence.
- `trace_ceas_source.py`: reproducible initial automated comparison script.

**Do not rewrite the original experiments' text to match CEAS.** Link the observed result texts to source IDs and disclose the encoding differences.
