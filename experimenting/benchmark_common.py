"""Local experiment helpers, adapted from the main benchmark.

This module is self-contained: experiment runs never import or mutate main scripts.
"""
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEMONE_QUESTION = {
    "spam": {
        "type": "choice",
        "instructions": "Classify this email as SPAM or NOT_SPAM based on its content.",
        "criteria": {
            "SPAM": "Unsolicited advertising, phishing, scams, fraudulent offers, or deceptive messages.",
            "NOT_SPAM": "Legitimate personal, business, or informational correspondence.",
        },
    }
}

COLUMNS = [
    "timestamp_utc", "defender", "backend_mode", "model_id", "input_id", "sample_id",
    "kind", "true_label", "source_attack_model", "predicted_label", "p_spam",
    "p_not_spam", "latency_ms", "status", "http_status", "error", "raw_sha256",
    "run_id", "dataset_sha256", "endpoint", "checkpoint_id", "tool_id",
    "raw_file", "raw_saved_sha256",
]

LABELS = {"SPAM", "NOT_SPAM"}

def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def canonical_label(s):
    if not isinstance(s, str):
        return None
    clean = s.strip().upper().replace(" ", "_").replace("-", "_")
    if clean in ("SPAM", "NOT_SPAM"):
        return clean
    # No prose/substring detection: wrong labels would corrupt a benchmark.
    return None

def normalize_probs(obj):
    if not isinstance(obj, dict):
        return "", ""
    norm = {canonical_label(str(k)): v for k, v in obj.items()}
    vals = []
    for lab in ("SPAM", "NOT_SPAM"):
        val = norm.get(lab)
        try:
            p = float(val)
        except (ValueError, TypeError):
            vals.append("")
            continue
        if not 0 <= p <= 1:
            vals.append("")
        else:
            vals.append(f"{p:.10g}")
    return tuple(vals)

def json_or_none(s):
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("```json") and s.endswith("```"):
        s = s[7:-3].strip()
    elif s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None

def structured_decisions(data, depth=0):
    """Find only *structured* label results; never parse natural-language claims.

    Handles Pipe JSON (model/label/probabilities) and native decision result,
    e.g. answers.spam.choice or nested tool result. Does not invent probabilities.
    """
    if depth > 12:
        return []
    data = json_or_none(data)
    if isinstance(data, list):
        found = []
        for x in data:
            found += structured_decisions(x, depth + 1)
        return found
    if not isinstance(data, dict):
        return []
    found = []
    # Avoid extracting the requested ground truth from input/body metadata.
    for key in ("predicted_label", "label", "choice"):
        label = canonical_label(data.get(key))
        if label:
            p = data.get("probabilities")
            if not isinstance(p, dict):
                p = data.get("probs")
            if not isinstance(p, dict):
                p = {}
            found.append((label, *normalize_probs(p)))
            break
    if not found:
        # Controlled nested keys, never blindly traverse the original prompt.
        for key in ("answers", "spam", "classification", "decision", "result", "output", "content", "data", "tool_result", "response"):
            if key in data:
                found += structured_decisions(data[key], depth + 1)
    return found

def choose_decision(tree):
    candidates = structured_decisions(tree)
    if not candidates:
        raise ValueError("No structured SPAM/NOT_SPAM decision found")
    labs = {x[0] for x in candidates}
    if len(labs) != 1:
        raise ValueError("Conflicting labels in structured response")
    # Prefer candidate that has probabilities if duplicates are present.
    candidates.sort(key=lambda x: int(bool(x[1])) + int(bool(x[2])), reverse=True)
    return candidates[0]

def request_json(url, key, payload, timeout, attempts):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = Request(url, method="POST", data=raw, headers=headers)
    start = time.perf_counter()
    for idx in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as resp:
                content = resp.read()
                code = resp.status
            obj = json.loads(content.decode("utf-8"))
            return obj, code, round((time.perf_counter() - start) * 1000, 2), hashlib.sha256(content).hexdigest()
        except HTTPError as exc:
            code = exc.code
            short = exc.read(500).decode("utf-8", errors="replace")
            if code not in (429, 500, 502, 503, 504) or idx == attempts - 1:
                raise RuntimeError(f"HTTP {code}: {short}") from exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            if idx == attempts - 1:
                raise RuntimeError(f"Network/timeout: {exc}") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(f"Invalid API JSON: {exc}") from exc
        time.sleep(min(2 ** idx, 12))
    raise RuntimeError("Unreachable retry state")

def load_dataset(path, kinds):
    # utf-8-sig deliberately strips the UTF-8 BOM in the supplied dataset.
    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"input_id", "sample_id", "kind", "true_label", "email_text"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"Dataset missing columns: {sorted(required - set(reader.fieldnames or []))}")
        selected = []
        seen = set()
        for row in reader:
            if row["kind"] not in kinds:
                continue
            if not row["email_text"].strip():
                raise SystemExit(f"Empty email_text for {row['input_id']}")
            if canonical_label(row["true_label"]) is None:
                raise SystemExit(f"Unknown true_label for {row['input_id']}")
            if row["input_id"] in seen:
                raise SystemExit(f"Repeated input_id: {row['input_id']}")
            seen.add(row["input_id"])
            selected.append(row)
    return selected

def existing_results(path, run_id):
    found = {}
    if not path.exists() or not path.stat().st_size:
        return found
    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames != COLUMNS:
            raise SystemExit("Existing output has an incompatible/legacy schema. Use a new --out file.")
        for row in reader:
            if row.get("run_id") != run_id:
                raise SystemExit("Output belongs to a different dataset/model/endpoint/checkpoint/configuration. Use a new --out file.")
            if None in row or any(v is None for v in row.values()):
                raise SystemExit("Existing output contains a truncated/malformed row; repair it before resuming.")
            found[row["input_id"]] = row
    return found

def save_checkpoint(path, output_path, run_id, identity, items, results, state, last_input=None):
    """Atomic progress snapshot; the fsynced prediction CSV remains authoritative."""
    successful, failed, pending = [], [], []
    for item in items:
        input_id = item["input_id"]
        result = results.get(input_id)
        if result and result.get("status") == "OK" and result.get("predicted_label") in LABELS:
            successful.append(input_id)
        elif result:
            failed.append(input_id)
        else:
            pending.append(input_id)
    snapshot = {
        "checkpoint_version": 1, "updated_utc": utc_now(), "state": state,
        "run_id": run_id, **identity, "output_csv": str(output_path.resolve()),
        "selected_count": len(items), "successful_count": len(successful),
        "failed_count": len(failed), "pending_count": len(pending),
        "successful_input_ids": successful, "failed_input_ids": failed,
        "pending_input_ids": pending, "last_saved_input_id": last_input,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(snapshot, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def progress(completed, total, elapsed, delay):
    remaining = total - completed
    eta = duration(remaining * (elapsed / completed + delay)) if completed else "estimating"
    wall_elapsed = elapsed + max(0, completed - 1) * delay
    return f"attempted={completed}/{total} elapsed={duration(wall_elapsed)} ETA={eta}"
