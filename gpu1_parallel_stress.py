#!/usr/bin/env python3
"""GPU 1 parallel API stress test: PPLX Flash, Clef-Flash, D3-Flash, EmbeddingGemma2.

Run on the Ubuntu GPU server (all ports are accessed via localhost).
Requires Python 3.9+ and nvidia-smi; no third-party Python packages.
This script only sends inference requests and writes CSV/JSON results.
It never stops, edits, replaces, or deletes Docker containers or model files.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SERVICES = {
    "pplx": {"port": 30002, "kind": "decision", "default_model": "perplexity-ai/pplx-decider-flash-v1.2"},
    "clef": {"port": 30008, "kind": "decision", "default_model": "clef-flash-q4"},
    "d3": {"port": 30009, "kind": "decision", "default_model": "d3-flash-q4"},
    "embedding": {"port": 30003, "kind": "embedding", "default_model": None},
}

SAMPLES = [
    ("The customer was charged twice for their subscription. Please refund the duplicate invoice.", "billing"),
    ("The API returns HTTP 401 after we updated our credentials and rotated the authentication key.", "technical"),
    ("Our organization wants to purchase 200 new seats and needs an enterprise quotation.", "sales"),
    ("I see an unrecognized payment transaction on my credit card statement.", "billing"),
    ("The integration fails with a network timeout when calling the production endpoint.", "technical"),
    ("Can your sales team provide a discount for an annual subscription?", "sales"),
]


def iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def request_json(url, payload=None, timeout=20):
    raw = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST" if raw is not None else "GET")
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            status = resp.status
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return status, None, time.perf_counter() - start, "Non-JSON response: " + text[:240]
        return status, data, time.perf_counter() - start, ""
    except HTTPError as e:
        msg = e.read(2048).decode("utf-8", errors="replace")
        return e.code, None, time.perf_counter() - start, msg[:400]
    except (URLError, TimeoutError, OSError) as e:
        return 0, None, time.perf_counter() - start, str(e)[:400]


def gpu_snapshot(gpu_id):
    cmd = ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7, check=True)
    line = result.stdout.strip().splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 4:
        raise ValueError("Unexpected nvidia-smi output: " + line)

    def to_num(x):
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "time_utc": iso_now(), "used_mib": to_num(parts[0]), "free_mib": to_num(parts[1]),
        "gpu_util_pct": to_num(parts[2]), "temp_c": to_num(parts[3])
    }


def decision_payload(model, i):
    state, _expected = SAMPLES[i % len(SAMPLES)]
    return {
        "model": model,
        "state": state,
        "questions": {
            "department": {
                "type": "choice", "instructions": "Which department should handle this customer request?",
                "criteria": {
                    "billing": "Invoices, refunds, charges, and payment disputes",
                    "technical": "API issues, integration errors, and software problems",
                    "sales": "Purchases, quotations, upgrades, and subscriptions",
                },
            }
        },
    }


def embedding_payload(model, i):
    state, _ = SAMPLES[i % len(SAMPLES)]
    return {"model": model, "input": f"Customer support embedding sample {i % len(SAMPLES)}: {state}"}


def run_one(name, i, model, timeout):
    service = SERVICES[name]
    endpoint = "/v1/systemone" if service["kind"] == "decision" else "/v1/embeddings"
    payload = decision_payload(model, i) if service["kind"] == "decision" else embedding_payload(model, i)
    code, data, seconds, err = request_json(f"http://127.0.0.1:{service['port']}{endpoint}", payload, timeout)
    ok = code == 200 and isinstance(data, dict) and (
        (bool((data.get("answers") or {}).get("department", {}).get("choice")) if name != "embedding" else
         bool(data.get("data") and isinstance(data["data"][0].get("embedding"), list) and data["data"][0]["embedding"]))
    )
    if not ok and not err:
        err = (json.dumps(data, ensure_ascii=False) if data is not None else "Empty response")[:400]
    return {"time_utc": iso_now(), "service": name, "model": model, "port": service["port"], "sequence": i,
            "http_status": code, "ok": int(ok), "latency_ms": round(seconds * 1000, 2), "error": err if not ok else ""}


def get_model_id(port, timeout):
    code, data, _, _ = request_json(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
    if code == 200 and isinstance(data, dict) and data.get("data"):
        return data["data"][0].get("id")
    return None


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    loc = (len(vals) - 1) * p / 100
    lo = int(loc)
    hi = min(len(vals) - 1, lo + 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (loc - lo), 1)


def print_stats(phase, rows, elapsed):
    print(f"\nPhase {phase} | elapsed={elapsed:.1f}s | total={len(rows)} | successes={sum(r['ok'] for r in rows)}/{len(rows)}")
    print(f"{'Service':<12}{'OK/Total':<14}{'p50 ms':>11}{'p95 ms':>11}{'RPS':>10}")
    for name in SERVICES:
        rs = [r for r in rows if r["service"] == name]
        if not rs:
            continue
        latencies = [r["latency_ms"] for r in rs if r["ok"]]
        print(f"{name:<12}{str(sum(r['ok'] for r in rs))+'/'+str(len(rs)):<14}{str(pct(latencies, 50)):>11}{str(pct(latencies, 95)):>11}{len(rs)/max(elapsed,.001):>10.2f}")
        failures = [r for r in rs if not r["ok"]]
        if failures:
            print(f"  First {name} failure: HTTP {failures[0]['http_status']} {failures[0]['error'][:250]}")


def main():
    ap = argparse.ArgumentParser(description="Staged four-service parallel GPU1 stress test; no changes to Docker or model files")
    ap.add_argument("--levels", default="1,2,3", help="Simultaneous requests PER service: default 1,2,3 (total 4,8,12)")
    ap.add_argument("--rounds", type=int, default=8, help="Parallel request waves at each level (default: 8)")
    ap.add_argument("--timeout", type=float, default=45, help="Seconds per inference request (default: 45)")
    ap.add_argument("--min-free-mib", type=float, default=3072, help="Abort further waves if GPU free VRAM drops below this (default: 3072)")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--pplx-model", default=os.environ.get("PPLX_MODEL", SERVICES["pplx"]["default_model"]))
    ap.add_argument("--clef-model", default="clef-flash-q4")
    ap.add_argument("--d3-model", default="d3-flash-q4")
    ap.add_argument("--embedding-model", default="", help="Optional exact ID; otherwise discover via /v1/models")
    ap.add_argument("--out", default="", help="Output directory (default: gpu1-stress-<UTC timestamp>)")
    args = ap.parse_args()

    try:
        levels = [int(v.strip()) for v in args.levels.split(",") if v.strip()]
    except ValueError:
        ap.error("--levels must be comma-separated integers")
    if not levels or any(x < 1 or x > 8 for x in levels) or args.rounds < 1:
        ap.error("levels must be between 1 and 8; rounds must be >= 1")

    output = Path(args.out or ("gpu1-stress-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    if output.exists():
        print(f"ERROR: output path already exists; will not overwrite: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    print(f"Reports: {output.resolve()}")

    try:
        baseline = gpu_snapshot(args.gpu)
    except Exception as e:
        print(f"ERROR: nvidia-smi baseline failed: {e}", file=sys.stderr)
        return 2
    print(f"GPU {args.gpu} baseline: used={baseline['used_mib']} MiB; free={baseline['free_mib']} MiB")
    if baseline["free_mib"] is None or baseline["free_mib"] < args.min_free_mib:
        print("ABORT: below requested GPU free-memory safety threshold", file=sys.stderr)
        return 2

    models = {"pplx": args.pplx_model, "clef": args.clef_model, "d3": args.d3_model,
              "embedding": args.embedding_model or get_model_id(30003, args.timeout)}
    for name in ("clef", "d3"):
        actual = get_model_id(SERVICES[name]["port"], min(args.timeout, 10))
        if actual and actual != models[name]:
            print(f"INFO: using served model ID for {name}: {actual}")
            models[name] = actual
    if not models["embedding"]:
        print("ABORT: embedding model ID not returned from 30003/v1/models. Provide --embedding-model 'exact-id'", file=sys.stderr)
        return 2

    print("\nPreflight: validating one real inference from each API before stressing GPU 1...")
    for name in SERVICES:
        row = run_one(name, 0, models[name], args.timeout)
        print(f"  {name:<10} port={row['port']} model={models[name]} HTTP={row['http_status']} {row['latency_ms']}ms {'OK' if row['ok'] else 'FAIL'}")
        if not row["ok"]:
            print("  Reason:", row["error"])
            if name == "pplx":
                print("  Check custom PPLX API (/openapi.json) or provide --pplx-model. No stress requests sent.")
            else:
                print("  No stress requests sent; verify model ID, route and server logs.")
            (output / "preflight_failure.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            return 2

    print("\nPreflight passed. Starting staged parallel test across all four endpoints.")
    gpu_rows = [baseline]
    all_rows = []
    phases = []
    stop = threading.Event()
    monitor_issues = []

    def monitor():
        while not stop.wait(0.5):
            try:
                sample = gpu_snapshot(args.gpu)
                gpu_rows.append(sample)
                if sample["free_mib"] is not None and sample["free_mib"] < args.min_free_mib:
                    monitor_issues.append(f"GPU {args.gpu} free VRAM {sample['free_mib']} MiB < threshold {args.min_free_mib} MiB")
                    stop.set()
            except Exception as e:
                monitor_issues.append("GPU monitoring failed: " + str(e))
                stop.set()

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    try:
        for level in levels:
            if stop.is_set():
                break
            print(f"\n=== Level {level}: up to {4 * level} simultaneous requests (each model x {level}) ===", flush=True)
            phase_rows = []
            begin = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4 * level) as pool:
                for wave in range(args.rounds):
                    if stop.is_set():
                        break
                    try:
                        current = gpu_snapshot(args.gpu)
                        gpu_rows.append(current)
                        if current["free_mib"] is None or current["free_mib"] < args.min_free_mib:
                            monitor_issues.append("GPU free VRAM below threshold before next wave")
                            stop.set()
                            break
                    except Exception as e:
                        monitor_issues.append("GPU sample before wave failed: " + str(e))
                        stop.set()
                        break
                    futures = []
                    for name in SERVICES:
                        for k in range(level):
                            sequence = (wave * level + k) % len(SAMPLES)
                            futures.append(pool.submit(run_one, name, sequence, models[name], args.timeout))
                    wave_results = [f.result() for f in as_completed(futures)]
                    phase_rows.extend(wave_results)
                    all_rows.extend(wave_results)
                    errors = [r for r in wave_results if not r["ok"]]
                    print(f"  wave {wave+1}/{args.rounds}: {len(wave_results)-len(errors)}/{len(wave_results)} OK", flush=True)
                    if errors:
                        monitor_issues.append(f"Request error at level {level}, wave {wave+1}; stopping further load")
                        stop.set()
                        break
            elapsed = time.perf_counter() - begin
            print_stats(level, phase_rows, elapsed)
            phases.append({"level_per_service": level, "elapsed_s": round(elapsed, 2), "requests": len(phase_rows),
                           "successes": sum(r["ok"] for r in phase_rows), "aggregate_rps": round(len(phase_rows) / max(elapsed, .001), 2)})
    except KeyboardInterrupt:
        monitor_issues.append("Interrupted by user")
        stop.set()
        print("\nKeyboardInterrupt: stopping additional test work...")
    finally:
        stop.set()
        sampler.join(timeout=3)
        try:
            gpu_rows.append(gpu_snapshot(args.gpu))
        except Exception:
            pass

    with (output / "requests.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_utc", "service", "model", "port", "sequence", "http_status", "ok", "latency_ms", "error"])
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "gpu1_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_utc", "used_mib", "free_mib", "gpu_util_pct", "temp_c"])
        writer.writeheader()
        writer.writerows(gpu_rows)

    free = [r["free_mib"] for r in gpu_rows if r["free_mib"] is not None]
    usage = [r["used_mib"] for r in gpu_rows if r["used_mib"] is not None]
    utils = [r["gpu_util_pct"] for r in gpu_rows if r["gpu_util_pct"] is not None]
    summary = {"timestamp_utc": iso_now(), "gpu": args.gpu, "models": models, "baseline": baseline,
               "levels": levels, "rounds": args.rounds, "timeout_s": args.timeout, "min_free_mib": args.min_free_mib,
               "phase_results": phases, "gpu_peak_used_mib": max(usage) if usage else None,
               "gpu_min_free_mib": min(free) if free else None,
               "gpu_peak_util_pct": max(utils) if utils else None,
               "total_requests": len(all_rows), "successful_requests": sum(r["ok"] for r in all_rows), "warnings": monitor_issues}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    print(f"Requests: {summary['successful_requests']}/{summary['total_requests']} successful")
    print(f"GPU peak used: {summary['gpu_peak_used_mib']} MiB; minimum free: {summary['gpu_min_free_mib']} MiB; peak utilization: {summary['gpu_peak_util_pct']}%")
    if monitor_issues:
        print("TEST HALTED OR WARNINGS:", *monitor_issues, sep="\n  - ")
    print(f"Saved: {output / 'summary.json'}, {output / 'requests.csv'}, {output / 'gpu1_samples.csv'}")
    print("No containers or model files were modified or removed.")
    return 1 if monitor_issues else 0


if __name__ == "__main__":
    sys.exit(main())
