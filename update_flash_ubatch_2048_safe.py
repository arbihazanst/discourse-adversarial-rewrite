#!/usr/bin/env python3
"""Safely raise Flash physical batch size to 2048, ONE service per invocation.

Preserves all previous Docker containers, weights, and caches. Tests health,
System-One inference and GPU VRAM before accepting the new configuration.
Rolls back on failed checks without deleting new or original containers.
Run as root: python3 update_flash_ubatch_2048_safe.py clef
or: python3 update_flash_ubatch_2048_safe.py d3
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

TARGETS = [
    "decision-clef-flash",
    "decision-d3-flash",
]
EXPECTED = {
    "decision-clef-flash": ("clef-flash-q4", "Clef-Flash-Q4_K_M.gguf", "30008"),
    "decision-d3-flash": ("d3-flash-q4", "vllm-d3-flash-Q4_K_M.gguf", "30009"),
}
BATCH_SIZE = "2048"
UBATCH_SIZE = "2048"
MIN_FREE_MIB = 3072


def docker(*argv, check=True):
    p = subprocess.run(["docker", *argv], text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError(f"docker {' '.join(argv)} failed: {p.stderr.strip()}")
    return p


def inspect(name):
    result = docker("inspect", name, check=False)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def set_arg(cmd, key, value):
    cmd = list(cmd)
    if key in cmd:
        idx = cmd.index(key)
        if idx + 1 >= len(cmd):
            raise ValueError(f"Missing value after {key}")
        cmd[idx + 1] = str(value)
    else:
        cmd.extend([key, str(value)])
    return cmd


def create_options(info, original):
    host = info["HostConfig"]
    config = info["Config"]
    opts = []

    # Reconstruct mounts exactly, retaining host-cache paths and read-only flags.
    for mount in info.get("Mounts") or []:
        kind = mount.get("Type")
        target = mount.get("Destination")
        readonly = not mount.get("RW", True)
        if kind == "bind":
            source = mount.get("Source")
        elif kind == "volume":
            source = mount.get("Name")
        else:
            raise ValueError(f"Unsupported mount type {kind!r}; manual review required")
        if not source or not target:
            raise ValueError("Incomplete mount data; refusing to recreate")
        spec = f"type={kind},source={source},target={target}"
        if readonly:
            spec += ",readonly"
        opts += ["--mount", spec]

    # Preserve every published host port, including loopback-only bindings.
    for container_port, bindings in (host.get("PortBindings") or {}).items():
        for binding in bindings or []:
            ip = binding.get("HostIp") or "0.0.0.0"
            hport = binding.get("HostPort")
            if not hport:
                raise ValueError(f"Dynamic host-port mapping: {container_port}")
            opts += ["-p", f"{ip}:{hport}:{container_port}"]

    # Preserve configured NVIDIA GPU assignments, without granting extra GPUs.
    requests = host.get("DeviceRequests") or []
    if len(requests) > 1:
        raise ValueError("More than one device request; needs manual recreation")
    if requests:
        req = requests[0]
        ids = req.get("DeviceIDs") or []
        count = req.get("Count", 0)
        if ids:
            opts += ["--gpus", "device=" + ",".join(ids)]
        elif count == -1:
            opts += ["--gpus", "all"]
        elif isinstance(count, int) and count > 0:
            opts += ["--gpus", str(count)]
        else:
            raise ValueError("Unrecognized GPU DeviceRequests; refusing to guess")

    # Carry existing environment, user, working directory, and custom resources.
    for entry in config.get("Env") or []:
        opts += ["-e", entry]
    if config.get("User"):
        opts += ["--user", config["User"]]
    if config.get("WorkingDir"):
        opts += ["--workdir", config["WorkingDir"]]
    for key, value in (config.get("Labels") or {}).items():
        if value is not None:
            opts += ["--label", f"{key}={value}"]

    network = host.get("NetworkMode") or "default"
    if network not in ("bridge", "default"):
        opts += ["--network", network]
    ipc = host.get("IpcMode") or "private"
    if ipc not in ("private", ""):
        opts += ["--ipc", ipc]
    if host.get("Privileged"):
        raise ValueError("Privileged container; manual review required")
    if host.get("NetworkMode") == "container":
        raise ValueError("Shared container network; manual review required")

    shm = host.get("ShmSize")
    if shm and shm != 67108864:
        opts += ["--shm-size", str(shm)]
    policy = host.get("RestartPolicy") or {}
    if policy.get("Name") and policy["Name"] != "no":
        restart = policy["Name"]
        if restart == "on-failure" and policy.get("MaximumRetryCount"):
            restart += ":" + str(policy["MaximumRetryCount"])
        opts += ["--restart", restart]
    return opts


def gpu_free_mib():
    p = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError("Cannot read GPU 1 VRAM: " + p.stderr.strip())
    return int(p.stdout.strip().splitlines()[0])


def request_json(url, payload=None, timeout=20):
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def verify_service(port, alias, timeout_seconds=180):
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_seconds
    last_err = None
    while time.monotonic() < deadline:
        try:
            resp = request_json(health_url, timeout=4)
            if resp.get("status") in ("ok", "ready") or resp.get("status") is None:
                break
        except Exception as exc:
            last_err = str(exc)
        time.sleep(3)
    else:
        raise RuntimeError(f"Health endpoint not ready after {timeout_seconds}s: {last_err}")

    response = request_json(
        f"http://127.0.0.1:{port}/v1/systemone",
        {"model": alias, "state": "A customer was charged twice.",
         "questions": {"department": {"type": "choice",
            "instructions": "Choose the responsible department.",
            "criteria": {"billing": "Charges and refunds", "technical": "Technical issues", "sales": "Purchases"}}}},
        timeout=60)
    if not (isinstance(response.get("answers"), dict) and "department" in response["answers"]):
        raise RuntimeError(f"System-One inference did not return answers.department: {str(response)[:500]}")
    print(f"  API check succeeded on port {port}: " + str(response["answers"]["department"].get("choice")))


def upgrade(name, stamp):
    info = inspect(name)
    if info is None:
        print(f"SKIP {name}: no such container")
        return
    image = info["Config"].get("Image") or ""
    cmd = info["Config"].get("Cmd") or []
    if not isinstance(cmd, list):
        raise ValueError("Unexpected command format")

    # Existing Flash containers were created using an immutable sha256 image ID;
    # that is legitimate and must not be mistaken for a non-llama.cpp image.
    expected_alias, expected_file, expected_port = EXPECTED[name]
    for flag in ("--model", "--alias", "--batch-size", "--ubatch-size"):
        if flag not in cmd or cmd.index(flag) + 1 >= len(cmd):
            raise ValueError(f"Expected llama.cpp argument {flag} missing; no changes made")
    if cmd[cmd.index("--alias") + 1] != expected_alias:
        raise ValueError("Model alias mismatch; refusing to update")
    model = cmd[cmd.index("--model") + 1]
    if (not model or not model.startswith("/opt/sglang/hf-cache/")
            or os.path.basename(model) != expected_file
            or not os.path.isfile(model) or os.path.getsize(model) < 1000000000):
        raise ValueError(f"Missing or unexpected GGUF model path: {model!r}; no changes made")
    bindings = info.get("HostConfig", {}).get("PortBindings") or {}
    hosts = [(b.get("HostIp"), b.get("HostPort"))
             for port, vals in bindings.items() for b in (vals or [])]
    if not any(p == expected_port and ip in ("127.0.0.1", "localhost") for ip,p in hosts):
        raise ValueError(f"Expected loopback host port {expected_port} not found; no changes made")
    reqs = info.get("HostConfig", {}).get("DeviceRequests") or []
    if not any("1" in (r.get("DeviceIDs") or []) for r in reqs):
        raise ValueError("GPU 1 DeviceRequest not present; no changes made")
    print(f"Verified {name}: image={image}, model={model}")

    if cmd[cmd.index("--batch-size") + 1] != BATCH_SIZE:
        raise ValueError("Logical batch is not 2048; expected existing tested configuration")
    previous_ubatch = cmd[cmd.index("--ubatch-size") + 1]
    updated = set_arg(cmd, "--ubatch-size", UBATCH_SIZE)
    if updated == cmd:
        print(f"SKIP {name}: already has --ubatch-size 2048; no changes made")
        return
    free = gpu_free_mib()
    print(f"  GPU 1 memory before: {free} MiB free")
    if free < MIN_FREE_MIB:
        raise RuntimeError("Less than 3 GiB GPU 1 VRAM free; no changes made")

    backup = f"{name}-before-ub2048-{stamp}"
    candidate = f"{name}-ub2048-candidate-{stamp}"
    if inspect(backup) or inspect(candidate):
        raise RuntimeError("Backup/candidate container name already exists. No changes made.")

    was_running = bool(info["State"].get("Running"))
    options = create_options(info, name)
    immutable_image_id = info["Image"]
    print(f"\n{name}: old state={'running' if was_running else 'stopped'}")
    print(f"  Previous physical batch: {previous_ubatch}")
    print(f"  New physical batch: {UBATCH_SIZE}; creating candidate (no deletions)")

    docker("create", "--name", candidate, *options, immutable_image_id, *updated)
    activated = False
    try:
        if was_running:
            docker("stop", "-t", "30", name)
        docker("rename", name, backup)
        docker("rename", candidate, name)
        if was_running:
            docker("start", name)
        activated = True
        print(f"  Candidate started for {name}; testing health, inference and GPU memory...")
        if was_running:
            verify_service(expected_port, expected_alias)
            free_after = gpu_free_mib()
            print(f"  GPU 1 free after test: {free_after} MiB")
            if free_after < MIN_FREE_MIB:
                raise RuntimeError(f"GPU free VRAM {free_after} MiB is below {MIN_FREE_MIB} MiB safety threshold")
        print(f"  SUCCESS: {name} physical batch=2048. Original retained: {backup}")
    except Exception:
        # Best-effort rollback, retaining every original/candidate container.
        print(f"  ERROR while validating {name}; attempting safe rollback")
        try:
            if inspect(backup):
                replacement = inspect(name)
                if replacement:
                    if replacement["State"].get("Running"):
                        docker("stop", "-t", "15", name)
                    failed = f"{name}-failed-ub2048-{stamp}"
                    docker("rename", name, failed)
                docker("rename", backup, name)
                if was_running:
                    docker("start", name)
                print(f"  Original {name} restored; candidates retained")
        except Exception as rollback_error:
            print(f"  Automatic rollback requires attention: {rollback_error}")
        print(f"  Check: docker ps -a (backup={backup}, candidate={candidate})")
        raise


def main():
    parser = argparse.ArgumentParser(description="Safely increase one Flash model's physical batch to 2048, retaining original container.")
    parser.add_argument("model", choices=("clef", "d3"), help="Only ONE service is changed per invocation")
    args = parser.parse_args()
    target = "decision-clef-flash" if args.model == "clef" else "decision-d3-flash"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Safe Flash physical batch-size update to 2048")
    print("Preserves all previous containers, images, mounts and caches")
    print("PPLX Flash, EmbeddingGemma and 27B models are NOT modified")
    print("Changes one live model at a time, and tests it; temporary interruption to target API is expected")
    try:
        upgrade(target, stamp)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Inspect docker ps -a and model logs; no containers were deleted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
