#!/usr/bin/env python3
"""Safely update the two Flash llama.cpp decision containers to --batch-size 2048.

Creates replacement containers, preserves all originals under timestamped backup
names, and restores their prior running/stopped states. Never deletes anything.
Run as: sudo python3 update_decision_batch_2048.py
"""
import json
import os
import subprocess
import sys
import time
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

    updated = set_arg(cmd, "--batch-size", BATCH_SIZE)
    # llama.cpp requires the physical batch not to exceed the logical batch.
    if "--ubatch-size" in updated:
        idx = updated.index("--ubatch-size")
        ubatch = int(updated[idx + 1])
        if ubatch > int(BATCH_SIZE):
            updated[idx + 1] = BATCH_SIZE
            print(f"  {name}: physical batch clamped from {ubatch} to {BATCH_SIZE}")
    if updated == cmd:
        print(f"SKIP {name}: already configured with --batch-size 2048")
        return

    backup = f"{name}-before-b2048-{stamp}"
    candidate = f"{name}-b2048-candidate-{stamp}"
    if inspect(backup) or inspect(candidate):
        raise RuntimeError("Backup/candidate container name already exists. No changes made.")

    was_running = bool(info["State"].get("Running"))
    options = create_options(info, name)
    immutable_image_id = info["Image"]
    print(f"\n{name}: old state={'running' if was_running else 'stopped'}")
    print(f"  Previous batch: {cmd[cmd.index('--batch-size') + 1] if '--batch-size' in cmd else '(default)'}")
    print(f"  New batch: {BATCH_SIZE}; creating candidate (no deletions)")

    docker("create", "--name", candidate, *options, immutable_image_id, *updated)
    try:
        if was_running:
            docker("stop", "-t", "30", name)
        docker("rename", name, backup)
        docker("rename", candidate, name)
        if was_running:
            docker("start", name)
        print(f"  UPDATED {name} (old container preserved: {backup})")
        if was_running:
            # Only check basic runtime; inference readiness is tested separately.
            time.sleep(4)
            state = inspect(name)["State"]
            if not state.get("Running"):
                print("  WARNING: replacement exited; attempting rollback")
                failed_name = f"{name}-failed-b2048-{stamp}"
                docker("rename", name, failed_name)
                docker("rename", backup, name)
                docker("start", name)
                print(f"  RESTORED original {name}; failed candidate retained as {failed_name}")
            else:
                print("  Process running; still verify /health and a real decision request")
    except Exception:
        # Best-effort rollback, retaining every original/candidate container.
        print(f"  ERROR while activating {name}; attempting safe rollback")
        try:
            if inspect(backup):
                replacement = inspect(name)
                if replacement:
                    if replacement["State"].get("Running"):
                        docker("stop", "-t", "15", name)
                    failed = f"{name}-failed-b2048-{stamp}"
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Safe Flash decision-model batch-size update to 2048")
    print("Preserves all previous containers, images, mounts, and caches")
    print("Leaves 27B models unchanged")
    print("PPLX Flash uses a separate custom image and is NOT modified")
    for name in TARGETS:
        try:
            upgrade(name, stamp)
        except Exception as exc:
            print(f"ERROR on {name}: {exc}", file=sys.stderr)
            print("Stopping before processing remaining models.", file=sys.stderr)
            sys.exit(1)
    print("\nFinished. Verify ports 30008/30009 and GPU 1 memory.")
    print("PPLX Flash is unchanged: inspect its application-specific batch settings separately.")


if __name__ == "__main__":
    main()
