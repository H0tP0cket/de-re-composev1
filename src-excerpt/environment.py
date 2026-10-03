"""Isolated Docker worker environment with native W/R capture and restoration.

The container is the worker's whole runtime. mini-SWE-agent executes every
command through ``docker exec bash -lc``, so no shell state survives between
actions; the runtime that can accumulate is the container filesystem outside
the workspace and any background processes.

Two kinds of manifest are kept separate on purpose:

* **Fidelity manifests** describe every entry (type, mode, owner, size, mtime,
  link target, content hash) and are used to validate that an archive equals
  the live tree at capture and that a restore reproduced it exactly.
* **Loop manifests** are content hashes of regular files excluding derived
  interpreter caches; they drive mutation relevance and the detector's
  workspace-equality rule. Dependency directories (.venv, node_modules, .git)
  are included so dependency and work changes count.

Capture policy (declared, not assumed):

* W = tar stream of ``/workspace`` taken inside the container.
* R = ``docker diff`` of the container versus its image restricted to paths
  outside the workspace: added/changed entries are archived with content
  hashes, deleted paths are listed; plus the process table. Filesystem R is
  restorable; processes are not. Any process beyond the keep-alive makes the
  capture R_UNSUPPORTED.

Every capture returns evidence of what was and was not covered. A failed
capture raises; nothing here ever synthesizes a valid checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shlex
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from minisweagent.exceptions import Submitted

from ..contracts import RuntimeClass
from .tool_transport import (SUPERVISOR, TRANSPORT_GRACE_SECONDS, VERSION as EXECUTION_VERSION,
                             ReceiptError, decode_receipt)

WORKSPACE = "/workspace"
KEEPALIVE = ["sleep", "infinity"]
CONTAINER_OWNER_LABEL = "trajectory-retention.owner"
LOOP_EXCLUDED_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
LOOP_EXCLUDED_SUFFIXES = (".pyc",)
LOOP_MANIFEST_RULE = ("w-loop-manifest/v2: sha256 of regular files under /workspace excluding "
                      + ",".join(LOOP_EXCLUDED_DIRS) + " dirs and " + ",".join(LOOP_EXCLUDED_SUFFIXES)
                      + "; dependency/vcs directories included")
FIDELITY_MANIFEST_RULE = ("fidelity-manifest/v2: every entry: type, mode, uid, gid, size(regular), mtime(ns via PAX), link target, "
                          "sha256(regular); directory mtime is recorded but not compared (parents are touched by deletions)")
# Docker's --init binary is injected at /usr/sbin/docker-init on merged-/usr images and at /sbin/docker-init where
# /sbin is a real directory (e.g. Debian bullseye python:3.10.11); both are harness, not worker, state.
RUNTIME_IGNORED_PREFIXES = ("/proc", "/sys", "/dev", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf",
                            "/usr/sbin/docker-init", "/sbin/docker-init")
RUNTIME_RULE_VERSION = ("r-capture/v4: docker diff outside /workspace; excluded paths match complete path components; "
                        "/run included; A/C entries archived with fidelity manifest, D paths listed; "
                        "process identities recorded; only initial harness PIDs/start-times/executables exempt; "
                        "processes not restorable; mutable mounts compared to initial baseline, divergence unsupported")
_FIND_FORMAT = r"%y\t%m\t%U\t%G\t%s\t%T@\t%l\t%p\n"


class EnvironmentError_(RuntimeError):
    pass


class CaptureError(EnvironmentError_):
    pass


class RestoreError(EnvironmentError_):
    pass


@dataclass
class ExecResult:
    output: str
    returncode: int
    exception_info: str
    duration_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)
    raw_output: bytes | None = None
    status: str | None = None
    transport_stdout: bytes | None = None
    transport_stderr: bytes | None = None

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = "ok" if self.returncode == 0 else "command_failed"

    def as_output(self) -> dict[str, Any]:
        out = {"output": self.output, "returncode": self.returncode, "exception_info": self.exception_info}
        if self.extra:
            out["extra"] = dict(self.extra)
        return out


class ToolTransportFailure(EnvironmentError_):
    def __init__(self, result: ExecResult) -> None:
        super().__init__(result.exception_info or "Command execution could not be established")
        self.result = result


def _run(cmd: list[str], *, input_bytes: bytes | None = None, timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise EnvironmentError_(f"{shlex.join(cmd)} failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[-2000:]}")
    return proc


def _vanished_ok(proc: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    """Accept a scan whose only errors are paths that disappeared mid-scan (e.g. a killed command's temp dir).

    find exits 1 and xargs 123 in that case; every other failure still raises.
    """
    if proc.returncode == 0:
        return proc
    errors = [line for line in proc.stderr.decode(errors="replace").splitlines() if line.strip()]
    if proc.returncode in (1, 123) and errors and all("No such file or directory" in line for line in errors):
        return proc
    raise EnvironmentError_(f"{what} failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[-2000:]}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def loop_excluded(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in LOOP_EXCLUDED_DIRS for p in parts[:-1]):
        return True
    return rel.endswith(LOOP_EXCLUDED_SUFFIXES)


def manifest_digest(manifest: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


def loop_manifest_from_fidelity(fidelity: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {rel: e["sha256"] for rel, e in sorted(fidelity.items()) if e["type"] == "f" and not loop_excluded(rel)}


def _parse_find(output: str, root: str) -> dict[str, dict[str, Any]]:
    """Parse find -printf output into a fidelity manifest without hashes."""
    entries: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        parts = line.split("\t", 7)
        if len(parts) != 8:
            continue
        ftype, mode, uid, gid, size, mtime, link, path = parts
        rel = path[len(root):].lstrip("/") if root != "/" else path.lstrip("/")
        entries[rel] = {"type": ftype, "mode": int(mode, 8), "uid": int(uid), "gid": int(gid),
                        "size": int(size) if ftype == "f" else None, "mtime": int(Decimal(mtime)),
                        "mtime_ns": int(Decimal(mtime) * 1000000000),
                        "link": link if ftype == "l" else None, "sha256": None}
    return entries


def fidelity_manifest_from_tar(data: bytes, *, root_prefix: str = "", legacy_seconds: bool = False) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        for m in tar.getmembers():
            name = m.name[2:] if m.name.startswith("./") else m.name
            name = name.rstrip("/")
            if name in ("", "."):
                continue
            if root_prefix:
                name = "/" + name if not name.startswith("/") else name
            # Same letters as find's %y in the live manifest; device nodes and FIFOs made by a worker
            # (e.g. mknod in /tmp) are archived by tar and must compare equal, not as "o".
            ftype = ("f" if m.isreg() else "d" if m.isdir() else "l" if m.issym() else "b" if m.isblk()
                     else "c" if m.ischr() else "p" if m.isfifo() else "o")
            digest = None
            if m.isreg():
                f = tar.extractfile(m)
                digest = sha256_bytes(f.read()) if f else ""
            out[name] = {"type": ftype, "mode": m.mode & 0o7777, "uid": m.uid, "gid": m.gid,
                         "size": m.size if m.isreg() else None, "mtime": int(m.mtime),
                         "link": m.linkname if m.issym() else None, "sha256": digest}
            # PAX may omit mtime when seconds are integral. Its ordinary header
            # then defines exact ns too. Legacy audit is an explicit separate
            # read mode, never the default for new capture/restore validation.
            if not legacy_seconds:
                out[name]["mtime_ns"] = int(Decimal(m.pax_headers.get("mtime", str(m.mtime))) * 1000000000)
    return dict(sorted(out.items()))


def comparable(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fidelity manifest view used for equality: directory mtimes dropped."""
    out = {}
    for k, e in manifest.items():
        if isinstance(e, dict) and e.get("type") == "d":
            out[k] = {kk: vv for kk, vv in e.items() if kk not in {"mtime", "mtime_ns"}}
        else:
            out[k] = e
    return out


def manifest_diff(a: dict[str, Any], b: dict[str, Any], limit: int = 5) -> list[str]:
    a, b = comparable(a), comparable(b)
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)} != {b.get(k)}")
            if len(out) >= limit:
                break
    return out


def _native_argv(cmdline: str) -> list[str]:
    """argv without a user-mode emulator launcher: under binfmt emulation (amd64 image on an arm64 host) the kernel
    reports `/usr/bin/qemu-x86_64 /usr/bin/sleep sleep infinity` for the keepalive `sleep infinity`."""
    argv = cmdline.split()
    if argv and argv[0].rsplit("/", 1)[-1].startswith("qemu-") and len(argv) >= 2:
        return argv[2:]
    return argv


class DockerWorkerEnvironment:
    """mini-SWE-agent Environment protocol plus capture/restore.

    Worker commands retain bash -lc, merged stdout/stderr and submission
    semantics. A separate supervisor receipt distinguishes their outcome from
    Docker transport. Command timeout is enforced inside the container; the
    client gets a bounded additional grace for startup/receipt transport.
    """

    def __init__(self, *, image: str, timeout: float = 120.0, cwd: str = WORKSPACE,
                 env: dict[str, str] | None = None, network: str = "bridge", name_prefix: str = "tr-worker",
                 memory: str = "2g", cpus: str = "1", pids_limit: int = 512, executable: str = "docker",
                 user: str | None = None, cap_drop: tuple[str, ...] = ()) -> None:
        self.image = image
        self.timeout = timeout
        self.cwd = cwd
        self.env = dict(env or {})
        self.network = network
        self.executable = executable
        self.container_name = f"{name_prefix}-{uuid.uuid4().hex[:10]}"
        self._container_owner = uuid.uuid4().hex
        self.container_id: str | None = None
        self._launch_pending = False
        self.limits = {"memory": memory, "cpus": cpus, "pids_limit": pids_limit}
        # Optional benchmark-native worker identity/capabilities (e.g. ProgramBench `--user agent --cap-drop
        # SYS_PTRACE`). Unset keeps the image default user and Docker's default capability set, as before.
        self.user = user
        self.cap_drop = tuple(cap_drop)
        inspect = json.loads(_run([executable, "image", "inspect", image]).stdout)[0]
        self.image_id = inspect["Id"]
        self.image_repo_digests = inspect.get("RepoDigests") or []
        self.image_architecture = inspect.get("Architecture")
        self.image_user = inspect.get("Config", {}).get("User") or "0"
        cmd = [executable, "run", "-d", "--name", self.container_name,
               "--label", f"{CONTAINER_OWNER_LABEL}={self._container_owner}", "--init", "--network", network,
               "--memory", memory, "--cpus", cpus, "--pids-limit", str(pids_limit), "-w", cwd,
               *(["--user", user] if user else []), *[a for c in self.cap_drop for a in ("--cap-drop", c)],
               "--entrypoint", KEEPALIVE[0], self.image_id, *KEEPALIVE[1:]]
        try:
            # Docker may have created the container even when its client call
            # is interrupted before returning the ID. Keep ownership available
            # before launch so that exceptional cleanup can resolve that case.
            self._launch_pending = True
            self.container_id = _run(cmd).stdout.decode().strip()
            if not self.container_id:
                raise EnvironmentError_("Docker launch returned no container ID")
            self._launch_pending = False
            self._initialize_state()
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup_error:
                original.add_note(f"Container cleanup failed for {self.container_name}: {cleanup_error!r}")
            raise

    def _initialize_state(self) -> None:
        identity = _run([self.executable, "exec", self.container_id, "sh", "-c", "id -u; id -g"]).stdout.splitlines()
        self.worker_uid, self.worker_gid = map(int, identity)
        self.started_at = time.time()
        self._exec(f"mkdir -p {shlex.quote(self.cwd)} && chown {self.worker_uid}:{self.worker_gid} {shlex.quote(self.cwd)}")
        self._harness_processes = {
            self._process_identity(p) for p in self.process_table()
            if p["pid"] == 1 or (p.get("ppid") == 1 and _native_argv(p["cmdline"]) == KEEPALIVE)
        }
        self._initial_mount_state = self.mount_state()
        self.config = {"image": self.image, "image_id": self.image_id, "cwd": self.cwd, "timeout": self.timeout, "env": self.env,
                       "network": self.network, "limits": self.limits, "interpreter": ["bash", "-lc"]}

    # ---------------------------------------------------------------- basics
    def binding(self) -> dict[str, Any]:
        return {"container_id": self.container_id, "container_name": self.container_name, "image": self.image,
                "image_id": self.image_id, "image_repo_digests": self.image_repo_digests,
                "architecture": self.image_architecture, "network": self.network, "limits": self.limits,
                "image_user": self.image_user, "worker_uid": self.worker_uid, "worker_gid": self.worker_gid,
                "recording_user": "0:0; worker commands retain image default user",
                "cwd": self.cwd, "env": self.env, "interpreter": ["bash", "-lc"], "loop_manifest_rule": LOOP_MANIFEST_RULE,
                **({"container_user": self.user} if self.user else {}),
                **({"cap_drop": list(self.cap_drop)} if self.cap_drop else {}),
                "command_execution": {"version": EXECUTION_VERSION, "supervisor_sha256": sha256_bytes(SUPERVISOR.encode()),
                    "supervisor_interpreter": ["python3", "-I", "-B"],
                    "timeout_seconds": self.timeout, "transport_grace_seconds": TRANSPORT_GRACE_SECONDS,
                    "timeout_cleanup": "SIGKILL command process group; uncertain transport stops the run"},
                "fidelity_manifest_rule": FIDELITY_MANIFEST_RULE, "runtime_rule": RUNTIME_RULE_VERSION}

    def _exec(self, command: str, *, timeout: float | None = None, input_bytes: bytes | None = None,
              cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        # Instrumentation needs to copy the captured numeric ownership even
        # when the benchmark's actual worker is an unprivileged image USER.
        # execute_recorded intentionally retains the original worker identity.
        cmd = [self.executable, "exec", "-i", "--user", "0:0", "-w", cwd or self.cwd]
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [self.container_id, "bash", "-lc", command]
        return _run(cmd, input_bytes=input_bytes, timeout=timeout, check=check)

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        uname = self._exec("uname -srvm", check=False).stdout.decode().split()
        system = dict(zip(("system", "release", "version", "machine"), uname + [""] * 4))
        return {**self.config, **system, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": self.config,
                                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}"}}}

    # -------------------------------------------------------------- execution
    def execute(self, action: dict[str, Any], cwd: str = "", *, timeout: float | None = None) -> dict[str, Any]:
        result = self.execute_recorded(action, cwd=cwd, timeout=timeout)
        if result.status in {"transport_error", "transport_timeout"}:
            raise ToolTransportFailure(result)
        output = result.as_output()
        self._check_finished(output)
        return output

    def execute_recorded(self, action: dict[str, Any], cwd: str = "", *, timeout: float | None = None) -> ExecResult:
        command = action.get("command", "")
        effective_timeout = self.timeout if timeout is None else timeout
        if (not isinstance(command, str) or type(effective_timeout) not in (int, float)
                or not math.isfinite(effective_timeout) or effective_timeout <= 0):
            raise ValueError("Command must be text and its timeout finite and positive")
        cmd = [self.executable, "exec", "-w", cwd or self.cwd]
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        spec = {"command": command, "timeout_seconds": effective_timeout,
                "receipt_id": uuid.uuid4().hex}
        cmd += [self.container_id, "python3", "-I", "-B", "-c", SUPERVISOR, json.dumps(spec)]
        start = time.monotonic()
        stdout, stderr, code, receipt, raw = b"", b"", None, None, b""
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=spec["timeout_seconds"] + TRANSPORT_GRACE_SECONDS)
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            if stdout:
                receipt, raw = decode_receipt(stdout, spec)
            if code != 0 or receipt is None or receipt["status"] == "supervisor_error":
                raise EnvironmentError_(f"Docker command transport failed ({code}); command completion is unconfirmed")
            timed_out = receipt["status"] == "timeout"
            status = "timeout" if timed_out else "ok" if receipt["command_returncode"] == 0 else "command_failed"
            extra = {"execution_request": spec, "execution_receipt": receipt, "transport_returncode": code,
                     "execution_version": EXECUTION_VERSION}
            if timed_out:
                extra.update(exception_type="TimeoutExpired", exception="Command timeout; command process group killed")
            return ExecResult(raw.decode("utf-8", errors="replace"), -1 if timed_out else receipt["command_returncode"],
                extra.get("exception", ""), time.monotonic() - start, extra, raw_output=raw, status=status,
                transport_stdout=stdout, transport_stderr=stderr)
        except Exception as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            if timed_out:
                stdout, stderr = error.output or b"", error.stderr or b""
                try:
                    receipt, raw = decode_receipt(stdout, spec)
                except ReceiptError as partial:
                    raw = partial.raw_output
            elif isinstance(error, ReceiptError):
                raw = error.raw_output
            return ExecResult(raw.decode("utf-8", errors="replace"), -1, str(error), time.monotonic() - start,
                {"exception_type": type(error).__name__, "exception": str(error),
                 "execution_version": EXECUTION_VERSION, "execution_request": spec, "execution_receipt": receipt,
                 "transport_returncode": code, "command_receipt_received": receipt is not None,
                 "execution_eligible": False}, raw_output=raw,
                status="transport_timeout" if timed_out else "transport_error",
                transport_stdout=stdout, transport_stderr=stderr)

    @staticmethod
    def _check_finished(output: dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission, "extra": {"exit_status": "Submitted", "submission": submission}})

    # -------------------------------------------------------------- manifests
    def _fidelity(self, root: str, paths: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Fidelity manifest of a tree (or an explicit path list, non-recursive)."""
        if paths is None:
            find = f"find {shlex.quote(root)} -mindepth 1 -printf {shlex.quote(_FIND_FORMAT)}"
            proc = _vanished_ok(self._exec(find, timeout=600, check=False), "find")
        else:
            if not paths:
                return {}
            listing = ("\0".join(paths) + "\0").encode()
            proc = _vanished_ok(self._exec(f"xargs -0 -r -I{{}} find {{}} -maxdepth 0 -printf {shlex.quote(_FIND_FORMAT)}",
                                           input_bytes=listing, timeout=600, check=False), "find")
        entries = _parse_find(proc.stdout.decode("utf-8", errors="replace"), root)
        regular = [("/" + rel if root == "/" else f"{root}/{rel}") for rel, e in entries.items() if e["type"] == "f"]
        if regular:
            listing = ("\0".join(regular) + "\0").encode()
            proc = _vanished_ok(self._exec("xargs -0 -r sha256sum --", input_bytes=listing, timeout=900, check=False),
                                "sha256sum")
            for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
                digest, _, path = line.partition("  ")
                rel = path[len(root):].lstrip("/") if root != "/" else path.lstrip("/")
                if rel in entries:
                    entries[rel]["sha256"] = digest
            # A regular file that vanished between listing and hashing is gone, not unhashed.
            entries = {rel: e for rel, e in entries.items() if e["type"] != "f" or e["sha256"] is not None}
        return dict(sorted(entries.items()))

    def workspace_fidelity_manifest(self) -> dict[str, dict[str, Any]]:
        return self._fidelity(WORKSPACE)

    def workspace_manifest(self) -> dict[str, str]:
        """Loop manifest (content hashes, caches excluded)."""
        return loop_manifest_from_fidelity(self.workspace_fidelity_manifest())

    def runtime_diff(self) -> list[tuple[str, str]]:
        proc = _run([self.executable, "diff", self.container_id])
        entries = []
        for line in proc.stdout.decode().splitlines():
            kind, _, path = line.partition(" ")
            if path == WORKSPACE or path.startswith(WORKSPACE + "/"):
                continue
            if any(path == p or path.startswith(p + "/") for p in RUNTIME_IGNORED_PREFIXES):
                continue
            entries.append((kind, path))
        return sorted(entries, key=lambda e: e[1])

    def runtime_manifest(self) -> dict[str, Any]:
        """Content-level R manifest: fidelity entries for A/C paths plus the deleted list."""
        diff = self.runtime_diff()
        changed = [p for k, p in diff if k in ("A", "C")]
        deleted = [p for k, p in diff if k == "D"]
        return {"entries": self._fidelity("/", changed), "deleted": deleted,
                "mounts": self.mount_state()}

    def mount_state(self) -> dict[str, Any]:
        """Record mounted storage omitted by docker diff; never read devices.

        These mounts currently have no restore implementation. Comparing their
        contents to the container's own initial baseline prevents unsupported
        state being certified merely because docker diff did not report it.
        Kernel proc/sys interfaces are recorded as topology, not file stores.
        """
        proc = self._exec("python3 -c " + shlex.quote(_MOUNT_STATE_SCRIPT), check=False, timeout=120)
        if proc.returncode != 0:
            raise CaptureError(f"mounted state probe failed ({proc.returncode}): {proc.stderr[-300:]!r}")
        try:
            result = json.loads(proc.stdout)
            if not isinstance(result, dict) or not {"topology", "entries"} <= result.keys():
                raise ValueError("missing mounted state fields")
            return result
        except (ValueError, TypeError) as exc:
            raise CaptureError(f"mounted state unreadable: {exc}") from exc

    # ---------------------------------------------------------------- capture
    def capture_workspace(self) -> tuple[bytes, dict[str, dict[str, Any]]]:
        """Tar of /workspace plus its fidelity manifest, validated against the archive."""
        live = self.workspace_fidelity_manifest()
        proc = self._exec("cd /workspace && tar --format=pax --numeric-owner -cf - .", timeout=900)
        data = proc.stdout
        archived = fidelity_manifest_from_tar(data)
        if comparable(archived) != comparable(live):
            differences = manifest_diff(live, archived)
            error = CaptureError("workspace changed during capture or archive incomplete: " + "; ".join(differences))
            error.capture_evidence = {"stage": "workspace_archive_mismatch", "live_manifest": live,
                                      "archive_manifest": archived, "archive_bytes": data, "differences": differences}
            raise error
        return data, live

    def process_table(self) -> list[dict[str, Any]]:
        """All container processes except the probe's own exec subtree."""
        proc = self._exec("python3 -c " + shlex.quote(_PROCESS_TABLE_SCRIPT), check=False, timeout=30)
        if proc.returncode != 0:
            raise CaptureError(f"process table probe failed ({proc.returncode}): {proc.stderr[-300:]!r}")
        try:
            result = json.loads(proc.stdout)
            if not isinstance(result, list) or not result:
                raise ValueError("missing harness process identities")
            return result
        except ValueError as e:
            raise CaptureError(f"process table unreadable: {e}: {proc.stdout[:200]!r}")

    @staticmethod
    def _process_identity(process: dict[str, Any]) -> tuple:
        return (process["pid"], process["start_time_ticks"], process["executable"], process["cmdline"])

    def capture_runtime(self) -> dict[str, Any]:
        """Filesystem diff outside the workspace plus process table.

        Returns keys: entries (fidelity manifest of A/C paths), deleted, archive (bytes|None),
        processes, foreign_processes, runtime_class, coverage, rule. The archive is validated
        against the live manifest.
        """
        manifest = self.runtime_manifest()
        changed = sorted(manifest["entries"])
        archive = None
        if changed:
            listing = "\n".join("/" + p for p in changed) + "\n"
            proc = self._exec("tar --format=pax --numeric-owner --no-recursion -cf - -T /dev/stdin", input_bytes=listing.encode(), timeout=900)
            archive = proc.stdout
            archived = fidelity_manifest_from_tar(archive)
            archived = {k.lstrip("/"): v for k, v in archived.items()}
            if comparable(archived) != comparable(manifest["entries"]):
                differences = manifest_diff(manifest["entries"], archived)
                error = CaptureError("runtime archive mismatch: " + "; ".join(differences))
                error.capture_evidence = {"stage": "runtime_archive_mismatch", "live_manifest": manifest,
                                          "archive_manifest": archived, "archive_bytes": archive, "differences": differences}
                raise error
        processes = self.process_table()
        foreign = [p for p in processes if self._process_identity(p) not in self._harness_processes]
        mount_changed = manifest["mounts"] != self._initial_mount_state
        runtime_class = RuntimeClass.UNSUPPORTED if foreign or mount_changed else RuntimeClass.CAPTURED
        return {
            "rule": RUNTIME_RULE_VERSION,
            "entries": manifest["entries"], "changed": changed, "deleted": manifest["deleted"], "archive": archive,
            "processes": processes, "foreign_processes": foreign, "runtime_class": runtime_class.value,
            "mounts": manifest["mounts"], "initial_mounts": self._initial_mount_state,
            "mount_state_changed": mount_changed,
            "coverage": {"filesystem_outside_workspace": "captured_with_fidelity_manifest", "processes": "recorded_not_restorable",
                         "network_state": "not_captured", "shell_state": "none_by_construction",
                         "excluded_path_roots": list(RUNTIME_IGNORED_PREFIXES),
                         "mounted_storage": "baseline divergence detected, not restorable; divergence makes R_UNSUPPORTED",
                         "kernel_interfaces": "proc/sys topology recorded; live kernel state not restored",
                         "docker_diff_limit": "image-layer filesystem changes; mounts separately baseline-checked"},
        }

    # ---------------------------------------------------------------- restore
    def restore_workspace(self, data: bytes, expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self._exec("find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +", timeout=300)
        self._exec("cd /workspace && tar --numeric-owner -xpf -", input_bytes=data, timeout=900)
        live = self.workspace_fidelity_manifest()
        if comparable(live) != comparable(expected):
            raise RestoreError("workspace restore mismatch: " + "; ".join(manifest_diff(expected, live)))
        return {"validated": True, "rule": FIDELITY_MANIFEST_RULE, "fidelity_digest": manifest_digest(live), "entries": len(live)}

    def restore_runtime(self, runtime: dict[str, Any]) -> dict[str, Any]:
        if runtime.get("runtime_class") == RuntimeClass.UNSUPPORTED.value or runtime.get("mount_state_changed"):
            raise RestoreError("Runtime contains unsupported process or mounted state")
        if runtime.get("mounts") and runtime["mounts"] != runtime.get("initial_mounts"):
            raise RestoreError("Runtime mounted state differs from its recorded baseline")
        if runtime.get("archive"):
            self._exec("cd / && tar --numeric-owner -xpf -", input_bytes=runtime["archive"], timeout=900)
        for path in runtime.get("deleted", []):
            self._exec(f"rm -rf -- {shlex.quote(path)}", timeout=60)
        live = self.runtime_manifest()
        if live["mounts"] != self._initial_mount_state:
            raise RestoreError("Restoration changed unsupported mounted state")
        expected_entries = runtime.get("entries", {})
        if comparable(live["entries"]) != comparable(expected_entries) or sorted(live["deleted"]) != sorted(runtime.get("deleted", [])):
            raise RestoreError("runtime restore mismatch: " + "; ".join(manifest_diff(expected_entries, live["entries"]))
                               + f"; deleted {sorted(live['deleted'])} vs {sorted(runtime.get('deleted', []))}")
        return {"validated": True, "rule": RUNTIME_RULE_VERSION, "entries": len(expected_entries),
                "deleted": len(runtime.get("deleted", [])), "processes_restored": False}

    # ---------------------------------------------------------------- cleanup
    def close(self) -> None:
        container_id = getattr(self, "container_id", None)
        if not container_id and getattr(self, "_launch_pending", False):
            result = _run([self.executable, "container", "inspect", self.container_name], check=False, timeout=30)
            if result.returncode:
                if b"no such container" in result.stderr.lower() or b"no such object" in result.stderr.lower():
                    self._launch_pending = False
                    return
                raise EnvironmentError_(f"Could not resolve interrupted container launch: {result.stderr.decode(errors='replace')}")
            instance = json.loads(result.stdout)[0]
            labels = instance.get("Config", {}).get("Labels") or {}
            if labels.get(CONTAINER_OWNER_LABEL) != self._container_owner:
                raise EnvironmentError_(f"Refusing cleanup of container with different ownership: {self.container_name}")
            container_id = instance["Id"]
        if container_id:
            result = _run([self.executable, "rm", "-f", container_id], check=False, timeout=30)
            if result.returncode and not (b"no such container" in result.stderr.lower() or b"no such object" in result.stderr.lower()):
                raise EnvironmentError_(f"Container cleanup failed: {result.stderr.decode(errors='replace')}")
            self.container_id = None
            self._launch_pending = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_MOUNT_STATE_SCRIPT = r"""
import hashlib, json, os, re, stat
def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)
mounts = []
with open('/proc/self/mountinfo') as f:
    for line in f:
        left, right = line.rstrip('\n').split(' - ', 1)
        a, b = left.split(), right.split()
        mounts.append({'target': unescape(a[4]), 'fs_type': b[0],
                       'source': unescape(b[1]), 'options': sorted(a[5].split(',')),
                       'super_options': sorted(b[2].split(','))})
mounts.sort(key=lambda m: m['target'])
targets = {m['target'] for m in mounts}
entries = {}
def inspect(path, root):
    if path != root and path in targets:
        return  # another mount is inspected independently
    if len(entries) >= 100000:
        raise RuntimeError('mounted-state probe entry limit exceeded')
    s = os.lstat(path)
    mode = s.st_mode
    item = {'mode': mode, 'uid': s.st_uid, 'gid': s.st_gid}
    if stat.S_ISREG(mode):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
        item.update(size=s.st_size, sha256=h.hexdigest(), mtime_ns=s.st_mtime_ns)
    elif stat.S_ISLNK(mode):
        item['target'] = os.readlink(path)
    elif stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        item['device'] = s.st_rdev  # do not read device bytes
    entries[path] = item
    if stat.S_ISDIR(mode):
        for child in sorted(os.listdir(path)):
            inspect(os.path.join(path, child), root)
for m in mounts:
    path = m['target']
    if path == '/' or path == '/workspace' or path.startswith('/workspace/'):
        continue
    if path in ('/proc', '/sys') or path.startswith(('/proc/', '/sys/')):
        continue  # live kernel interfaces, not persistent file stores
    if 'rw' in m['options']:
        inspect(path, path)
print(json.dumps({'topology': mounts, 'entries': entries}, sort_keys=True))
"""


_PROCESS_TABLE_SCRIPT = r"""
import json, os
rows, ppid = {}, {}
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmd = f.read().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
        with open(f'/proc/{pid}/stat') as f:
            start_time_ticks = int(f.read().rpartition(')')[2].split()[19])
        try:
            executable = os.readlink(f'/proc/{pid}/exe')
        except PermissionError:
            # Root in an unprivileged container may lack ptrace permission
            # across UIDs. Preserve the process instead of dropping it from
            # the table; PID/start-time/command still bind the initial keeper.
            executable = None
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('PPid:'):
                    ppid[pid] = int(line.split()[1])
    except OSError:
        continue
    rows[pid] = {'cmdline': cmd, 'start_time_ticks': start_time_ticks, 'executable': executable}
# Exclude this probe and its exec'd shell. bash -lc may exec python directly, in which
# case the parent is outside the pid namespace (ppid 0) and must not be excluded.
excluded = {os.getpid()}
if os.getppid() not in (0, 1):
    excluded.add(os.getppid())
changed = True
while changed:
    changed = False
    for pid, parent in ppid.items():
        if parent in excluded and pid not in excluded:
            excluded.add(pid); changed = True
print(json.dumps([{'pid': p, 'ppid': ppid.get(p), **c} for p, c in sorted(rows.items()) if p not in excluded]))
"""
