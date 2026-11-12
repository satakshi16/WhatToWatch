"""Check built images for baked-in secrets (P2-01 acceptance, NFR-7).

    uv run python -m scripts.check_image_secrets [IMAGE ...] [--env-file .env]

For each image (default: the two `make dev` images):

1. no `.env` / `.env.*` file anywhere in the filesystem (`.env.example` is fine);
2. none of the secret values in your local `.env` appear in `docker history --no-trunc`,
   the image config env, or the files under `/app`;
3. `detect-secrets scan` of `/app` finds nothing beyond what `.secrets.baseline` already
   audits for the same file.

Exit 0 when clean, 1 with a list of findings otherwise. Needs Docker; run `make dev` (or
`docker compose build`) first.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

DEFAULT_IMAGES = ("whattowatch-app:dev", "whattowatch-ui:dev")
APP_ROOT = "app/"  # `/app` inside the container, as a tar member prefix
ROOT = Path(__file__).resolve().parents[1]

_SECRET_KEY_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|_KEY$|_KEYS$|API_KEY)", re.IGNORECASE)
_PLACEHOLDERS = frozenset({"change-me", "changeme", "whattowatch", "none", "null"})
MIN_SECRET_LEN = 8


def parse_env(text: str) -> dict[str, str]:
    """Minimal dotenv parser: `KEY=value  # comment`, optional quotes, `#` lines skipped."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        out[key.strip()] = value
    return out


def secret_values(env: Mapping[str, str]) -> dict[str, str]:
    """Secret-looking keys with real values, comma lists split (e.g. JWT key rotation).

    Returns value → key name, so findings name the key and never print the value.
    """
    found: dict[str, str] = {}
    for key, value in env.items():
        if not _SECRET_KEY_RE.search(key):
            continue
        for part in (p.strip() for p in value.split(",")):
            if (
                len(part) >= MIN_SECRET_LEN
                and part.lower() not in _PLACEHOLDERS
                and not part.startswith("<")
            ):
                found[part] = key
    return found


def find_leaks(text: str, secrets: Mapping[str, str], where: str) -> list[str]:
    return [
        f"{where}: contains the value of {key}" for value, key in secrets.items() if value in text
    ]


def is_env_file(member_name: str) -> bool:
    name = member_name.rstrip("/").rsplit("/", 1)[-1]
    return (name == ".env" or name.startswith(".env.")) and name != ".env.example"


def new_findings(scan: Mapping[str, object], baseline: Mapping[str, object]) -> list[str]:
    """detect-secrets results in `scan` that the repo baseline doesn't already cover."""
    known: set[tuple[str, str]] = set()
    for filename, items in _results(baseline).items():
        known |= {(filename, str(i.get("hashed_secret"))) for i in items}
    out = []
    for filename, items in _results(scan).items():
        for item in items:
            if (filename, str(item.get("hashed_secret"))) not in known:
                out.append(f"detect-secrets: {item.get('type')} in /app/{filename}")
    return out


def _results(report: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    results = report.get("results")
    if not isinstance(results, dict):
        return {}
    return {str(k).removeprefix("./"): list(v) for k, v in results.items()}


def _docker(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _export_app_tree(image: str, dest: Path) -> list[str]:
    """Extract `/app` of `image` into `dest`; return any env-file paths in the whole fs."""
    env_files: list[str] = []
    container = _docker("create", image).strip()
    try:
        # Stream the export (the app image is GBs with torch); keep only /app on disk.
        with subprocess.Popen(  # noqa: S603
            ["docker", "export", container],  # noqa: S607
            stdout=subprocess.PIPE,
        ) as proc:
            assert proc.stdout is not None  # noqa: S101 - stdout=PIPE
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    if is_env_file(member.name):
                        env_files.append("/" + member.name)
                    if member.name.startswith(APP_ROOT) and (member.isfile() or member.isdir()):
                        member.name = member.name.removeprefix(APP_ROOT) or "."
                        tar.extract(member, dest, filter="data")
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, ["docker", "export", container])
    finally:
        _docker("rm", "-f", container)
    return env_files


def _scan_files(root: Path, secrets: Mapping[str, str]) -> list[str]:
    leaks: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            text = path.read_bytes().decode("utf-8", errors="ignore")
            leaks += find_leaks(text, secrets, f"/app/{path.relative_to(root)}")
    return leaks


def _detect_secrets(root: Path) -> dict[str, object]:
    out = subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", "--all-files", "."],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    ).stdout
    report: dict[str, object] = json.loads(out)
    return report


def check_image(
    image: str, secrets: Mapping[str, str], baseline: Mapping[str, object]
) -> list[str]:
    findings = find_leaks(_docker("history", "--no-trunc", image), secrets, "docker history")
    findings += find_leaks(
        _docker("image", "inspect", "--format", "{{json .Config.Env}}", image),
        secrets,
        "image env",
    )
    with tempfile.TemporaryDirectory() as tmp:
        app_dir = Path(tmp) / "app"
        app_dir.mkdir()
        findings += [f"env file in image: {p}" for p in _export_app_tree(image, app_dir)]
        findings += _scan_files(app_dir, secrets)
        findings += new_findings(_detect_secrets(app_dir), baseline)
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.check_image_secrets")
    parser.add_argument("images", nargs="*", default=list(DEFAULT_IMAGES))
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--baseline", type=Path, default=ROOT / ".secrets.baseline")
    args = parser.parse_args(argv)

    env = parse_env(args.env_file.read_text(encoding="utf-8")) if args.env_file.is_file() else {}
    secrets = secret_values(env)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    print(f"Checking {len(secrets)} secret value(s) from {args.env_file.name} in: {args.images}")

    findings: list[str] = []
    for image in args.images:
        found = check_image(image, secrets, baseline)
        print(f"  {image}: {'clean' if not found else f'{len(found)} finding(s)'}")
        findings += [f"{image}: {f}" for f in found]
    _report(findings)
    return 1 if findings else 0


def _report(findings: Iterable[str]) -> None:
    for line in findings:
        print(f"  ✗ {line}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
