#!/usr/bin/env python3
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OVERRIDES = ROOT / "docs" / "AUDIT_OVERRIDES.json"

VALID = {
    "VERIFIED",
    "VERIFIED_SYNTHETIC",
    "SUPPORTIVE_ONLY",
    "BOUNDARY_ONLY",
    "INVALIDATED",
    "FAILED_RUN",
    "PENDING",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def report_directories() -> list[Path]:
    directories = set()
    for path in REPORTS.rglob("*"):
        if path.is_file():
            relative = path.parent.relative_to(REPORTS)
            if relative.parts:
                directories.add(path.parent)
    return sorted(directories)


def main() -> None:
    override_doc = read_json(OVERRIDES) or {"overrides": {}}
    overrides = override_doc.get("overrides", {})
    entries = []
    for directory in report_directories():
        relative = directory.relative_to(REPORTS).as_posix()
        evidence_path = directory / "EVIDENCE_STATUS.json"
        evidence = read_json(evidence_path)
        metrics = sorted(directory.glob("*metrics*.json"))
        provenance = directory / "provenance.json"
        logs = sorted(directory.glob("*.log"))

        status = "PENDING"
        evidence_status = None
        failures: list[str] = []
        if evidence is not None:
            evidence_status = str(evidence.get("status", "UNKNOWN")).upper()
            failures = [str(item) for item in evidence.get("failures", [])]
            if evidence_status == "VERIFIED" and metrics:
                status = "VERIFIED"
            elif evidence_status in {"FAILED", "INVALID", "ERROR"}:
                status = "FAILED_RUN"
        elif metrics and provenance.exists():
            status = "SUPPORTIVE_ONLY"

        override = overrides.get(relative)
        if override:
            override_status = str(override.get("status", "PENDING")).upper()
            if override_status not in VALID:
                raise ValueError(f"invalid override status for {relative}: {override_status}")
            status = override_status

        entries.append(
            {
                "audit": relative,
                "status": status,
                "evidence_status": evidence_status,
                "evidence_sha256": sha256(evidence_path) if evidence_path.exists() else None,
                "metric_files": [
                    {"path": p.relative_to(ROOT).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
                    for p in metrics
                ],
                "provenance": provenance.relative_to(ROOT).as_posix() if provenance.exists() else None,
                "logs": [p.relative_to(ROOT).as_posix() for p in logs],
                "failures": failures,
                "override_reason": override.get("reason") if override else None,
            }
        )

    registry = {
        "schema": "se-oct-audit-registry-v1",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rule": "A commit or workflow trigger is not evidence. VERIFIED requires a verified evidence file and at least one metrics file; overrides can only downgrade.",
        "entries": entries,
    }
    (REPORTS / "AUDIT_REGISTRY.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    lines = [
        "# Audit registry",
        "",
        registry["rule"],
        "",
        "| Audit | Status | Evidence | Metrics | Boundary / reason |",
        "|---|---|---|---:|---|",
    ]
    for entry in entries:
        reason = entry["override_reason"] or ("; ".join(entry["failures"]) if entry["failures"] else "")
        lines.append(
            f"| `{entry['audit']}` | **{entry['status']}** | {entry['evidence_status'] or '—'} | "
            f"{len(entry['metric_files'])} | {reason.replace('|', '/')} |"
        )
    (REPORTS / "AUDIT_REGISTRY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
