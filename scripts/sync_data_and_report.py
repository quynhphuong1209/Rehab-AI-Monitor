import argparse
import os
from datetime import datetime
from pathlib import Path


GENERATED_DIR = Path("docs") / "generated"
REPORT_PATH = GENERATED_DIR / "sync_report.md"
ALLOWED_OUTPUT_ROOT = GENERATED_DIR.resolve()


def configured_dataset_id() -> str | None:
    value = os.environ.get("HF_DATASET_ID", "").strip()
    return value or None


def generate_report(dataset_id: str | None = None, synced: bool = False) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_status = "configured" if dataset_id else "not configured"
    sync_status = "completed" if synced else "not run"
    return f"""# Sanitized Sync Report

Generated: {generated_at}

This generated report intentionally excludes patient identifiers, emails, clinical notes, credentials, tokens, and login links.

## Status

- Dataset: {dataset_status}
- Sync: {sync_status}

## Notes

- README is not overwritten by this script.
- Review any generated report for PII before committing it.
- Configure `HF_DATASET_ID` and `HF_TOKEN` outside the repository when sync functionality is needed.
"""


def write_generated_report(content: str, output_path: Path = REPORT_PATH) -> Path:
    resolved_parent = output_path.parent.resolve()
    if output_path.resolve() != REPORT_PATH.resolve() and ALLOWED_OUTPUT_ROOT not in [resolved_parent, *resolved_parent.parents]:
        raise ValueError(f"Output must stay under {GENERATED_DIR}/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a sanitized sync/report note without modifying README."
    )
    parser.add_argument(
        "--output",
        default=str(REPORT_PATH),
        help="Generated report path. Default: docs/generated/sync_report.md",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Reserved for future sanitized sync. Current script does not download runtime data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print where the sanitized report would be written without creating files.",
    )
    args = parser.parse_args()

    dataset_id = configured_dataset_id()
    if args.sync:
        print("Sync is disabled in this sanitized Phase 1A script; no runtime data was downloaded.")
    report = generate_report(dataset_id=dataset_id, synced=False)
    output_path = Path(args.output)
    if args.dry_run:
        print(f"Dry run: sanitized report would be written to {output_path}")
        return 0
    out = write_generated_report(report, output_path)
    print(f"Wrote sanitized report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
