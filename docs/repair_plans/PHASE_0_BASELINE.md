# Phase 0 Baseline - 2026-06-18

This file records sanitized containment findings before code repair. It intentionally omits tokens, passwords, patient names, emails, and clinical details.

## Public Exposure Gate

- Local Git remote exists: GitHub origin.
- Public exposure status: unknown from local workspace. Treat as public until the team confirms otherwise.
- Required operational follow-up: rotate Hugging Face tokens, reset exposed/default passwords, and review Git/Hugging Face history before re-enabling public demos.

## Backup

- Local backup created: `backups/pre_repair_20260618_105524/`
- Contents copied: `database/`, `debug_files/`
- Backup is local runtime material and must not be committed.

## Static Baseline Counts

Counts below are line counts from `rg` baseline commands, not excerpts.

| Check | Count |
| --- | ---: |
| Auth bypass terms | 40 |
| HF token/config terms | 86 |
| `unsafe_allow_html=True` | 139 |
| Bare `except:` in `app.py` | 61 |
| `except Exception: pass` in `app.py` | 0 |
| Upload/ZIP/server/CORS hot patterns | 5 |
| Sensitive data terms in runtime/debug JSON | 388 |

## Compile Baseline

Command:

```powershell
python -m py_compile app.py utils\reference_utils.py utils\pose_classifier_utils.py utils\checkpoint_utils.py scripts\sync_from_hf.py scripts\sync_data_and_report.py scripts\reset_data.py
```

Result:

- Pass.
- Existing warning: `scripts\sync_data_and_report.py` has an invalid escape sequence warning.

## Data And Ignore Notes

- `database/schedules.json` parsed as JSON `null`, not a list.
- `.gitignore` currently ignores `*.json` but whitelists multiple runtime JSON files under `database/`.
- `debug_files/*.json` and runtime database JSON contain sensitive-field patterns and must be sanitized or removed from tracking in Phase 1B.
