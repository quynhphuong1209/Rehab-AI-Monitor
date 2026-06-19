# Production Gate Checklist

This checklist is the deploy gate for the React/backend migration. Treat a failed
item as a release blocker unless the owner records an explicit exception.

## Automated Gates

- [ ] GitHub Actions `CI` passes on the exact commit being deployed.
- [ ] Python compile passes: `python -m compileall backend auth cloud models storage video utils scripts tests`.
- [ ] Backend unit tests pass: `python -m pytest tests/unit`.
- [ ] JSON to SQLite migration dry-run passes: `python scripts/migrate_json_to_sqlite.py --repo-root . --dry-run`.
- [ ] Frontend typecheck passes: `cd web && npm run lint`.
- [ ] Frontend production build passes: `cd web && npm run build`.
- [ ] Playwright smoke passes: `cd web && npm run e2e:smoke`.

## Config Gates

- [ ] `REHAB_BACKEND_CORS_ORIGINS` is set to the production frontend origin only.
- [ ] `HF_TOKEN` is set only in server/deploy secrets, never in frontend env or JSON.
- [ ] `HF_DATASET_ID` or `REHAB_HF_DATASET_ID` is set explicitly for the deployment.
- [ ] Runtime data directories are outside versioned source or mounted as private storage.
- [ ] `database/users.json` in production is not copied from demo/default credentials.
- [ ] Backups are written outside public web/static paths.

## Data Migration And Rollback

- [ ] Run migration dry-run and save the JSON report.
- [ ] Run migration apply only after a fresh backup snapshot exists.
- [ ] Verify SQLite row counts against the dry-run report.
- [ ] Restore drill completed with `scripts/migrate_json_to_sqlite.py --rollback` on a staging copy.
- [ ] Rollback owner and backup retention window are recorded for the release.

## Privacy And Compliance

- [ ] Research export is pseudonymized and excludes patient identifiers.
- [ ] Feedback, audit, report, and HF sync responses do not expose tokens or password fields.
- [ ] Users have role-appropriate access only; patient and doctor scope tests pass.
- [ ] Consent/participation text shown in the app matches the approved protocol.
- [ ] Incident response contacts and token/session revoke procedure are known to operators.

## Release

- [ ] Release notes include migration status, known limitations, and verification result.
- [ ] Production deploy requires manual approval after automated gates.
- [ ] Smoke login for patient, doctor/NCV, and admin roles is completed after deploy.
- [ ] Monitor upload, analysis job, auth failure, and disk quota logs after deploy.
