# Repository completion

Prepared on 2026-10-01. This is the current public-companion task list; older submission-readiness checklists and the June migration plan are historical.

## Prepared in the local review candidate

- [x] Identify the paper through IEEE-deposited DOI metadata and preserve its exact title and author order.
- [x] Add the paper landing page, BibTeX, and schema-valid `CITATION.cff`.
- [x] Select the existing local stereo implementation, research scripts, recorder, benchmark, and tests without moving the owner's files.
- [x] Publish architecture, commands, source ownership, and research/attendance boundaries in documentation.
- [x] Archive the two numeric pilot CSV/JSON sets and check their counts and metrics.
- [x] Record imported source/evidence hashes and distinguish a working-copy snapshot from a paper-run revision.
- [x] Exclude private biometric media, mutable runtime data, weights, secrets, and internal agent/session material.

These checks describe preparation, not merge or deployment completion.

## Required decisions and validation

| Priority | Item | Acceptance |
| --- | --- | --- |
| Done | Owner approves the proposed repository name and public PR contents | Renamed the same repository and published PR #1 after approval |
| P0 | Identify the exact accepted/camera-ready artifact | Owner-supplied artifact identity/hash; add a permitted manuscript link when available |
| P0 | Choose a license for owner-controlled code | License scope agreed by owners; paper/data/model rights kept separate |
| Done | Verify a fresh Python 3.11 application environment on Linux AMD64 | Installation with documented transitive constraints, `pip check`, 65 non-integration tests, hardware-free startup/API smoke |
| P1 | Verify the actual camera demo on the chosen machine | Two streams, missing/poor input handling, verdicts, and no unintended attendance recording |
| P1 | Preserve calibration and Pi trial evidence in shareable form | Numeric/configuration provenance audited and matched to the paper; no private subject media |
| P1 | Merge the reviewed companion and record a release revision | Reviewed commit, verified public links/citation, matching evidence hashes |

## Follow-on research

Print attacks, held-out multi-subject studies, calibrated dense/deep trials on Pi4, and sustained target-device stability are research extensions. They do not change the claims or evidence of the accepted publication and are not required to invent a stronger conclusion for this release.
