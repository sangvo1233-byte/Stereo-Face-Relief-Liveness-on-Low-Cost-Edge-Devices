# Repository guidance

Read `README.md`, `docs/architecture.md`, `docs/reproduction.md`, then `docs/development.md`.

- Maintain this as the research companion for the cited ICERA stereo face-relief paper.
- Canonical software is `core/`, `app/`, `web/`, and the documented scripts. Research prototypes are indexed in `research/stereo_matching/README.md`.
- The experimental stereo route and the single-camera attendance route are separate. Do not describe stereo as the deployed attendance decision gate.
- Keep numeric evidence under `results/pilot/` immutable. New measurements get a new run directory and provenance record.
- Keep `datasets/`, `models/`, `database/`, `logs/`, `.env`, credentials, identifiable face media, and real embeddings out of Git.
- Do not equate pilot non-acceptance with general PAD effectiveness, host timing with Pi timing, or configured container limits with observed enforcement.
- Paper metadata and citation are under `paper/`; do not rewrite the accepted publication when documenting later software work.
- Work on a task branch, stage explicit task files, validate the relevant commands, and deliver through a reviewable PR. Preserve unrelated changes; do not rewrite history or deploy without explicit scope.
- Verification commands and runtime boundaries are in `docs/development.md`. A successful static check does not establish camera or Pi performance.
- Historical source provenance is recorded in `docs/reference/source-manifest.json`.
