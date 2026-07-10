# GDNX Repository Organization Design

**Date:** 2026-07-09

## Context

GDNX is already a Git repository with ten commits on `kmd2-fable`. The history
contains the evolution from the GDN3 two-timescale release through the KMD-2
research and heal work. That history is valuable and must not be rewritten.

The working tree is an unfinished preservation-oriented reorganization. It
contains active code changes, new KMD-2 code, archived research, and moved raw
experiment evidence. Most apparent deletions under `research/` have exact copies
in either `research/archive_autoresearch/` or
`research/runs_fable/experiment_data/`. The tracked phase-0 export also contains
Windows-hostile literal backslash paths such as `code\\kernels.py`; canonical
copies already exist under `gdn3/` and `train/`.

The repository has no remote, uses placeholder local Git author settings, and
lacks standard packaging and cross-platform text/LFS metadata. The current root
README describes an older release, while `HANDOFF.md` describes the current
project.

## Goals

1. Preserve the existing commit graph and all current code changes.
2. Preserve all experiment evidence, including logs, result data, plots,
   configuration, research notes, and non-regenerable tensor artifacts.
3. Make the repository understandable to a new contributor from the root.
4. Make the Python packages installable without forcing a disruptive `src/`
   migration.
5. Keep large derived training data out of Git while recording enough metadata
   to identify and reproduce it.
6. Leave a clean working tree whose staged history clearly records moves,
   metadata, documentation, and large-artifact handling.

## Non-goals

- Rewriting or squashing existing commits.
- Refactoring the GDN/KMD-2 architecture or changing experiment conclusions.
- Renaming every historical script or normalizing historical result formats.
- Publishing to a remote or selecting a software license without owner input.
- Committing caches, transient training output, or reproducible prepared data.

## Chosen Approach

Use a preservation-first cleanup. Keep the established top-level domains and
finish the reorganization already present in the working tree. This avoids the
path breakage and provenance loss of a full `src/` migration while still adding
the metadata expected of a usable Python research repository.

The alternatives were rejected for this pass:

- A full `src/gdnx/` refactor would be aesthetically cleaner but would invalidate
  historical commands and imports throughout the research archive.
- A minimal archival commit would preserve files quickly but would leave the
  repository difficult to install, navigate, and reproduce.

## Repository Boundaries

The target layout is:

```text
GDNX/
|-- gdn3/                         Core GDN3 and KMD-2 Python package
|-- train/                        Training, verification, and plotting tools
|-- data_pipeline/                Reproducible dataset construction package
|-- tests/                        Focused correctness and parity tests
|-- research/
|   |-- README.md                 Map of active and archived research
|   |-- EXPERIMENT_LEDGER.md      Canonical findings ledger
|   |-- KMD2_STATUS.md            Current KMD-2 decisions and status
|   |-- kernel_ab/                Kernel optimization study and evidence
|   |-- runs_fable/               Curated KMD-2 evaluation evidence
|   |   `-- experiment_data/      Historical raw result payloads and scripts
|   `-- archive_autoresearch/     Superseded auto-research loop, kept intact
|-- data/
|   `-- README.md                 Dataset identity and regeneration manifest
|-- runs/                         Local runtime outputs; allowlisted LFS results
|-- docs/
|   |-- history/                  Superseded release documentation
|   `-- superpowers/              Approved designs and implementation plans
|-- HANDOFF.md                    Detailed current-state handoff
|-- README.md                     Current project overview and quick start
|-- pyproject.toml                Packaging, dependencies, and test configuration
|-- .gitignore                    Cache/output/secret policy and LFS exceptions
|-- .gitattributes                Line endings, text classification, and Git LFS
`-- .editorconfig                 Basic cross-editor whitespace policy
```

Each top-level Python directory retains one clear responsibility. Historical
research remains executable in place when its original relative assumptions
permit it; archived material is not presented as supported package API.

## Preservation and Move Policy

All existing working-tree changes are treated as user-owned. Cleanup must not
reset, overwrite, or silently regenerate them.

Before staging the reorganization:

1. Match deleted tracked paths to untracked files by Git blob hash.
2. Record exact matches as moves wherever Git's rename detection can express
   them.
3. Review every unmatched deletion individually.
4. Remove the broken flattened `code\\...` index entries only after confirming
   that their canonical code remains under `gdn3/` or `train/`.
5. Preserve superseded documentation under `docs/history/` instead of replacing
   it in place.

Nothing classified as experiment evidence may be deleted merely because it is
old, negative, duplicated in a summary, or no longer part of the active path.

## Experiment Evidence Policy

Commit ordinary experiment evidence directly to Git:

- JSON, JSONL, YAML, and other configuration or result records
- logs and textual diagnostics
- plots and other compact visual summaries
- benchmark, probe, launch, and analysis scripts
- research notes, briefs, postmortems, ledgers, and status documents

The generic `runs/` directory remains ignored because it is a runtime output
location. Explicit, non-regenerable milestone artifacts are allowlisted rather
than broadly tracking all run output.

Git LFS will track non-regenerable binary experiment artifacts, including:

- `runs/kmd2_native_heal/final/gdn3_layers.pt`
- `research/runs_fable/experiment_data/qk_probe_init.pt`

The exact allowlist will be verified against the final artifact inventory before
staging. If Git LFS is unavailable, implementation must stop before adding these
binaries to ordinary Git and report the blocker; it must not silently omit them.

## Generated Dataset Policy

`data/mix_v1/blocks.pt` is derived training data and remains ignored. A committed
`data/README.md` will record:

- relative path and byte size
- SHA-256 checksum
- generating recipe and command
- source/recipe references in `data_pipeline/`
- a clear statement that the binary is intentionally not versioned

This preserves identity and reproducibility without adding a 160 MB rebuildable
artifact to repository history.

## Packaging and Environment Metadata

Add `pyproject.toml` using the existing `gdn3`, `train`, and `data_pipeline`
packages. Dependencies will be split by responsibility so that repository
installation does not unnecessarily force every plotting or GPU research tool:

- core model/runtime dependencies
- training and data-pipeline extras
- research/plotting extras
- test tooling

Version bounds must reflect imports and the verified local environment; they
must not invent narrow pins without evidence. Triton and GPU-specific behavior
will be documented explicitly because not every platform supports the optimized
kernel path.

Pytest configuration will point to `tests/` and use existing import boundaries.
Packaging changes may fix import paths that are demonstrably broken by the
reorganization, but must not alter model behavior.

## Documentation Design

The new root README will explain the current GDNX/KMD-2 state, repository map,
installation, artifact policy, high-value verification commands, and links to
the handoff and research ledgers.

The existing phase-0 README content will move unchanged in substance to
`docs/history/phase-0-two-timescale-release.md`. `HANDOFF.md` remains the detailed
continuation guide and will be updated only where paths or setup commands change.
`research/README.md` will distinguish active evidence from preserved archives.

Documentation must not claim that a test, checkpoint, or setup path works unless
it is verified during implementation.

## Git Hygiene

Add:

- `.gitignore` rules for Python caches, test/build output, local environments,
  editor state, secrets, transient runs, and generated datasets
- narrow negation rules for curated evidence and LFS-managed artifacts
- `.gitattributes` rules for LF-normalized source/docs, binary result formats,
  and the explicit Git LFS allowlist
- `.editorconfig` for UTF-8, final newlines, and consistent indentation

The placeholder local Git author override will be removed only after confirming
that the existing global identity is usable. Existing commits are not amended.
No remote will be configured in this work.

## Commit Strategy

Use focused commits so preservation decisions remain reviewable:

1. Record and review this design.
2. Finish research/result moves and remove only verified duplicate broken-path
   exports.
3. Add Git, packaging, and dataset-manifest metadata.
4. Add current documentation while preserving the historical README.
5. Add curated experiment evidence and LFS-managed result artifacts.
6. Apply only verification-driven path or import corrections, if required.

Each commit stages an explicit path list. Unrelated working-tree changes must not
be swept into a commit accidentally.

## Verification

Completion requires fresh evidence for all of the following:

1. `git status --short --branch --untracked-files=all` is clean except for files
   intentionally documented as local-only.
2. `git diff --cached` is empty after the final commit.
3. Git rename detection shows the archival moves rather than unexplained loss
   where content is identical.
4. Every pre-cleanup tracked or untracked experiment-evidence file is either
   committed at a documented path or explicitly classified as a generated local
   artifact.
5. `git lfs ls-files` lists every allowlisted result tensor, and no large tensor
   was added as a normal Git blob.
6. The dataset checksum in `data/README.md` matches the local file.
7. Package/import smoke checks pass in the verified environment.
8. The existing parity test is run when its CUDA/dependency prerequisites are
   available; an unavailable prerequisite is reported distinctly from a failure.
9. README commands and paths resolve against the final tree.
10. A final size audit confirms no cache, secret, or unintended runtime artifact
    is tracked.

## Failure Handling

- A missing Git LFS installation blocks binary staging, not the rest of the
  preservation audit.
- A failed or unavailable GPU test does not authorize code changes by itself;
  diagnose it and report the exact prerequisite or failure.
- Any ambiguous deletion is preserved in an archive location until reviewed.
- Any suspected secret is excluded immediately and reported by filename without
  printing its value.
- No destructive Git command, history rewrite, remote push, or cleanup of
  unrelated local files is part of this design.
