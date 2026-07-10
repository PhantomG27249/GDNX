# GDNX Repository Organization Design

**Date:** 2026-07-09

## Context

GDNX is already a Git repository with ten project commits on `kmd2-fable`. The
last pre-cleanup project commit is `e1b99c4`. The history
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

Before moving or staging any existing project file, generate and commit
`docs/repository-inventory-2026-07-09.json`. This immutable pre-cleanup inventory
must cover every workspace file outside `.git/` that is code, configuration,
documentation, experiment evidence, a model/data tensor, or an ignored runtime
artifact. Each record must include:

- repository-relative path
- byte size and SHA-256 checksum
- tracked, untracked, or ignored state
- classification: code, documentation, configuration, experiment evidence,
  non-regenerable LFS artifact, reproducible generated data, transient cache,
  local tooling state, or suspected secret
- intended final path or explicit local-only disposition

The inventory generator must not print secret contents. If a relevant file
cannot be classified, implementation stops before cleanup. After organization,
a corresponding final inventory is compared against the committed baseline so
every code and result record has an explicit surviving path and matching content
or an intentional, reviewed modification.

After freezing that inventory and before staging the reorganization:

1. Match deleted tracked paths to untracked files by Git blob hash.
2. Record exact matches as moves wherever Git's rename detection can express
   them.
3. Review every unmatched deletion individually.
4. Enumerate every tracked path containing a literal backslash at `e1b99c4` and
   resolve it explicitly. The complete current set is:
   `code\\kernels.py`, `code\\module.py`,
   `code\\training\\gdn3_upgrade.py`,
   `code\\training\\plot_loss_loglog.py`,
   `code\\training\\plot_training.py`,
   `code\\training\\train_gdn3_distill.py`,
   `code\\training\\verify_ruler.py`,
   `code\\training\\verify_trend.py`,
   `docs\\COMPACTION_MQAR_RESULTS.md`, and
   `docs\\RELEASE_NOTES.md`.
5. Remove a malformed index entry only after its content is hash-matched or
   semantically compared with the canonical `gdn3/`, `train/`, or `docs/` copy
   and that disposition is recorded in the inventory.
6. Preserve superseded documentation under `docs/history/` instead of replacing
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

An artifact is non-regenerable when it is required to reproduce a recorded
result and the repository does not contain a deterministic generation command,
inputs, and seed that recreate it. Such artifacts must use Git LFS. Reproducible
generated data must remain ignored and receive a manifest entry instead.

The complete pre-cleanup tensor inventory contains exactly three files, with
these fixed dispositions:

- `runs/kmd2_native_heal/final/gdn3_layers.pt` - Git LFS, milestone checkpoint
- `research/runs_fable/experiment_data/qk_probe_init.pt` - Git LFS,
  non-regenerable probe evidence
- `data/mix_v1/blocks.pt` - ignored reproducible generated dataset

Any additional tensor found by the immutable inventory is a blocking discrepancy
until the spec is updated with an explicit disposition. If Git LFS is
unavailable, implementation must stop before adding the two allowlisted binaries
to ordinary Git and report the blocker; it must not silently omit them.

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

The authoritative historical source is the complete pre-cleanup working-tree
`README.md`, including its uncommitted historical-warning header. Its Git blob ID
is `513bca270b4e06e1f2dbf0a72835c99befd8e4d8` and its SHA-256 is
`864dc13e83e1fe520be1015fe78cadd4e8212a0486460636456b4b2a179d4219`.
For reference, the README at original project HEAD `e1b99c4` has blob ID
`d4c3958f6aaa4f3aebf423a7fdebe87c322be0da`. The complete pre-cleanup working
README will move byte-for-byte to
`docs/history/phase-0-two-timescale-release.md`; the destination must reproduce
both recorded hashes before the root README is replaced. `HANDOFF.md` remains the detailed
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

Before repository mutations, record the complete branch-to-commit mapping and
the set of commits reachable from all local branches. Original project commit
`e1b99c4` and every commit reachable from the pre-cleanup refs must remain
reachable after cleanup. `e1b99c4` must remain an ancestor of final `HEAD`; no
existing branch may be deleted or moved by this work.

## Commit Strategy

Use focused commits so preservation decisions remain reviewable:

1. Record and review this design.
2. Generate and commit the immutable pre-cleanup file and ref inventory.
3. Finish research/result moves and remove only verified duplicate broken-path
   exports.
4. Add Git, packaging, and dataset-manifest metadata.
5. Add current documentation while preserving the historical README.
6. Add curated experiment evidence and LFS-managed result artifacts.
7. Apply only verification-driven path or import corrections, if required.

Each commit stages an explicit path list. Unrelated working-tree changes must not
be swept into a commit accidentally.

## Verification

Completion requires fresh evidence for all of the following:

1. `git status --short --branch --untracked-files=all` is clean except for files
   intentionally documented as local-only.
2. `git diff --cached` is empty after the final commit.
3. The pre-cleanup and final inventories account for every code and experiment
   result path, including ignored artifacts, by checksum and disposition.
4. Git rename detection shows the archival moves rather than unexplained loss
   where content is identical.
5. Every pre-cleanup tracked, untracked, or ignored experiment-evidence file is either
   committed at a documented path or explicitly classified as a generated local
   artifact.
6. `git lfs ls-files` lists both allowlisted result tensors, `.gitattributes`
   resolves them to the LFS filter, their committed objects are LFS pointers,
   `git lfs fsck` passes, and no result tensor was added as a normal Git blob.
7. The dataset checksum in `data/README.md` matches the local file.
8. `python -m build` produces an sdist and wheel. The wheel is installed into a
   newly created isolated virtual environment, where `gdn3`, `train`, and
   `data_pipeline` all import and their expected modules are present. The
   temporary environment and build output remain ignored.
9. The existing parity test is run when its CUDA/dependency prerequisites are
   available; an unavailable prerequisite is reported distinctly from a failure.
10. The archived historical README matches pre-cleanup blob
    `513bca270b4e06e1f2dbf0a72835c99befd8e4d8` and SHA-256
    `864dc13e83e1fe520be1015fe78cadd4e8212a0486460636456b4b2a179d4219`.
11. `git merge-base --is-ancestor e1b99c4 HEAD` succeeds, every pre-cleanup
    commit remains reachable, and the recorded pre-existing branch refs are
    unchanged.
12. README commands and paths resolve against the final tree.
13. A final size audit confirms no cache, secret, or unintended runtime artifact
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
