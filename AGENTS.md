# AGENTS.md

Instructions for AI coding agents working in this repository.

This repo is a **curated citation list**. There is no build, no test suite and no
application code. The product is the accuracy of the claims in `README.md`.
That changes what "done" means: a plausible-looking entry is worse than no entry.

## The one rule that matters

**Never write a citation you have not verified against a primary source.**

A wrong citation here propagates into other people's bibliographies and is very
hard to undo. Every other instruction in this file follows from this one.

### What counts as a primary source

- The publisher's page for the paper (IEEE Xplore, ScienceDirect, SpringerLink).
- The official proceedings listing (`openaccess.thecvf.com`, `proceedings.neurips.cc`,
  `proceedings.mlr.press`, `icml.cc`, `ecva.net`, `ijcai.org`, `aaai.org`).
- The arXiv abstract page, specifically its **Comments** or **Journal reference**
  field.

### What does NOT count

- **A project's own README.** Repos routinely announce acceptance that never
  happened, or state the wrong year. This has already produced one wrong entry
  in this list.
- **A search engine summary.** Useful for finding candidates, not for confirming
  facts. Entries resting only on this carry a `†` marker.
- **Another awesome-list.** Errors propagate between lists; do not launder them.
- **Your own recollection.** You may well know this literature. It is not
  evidence.

### Never construct an identifier

Do not guess an arXiv ID, DOI, or IEEE document number from a pattern, and do not
assume a proceedings URL resolves because similar ones do. Fetch it and read the
title back.

A real example from this repo's history: an agent guessed an arXiv ID for a paper
it expected to be about remote sensing. The ID resolved to *"Text-to-Song:
Towards Controllable Music Generation"*, an ACL paper on music synthesis. The
identifier looked entirely plausible. Always read the returned title.

### When you cannot verify

Mark it, do not guess:

- Venue unconfirmed → tag the entry `preprint`.
- Venue from a secondary source only → tag it with the venue plus `†`.
- Cannot confirm the paper exists at all → **do not add it**, and say so in
  your PR description.

Reporting "I could not verify these six" is a good outcome. Filling them in with
confident-looking guesses is a failure, even if most turn out correct.

## Scope rules

An entry belongs in this list only if **all** of these hold:

1. Published **2021 or later**.
2. At a venue in the policy table in `README.md`. Exceptions are marked `⚠` and
   justified by the work being canonical for its sub-area.
3. It contributes an **architecture**. Applying an existing model to a new region
   or crop is out of scope.
4. It is remote sensing / Earth observation.

**Code is strongly preferred.** Verify the repository actually resolves and
belongs to that paper. Do not record a star count you did not read off the page.

## Entry format

```
- **ModelName** — One sentence on the architectural idea, not the results.
  [`paper`](url) [`code`](url) `Venue'YY` `task-tag` `NNN★`
```

The sentence describes what is **structurally new**. "Replaces full attention with
rotated varied-size windows" is useful. "Achieves state-of-the-art on DOTA" is
not — benchmark numbers age badly and every paper claims them.

Ordering within a sub-area: by year, then alphabetically.

## Tasks that suit an agent

Well suited — mechanical, verifiable, low risk:

- Fixing broken or redirected links (verify the new target is the same paper).
- Normalising entry formatting to the template above.
- Finding duplicate entries listed under two sub-areas.
- Updating the task index when sections are added or renamed.
- Checking that every anchor link in the contents and task index resolves.
- Refreshing star counts **from fetched pages**.

Poorly suited — do not attempt autonomously:

- Deciding whether a paper is canonical enough to justify a venue exception.
- Writing the one-sentence contribution for a paper you have not read.
- Bulk-adding entries from another list without individually verifying each.
- Resolving a venue dispute between two conflicting sources without fetching a
  primary source.

## Commit and PR conventions

- **Do not add AI attribution trailers** of any kind — no `Co-Authored-By`, no
  tool signatures.
- Commit messages: imperative mood, explain *why* a change was made, not just
  what.
- In PR descriptions, state explicitly which sources you verified against and
  list anything you could **not** verify. A PR that hides its uncertainty is
  harder to review than one that admits it.

## Repository layout

```
README.md          The list itself. The deliverable.
CONTRIBUTING.md    Human-facing contribution rules.
AGENTS.md          This file.
.github/
  ISSUE_TEMPLATE/  Templates for additions and corrections.
  workflows/       Monthly link check (lychee).
```

There is nothing to build or run. To validate changes, check that Markdown
renders and that links resolve.
