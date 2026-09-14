# Contributing

Thanks for helping keep this list useful. The goal is a **curated** list, not an
exhaustive one — every addition should earn its place.

## Inclusion criteria

A paper belongs here if it meets **all** of these:

1. **Published 2021 or later.** This list tracks the modern era of remote sensing
   deep learning. Older work is covered well elsewhere.
2. **Published at a listed venue.** See the venue table in the README. Preprints
   are accepted only when they are already widely adopted *and* ship weights or
   code; they are tagged `preprint`.
3. **Contributes an architecture.** This is a list about model design. A paper
   applying an existing architecture to a new region or crop type is good science
   but out of scope here.
4. **Remote sensing / Earth observation.** Satellite, aerial, UAV, SAR,
   hyperspectral, LiDAR.

**Code is strongly preferred.** Entries without a public implementation are
accepted only when the work is foundational to its sub-area, and are marked
`no code`.

## Entry format

Keep one line per entry, in this exact shape:

```
- **ModelName** — One sentence on the architectural idea, not the results.
  [`paper`](url) [`code`](url) `Venue'YY` `task-tag` `NNN★`
```

Rules for the sentence: say what is *structurally new*, not how well it scored.
"Replaces full attention with rotated varied-size windows" is useful.
"Achieves state-of-the-art on DOTA" is not.

## Before you open a PR

- [ ] Verify the venue and year from a primary source (the publisher page, the
      proceedings listing, or the arXiv comments field). Do not trust a repo
      README alone.
- [ ] Click the code link. Dead or empty repos do not count as having code.
- [ ] Check the entry is not already listed under a different sub-area.
- [ ] Put the entry in the correct architecture family, then the correct
      sub-area. If it genuinely spans two families, list it in the primary one
      and cross-reference from the other.
- [ ] Keep the list within a sub-area ordered by year, then alphabetically.

## Automated verification

Every pull request that touches `README.md` runs `scripts/verify_citations.py`,
which resolves the identifiers in your changed entries against primary sources
and posts the result as a comment.

You can run it yourself before opening the PR:

```bash
python scripts/verify_citations.py --only YourModelName
```

It reports three levels:

- **Failure** blocks the merge. An identifier resolved to a different paper, or a
  linked repository does not exist.
- **Warning** asks for review. An unusual venue, a large star drift, a duplicate.
- **Info** is resolved metadata, printed so a reviewer can eyeball the fetched
  title against the entry.

A check that could not run (an API rate limit, a service outage) is reported as
skipped, never as a failure. The tool never blocks a PR because it could not
reach a server.

It cannot check whether your one-sentence description of the contribution is
accurate. That is still a human reading the paper.

## Star counts

Star counts are indicative, refreshed periodically, and deliberately not
load-bearing — do not open a PR solely to bump a number.

## Corrections

Wrong venue, wrong year, dead link, mis-attributed contribution: please open an
issue or PR. Citation accuracy matters more here than coverage, and corrections
are the most valuable contribution you can make.
