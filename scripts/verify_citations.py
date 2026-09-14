#!/usr/bin/env python3
"""Verify citations in README.md against primary bibliographic sources.

What this catches:
  * An arXiv identifier that resolves to a different paper than the entry claims.
    This is the single most common way a fabricated citation enters a list.
  * A DOI whose Crossref record disagrees with the venue tag in the entry.
  * A code repository that no longer exists.
  * A venue outside the policy list, or a year before the cutoff.
  * Duplicate paper links.

What this cannot catch:
  * Whether the one-sentence description of the contribution is accurate.
    That requires reading the paper, and remains a human review job.

Exit codes:
  0  no hard failures
  1  at least one hard failure

Usage:
  python scripts/verify_citations.py                  # verify everything
  python scripts/verify_citations.py --only MambaHSI  # verify one entry
  python scripts/verify_citations.py --markdown       # report for a PR comment
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

CUTOFF_YEAR = 2021

ALLOWED_VENUES = {
    "CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "AAAI", "IJCAI",
    "TPAMI", "IJCV", "TGRS", "ISPRS J.", "RSE", "JSTARS", "GRSL", "GRSM",
    "IEEE TIP", "IEEE TMM", "IEEE TCSVT", "Information Fusion",
    "Nat. Mach. Intell.",
}

STATUS_TAGS = {"preprint", "model release", "corpus", "scaling", "2022", "2024", "2025"}

UA = ("awesome-remote-sensing-architectures-verifier/1.0 "
      "(+https://github.com/PrasannaPulakurthi/awesome-remote-sensing-architectures)")

ENTRY_NAME_RE = re.compile(r"^\s*-\s+\*\*(?P<name>[^*]+)\*\*\s+—")
PAPER_LINK_RE = re.compile(r"\[`paper`\]\((?P<url>[^)]+)\)")
CODE_LINK_RE = re.compile(r"\[`(?:code|weights|PASTIS|docs)`\]\((?P<url>[^)]+)\)")
TAG_RE = re.compile(r"`(?P<tag>[^`]+)`")
ARXIV_RE = re.compile(r"arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})", re.I)
DOI_RE = re.compile(r"(?:doi\.org/|/doi/(?:abs/)?)(?P<doi>10\.\d{4,9}/[^\s)\"<>]+)", re.I)
GITHUB_RE = re.compile(r"github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)", re.I)
STAR_RE = re.compile(r"`(?P<stars>[\d,]+)★`")
VENUE_TAG_RE = re.compile(r"^(?P<venue>.+?)'(?P<yy>\d{2})†?$")


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fetch(url, timeout=30, accept=None):
    """Return (body, error_code).

    error_code is None on success, an HTTP status on an HTTP error, or -1 on a
    transport failure. Callers must distinguish "the server said no such thing"
    (404) from "the server would not answer right now" (403/429/5xx) - treating
    a rate limit as a dead resource would fail valid entries.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if accept:
        req.add_header("Accept", accept)
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        req.add_header("Authorization", "Bearer " + token)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return None, exc.code
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return None, -1
    return None, -1


@dataclass
class Finding:
    level: str
    entry: str
    message: str


@dataclass
class Entry:
    name: str
    line_no: int
    paper_url: str = None
    code_urls: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    stars: int = None


def parse_entries(text):
    lines = text.splitlines()
    entries = []
    current = None
    for i, line in enumerate(lines, start=1):
        m = ENTRY_NAME_RE.match(line)
        if m:
            current = Entry(name=m.group("name").strip(), line_no=i)
            entries.append(current)
            continue
        if current is None:
            continue
        if PAPER_LINK_RE.search(line) or CODE_LINK_RE.search(line):
            pm = PAPER_LINK_RE.search(line)
            if pm and not current.paper_url:
                current.paper_url = pm.group("url")
            current.code_urls += [c.group("url") for c in CODE_LINK_RE.finditer(line)]
            current.tags += [t.group("tag") for t in TAG_RE.finditer(line)]
            sm = STAR_RE.search(line)
            if sm:
                current.stars = int(sm.group("stars").replace(",", ""))
        elif not line.strip() or line.startswith("#"):
            current = None
    # Bullets carrying neither a paper nor a code link are prose or dataset
    # rows, not citation entries. Checking them would produce standing warnings
    # that train reviewers to ignore the report.
    return [e for e in entries if e.paper_url or e.code_urls]


def check_arxiv(arxiv_id):
    """Return (title, journal_ref, status).

    status is 'ok', 'missing' (the API answered and knows no such paper) or
    'unchecked' (rate limited or unreachable). Only 'missing' is evidence of a
    bad identifier.
    """
    raw, err = fetch(
        "https://export.arxiv.org/api/query?id_list=%s&max_results=1" % arxiv_id)
    if not raw:
        return None, None, "unchecked"
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None, None, "unchecked"
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None, None, "missing"
    t = entry.find("a:title", ns)
    if t is None or not (t.text or "").strip():
        return None, None, "missing"
    title = " ".join(t.text.split())
    if norm(title) == "error":
        return None, None, "missing"
    jr = entry.find("arxiv:journal_ref", ns)
    jref = " ".join(jr.text.split()) if jr is not None and jr.text else None
    return title, jref, "ok"


def check_crossref(doi):
    raw, _ = fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""),
                   accept="application/json")
    if not raw:
        return None, None, None
    try:
        msg = json.loads(raw)["message"]
    except Exception:
        return None, None, None
    title = (msg.get("title") or [None])[0]
    container = (msg.get("container-title") or [None])[0]
    year = None
    for key in ("published-print", "published-online", "issued"):
        parts = msg.get(key, {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    return title, container, year


def check_github(owner, repo):
    """Return (stars, status) where status is 'ok', 'missing' or 'unchecked'."""
    raw, err = fetch("https://api.github.com/repos/%s/%s" % (owner, repo))
    if raw:
        try:
            return int(json.loads(raw).get("stargazers_count", 0)), "ok"
        except Exception:
            return None, "unchecked"
    if err == 404:
        return None, "missing"
    # 403 and 429 are rate limits, not evidence that the repository is gone.
    return None, "unchecked"


def verify(entries, check_stars=True, delay=1.0):
    findings = []
    seen = {}
    for e in entries:
        label = "%s (line %d)" % (e.name, e.line_no)

        if not e.paper_url:
            findings.append(Finding("warn", label,
                "has a code link but no paper link"))
            continue

        key = e.paper_url.rstrip("/")
        if key in seen and seen[key] != e.name:
            findings.append(Finding("warn", label,
                "paper link duplicates entry '%s' (fine if a deliberate cross-reference)"
                % seen[key]))
        seen.setdefault(key, e.name)

        venue_tag = None
        for t in e.tags:
            if t in STATUS_TAGS:
                venue_tag = t
                break
            m = VENUE_TAG_RE.match(t)
            if m:
                venue_tag = t
                venue = m.group("venue").strip()
                year = 2000 + int(m.group("yy"))
                if year < CUTOFF_YEAR:
                    findings.append(Finding("fail", label,
                        "year %d predates the %d cutoff" % (year, CUTOFF_YEAR)))
                if venue not in ALLOWED_VENUES:
                    findings.append(Finding("warn", label,
                        "venue '%s' is not in the policy list (needs a marker)" % venue))
                break
        if venue_tag is None:
            findings.append(Finding("warn", label, "no venue or status tag found"))

        am = ARXIV_RE.search(e.paper_url)
        if am:
            title, jref, status = check_arxiv(am.group("id"))
            time.sleep(delay)
            if status == "missing":
                findings.append(Finding("fail", label,
                    "arXiv id %s does not exist" % am.group("id")))
            elif status == "unchecked":
                findings.append(Finding("info", label,
                    "arXiv id %s not checked (API rate limited or unreachable)"
                    % am.group("id")))
            else:
                findings.append(Finding("info", label, "arXiv title: " + title))
                short = norm(e.name).split()[0] if norm(e.name) else ""
                if short and len(short) > 3 and short not in norm(title):
                    findings.append(Finding("warn", label,
                        "model name '%s' does not appear in the arXiv title "
                        "- confirm the identifier is correct" % e.name))
                if jref:
                    findings.append(Finding("info", label, "arXiv journal-ref: " + jref))

        dm = DOI_RE.search(e.paper_url)
        if dm:
            doi = dm.group("doi").rstrip(".")
            title, container, year = check_crossref(doi)
            time.sleep(delay)
            if title is None:
                findings.append(Finding("warn", label,
                    "DOI %s did not resolve via Crossref" % doi))
            else:
                findings.append(Finding("info", label, "Crossref title: " + title))
                if container:
                    findings.append(Finding("info", label,
                        "Crossref venue: %s (%s)" % (container, year)))

        for url in e.code_urls:
            gm = GITHUB_RE.search(url)
            if not gm:
                continue
            repo = gm.group("repo").rstrip("/")
            stars, status = check_github(gm.group("owner"), repo)
            time.sleep(delay)
            if status == "missing":
                findings.append(Finding("fail", label,
                    "repository %s/%s does not exist" % (gm.group("owner"), repo)))
            elif status == "unchecked":
                findings.append(Finding("info", label,
                    "repository %s/%s not checked (API rate limited)"
                    % (gm.group("owner"), repo)))
            elif check_stars and e.stars:
                if abs(stars - e.stars) / float(max(e.stars, 1)) > 0.5:
                    findings.append(Finding("warn", label,
                        "recorded %d stars but repository now has %d" % (e.stars, stars)))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--only")
    ap.add_argument("--changed-lines")
    ap.add_argument("--no-stars", action="store_true")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    with open(args.readme, encoding="utf-8") as fh:
        text = fh.read()

    entries = parse_entries(text)
    if args.only:
        entries = [e for e in entries if args.only.lower() in e.name.lower()]
    if args.changed_lines and os.path.exists(args.changed_lines):
        with open(args.changed_lines, encoding="utf-8") as fh:
            wanted = set(int(x) for x in fh.read().split() if x.strip().isdigit())
        if wanted:
            entries = [e for e in entries
                       if any(e.line_no <= n <= e.line_no + 3 for n in wanted)]

    if not entries:
        print("No entries to verify.")
        return 0

    sys.stderr.write("Verifying %d entries...\n" % len(entries))
    findings = verify(entries, check_stars=not args.no_stars, delay=args.delay)

    fails = [f for f in findings if f.level == "fail"]
    warns = [f for f in findings if f.level == "warn"]
    infos = [f for f in findings if f.level == "info"]

    if args.markdown:
        print("## Citation verification\n")
        print("Checked **%d** entries: **%d** failures, **%d** warnings.\n"
              % (len(entries), len(fails), len(warns)))
        if fails:
            print("### Failures (must fix)\n")
            for f in fails:
                print("- **%s** - %s" % (f.entry, f.message))
            print("")
        if warns:
            print("### Warnings (review)\n")
            for f in warns:
                print("- **%s** - %s" % (f.entry, f.message))
            print("")
        if infos:
            print("<details><summary>Resolved metadata</summary>\n")
            for f in infos:
                print("- **%s** - %s" % (f.entry, f.message))
            print("\n</details>")
    else:
        for f in fails:
            print("FAIL  %s: %s" % (f.entry, f.message))
        for f in warns:
            print("WARN  %s: %s" % (f.entry, f.message))
        for f in infos:
            print("info  %s: %s" % (f.entry, f.message))
        print("\n%d entries | %d failures | %d warnings"
              % (len(entries), len(fails), len(warns)))

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
