"""Render the multi-page audit site into ``site/``.

Design mirrors the Marathon Trainer dashboard -- same Inter typeface, blue
system, sidebar shell, card and table language -- so the two tools read as one
family. The visual layer lives in ``web/style.css`` and ``web/app.js`` as plain
files; this script only produces the HTML shells and the data bundle, then
copies those two through untouched. Tweaking the look means editing the CSS,
not this file.

The output is static and offline: data ships as ``site/static/data.js`` setting
``window.DOE`` rather than being fetched, because ``fetch`` against ``file://``
is blocked by browsers. So the site opens by double-clicking ``index.html`` --
no server, no build step, no network. ``python3 serve.py`` is available for
anyone who prefers a real origin.

    python3 src/build_site.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "webapp_data.json"
WEB = ROOT / "web"
SITE = ROOT / "site"
FONTS_SRC = Path.home() / "Documents" / "marathon_agent" / "static" / "fonts"

NAV = [
    ("index.html", "score", "Score a contract", "gauge"),
    ("review.html", "review", "Review queue", "flag"),
]

# 24x24 stroke icons, matching the trainer's set.
ICONS = {
    "gauge": '<path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"/><path d="m13.4 10.6 4.1-4.1"/>'
             '<path d="M20.5 17.5a9.5 9.5 0 1 0-17 0"/>',
    "flag": '<path d="M4 21V4"/><path d="M4 4h11l-1.5 3.5L15 11H4"/>',
    "capitol": '<path d="M12 3v3"/><path d="M7 10h10"/><path d="M8.5 10v8M15.5 10v8M12 10v8"/>'
               '<path d="M5 18h14M4 21h16"/><path d="M12 6a4 4 0 0 0-4 4h8a4 4 0 0 0-4-4Z"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.5.5l2-2A5 5 0 0 0 12.5 4.5l-1 1"/>'
            '<path d="M14 11a5 5 0 0 0-7.5-.5l-2 2A5 5 0 0 0 11.5 19.5l1-1"/>',
    "shield": '<path d="M12 3 5 6v6c0 4.5 3 7.6 7 9 4-1.4 7-4.5 7-9V6l-7-3Z"/>',
}


def icon(name: str, cls: str = "") -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" '
            f'stroke-linecap="round" stroke-linejoin="round"{f" class={cls!r}" if cls else ""}>'
            f'{ICONS[name]}</svg>')


def shell(page: str, title: str, body: str, meta: dict) -> str:
    nav = "".join(
        f'<a class="nav-item{" active" if key == page else ""}" href="{href}">'
        f'{icon(ico)}<span>{label}</span></a>'
        for href, key, label, ico in NAV
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="referrer" content="no-referrer">
<title>{title} · DOE Procurement Auditor</title>
<link rel="stylesheet" href="static/fonts/inter.css">
<link rel="stylesheet" href="static/style.css">
</head>
<body data-page="{page}">

<div class="shell">
  <aside class="sidebar">
    <div class="logo">
      <span class="logo-mark">{icon('shield')}</span>
      <span class="logo-text">
        <span class="l1">PROCUREMENT</span>
        <span class="l2">AUDITOR</span>
      </span>
    </div>

    <nav class="nav">{nav}</nav>

    <div class="sidebar-spacer"></div>

    <div class="coverage">
      <div class="cov-label">Contracts analysed</div>
      <div class="cov-big">{meta['contracts_total']:,}</div>
      <div class="cov-note">NYC DOE &middot; <span id="covyears"></span></div>
    </div>
  </aside>

  <main class="main">
{body}
    <footer class="foot">
      Built from Checkbook NYC and NYC City Clerk eLobbyist open data.
      Scoring runs locally in your browser; nothing is sent anywhere.
    </footer>
  </main>
</div>

<script src="static/data.js"></script>
<script src="static/app.js"></script>
</body>
</html>
"""


def page_head(title: str, sub: str) -> str:
    return f'    <div class="page-head">\n      <h1 class="page-title">{title}</h1>\n' \
           f'      <p class="page-sub">{sub}</p>\n    </div>\n'


# --------------------------------------------------------------------------
# page bodies
# --------------------------------------------------------------------------

def body_score(payload: dict) -> str:
    """Verdict-led: inputs sit in a narrow rail, the assessment is the page.

    The earlier version gave the inputs more width than the answer and stacked
    three cards of methodology underneath. An auditor opens this to get a
    reading, so the reading is the largest thing on screen.
    """
    am = payload["award_model"]
    return page_head(
        "Score a contract",
        "Set a contract's characteristics on the left. The model returns how likely DOE would "
        "be to award work like it <em>without a formal solicitation</em> &mdash; and a low "
        "reading on a contract that <em>was</em> awarded that way is the thing worth pulling."
    ) + f"""
    <div class="grid">
      <div class="card sp4 rail">
        <div class="card-head"><h2 class="card-title">Characteristics</h2></div>
        <div class="field">
          <label for="amt">Contract amount <b id="amtL"></b></label>
          <input type="range" id="amt" min="3" max="9.4" step="0.05" value="5.6">
        </div>
        <div class="field">
          <label for="dur">Contract term <b id="durL"></b></label>
          <input type="range" id="dur" min="0" max="2200" step="5" value="365">
        </div>
        <div class="field">
          <label>Industry</label>
          <div class="chips" id="inds"></div>
        </div>
        <div class="field" style="margin-bottom:0">
          <label>Contract type</label>
          <div class="chips" id="ctypes"></div>
        </div>
      </div>

      <div class="card sp8 verdict verdict-lg">
        <span class="eyebrow">Likelihood of a no-bid award</span>
        <svg class="gauge" id="gauge" width="300" height="172" viewBox="0 0 300 172"
             role="img" aria-label="Probability of a non-competitive award"></svg>
        <div class="pval" id="pv">&ndash;</div>
        <div class="vlabel" id="vl"></div>
        <div class="vnote" id="vn"></div>
        <p class="card-body-text vctx" id="ctx"></p>
      </div>

      <div class="card sp12">
        <div class="card-head"><h2 class="card-title">What is driving this reading</h2></div>
        <div class="contrib" id="contrib" style="margin-top:0;border-top:none;padding-top:0"></div>
        <p class="card-body-text" style="margin-top:14px;font-size:13px">
          Fitted on <b>{am['n_labeled']:,}</b> DOE contracts above $25,000 where DOE chose how
          much competition to run. Terms marked <span class="tag">structural</span> follow from
          the contract's length, which procurement rules tie to the award method &mdash; they
          predict well but explain little.</p>
      </div>
    </div>
"""


def body_review(payload: dict) -> str:
    """Anomaly-led case list. Four columns, so nothing scrolls sideways."""
    queue = payload.get("review_queue") or {}
    anomalies = queue.get("anomalies", 0)
    disc = queue.get("discretionary_total", 0)
    pctile = queue.get("anomaly_percentile", 0.01)
    total = queue.get("total", 0)
    retro_days = queue.get("retro_flag_days", 365)
    long_years = queue.get("long_term_days", 1460) // 365

    return page_head(
        "Review queue",
        f"<strong>{anomalies}</strong> no-bid awards that do not look like no-bid awards "
        f"&mdash; the most unusual {pctile*100:.0f}% within their own industry, out of "
        f"{disc:,} discretionary awards. Listed first, most unusual at the top."
    ) + f"""
    <div class="tiles" id="qtiles"></div>

    <div class="card sp12">
      <div class="card-head"><h2 class="card-title">The checks, and what they mean</h2></div>
      <p class="card-body-text">The first is the primary signal and the queue is ranked by it.
      The other four are corroboration &mdash;
      an anomaly that also trips one of these is a better use of an afternoon than one that does
      not. Some background: in New York City a contract is <b>registered</b> by the Comptroller,
      the final approval step that records it and makes it payable. Registration is meant to
      happen before the work starts.</p>
      <div class="table-wrap"><table class="data">
        <thead><tr><th>Check</th><th class="num">Cases</th>
          <th class="wrap">What it means, and why it is set there</th></tr></thead>
        <tbody id="flagkey"></tbody>
      </table></div>
      <p class="card-body-text" style="margin-top:18px"><b>The unusual-for-industry bar sits in
      a different place in every industry</b>, because the industries are not comparable
      &mdash; Professional Services goes to bid far more readily than Goods:</p>
      <div class="bars" id="barsind"></div>
      <div class="note warn">
        <h3>None of this is a finding</h3>
        <p>Awarding without competition is lawful and often correct &mdash; sometimes one firm
        genuinely can do the work. Lobbying is lawful, disclosed, and commoner among large
        vendors precisely <em>because</em> they are large. A long term can be the efficient
        choice. What a case gives you is a named reason, a dollar figure, and a document you can
        request. The model also learns what DOE <em>does</em>, not what it <em>should</em> do:
        where a whole category is routinely sole-sourced, that becomes the expected pattern and
        will not be flagged at all.</p>
      </div>
    </div>

    <div class="card sp12">
      <div class="card-head"><h2 class="card-title">Cases</h2></div>
      <div class="toolbar">
        <input type="text" id="q" placeholder="Filter vendor or award method…"
               aria-label="Filter cases">
        <select id="flagsel" aria-label="Filter by check"></select>
        <span class="count" id="count"></span>
      </div>
      <div class="table-wrap"><table class="data" id="tbl">
        <thead><tr>
          <th class="wrap-tight">Vendor</th>
          <th class="num">FY</th>
          <th class="num">Amount</th>
          <th class="wrap">Checks</th>
        </tr></thead><tbody></tbody>
      </table></div>
      <p class="card-body-text" style="margin-top:14px;font-size:13.5px">Click a row for the
      award method, term, registration lag, how far into its industry's tail it sits, and any
      lobbying filing &mdash; including the officials named in it.</p>
    </div>
"""


def main() -> int:
    if not DATA.exists():
        print("no webapp_data.json -- run src/export_webapp.py first")
        return 1
    payload = json.loads(DATA.read_text())

    static = SITE / "static"
    static.mkdir(parents=True, exist_ok=True)

    # Data rides as a script that assigns a global, because fetch() against
    # file:// is blocked and this site has to open by double-click.
    (static / "data.js").write_text(
        "window.DOE = " + json.dumps(payload, separators=(",", ":")) + ";\n")

    shutil.copy2(WEB / "style.css", static / "style.css")
    shutil.copy2(WEB / "app.js", static / "app.js")

    fonts_dst = static / "fonts"
    if FONTS_SRC.exists():
        shutil.copytree(FONTS_SRC, fonts_dst, dirs_exist_ok=True)
    elif not fonts_dst.exists():
        # Inter is a nicety, not a dependency: the stylesheet falls back to the
        # system UI face, so a missing font directory must not break the build.
        fonts_dst.mkdir(parents=True, exist_ok=True)
        (fonts_dst / "inter.css").write_text("/* Inter not vendored; system fallback in use */\n")
        print("  note: Inter fonts not found, falling back to system faces")

    pages = {
        "index.html": ("score", "Score a contract", body_score(payload)),
        "review.html": ("review", "Review queue", body_review(payload)),
    }
    for filename, (key, title, body) in pages.items():
        (SITE / filename).write_text(shell(key, title, body, payload["meta"]), encoding="utf-8")

    total = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"wrote {SITE.name}/  {len(pages)} pages, {total/1e6:.2f} MB total")
    for filename in pages:
        print(f"  {filename:<16} {(SITE / filename).stat().st_size/1024:6.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
