/* DOE Procurement Auditor — page logic.
 *
 * Every page reads window.DOE, set by static/data.js. Nothing is fetched at
 * runtime, so the site works from file:// with no server and no network. Which
 * page is running is read off <body data-page>.
 *
 * Edit this file directly; the build copies it into site/static/ untouched.
 */
"use strict";

const D = window.DOE || {};
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => [...(r || document).querySelectorAll(s)];

/* ── formatting ─────────────────────────────────────────────────────── */

const money = n =>
  n >= 1e9 ? "$" + (n / 1e9).toFixed(2) + "B" :
  n >= 1e6 ? "$" + (n / 1e6).toFixed(1) + "M" :
  n >= 1e3 ? "$" + (n / 1e3).toFixed(0) + "K" : "$" + Math.round(n);

const money0 = n => "$" + Math.round(n).toLocaleString();
const pct = (x, d) => (x * 100).toFixed(d === undefined ? 1 : d) + "%";
const esc = s => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

/* Severity is always relative to how often this KIND of work goes
 * non-competitive. A negotiated acquisition in an 8.8% industry is a different
 * object from the same award in a 31% industry. */
const rateTone = r => (r < 0.15 ? "risk" : r < 0.32 ? "warn" : "neutral");
/* Expandable detail rows, shared by every table that has them. */
function wireExpand(table) {
  const toggle = e => {
    const row = e.target.closest("tr.clickable");
    if (!row) return;
    const det = $(`tr.detail[data-d="${row.dataset.i}"]`, table);
    if (det) det.hidden = !det.hidden;
  };
  table.addEventListener("click", toggle);
  table.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(e); }
  });
}

/* ── page 1: score a contract ───────────────────────────────────────── */

function initScore() {
  const AM = D.award_model;
  const F = AM.features;
  const at = name => F.indexOf(name);

  /* Inputs. Duration is in DAYS, not years: the median DOE contract runs 212
   * days and 15.6% run under 91, so a whole-year control cannot express most of
   * the file. The two threshold flags are what the data actually shows —
   * non-formal share is flat at 88–98% below ~400 days and ~2% above 1,460 —
   * so the score steps at those lines rather than sliding smoothly. */
  /* Industry and contract type are the feed's own categories now, one chip per
   * real value, with the residual category as the reference level. An earlier
   * version offered Goods / Services / Construction / Other, which collapsed a
   * 10.3%-to-56.7% spread into two flags — and DOE has no construction
   * contracts at all, so that chip scored nothing. */
  let industry = null, ctype = null;

  const industryRate = key => (AM.industry_rates || []).find(r => r.key === key) || null;
  const typeRate = key => (AM.type_rates || []).find(r => r.key === key) || null;

  function vector() {
    const amount = Math.pow(10, +$("#amt").value);
    const days = +$("#dur").value;
    const med = AM.industry_medians[industry || ""] ?? AM.industry_medians.all;

    const v = new Array(F.length).fill(0);
    v[at("log_duration")] = Math.log1p(Math.max(days, 0));
    v[at("dur_under_1y")] = days <= 400 ? 1 : 0;
    v[at("dur_1_to_2y")] = days > 400 && days <= 730 ? 1 : 0;
    v[at("dur_over_4y")] = days >= 1460 ? 1 : 0;
    v[at("log_amount")] = Math.log1p(amount);
    v[at("amount_vs_industry")] = Math.log1p(amount) - Math.log1p(med);
    if (industry && at(industry) >= 0) v[at(industry)] = 1;
    if (ctype && at(ctype) >= 0) v[at(ctype)] = 1;
    return { v, amount, days };
  }

  function score() {
    const { v, amount, days } = vector();
    let z = AM.intercept;
    const parts = v.map((val, i) => {
      const c = ((val - AM.means[i]) / AM.stds[i]) * AM.coefficients[i];
      z += c;
      return { name: AM.feature_labels?.[F[i]] || F[i], c, key: F[i] };
    });
    render(1 / (1 + Math.exp(-z)), parts, amount, days);
  }

  const showDays = d =>
    d >= 730 ? (d / 365).toFixed(1).replace(/\.0$/, "") + " yr"
    : d === 365 ? "1 yr"
    : d > 365 ? "1 yr " + (d - 365) + " d"
    : d >= 60 ? Math.round(d / 30.4) + " mo"
    : d + " d";

  function render(p, parts, amount, days) {
    $("#amtL").textContent = money(amount);
    $("#durL").textContent = showDays(days);

    const p100 = p * 100;
    $("#pv").textContent = p100 < 0.1 ? "<0.1%" : p100 > 99.9 ? ">99.9%" : p100.toFixed(1) + "%";

    const hi = p >= 0.5, mid = p >= 0.2 && p < 0.5;
    const tone = hi ? "risk" : mid ? "warn" : "good";
    const col = `var(--${tone})`;
    $("#pv").style.color = col;
    $("#vl").textContent = hi ? "A formal solicitation would be unusual here"
      : mid ? "Could go either way"
      : "A formal solicitation is typical here";
    $("#vn").textContent = hi
      ? "Awarding this without a competitive solicitation matches DOE's normal pattern for contracts like it."
      : mid ? "The pattern is mixed for contracts with these characteristics."
      : "If this was awarded without a formal solicitation, it departs from the pattern — worth pulling the file.";

    const r = 118, cx = 150, cy = 142, a = Math.PI * (1 - p);
    const x = cx + r * Math.cos(a), y = cy - r * Math.sin(a);
    $("#gauge").innerHTML =
      `<path d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}" fill="none"
         stroke="var(--border)" stroke-width="18" stroke-linecap="round"/>
       <path d="M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${x} ${y}" fill="none"
         stroke="${col}" stroke-width="18" stroke-linecap="round"/>
       <text x="${cx - r}" y="${cy + 24}" fill="var(--ink-3)" font-size="12" font-weight="600">0%</text>
       <text x="${cx + r}" y="${cy + 24}" fill="var(--ink-3)" font-size="12" font-weight="600"
         text-anchor="end">100%</text>`;

    /* Structural terms are tinted differently: their value is fixed by the term
     * length, which procurement rules tie to the award method. Separating them
     * on screen stops the biggest bars from being read as findings. */
    const structural = new Set(AM.structural || []);
    const max = Math.max(...parts.map(o => Math.abs(o.c)), 0.01);
    $("#contrib").innerHTML =
      parts.filter(o => Math.abs(o.c) > 0.005)
        .sort((a, b) => Math.abs(b.c) - Math.abs(a.c)).map(o => {
        const w = Math.abs(o.c) / max * 50, pos = o.c > 0;
        const st = structural.has(o.key);
        return `<div class="crow"><span${st ? ' style="opacity:.72"' : ""}>${esc(o.name)}${
          st ? ' <span class="tag">structural</span>' : ""}</span>
          <span class="cbar"><i style="${pos ? "left:50%" : "right:50%"};width:${w}%;
          background:${pos ? "var(--risk)" : "var(--good)"}"></i></span>
          <span class="cval">${o.c > 0 ? "+" : ""}${o.c.toFixed(2)}</span></div>`;
      }).join("");

    /* Always name the comparison set. "Unusual" with no denominator is the
     * failure mode this whole tool exists to avoid. */
    const ir = industryRate(industry), tr = typeRate(ctype);
    const parts2 = [];
    if (ir) parts2.push(`<b class="tone-${rateTone(ir.rate)}">${pct(ir.rate)}</b> of
      ${ir.n.toLocaleString()} <b>${esc(ir.label)}</b> contracts`);
    if (tr) parts2.push(`<b class="tone-${rateTone(tr.rate)}">${pct(tr.rate)}</b> of
      ${tr.n.toLocaleString()} <b>${esc(tr.label)}</b> contracts`);
    $("#ctx").innerHTML = parts2.length
      ? "In this scope DOE awarded " + parts2.join(" and ") +
        " without a formal solicitation. Those are the base rates this score is judged against."
      : `No published base rate for <b>${esc(AM.industry_reference)}</b> /
         <b>${esc(AM.type_reference)}</b> — too few contracts to quote one honestly.`;
  }

  $$("#amt, #dur").forEach(el => el.addEventListener("input", score));

  function chips(host, options, initial, apply) {
    $(host).innerHTML = options.map(([v, l]) =>
      `<button class="chip" data-v="${v}" aria-pressed="${v === initial}">${l}</button>`).join("");
    $(host).addEventListener("click", e => {
      const b = e.target.closest(".chip"); if (!b) return;
      apply(b.dataset.v);
      $$(".chip", $(host)).forEach(c => c.setAttribute("aria-pressed", c === b));
      score();
    });
  }

  /* Reference categories are real options, not omissions: picking them sets
   * every dummy in that family to zero, which is exactly what the fit means by
   * "Not classified" / "Other". */
  const indOpts = [...Object.entries(AM.industry_labels || {}).map(([k, l]) => [k, l]),
                   ["", AM.industry_reference]];
  const typeOpts = [...Object.entries(AM.contract_type_labels || {}).map(([k, l]) => [k, l]),
                    ["", AM.type_reference]];

  chips("#inds", indOpts, "ind_human_services", v => { industry = v || null; });
  chips("#ctypes", typeOpts, "type_programs", v => { ctype = v || null; });
  industry = "ind_human_services"; ctype = "type_programs";

  score();
}

/* ── page 2: review queue ───────────────────────────────────────────── */

/* Replaced three pages (Flagged / Lobbying / Overlap) with one case list.
 *
 * Four columns only, so the table never scrolls sideways. Everything else goes
 * in the expandable detail row — the table is for triage, the detail is for
 * deciding whether to pull the file. The old lobbying "match" column is gone:
 * whether a name matched is a precondition for showing lobbying evidence, not
 * something a reader should have to interpret. */
function initReview() {
  const Q = D.review_queue;
  if (!Q) return;
  const rows0 = Q.cases, L = Q.flag_labels;
  const pctile = (Q.anomaly_percentile || 0.01) * 100;
  let onlyAnomalies = true;   // the tool opens on the outliers

  const FLAG_TONE = {
    unusual_for_industry: "risk",
    lobbied: "risk",
    registered_after_expiry: "warn",
    retroactive: "neutral",
    long_no_bid: "neutral",
  };

  /* Plain-English, and each one says why its threshold sits where it does. The
   * numbers are the reason a reader can trust the bar wasn't picked to produce a
   * pleasing count. */
  const WHY = {
    unusual_for_industry:
      `The model reads this no-bid award as one DOE would normally have put out to bid — ` +
      `it sits in the most unusual ${pctile.toFixed(0)}% of no-bid awards in its own industry. ` +
      `This is the signal the queue is ranked by; the rest is corroboration.`,
    lobbied:
      "The vendor filed procurement lobbying against DOE in the award year or the year before. " +
      "Filings after an award cannot have influenced it, so later ones are ignored. The filing " +
      "names the officials lobbied — the most directly requestable thing in the tool.",
    registered_after_expiry:
      "The Comptroller registered this contract after its own end date: the work was performed, " +
      "finished, and only then was the contract formally approved. No model is involved — the " +
      "paperwork post-dates the work it authorises.",
    retroactive:
      "Work began at least a year before the contract was registered. The bar is a full year " +
      "because 92.6% of contracts in scope are registered after their start date at all — " +
      "being late is normal at DOE, so only the extreme tail is informative.",
    long_no_bid:
      "A no-bid award running four years or more. Long terms are how one non-competitive " +
      "decision compounds into years of committed spend without a further look.",
  };

  $("#qtiles").innerHTML = [
    [Q.anomalies.toLocaleString(), "unusual for their industry"],
    [Q.multi_flag.toLocaleString(), "corroborated by 2+ checks"],
    [(Q.flag_counts.lobbied || 0).toLocaleString(), "with lobbying on record"],
    [Q.total.toLocaleString(), "cases in total"],
  ].map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join("");

  /* Per-industry bars, to show the threshold really does move. */
  const bars = Q.industry_bars || {};
  const labels = D.award_model?.industry_labels || {};
  const mx = Math.max(...Object.values(bars), 0.01);
  $("#barsind").innerHTML = Object.entries(bars)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class="bar-row">
      <span class="bl">${esc(labels[k] || "Not classified")}</span>
      <span class="cbar"><i style="left:0;width:${(v / mx * 100).toFixed(1)}%;
        background:var(--blue)"></i></span>
      <span class="tone-neutral">flagged below ${pct(v)}</span></div>`).join("");

  $("#flagkey").innerHTML = Q.flag_order.map(f =>
    `<tr><td class="wrap-tight"><span class="badge ${FLAG_TONE[f]}">${esc(L[f])}</span></td>
      <td class="num strong">${(Q.flag_counts[f] || 0).toLocaleString()}</td>
      <td class="wrap tone-neutral" style="font-size:13px">${esc(WHY[f])}</td></tr>`).join("");

  $("#flagsel").innerHTML =
    `<option value="_anom">Unusual for its industry (${Q.anomalies})</option>` +
    `<option value="">Every check (${Q.total.toLocaleString()})</option>` +
    Q.flag_order.filter(f => f !== "unusual_for_industry").map(f =>
      `<option value="${f}">${esc(L[f])} (${Q.flag_counts[f] || 0})</option>`).join("");

  const term = d => d == null ? "—" : d >= 730 ? (d / 365).toFixed(1).replace(/\.0$/, "") + " yr"
    : d >= 365 ? "1 yr " + (d - 365) + " d" : d + " d";

  function detail(c) {
    const e = c.evidence || {};
    const bits = [
      `<b>${esc(c.method)}</b> &middot; ${esc(c.industry)} &middot; ${esc(c.type)}`,
      `Term <b>${term(c.term)}</b>`,
      c.retro > 0 ? `Registered <b>${c.retro} d</b> after work began`
                  : "Registered on or before start",
      `Vendor's <b>${c.prior + 1}</b>${c.prior === 0 ? "st" : c.prior === 1 ? "nd"
        : c.prior === 2 ? "rd" : "th"} DOE contract`,
    ];
    let out = `<div style="margin-bottom:8px">${bits.join(" &nbsp;·&nbsp; ")}</div>`;

    if (c.anomaly != null) {
      out += `<div class="note" style="margin:10px 0 0">
        <h3>Why this is unusual</h3>
        <p>The model puts a contract like this at <b>${pct(c.p)}</b> likely to be a no-bid
        award. ${c.anomaly === 0
          ? `No other no-bid award in <b>${esc(c.industry)}</b> scores lower.`
          : `Among DOE's own no-bid awards in <b>${esc(c.industry)}</b>, just
             <b>${c.anomaly.toFixed(2)}%</b> score lower.`}
        The flag threshold for that industry is
        ${e.industry_bar != null ? pct(e.industry_bar) : "—"}.
        ${e.industry_rate != null
          ? ` For scale, ${pct(e.industry_rate)} of ${esc(c.industry)} contracts in scope go
             without a formal solicitation.` : ""}</p></div>`;
    }

    if (e.lobbying) {
      const lb = e.lobbying;
      out += `<div class="note" style="margin:10px 0 0">
        <h3>Lobbying on record</h3>
        <p><b>${lb.filings}</b> filing${lb.filings === 1 ? "" : "s"} FY${lb.years[0]}–${lb.years[1]},
        ${money0(lb.compensation)} reported compensation, ${lb.lobbyists}
        lobbyist${lb.lobbyists === 1 ? "" : "s"}.
        ${lb.clients ? "Filed as <b>" + esc(lb.clients) + "</b>." : ""}</p>
        ${lb.officials ? `<p><b>Officials named:</b> ${esc(lb.officials)}</p>` : ""}</div>`;
    }
    return out;
  }

  function draw() {
    const q = $("#q").value.toLowerCase();
    const sel = $("#flagsel").value;
    const rows = rows0.filter(c =>
      (!q || c.vendor.toLowerCase().includes(q) || c.method.toLowerCase().includes(q)) &&
      (sel === "_anom" ? c.anomaly != null : (!sel || c.flags.includes(sel))));

    $("#count").textContent = rows.length.toLocaleString() + " shown";
    $("#tbl tbody").innerHTML = rows.map((c, i) => `
      <tr class="clickable${c.anomaly != null ? " lead" : ""}" data-i="${i}" tabindex="0">
        <td class="strong wrap-tight">${esc(c.vendor)}${c.anomaly != null
          ? `<span class="sub">${c.anomaly === 0
              ? "least like a no-bid award in " + esc(c.industry)
              : "more unusual than " + (100 - c.anomaly).toFixed(1) + "% of "
                + esc(c.industry) + " no-bid awards"}</span>`
          : ""}</td>
        <td class="num">${c.fy}</td>
        <td class="num strong">${money(c.amount)}</td>
        <td class="badges">${c.flags.map(f =>
          `<span class="badge ${FLAG_TONE[f]}" style="margin:1px 3px 1px 0">${esc(L[f])}</span>`
        ).join("")}</td>
      </tr>
      <tr class="detail" data-d="${i}" hidden><td colspan="4">${detail(c)}</td></tr>`
    ).join("") || `<tr><td colspan="4" class="tone-neutral">No cases match that filter.</td></tr>`;
  }

  $("#q").addEventListener("input", draw);
  $("#flagsel").addEventListener("change", draw);
  wireExpand($("#tbl"));
  draw();
}

/* ── boot ───────────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  const M = D.meta || {};
  const cov = $("#covyears");
  if (cov) cov.textContent = "FY" + M.years[0] + "–FY" + M.years[1];

  ({ score: initScore, review: initReview }[document.body.dataset.page] || (() => {}))();
});
