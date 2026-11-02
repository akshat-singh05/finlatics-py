/*
 * Finlatics Ad Analytics web app.
 * Plain JavaScript single-page app over the Flask API (no build step).
 * Every number comes from the API, which runs the same pipeline as main.py.
 */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // ------------------------------------------------------------------
  // Per-viewer preferences (best effort: storage can be unavailable)
  // ------------------------------------------------------------------
  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem("finlatics-" + key); return v === null ? fallback : JSON.parse(v); }
      catch (e) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem("finlatics-" + key, JSON.stringify(value)); } catch (e) { /* ignore */ }
    },
  };

  // ------------------------------------------------------------------
  // API
  // ------------------------------------------------------------------
  class ApiError extends Error {
    constructor(message, messages, status) { super(message); this.messages = messages; this.status = status; }
  }
  const cache = new Map();

  function buildUrl(path, params) {
    const url = new URL(path, location.origin);
    for (const [k, v] of Object.entries(params || {})) {
      if (v !== "" && v !== null && v !== undefined) url.searchParams.set(k, v);
    }
    return url.pathname + url.search;
  }

  async function getJson(path, params) {
    const url = buildUrl(path, params);
    if (cache.has(url)) return cache.get(url);
    const promise = fetch(url).then(async (res) => {
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new ApiError(body.error || res.statusText, body.messages, res.status);
      return body;
    });
    cache.set(url, promise);
    promise.catch(() => cache.delete(url));
    return promise;
  }

  async function getText(path) {
    if (cache.has(path)) return cache.get(path);
    const promise = fetch(path).then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new ApiError(body.error || res.statusText, null, res.status);
      }
      return res.text();
    });
    cache.set(path, promise);
    promise.catch(() => cache.delete(path));
    return promise;
  }

  async function postJson(path, data) {
    const res = await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(body.error || res.statusText, body.messages, res.status);
    return body;
  }

  // ------------------------------------------------------------------
  // Formatting
  // ------------------------------------------------------------------
  const isNum = (x) => typeof x === "number" && isFinite(x);
  const fmt = {
    money(x, d = 0) {
      if (!isNum(x)) return "—";
      const a = Math.abs(x);
      // No minus sign on values that round to zero ("−$0")
      const sign = x < 0 && Number(a.toFixed(d)) > 0 ? "−$" : "$";
      return sign + a.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
    },
    moneyCompact(x) {
      if (!isNum(x)) return "—";
      const a = Math.abs(x);
      const s = a >= 1e6 ? (a / 1e6).toFixed(2) + "M" : a >= 1e4 ? (a / 1e3).toFixed(1) + "k" : a.toLocaleString("en-US", { maximumFractionDigits: 0 });
      return (x < 0 && s !== "0" ? "−$" : "$") + s;
    },
    int: (x) => (isNum(x) ? Math.round(x).toLocaleString("en-US") : "—"),
    compact(x) {
      if (!isNum(x)) return "—";
      const a = Math.abs(x);
      if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e8 ? 0 : 1) + "M";
      if (a >= 1e4) return (x / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
      return Math.round(x).toLocaleString("en-US");
    },
    pct: (x, d = 1) => (isNum(x) ? (x * 100).toFixed(d) + "%" : "—"),
    times: (x) => (isNum(x) ? x.toFixed(x < 10 ? 1 : 0) + "×" : "—"),
    dec: (x, d = 3) => (isNum(x) ? x.toFixed(d) : "—"),
    date(s) {
      const d = new Date(s + "T00:00:00");
      return isNaN(d) ? s : d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
    },
  };

  // ------------------------------------------------------------------
  // Metric catalogue (thresholds match analysis/insights.py)
  // ------------------------------------------------------------------
  const MIN_CLICKS = 1000, MIN_DISPLAYS = 10000, MIN_CONVERSIONS = 100, MIN_COST_FOR_ROI = 20;

  const METRICS = {
    revenue: { label: "Revenue", long: "Revenue (Company X ad spend)", f: (v) => fmt.money(v), short: fmt.moneyCompact, tick: "$~s", total: "total_revenue" },
    cost: { label: "Cost", long: "Cost paid to publishers", f: (v) => fmt.money(v), short: fmt.moneyCompact, tick: "$~s", total: "total_cost" },
    firm_margin: { label: "Firm A margin", long: "Advert Firm A margin (revenue − cost)", f: (v) => fmt.money(v), short: fmt.moneyCompact, tick: "$~s", total: "total_firm_margin", diverging: true },
    firm_roi: { label: "Firm A ROI", long: "Advert Firm A ROI (margin ÷ cost)", f: (v) => fmt.pct(v, 0), tick: ".0%", ratio: ["firm_margin", "cost", 1], min: ["cost", MIN_COST_FOR_ROI, "less than $20 of cost"], diverging: true },
    displays: { label: "Displays", f: fmt.int, short: fmt.compact, tick: "~s", total: "total_displays" },
    clicks: { label: "Clicks", f: fmt.int, short: fmt.compact, tick: "~s", total: "total_clicks" },
    ctr: { label: "CTR", long: "Click-through rate (clicks ÷ displays)", f: (v) => fmt.pct(v, 2), tick: ".1%", ratio: ["clicks", "displays", 1], min: ["displays", MIN_DISPLAYS, "fewer than 10,000 displays"] },
    post_click_conversions: { label: "Conversions", long: "Post-click conversions", f: fmt.int, short: fmt.compact, tick: "~s", total: "total_post_click_conversions" },
    conversion_rate: { label: "Conversion rate", long: "Conversion rate (conversions ÷ clicks)", f: (v) => fmt.pct(v, 1), tick: ".0%", ratio: ["post_click_conversions", "clicks", 1], min: ["clicks", MIN_CLICKS, "fewer than 1,000 clicks"] },
    post_click_sales_amount: { label: "Attributed sales", long: "Post-click sales amount", f: (v) => fmt.money(v), short: fmt.moneyCompact, tick: "$~s", total: "total_post_click_sales_amount" },
    advertiser_roas: { label: "ROAS", long: "Company X ROAS (sales ÷ ad spend)", f: fmt.times, tick: ".0f", suffix: "×", ratio: ["post_click_sales_amount", "revenue", 1], min: ["clicks", MIN_CLICKS, "fewer than 1,000 clicks"] },
    advertiser_cpc: { label: "CPC (Company X)", long: "Company X cost per click (ad spend ÷ clicks)", f: (v) => fmt.money(v, 3), tick: "$.2f", ratio: ["revenue", "clicks", 1], min: ["clicks", MIN_CLICKS, "fewer than 1,000 clicks"], lower: true },
    advertiser_cpa: { label: "CPA (Company X)", long: "Company X cost per conversion (ad spend ÷ conversions)", f: (v) => fmt.money(v, 3), tick: "$.2f", ratio: ["revenue", "post_click_conversions", 1], min: ["post_click_conversions", MIN_CONVERSIONS, "fewer than 100 conversions"], lower: true },
    firm_cpc: { label: "Firm cost per click", long: "Advert Firm A cost per click (cost ÷ clicks)", f: (v) => fmt.money(v, 3), tick: "$.2f", ratio: ["cost", "clicks", 1], min: ["clicks", MIN_CLICKS, "fewer than 1,000 clicks"], lower: true },
    cpm: { label: "Cost per 1,000 displays", f: (v) => fmt.money(v, 2), tick: "$.2f", ratio: ["cost", "displays", 1000], min: ["displays", MIN_DISPLAYS, "fewer than 10,000 displays"], lower: true },
  };
  const metricShort = (key, v) => (METRICS[key].short || METRICS[key].f)(v);
  const lowVolume = (key, row) => {
    const m = METRICS[key].min;
    return Boolean(m) && !(row[m[0]] >= m[1]);
  };

  const DIMENSIONS = {
    campaign_number: { label: "Campaign", order: ["camp 1", "camp 2", "camp 3"] },
    user_engagement: { label: "Engagement", order: ["Low", "Medium", "High"] },
    banner: { label: "Banner size" },
    placement: { label: "Placement" },
    month: { label: "Month", order: ["April", "May", "June"], natural: true },
    weekday: { label: "Weekday", order: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], natural: true },
  };
  // Entities with a fixed categorical colour (slot order as in utils/theme.py)
  const ENTITY_SLOTS = {
    campaign_number: { "camp 1": 0, "camp 2": 1, "camp 3": 2 },
    user_engagement: { Low: 0, Medium: 1, High: 2 },
  };

  function orderValues(dim, values) {
    const order = DIMENSIONS[dim] && DIMENSIONS[dim].order;
    if (!order) return [...values].sort((a, b) => String(a).localeCompare(String(b), "en", { numeric: true }));
    return [...values].sort((a, b) => order.indexOf(a) - order.indexOf(b));
  }

  const options = (pairs, selected) =>
    pairs.map(([v, l]) => `<option value="${esc(v)}"${v === selected ? " selected" : ""}>${esc(l)}</option>`).join("");
  const metricOptions = (keys, selected) => options(keys.map((k) => [k, METRICS[k].label]), selected);
  const dimOptions = (keys, selected) => options(keys.map((k) => [k, DIMENSIONS[k].label]), selected);

  // ------------------------------------------------------------------
  // Charts (Plotly, styled from the CSS tokens so both themes work)
  // ------------------------------------------------------------------
  function tokens() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    return {
      ink: g("--ink"), ink2: g("--ink-2"), muted: g("--muted"), grid: g("--grid"), axis: g("--axis"),
      surface: g("--surface"), series: [g("--series-1"), g("--series-2"), g("--series-3")],
      neg: g("--neg"), neutral: g("--neutral"), seq: g("--seq").split(",").map((s) => s.trim()),
    };
  }

  function merge(a, b) {
    const out = { ...a };
    for (const [k, v] of Object.entries(b || {})) {
      out[k] = v && typeof v === "object" && !Array.isArray(v) && a[k] && typeof a[k] === "object" ? merge(a[k], v) : v;
    }
    return out;
  }

  function baseLayout(extra) {
    const t = tokens();
    const axis = {
      gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickcolor: t.axis,
      tickfont: { color: t.muted, size: 11 }, title: { font: { color: t.ink2, size: 12 } }, automargin: true,
    };
    return merge({
      font: { family: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', size: 12, color: t.ink2 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 8, r: 12, t: 8, b: 8 },
      xaxis: { ...axis }, yaxis: { ...axis },
      hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink, size: 12 } },
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: t.ink2 } },
      showlegend: false, bargap: 0.35, hovermode: "closest",
    }, extra);
  }
  const PLOT_CONFIG = { displayModeBar: false, responsive: true };
  // Plotly's own `responsive` only follows window resizes; also follow the container
  const resizer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver((entries) => {
    for (const e of entries) if (e.target.isConnected && e.target.data) Plotly.Plots.resize(e.target);
  });
  function plot(el, data, layout) {
    for (const s of el.querySelectorAll(":scope > .skeleton")) s.remove();
    Plotly.react(el, data, baseLayout(layout), PLOT_CONFIG);
    if (resizer && !el.dataset.observed) { resizer.observe(el); el.dataset.observed = "1"; }
  }

  function axisFor(key) {
    const m = METRICS[key];
    return { tickformat: m.tick, ticksuffix: m.suffix || "" };
  }

  /** Bar colours: entity colours for campaign/engagement, sign colours for margin/ROI, else series 1. */
  function barColors(dim, labels, values, key) {
    const t = tokens();
    return labels.map((l, i) => {
      if (METRICS[key].diverging) return values[i] < 0 ? t.neg : t.series[0];
      const slots = ENTITY_SLOTS[dim];
      return slots && l in slots ? t.series[slots[l]] : t.series[0];
    });
  }

  /** Bar chart of one metric per group; low-volume groups are faded. */
  function barChart(el, dim, rows, key, { horizontal = false } = {}) {
    const labels = rows.map((r) => String(r[dim]));
    const values = rows.map((r) => r[key]);
    const faded = rows.map((r) => lowVolume(key, r));
    const colors = barColors(dim, labels, values, key);
    const hover = rows.map((r, i) =>
      `<b>${esc(labels[i])}</b><br>${esc(METRICS[key].label)}: ${METRICS[key].f(values[i])}` +
      `<br>Clicks: ${fmt.int(r.clicks)} · Cost: ${fmt.money(r.cost)}` +
      (faded[i] ? `<br><i>Low volume: ${METRICS[key].min[2]}</i>` : ""));
    const trace = {
      type: "bar", orientation: horizontal ? "h" : "v",
      x: horizontal ? values : labels, y: horizontal ? labels : values,
      marker: { color: colors, opacity: faded.map((f) => (f ? 0.35 : 1)), cornerradius: 4 },
      text: values.map((v) => metricShort(key, v)), textposition: "outside", cliponaxis: false,
      textfont: { color: tokens().ink2, size: 11 },
      customdata: hover, hovertemplate: "%{customdata}<extra></extra>",
    };
    const valueAxis = { ...axisFor(key), zeroline: true, showgrid: true };
    const catAxis = { showgrid: false, type: "category", tickfont: { color: tokens().ink2, size: 12 } };
    plot(el, [trace], horizontal
      ? { xaxis: valueAxis, yaxis: { ...catAxis, autorange: "reversed" }, margin: { r: 48 } }
      : { yaxis: valueAxis, xaxis: catAxis, margin: { t: 20 } });
    return faded.some(Boolean);
  }

  const lowVolumeNote = (key) =>
    `Faded bars have ${METRICS[key].min[2]}, so their ${METRICS[key].label} is not reliable and is not ranked in the report.`;

  // ------------------------------------------------------------------
  // App state, filters and routing
  // ------------------------------------------------------------------
  const state = {
    info: null,
    filters: store.get("filters", {}),
    renderId: 0,
  };
  const view = $("#view");

  function activeFilters() {
    const f = {};
    for (const [k, v] of Object.entries(state.filters)) if (v) f[k] = v;
    return f;
  }
  const filterCount = () => Object.keys(activeFilters()).length;

  function initFilters(opts) {
    for (const sel of $$("#filters select")) {
      const key = sel.dataset.filter;
      const values = orderValues(key, opts[key] || []);
      sel.insertAdjacentHTML("beforeend", values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join(""));
    }
    syncFilterInputs();
    let timer;
    $("#filters").addEventListener("change", (e) => {
      const key = e.target.dataset.filter;
      if (!key) return;
      state.filters[key] = e.target.value;
      store.set("filters", state.filters);
      clearTimeout(timer);
      timer = setTimeout(render, 150);
    });
    $("#filters-reset").addEventListener("click", () => {
      state.filters = {};
      store.set("filters", {});
      syncFilterInputs();
      render();
    });
  }

  function syncFilterInputs() {
    for (const el of $$("#filters [data-filter]")) el.value = state.filters[el.dataset.filter] || "";
    $("#filters-reset").disabled = filterCount() === 0;
  }

  const PAGES = {
    overview: { title: "Overview", subtitle: "Headline KPIs for Advert Firm A and Company X, with daily trends.", filters: true, render: renderOverview },
    breakdown: { title: "Breakdown", subtitle: "Compare any metric across campaigns, engagement levels, banners, placements, months or weekdays.", filters: true, render: renderBreakdown },
    segments: { title: "Segments", subtitle: "Two dimensions at once, such as banner × placement, to find the segments that win or lose.", filters: true, render: renderSegments },
    questions: { title: "20 Questions", subtitle: "Answers to the 20 official questions in the problem statement, with the supporting charts.", filters: false, render: renderQuestions },
    predict: { title: "Predict a buy", subtitle: "Estimate revenue, margin and ROI for a planned inventory buy with the trained model.", filters: false, render: renderPredict },
    model: { title: "Model", subtitle: "How the revenue model was tested, and how it compares with simple rate-card rules.", filters: false, render: renderModel },
    data: { title: "Data", subtitle: "Browse, sort and download the cleaned rows behind every number.", filters: true, render: renderData },
    api: { title: "API", subtitle: "The JSON endpoints this app is built on.", filters: false, render: renderApi },
  };

  function parseRoute() {
    const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
    const page = PAGES[parts[0]] ? parts[0] : "overview";
    return { page, args: parts.slice(1).map(decodeURIComponent) };
  }

  async function render() {
    const { page, args } = parseRoute();
    const def = PAGES[page];
    const id = ++state.renderId;
    document.title = `${def.title} · Finlatics Ad Analytics`;
    $("#page-title").textContent = def.title;
    $("#page-subtitle").textContent = def.subtitle;
    $("#filters").hidden = !def.filters;
    syncFilterInputs();
    for (const a of $$("#nav a")) a.classList.toggle("active", a.dataset.page === page);
    const current = () => id === state.renderId;
    try {
      await def.render({ args, current });
    } catch (err) {
      if (current()) showError(view, err);
    }
  }

  function showError(el, err) {
    const msgs = err.messages ? (Array.isArray(err.messages) ? err.messages : [JSON.stringify(err.messages)]) : [];
    const hint = err.status === 404 && /No data matches/.test(err.message)
      ? "<p class='muted'>No rows match the current filters. Widen the date range or reset the filters.</p>" : "";
    el.innerHTML = `<div class="card empty"><p class="error"><strong>${esc(err.message || "Something went wrong")}</strong></p>` +
      `${msgs.map((m) => `<p class="muted small">${esc(m)}</p>`).join("")}${hint}</div>`;
  }

  const skeleton = (n = 3) => Array.from({ length: n }, () => '<div class="skeleton"></div>').join("");

  function filterSummary() {
    const f = activeFilters();
    const parts = [];
    for (const k of ["campaign_number", "user_engagement", "banner", "placement"]) if (f[k]) parts.push(f[k]);
    if (f.start || f.end) parts.push(`${f.start ? fmt.date(f.start) : "start"} – ${f.end ? fmt.date(f.end) : "end"}`);
    return parts.length ? `Filtered: ${parts.join(" · ")}` : "All data";
  }

  // ------------------------------------------------------------------
  // Overview
  // ------------------------------------------------------------------
  const TREND_METRICS = ["revenue", "cost", "firm_margin", "firm_roi", "displays", "clicks", "ctr",
    "post_click_conversions", "conversion_rate", "post_click_sales_amount", "advertiser_cpc"];

  function kpi(label, value, sub, cls = "") {
    return `<div class="card kpi"><div class="label">${esc(label)}</div>` +
      `<div class="value num ${cls}" title="${esc(value)}">${esc(value)}</div>` +
      `<div class="sub">${esc(sub || "")}</div></div>`;
  }
  const signClass = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");

  async function renderOverview({ current }) {
    const metric = store.get("ov-metric", "revenue");
    const smooth = store.get("ov-smooth", true);
    view.innerHTML = `
      <div class="kpi-group"><h2>Advert Firm A</h2><div class="kpis" id="k-firm">${skeleton(4)}</div></div>
      <div class="kpi-group"><h2>Company X</h2><div class="kpis" id="k-cx">${skeleton(4)}</div></div>
      <div class="kpi-group"><h2>Audience</h2><div class="kpis" id="k-aud">${skeleton(4)}</div></div>
      <div class="card">
        <div class="card-head">
          <div><h2>Daily trend</h2><p class="muted small" id="trend-sub"></p></div>
          <div class="controls">
            <label>Metric<select id="ov-metric">${metricOptions(TREND_METRICS, metric)}</select></label>
            <label class="check"><input type="checkbox" id="ov-smooth" ${smooth ? "checked" : ""}>7-day average</label>
          </div>
        </div>
        <div id="ov-trend" class="chart"></div>
      </div>
      <div class="grid two">
        <div class="card"><div class="card-head"><h2 id="ov-camp-title">By campaign</h2></div><div id="ov-camp" class="chart short"></div><p class="note" id="ov-camp-note" hidden></p></div>
        <div class="card"><div class="card-head"><h2 id="ov-eng-title">By engagement level</h2></div><div id="ov-eng" class="chart short"></div><p class="note" id="ov-eng-note" hidden></p></div>
      </div>`;

    const f = activeFilters();
    const [m, daily, camp, eng] = await Promise.all([
      getJson("/metrics", f), getJson("/metrics/date", f),
      getJson("/metrics/campaign_number", f), getJson("/metrics/user_engagement", f),
    ]);
    if (!current()) return;

    $("#k-firm").innerHTML =
      kpi("Revenue earned", fmt.money(m.total_revenue), "paid by Company X for clicks") +
      kpi("Cost", fmt.money(m.total_cost), "paid to publishers") +
      kpi("Margin", fmt.money(m.total_firm_margin), "revenue − cost", signClass(m.total_firm_margin)) +
      kpi("ROI", fmt.pct(m.firm_roi, 1), "margin ÷ cost", signClass(m.firm_roi));
    $("#k-cx").innerHTML =
      kpi("Attributed sales", fmt.money(m.total_post_click_sales_amount), "30-day post-click window") +
      kpi("ROAS", fmt.times(m.advertiser_roas), "sales ÷ ad spend") +
      kpi("Cost per click", fmt.money(m.advertiser_cpc, 3), "ad spend ÷ clicks") +
      kpi("Cost per conversion", fmt.money(m.advertiser_cpa, 3), "ad spend ÷ conversions");
    $("#k-aud").innerHTML =
      kpi("Displays", fmt.compact(m.total_displays), fmt.int(m.total_displays) + " impressions") +
      kpi("Clicks", fmt.compact(m.total_clicks), fmt.int(m.total_clicks) + " clicks") +
      kpi("CTR", fmt.pct(m.ctr, 2), "clicks ÷ displays") +
      kpi("Conversion rate", fmt.pct(m.conversion_rate, 1), fmt.int(m.total_post_click_conversions) + " conversions");

    const draw = () => {
      const key = $("#ov-metric").value;
      const sm = $("#ov-smooth").checked;
      store.set("ov-metric", key);
      store.set("ov-smooth", sm);
      drawTrend($("#ov-trend"), daily.groups, key, sm);
      $("#trend-sub").textContent = `${METRICS[key].long || METRICS[key].label} · ${filterSummary()} · ${m.date_range}`;
      for (const [id, dim, data] of [["ov-camp", "campaign_number", camp], ["ov-eng", "user_engagement", eng]]) {
        const rows = orderValues(dim, data.groups.map((g) => g[dim])).map((v) => data.groups.find((g) => g[dim] === v));
        $(`#${id}-title`).textContent = `${METRICS[key].label} by ${dim === "campaign_number" ? "campaign" : "engagement level"}`;
        const anyFaded = barChart($(`#${id}`), dim, rows, key);
        const note = $(`#${id}-note`);
        note.hidden = !anyFaded;
        if (anyFaded) note.textContent = lowVolumeNote(key);
      }
    };
    $("#ov-metric").addEventListener("change", draw);
    $("#ov-smooth").addEventListener("change", draw);
    draw();
  }

  function drawTrend(el, rows, key, smooth) {
    const t = tokens();
    const m = METRICS[key];
    const x = rows.map((r) => r.date);
    const y = rows.map((r) => r[key]);
    const traces = [];
    const hover = (v) => m.f(v);
    if (smooth) {
      // Rolling ratio-of-totals for rates, rolling mean for totals
      const w = 7;
      const avg = rows.map((_, i) => {
        const win = rows.slice(Math.max(0, i - w + 1), i + 1);
        if (win.length < w) return null;
        if (m.ratio) {
          const [n, d, mult] = m.ratio;
          const den = win.reduce((s, r) => s + (r[d] || 0), 0);
          return den ? (win.reduce((s, r) => s + (r[n] || 0), 0) / den) * mult : null;
        }
        return win.reduce((s, r) => s + (r[key] || 0), 0) / w;
      });
      traces.push({ type: "scatter", mode: "lines", name: "Daily", x, y, line: { color: t.series[0], width: 1 }, opacity: 0.35,
        customdata: y.map(hover), hovertemplate: "Daily: %{customdata}<extra></extra>" });
      traces.push({ type: "scatter", mode: "lines", name: "7-day average", x, y: avg, line: { color: t.series[0], width: 2 },
        customdata: avg.map((v) => (v === null ? "—" : hover(v))), hovertemplate: "7-day average: %{customdata}<extra></extra>" });
    } else {
      traces.push({ type: "scatter", mode: "lines", name: m.label, x, y, line: { color: t.series[0], width: 2 },
        customdata: y.map(hover), hovertemplate: `${esc(m.label)}: %{customdata}<extra></extra>` });
    }
    plot(el, traces, {
      showlegend: smooth, hovermode: "x unified",
      xaxis: { type: "date", showgrid: false, tickformat: "%-d %b", hoverformat: "%a %-d %b %Y", showspikes: true, spikecolor: t.axis, spikethickness: 1, spikedash: "solid", spikemode: "across" },
      yaxis: { ...axisFor(key), rangemode: m.diverging ? "normal" : "tozero" },
      margin: { t: 28 },
    });
  }

  // ------------------------------------------------------------------
  // Breakdown
  // ------------------------------------------------------------------
  const TABLE_METRICS = ["displays", "clicks", "ctr", "post_click_conversions", "conversion_rate", "revenue", "cost",
    "firm_margin", "firm_roi", "advertiser_cpc", "advertiser_cpa", "advertiser_roas", "post_click_sales_amount"];
  const BREAKDOWN_DIMS = ["campaign_number", "user_engagement", "banner", "placement", "month", "weekday"];

  async function renderBreakdown({ current }) {
    const dim = store.get("bd-dim", "banner");
    const key = store.get("bd-metric", "revenue");
    view.innerHTML = `
      <div class="card">
        <div class="card-head">
          <div><h2 id="bd-title"></h2><p class="muted small" id="bd-sub"></p></div>
          <div class="controls">
            <label>Group by<select id="bd-dim">${dimOptions(BREAKDOWN_DIMS, dim)}</select></label>
            <label>Metric<select id="bd-metric">${metricOptions(Object.keys(METRICS), key)}</select></label>
          </div>
        </div>
        <div id="bd-chart" class="chart tall">${skeleton(1)}</div>
        <p class="note" id="bd-note" hidden></p>
      </div>
      <div class="card">
        <div class="card-head"><div><h2>All metrics</h2><p class="muted small">Click a column to sort. Rates are ratio-of-totals for each group.</p></div></div>
        <div class="table-wrap" id="bd-table"></div>
      </div>`;

    let sort = { col: null, desc: true };
    const draw = async () => {
      const d = $("#bd-dim").value;
      const k = $("#bd-metric").value;
      store.set("bd-dim", d);
      store.set("bd-metric", k);
      const data = await getJson(`/metrics/${d}`, activeFilters());
      if (!current()) return;
      let rows = data.groups;
      const natural = DIMENSIONS[d].natural;
      if (natural) {
        rows = orderValues(d, rows.map((r) => r[d])).map((v) => rows.find((r) => r[d] === v));
      } else {
        // Best first; low-volume groups go last so they are not read as winners
        rows = [...rows].sort((a, b) =>
          (lowVolume(k, a) - lowVolume(k, b)) || ((b[k] ?? -Infinity) - (a[k] ?? -Infinity)));
      }
      $("#bd-title").textContent = `${METRICS[k].label} by ${DIMENSIONS[d].label.toLowerCase()}`;
      $("#bd-sub").textContent = `${METRICS[k].long || METRICS[k].label} · ${filterSummary()}`;
      $("#bd-chart").style.height = natural ? "" : `${Math.max(260, rows.length * 44 + 60)}px`;
      const anyFaded = barChart($("#bd-chart"), d, rows, k, { horizontal: !natural });
      $("#bd-note").hidden = !anyFaded;
      if (anyFaded) $("#bd-note").textContent = lowVolumeNote(k);
      drawTable(d, data.groups, k);
    };

    const drawTable = (d, groups, k) => {
      let rows = groups;
      if (sort.col) {
        rows = [...groups].sort((a, b) => {
          const va = a[sort.col], vb = b[sort.col];
          const cmp = typeof va === "number" || typeof vb === "number" ? (va ?? -Infinity) - (vb ?? -Infinity) : String(va).localeCompare(String(vb), "en", { numeric: true });
          return sort.desc ? -cmp : cmp;
        });
      } else {
        rows = orderValues(d, groups.map((g) => g[d])).map((v) => groups.find((g) => g[d] === v));
      }
      const cols = [d, ...TABLE_METRICS];
      const head = cols.map((c) => {
        const label = c === d ? DIMENSIONS[d].label : METRICS[c].label;
        const cls = ["sortable", c === d ? "" : "n", c === k ? "hl" : "", sort.col === c ? "sorted" : ""].join(" ");
        const arrow = sort.col === c ? (sort.desc ? " ↓" : " ↑") : "";
        return `<th class="${cls}" data-col="${c}" title="${esc(c === d ? "" : METRICS[c].long || "")}">${esc(label)}${arrow}</th>`;
      }).join("");
      const body = rows.map((r) => `<tr>${cols.map((c) => c === d
        ? `<td>${esc(r[c])}</td>`
        : `<td class="n${c === k ? " hl" : ""}"${lowVolume(c, r) ? ' style="opacity:.5" title="Low volume"' : ""}>${METRICS[c].f(r[c])}</td>`).join("")}</tr>`).join("");
      $("#bd-table").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
      for (const th of $$("#bd-table th")) {
        th.addEventListener("click", () => {
          const c = th.dataset.col;
          sort = sort.col === c ? { col: c, desc: !sort.desc } : { col: c, desc: c !== d };
          drawTable(d, groups, k);
        });
      }
    };

    $("#bd-dim").addEventListener("change", () => { sort = { col: null, desc: true }; draw().catch((e) => showError(view, e)); });
    $("#bd-metric").addEventListener("change", () => draw().catch((e) => showError(view, e)));
    await draw();
  }

  // ------------------------------------------------------------------
  // Segments (two-way heatmap)
  // ------------------------------------------------------------------
  const PIVOT_DIMS = ["campaign_number", "user_engagement", "banner", "placement", "month", "weekday"];
  const SEGMENT_METRICS = ["firm_margin", "firm_roi", "revenue", "cost", "clicks", "displays", "ctr",
    "conversion_rate", "post_click_conversions", "advertiser_cpc", "advertiser_cpa", "advertiser_roas"];

  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }
  function colorAt(scale, t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < scale.length; i++) {
      if (t <= scale[i][0]) {
        const [p0, c0] = scale[i - 1], [p1, c1] = scale[i];
        const u = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
        const a = hexToRgb(c0), b = hexToRgb(c1);
        return a.map((v, j) => Math.round(v + (b[j] - v) * u));
      }
    }
    return hexToRgb(scale[scale.length - 1][1]);
  }
  const luminance = ([r, g, b]) => {
    const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };

  async function renderSegments({ current }) {
    const rowsDim = store.get("sg-rows", "banner");
    const colsDim = store.get("sg-cols", "placement");
    const key = store.get("sg-metric", "firm_margin");
    const hide = store.get("sg-hide", true);
    view.innerHTML = `
      <div class="card">
        <div class="card-head">
          <div><h2 id="sg-title"></h2><p class="muted small" id="sg-sub"></p></div>
          <div class="controls">
            <label>Rows<select id="sg-rows">${dimOptions(PIVOT_DIMS, rowsDim)}</select></label>
            <label>Columns<select id="sg-cols">${dimOptions(PIVOT_DIMS, colsDim)}</select></label>
            <label>Metric<select id="sg-metric">${metricOptions(SEGMENT_METRICS, key)}</select></label>
            <label class="check"><input type="checkbox" id="sg-hide" ${hide ? "checked" : ""}>Hide low-volume cells</label>
          </div>
        </div>
        <div id="sg-chart" class="chart tall">${skeleton(1)}</div>
        <p class="note" id="sg-note"></p>
      </div>
      <div class="grid two">
        <div class="card"><h2 id="sg-top-title">Highest</h2><div class="table-wrap" id="sg-top" style="margin-top:10px"></div></div>
        <div class="card"><h2 id="sg-bottom-title">Lowest</h2><div class="table-wrap" id="sg-bottom" style="margin-top:10px"></div></div>
      </div>`;

    const draw = async () => {
      let r = $("#sg-rows").value, c = $("#sg-cols").value;
      if (r === c) {
        c = PIVOT_DIMS.find((d) => d !== r);
        $("#sg-cols").value = c;
      }
      const k = $("#sg-metric").value;
      const hideLow = $("#sg-hide").checked;
      store.set("sg-rows", r); store.set("sg-cols", c); store.set("sg-metric", k); store.set("sg-hide", hideLow);
      const data = await getJson("/pivot", { rows: r, cols: c, ...activeFilters() });
      if (!current()) return;

      const cells = data.cells;
      const rowVals = orderValues(r, [...new Set(cells.map((x) => x[r]))]);
      const colVals = orderValues(c, [...new Set(cells.map((x) => x[c]))]);
      const lookup = new Map(cells.map((x) => [`${x[r]}\u0000${x[c]}`, x]));
      const shown = (cell) => cell && isNum(cell[k]) && !(hideLow && lowVolume(k, cell));
      const vals = cells.filter(shown).map((x) => x[k]);

      $("#sg-title").textContent = `${METRICS[k].label}: ${DIMENSIONS[r].label.toLowerCase()} × ${DIMENSIONS[c].label.toLowerCase()}`;
      $("#sg-sub").textContent = `${METRICS[k].long || METRICS[k].label} · ${filterSummary()}`;

      const t = tokens();
      let scale, zmin, zmax, capped = false;
      if (METRICS[k].diverging) {
        // Grey at zero. Colours are capped so one extreme cell cannot wash out the rest:
        // ROI at −100%…+200%, money at the 90th percentile of absolute values
        const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
        if (k === "firm_roi") { zmin = Math.max(lo, -1); zmax = Math.min(hi, 2); }
        else {
          const abs = vals.map(Math.abs).sort((a, b) => a - b);
          const cap = abs[Math.floor(0.9 * (abs.length - 1))] || abs[abs.length - 1] || 1;
          zmin = -cap; zmax = cap;
          capped = cap < Math.max(-lo, hi);
        }
        if (zmin === zmax) zmax = zmin + 1;
        const mid = (0 - zmin) / (zmax - zmin);
        scale = [[0, t.neg], [mid, t.neutral], [1, t.series[0]]];
        if (mid <= 0) scale = [[0, t.neutral], [1, t.series[0]]];
        if (mid >= 1) scale = [[0, t.neg], [1, t.neutral]];
      } else {
        zmin = Math.min(...vals); zmax = Math.max(...vals);
        if (!(zmax > zmin)) zmax = zmin + 1;
        scale = t.seq.map((col, i) => [i / (t.seq.length - 1), col]);
      }

      const z = rowVals.map((rv) => colVals.map((cv) => { const x = lookup.get(`${rv}\u0000${cv}`); return shown(x) ? x[k] : null; }));
      const hover = rowVals.map((rv) => colVals.map((cv) => {
        const x = lookup.get(`${rv}\u0000${cv}`);
        if (!x) return `<b>${esc(rv)} · ${esc(cv)}</b><br>No rows`;
        return `<b>${esc(rv)} · ${esc(cv)}</b><br>${esc(METRICS[k].label)}: ${METRICS[k].f(x[k])}` +
          `<br>Clicks: ${fmt.int(x.clicks)} · Cost: ${fmt.money(x.cost)} · Rows: ${fmt.int(x.record_count)}` +
          (lowVolume(k, x) ? `<br><i>Low volume: ${METRICS[k].min[2]}</i>` : "");
      }));
      const annotations = [];
      rowVals.forEach((rv, i) => colVals.forEach((cv, j) => {
        const x = lookup.get(`${rv}\u0000${cv}`);
        const v = z[i][j];
        let text = "", color = t.muted;
        if (v !== null) {
          const rgb = colorAt(scale, (Math.max(zmin, Math.min(zmax, v)) - zmin) / (zmax - zmin));
          color = luminance(rgb) > 0.4 ? "#0b0b0b" : "#ffffff";
          text = metricShort(k, v);
        } else if (x) {
          text = "·";
        }
        if (text) annotations.push({ x: cv, y: rv, text, showarrow: false, font: { color, size: 11 } });
      }));

      const height = Math.max(300, rowVals.length * 46 + 90);
      $("#sg-chart").style.height = `${height}px`;
      plot($("#sg-chart"), [{
        type: "heatmap", x: colVals, y: rowVals, z, zmin, zmax, colorscale: scale, xgap: 2, ygap: 2,
        customdata: hover, hovertemplate: "%{customdata}<extra></extra>", hoverongaps: true,
        colorbar: { thickness: 10, outlinewidth: 0, tickformat: METRICS[k].tick, ticksuffix: METRICS[k].suffix || "", tickfont: { color: t.muted, size: 11 }, len: 0.9 },
      }], {
        annotations,
        xaxis: { type: "category", showgrid: false, side: "top", tickfont: { color: t.ink2, size: 12 } },
        yaxis: { type: "category", showgrid: false, autorange: "reversed", tickfont: { color: t.ink2, size: 12 } },
        plot_bgcolor: "rgba(0,0,0,0)",
      });

      const lowNote = METRICS[k].min ? ` Cells with ${METRICS[k].min[2]} ${hideLow ? "are shown as “·”" : "are included but unreliable"}.` : "";
      const capNote = capped ? ` Colours are capped at ±${metricShort(k, zmax > -zmin ? zmax : -zmin)}; cells beyond that use the strongest colour.` : "";
      $("#sg-note").textContent = (METRICS[k].diverging
        ? "Blue = positive, red = negative, grey = break-even."
        : "Darker = higher.") + capNote + lowNote + " Empty cells have no rows.";

      // Ranked lists of eligible cells
      const eligible = cells.filter((x) => isNum(x[k]) && !lowVolume(k, x)).sort((a, b) => b[k] - a[k]);
      const list = (items) => items.length
        ? `<table><thead><tr><th>Segment</th><th class="n">${esc(METRICS[k].label)}</th><th class="n">Clicks</th><th class="n">Cost</th></tr></thead><tbody>` +
          items.map((x) => `<tr><td>${esc(x[r])} · ${esc(x[c])}</td><td class="n">${METRICS[k].f(x[k])}</td><td class="n">${fmt.int(x.clicks)}</td><td class="n">${fmt.money(x.cost)}</td></tr>`).join("") +
          "</tbody></table>"
        : '<p class="empty">No segments with enough volume.</p>';
      const n = Math.min(5, Math.floor(eligible.length / 2) || eligible.length);
      $("#sg-top-title").textContent = `Highest ${METRICS[k].label}`;
      $("#sg-bottom-title").textContent = `Lowest ${METRICS[k].label}`;
      $("#sg-top").innerHTML = list(eligible.slice(0, n));
      $("#sg-bottom").innerHTML = list(eligible.slice(-n).reverse());
    };

    for (const id of ["#sg-rows", "#sg-cols", "#sg-metric", "#sg-hide"]) {
      $(id).addEventListener("change", () => draw().catch((e) => showError(view, e)));
    }
    await draw();
  }

  // ------------------------------------------------------------------
  // 20 Questions (rendered from reports/eda_findings.md)
  // ------------------------------------------------------------------
  function splitReport(md) {
    const lines = md.replace(/\r\n?/g, "\n").split("\n");
    const sections = [];
    let cur = { id: "intro", title: "About this report", lines: [] };
    for (const line of lines) {
      const m = line.match(/^##\s+(.*)$/);
      if (m) {
        sections.push(cur);
        const title = m[1].trim();
        const q = title.match(/^Q(\d+)\s*·\s*(.*)$/);
        cur = q
          ? { id: `q${q[1]}`, num: +q[1], title: q[2], lines: [] }
          : { id: title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""), title: title.replace(/^Appendix\s*·\s*/, "Appendix: "), lines: [] };
      } else {
        cur.lines.push(line);
      }
    }
    sections.push(cur);
    for (const s of sections) {
      s.md = s.lines.join("\n").replace(/\n-{3,}\s*$/g, "").trim();
      s.text = s.md.replace(/[#*`|>!\[\]()_-]/g, " ").toLowerCase();
    }
    const intro = sections[0];
    intro.md = intro.md.replace(/^#\s+.*\n?/, "").trim(); // the page already has a title
    return sections;
  }

  const resolveFigure = (src) => (/^https?:/.test(src) ? src : `/report/${src.replace(/^\.?\/?/, "")}`);

  function bindLightbox(root) {
    for (const img of $$("img", root)) {
      img.addEventListener("click", () => {
        const dlg = $("#lightbox");
        $("img", dlg).src = img.src;
        $("img", dlg).alt = img.alt;
        $("p", dlg).textContent = img.alt;
        dlg.showModal();
      });
    }
  }

  async function renderQuestions({ args, current }) {
    view.innerHTML = `<div class="qlayout"><div class="card qlist">${skeleton(1)}</div><div class="card">${skeleton(2)}</div></div>`;
    const md = await getText("/report/eda");
    if (!current()) return;
    const sections = splitReport(md);
    const questions = sections.filter((s) => s.num);
    const start = sections.filter((s) => !s.num && sections.indexOf(s) < sections.indexOf(questions[0]));
    const more = sections.filter((s) => !s.num && sections.indexOf(s) > sections.indexOf(questions[0]));
    const selectedId = args[0] && sections.some((s) => s.id === args[0]) ? args[0] : "q1";
    const sel = sections.find((s) => s.id === selectedId);
    const order = [...start, ...questions, ...more];
    const idx = order.indexOf(sel);
    const prev = order[idx - 1], next = order[idx + 1];

    const item = (s) => `<li data-text="${esc(s.title.toLowerCase() + " " + s.text)}"><a href="#/questions/${s.id}" class="${s.id === selectedId ? "active" : ""}">` +
      `<b>${s.num ? "Q" + s.num : "•"}</b><span>${esc(s.title)}</span></a></li>`;
    const heading = sel.num
      ? `<p class="muted small" style="margin:0 0 4px">Question ${sel.num} of ${questions.length}</p><h2>${esc(sel.title)}</h2>`
      : `<h2>${esc(sel.title)}</h2>`;
    const navLink = (s, label) => s
      ? `<a class="btn ghost small" href="#/questions/${s.id}">${label === "prev" ? "← " : ""}${esc(s.num ? "Q" + s.num : s.title)}${label === "next" ? " →" : ""}</a>`
      : "<span></span>";

    view.innerHTML = `
      <div class="qlayout">
        <nav class="card qlist" aria-label="Questions">
          <input type="search" id="q-search" placeholder="Search questions and answers…" aria-label="Search questions and answers">
          <ol id="q-items">
            <li class="group">Start here</li>${start.map(item).join("")}
            <li class="group">Official questions</li>${questions.map(item).join("")}
            <li class="group">More</li>${more.map(item).join("")}
          </ol>
        </nav>
        <article class="card">
          <div class="prose">${heading}<div style="margin-top:14px">${Markdown.render(sel.md, { resolveImage: resolveFigure })}</div></div>
          <div class="qnav">${navLink(prev, "prev")}${navLink(next, "next")}</div>
        </article>
      </div>`;
    bindLightbox(view);
    const active = $("#q-items a.active");
    if (active) active.scrollIntoView({ block: "nearest" });
    $("#q-search").addEventListener("input", (e) => {
      const q = e.target.value.trim().toLowerCase();
      for (const li of $$("#q-items li[data-text]")) li.hidden = Boolean(q) && !li.dataset.text.includes(q);
    });
  }

  // ------------------------------------------------------------------
  // Predict
  // ------------------------------------------------------------------
  const COMPARE_DIMS = ["placement", "banner", "user_engagement", "campaign_number"];

  async function renderPredict({ current }) {
    const info = state.info;
    const opts = info.filters;
    const saved = store.get("predict", {});
    const v = {
      date: saved.date || info.dataset.end, campaign_number: saved.campaign_number || "camp 1",
      user_engagement: saved.user_engagement || "High", banner: saved.banner || "240 x 400",
      placement: saved.placement || "ghi", displays: saved.displays ?? 30000, cost: saved.cost ?? 20,
    };
    const sel = (name, label) =>
      `<label>${label}<select name="${name}">${options(orderValues(name, opts[name]).map((x) => [x, x]), v[name])}</select></label>`;
    const compareDim = store.get("cmp-dim", "placement");

    view.innerHTML = `
      ${info.model.available ? "" : `<div class="banner">No trained model found. Run <code>python main.py</code> to train it, then restart the app.</div>`}
      <div class="grid side">
        <form class="card" id="pf" novalidate>
          <div class="card-head"><div><h2>Planned buy</h2><p class="muted small">One row = one day of one campaign × engagement × banner × placement.</p></div></div>
          <div class="form-grid">
            <label style="grid-column:1/-1">Date<input type="date" name="date" value="${esc(v.date)}" min="${esc(info.dataset.start)}" required></label>
            ${sel("campaign_number", "Campaign")}${sel("user_engagement", "Engagement")}
            ${sel("banner", "Banner size")}${sel("placement", "Placement")}
            <label>Displays<input type="number" name="displays" value="${esc(v.displays)}" min="0" step="1" required></label>
            <label>Cost ($)<input type="number" name="cost" value="${esc(v.cost)}" min="0" step="0.01" required></label>
          </div>
          <div class="form-actions">
            <button class="btn" type="submit" id="pf-go">Predict</button>
            <button class="btn ghost" type="button" id="pf-avg">Use segment averages</button>
          </div>
          <p class="note">Displays and cost are what Advert Firm A pays the publisher for this row. The model was trained on data up to ${fmt.date(info.dataset.end)}.</p>
        </form>
        <div style="display:flex;flex-direction:column;gap:16px;min-width:0">
          <section class="card" id="pr" aria-live="polite"><div class="empty">Fill in the planned buy and press Predict.</div></section>
          <section class="card"><div class="card-head"><div><h2>History for this segment</h2><p class="muted small" id="hist-sub"></p></div></div><div id="hist"></div></section>
        </div>
      </div>
      <section class="card">
        <div class="card-head">
          <div><h2>Compare options</h2><p class="muted small">The same buy with one choice changed. Every option keeps the same date, displays and cost.</p></div>
          <div class="segmented" id="cmp-dim" role="group" aria-label="Option to compare">
            ${COMPARE_DIMS.map((d) => `<button type="button" data-dim="${d}" class="${d === compareDim ? "on" : ""}">${DIMENSIONS[d].label}</button>`).join("")}
          </div>
        </div>
        <div id="cmp-chart" class="chart short"></div>
        <div class="table-wrap" id="cmp-table" style="margin-top:12px"></div>
        <p class="note">Real prices differ by placement and banner, so an option can look better here only because it is given the same cost. Press “Use segment averages” first for a like-for-like plan.</p>
      </section>`;

    const form = $("#pf");
    const read = () => {
      const fd = new FormData(form);
      return {
        date: fd.get("date"), campaign_number: fd.get("campaign_number"), user_engagement: fd.get("user_engagement"),
        banner: fd.get("banner"), placement: fd.get("placement"),
        displays: fd.get("displays") === "" ? null : Number(fd.get("displays")),
        cost: fd.get("cost") === "" ? null : Number(fd.get("cost")),
      };
    };
    const segmentFilters = (b) => ({ campaign_number: b.campaign_number, user_engagement: b.user_engagement, banner: b.banner, placement: b.placement });

    async function showHistory() {
      const b = read();
      $("#hist-sub").textContent = `${b.campaign_number} · ${b.user_engagement} · ${b.banner} · ${b.placement}, ${fmt.date(info.dataset.start)} – ${fmt.date(info.dataset.end)}`;
      try {
        const m = await getJson("/metrics", segmentFilters(b));
        const avgDisplays = m.total_displays / m.total_records, avgCost = m.total_cost / m.total_records;
        $("#hist").innerHTML = `<dl class="ref">
          <dt>Rows (days with delivery)</dt><dd>${fmt.int(m.total_records)}</dd>
          <dt>Average displays per row</dt><dd>${fmt.int(avgDisplays)}</dd>
          <dt>Average cost per row</dt><dd>${fmt.money(avgCost, 2)}</dd>
          <dt>Revenue per 1,000 displays</dt><dd>${fmt.money(m.total_revenue / m.total_displays * 1000, 3)}</dd>
          <dt>CTR</dt><dd>${fmt.pct(m.ctr, 2)}</dd>
          <dt>Actual firm ROI</dt><dd>${fmt.pct(m.firm_roi, 0)}</dd></dl>`;
        return { displays: Math.round(avgDisplays), cost: Math.round(avgCost * 100) / 100 };
      } catch (err) {
        $("#hist").innerHTML = `<p class="muted">${err.status === 404 ? "This combination never ran in the data, so the prediction is an extrapolation." : esc(err.message)}</p>`;
        return null;
      }
    }

    function showResult(res, b) {
      const roi = res.predicted_roi;
      $("#pr").innerHTML = `
        <div class="result-head">
          <div><h2>Prediction</h2><p class="muted small" style="margin:2px 0 0">${esc(b.campaign_number)} · ${esc(b.user_engagement)} · ${esc(b.banner)} · ${esc(b.placement)} on ${fmt.date(b.date)}</p></div>
          <span class="badge ${res.profitable ? "good" : "bad"}">${res.profitable ? "Profitable" : "Not profitable"}</span>
        </div>
        <div class="result-kpis">
          ${kpi("Predicted revenue", fmt.money(res.predicted_revenue, 2), "paid by Company X")}
          ${kpi("Cost", fmt.money(res.cost, 2), `${fmt.int(b.displays)} displays`)}
          ${kpi("Predicted margin", fmt.money(res.predicted_margin, 2), "revenue − cost", signClass(res.predicted_margin))}
          ${kpi("Predicted ROI", roi === null ? "—" : fmt.pct(roi, 0), "margin ÷ cost", signClass(roi))}
        </div>
        ${res.warnings.length ? `<ul class="warn-list">${res.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}`;
    }

    async function compare(b) {
      const d = $("#cmp-dim .on").dataset.dim;
      store.set("cmp-dim", d);
      const values = orderValues(d, opts[d]);
      const results = await postJson("/predict", values.map((x) => ({ ...b, [d]: x })));
      const rows = values.map((x, i) => ({ option: x, ...results[i] }));
      const t = tokens();
      plot($("#cmp-chart"), [{
        type: "bar", x: rows.map((r) => r.option), y: rows.map((r) => r.predicted_margin),
        marker: {
          color: rows.map((r) => (r.predicted_margin < 0 ? t.neg : t.series[0])), cornerradius: 4,
          line: { color: rows.map((r) => (r.option === b[d] ? t.ink : "rgba(0,0,0,0)")), width: 2 },
        },
        text: rows.map((r) => fmt.money(r.predicted_margin, 2)), textposition: "outside", cliponaxis: false,
        textfont: { color: t.ink2, size: 11 },
        customdata: rows.map((r) => `<b>${esc(r.option)}</b>${r.option === b[d] ? " (current)" : ""}<br>Revenue: ${fmt.money(r.predicted_revenue, 2)}<br>Margin: ${fmt.money(r.predicted_margin, 2)}<br>ROI: ${r.predicted_roi === null ? "—" : fmt.pct(r.predicted_roi, 0)}`),
        hovertemplate: "%{customdata}<extra></extra>",
      }], {
        xaxis: { type: "category", showgrid: false, tickfont: { color: t.ink2, size: 12 } },
        yaxis: { tickformat: "$.2f", zeroline: true, title: { text: "Predicted margin" } },
        margin: { t: 20 },
      });
      $("#cmp-table").innerHTML = `<table><thead><tr><th>${esc(DIMENSIONS[d].label)}</th><th class="n">Predicted revenue</th><th class="n">Predicted margin</th><th class="n">ROI</th><th>Call</th></tr></thead><tbody>` +
        rows.map((r) => `<tr class="${r.option === b[d] ? "best" : ""}"><td>${esc(r.option)}${r.option === b[d] ? ' <span class="badge plain">current</span>' : ""}</td>` +
          `<td class="n">${fmt.money(r.predicted_revenue, 2)}</td><td class="n">${fmt.money(r.predicted_margin, 2)}</td>` +
          `<td class="n">${r.predicted_roi === null ? "—" : fmt.pct(r.predicted_roi, 0)}</td>` +
          `<td><span class="badge ${r.profitable ? "good" : "bad"}">${r.profitable ? "Profitable" : "Loss"}</span></td></tr>`).join("") +
        "</tbody></table>";
    }

    async function run() {
      const b = read();
      store.set("predict", b);
      const go = $("#pf-go");
      go.disabled = true;
      go.textContent = "Predicting…";
      try {
        const res = await postJson("/predict", b);
        if (!current()) return;
        showResult(res, b);
        await compare(b);
      } catch (err) {
        if (!current()) return;
        const msgs = Array.isArray(err.messages) ? err.messages : err.messages ? [JSON.stringify(err.messages)] : [];
        $("#pr").innerHTML = `<p class="error"><strong>${esc(err.message)}</strong></p>${msgs.map((m) => `<p class="muted small">${esc(m)}</p>`).join("")}`;
        $("#cmp-chart").innerHTML = "";
        $("#cmp-table").innerHTML = "";
      } finally {
        go.disabled = false;
        go.textContent = "Predict";
      }
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); showHistory(); run(); });
    form.addEventListener("change", (e) => { if (e.target.tagName === "SELECT") showHistory(); });
    $("#pf-avg").addEventListener("click", async () => {
      const avg = await showHistory();
      if (!avg) return;
      form.elements.displays.value = avg.displays;
      form.elements.cost.value = avg.cost;
      run();
    });
    $("#cmp-dim").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-dim]");
      if (!btn) return;
      for (const b of $$("#cmp-dim button")) b.classList.toggle("on", b === btn);
      compare(read()).catch(() => {});
    });

    showHistory();
    if (info.model.available) await run();
  }

  // ------------------------------------------------------------------
  // Model
  // ------------------------------------------------------------------
  async function renderModel({ current }) {
    view.innerHTML = skeleton(3);
    const [rep, md] = await Promise.all([getJson("/model"), getText("/report/model").catch(() => null)]);
    if (!current()) return;
    const results = rep.results;
    const modelKey = Object.keys(results).find((k) => k.startsWith("Model:"));
    const best = results[modelKey];
    const t = tokens();
    const names = Object.keys(results);
    const shortName = (k) => k.replace(/^Baseline:\s*/, "Rate card: ").replace(/^Model:\s*/, "");

    const cv = rep.metadata.model_comparison || {};
    view.innerHTML = `
      <div class="kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">
        ${kpi("Selected model", rep.selected_model, `target: ${rep.target}`)}
        ${kpi("Test R² (log revenue)", fmt.dec(best.revenue.r2_log, 3), "share of variation explained")}
        ${kpi("Average error per row", fmt.money(best.revenue.mae_usd, 2), "mean absolute error, test period")}
        ${kpi("Total forecast vs actual", fmt.dec(best.revenue.total_predicted_vs_actual, 2) + "×", "1.00× = exact total")}
        ${kpi("Profitable-call F1", fmt.dec(best.profitability.f1, 3), `precision ${fmt.dec(best.profitability.precision, 2)} · recall ${fmt.dec(best.profitability.recall, 2)}`)}
        ${kpi("Margin captured", fmt.pct(best.margin.share_of_oracle, 0), "of the margin available with perfect foresight")}
      </div>
      <p class="muted small" style="margin:0">Trained on ${fmt.int(rep.train_rows)} rows before ${fmt.date(rep.test_start)}; tested once on the ${fmt.int(rep.test_rows)} rows from ${fmt.date(rep.test_start)} onward. Inputs are only things known before the ads run: ${rep.features.map((f) => `<code>${esc(f)}</code>`).join(", ")}.</p>
      <div class="grid two">
        <div class="card"><div class="card-head"><div><h2>Margin captured in the test period</h2><p class="muted small">Buying only the rows each approach calls profitable, as a share of perfect foresight. Grey bars are rate-card baselines.</p></div></div><div id="m-oracle" class="chart short"></div></div>
        <div class="card"><div class="card-head"><div><h2>Average error per row</h2><p class="muted small">Mean absolute revenue error in dollars (lower is better).</p></div></div><div id="m-mae" class="chart short"></div></div>
      </div>
      <div class="card">
        <div class="card-head"><div><h2>Model vs rate-card baselines</h2><p class="muted small">Rate card = displays × historical revenue per display for the segment.</p></div></div>
        <div class="table-wrap"><table><thead><tr><th>Approach</th><th class="n">R² (log)</th><th class="n">MAE ($)</th><th class="n">Total vs actual</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">ROC-AUC</th><th class="n">Margin if followed</th><th class="n">Share of perfect</th></tr></thead><tbody>
          ${names.map((k) => { const r = results[k]; return `<tr class="${k === modelKey ? "best" : ""}"><td>${esc(shortName(k))}</td>
            <td class="n">${fmt.dec(r.revenue.r2_log, 3)}</td><td class="n">${fmt.money(r.revenue.mae_usd, 2)}</td><td class="n">${fmt.dec(r.revenue.total_predicted_vs_actual, 2)}×</td>
            <td class="n">${fmt.dec(r.profitability.precision, 3)}</td><td class="n">${fmt.dec(r.profitability.recall, 3)}</td><td class="n">${fmt.dec(r.profitability.f1, 3)}</td><td class="n">${fmt.dec(r.profitability.roc_auc, 3)}</td>
            <td class="n">${fmt.money(r.margin.buy_predicted_profitable, 0)}</td><td class="n">${fmt.pct(r.margin.share_of_oracle, 0)}</td></tr>`; }).join("")}
          </tbody></table></div>
        <p class="note">Buying every row in the test period would have made ${fmt.money(best.margin.buy_everything, 0)}; perfect foresight ${fmt.money(best.margin.oracle, 0)}.</p>
      </div>
      ${Object.keys(cv).length ? `<div class="card"><div class="card-head"><div><h2>Model selection</h2><p class="muted small">5-fold time-series cross-validation inside the training period, then the held-out test.</p></div></div>
        <div class="table-wrap"><table><thead><tr><th>Model</th><th class="n">CV R² (mean ± sd)</th><th class="n">Test R²</th><th class="n">MAE (log)</th><th class="n">RMSE (log)</th><th class="n">Training time</th></tr></thead><tbody>
        ${Object.entries(cv).map(([k, r]) => `<tr class="${k === rep.selected_model ? "best" : ""}"><td>${esc(k)}${k === rep.selected_model ? ' <span class="badge plain">selected</span>' : ""}</td><td class="n">${fmt.dec(r.cv_mean, 3)} ± ${fmt.dec(r.cv_std, 3)}</td><td class="n">${fmt.dec(r.r2_score, 3)}</td><td class="n">${fmt.dec(r.mae, 3)}</td><td class="n">${fmt.dec(r.rmse, 3)}</td><td class="n">${isNum(r.train_time_s) ? Math.round(r.train_time_s) + " s" : "—"}</td></tr>`).join("")}
        </tbody></table></div></div>` : ""}
      ${md ? `<details class="card report"><summary>Full model report</summary><div class="prose">${Markdown.render(md.replace(/^#\s+.*\n/, ""), { resolveImage: resolveFigure })}</div></details>` : ""}`;

    // Chart labels drop the "Rate card:" prefix; the card subtitle explains the baselines
    const chartName = (k) => shortName(k).replace(/^Rate card: segment rate \((.*)\)$/, "Segment rate, $1").replace(/^Rate card: global rate$/, "Global rate");
    const bar = (el, key, f, tick) => {
      const vals = names.map((k) => key(results[k]));
      const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1;
      plot(el, [{
        type: "bar", orientation: "h", y: names.map(chartName), x: vals,
        marker: { color: names.map((k) => (k === modelKey ? t.series[0] : t.axis)), cornerradius: 4 },
        text: vals.map(f), textposition: "outside", cliponaxis: false, textfont: { color: t.ink2, size: 11 },
        customdata: names.map((k, i) => `<b>${esc(shortName(k))}</b><br>${f(vals[i])}`), hovertemplate: "%{customdata}<extra></extra>",
      }], {
        xaxis: { tickformat: tick, zeroline: true, nticks: 5, range: [lo < 0 ? lo - span * 0.32 : 0, hi + span * 0.15] },
        yaxis: { type: "category", autorange: "reversed", showgrid: false, tickfont: { color: t.ink2, size: 12 } },
      });
    };
    bar($("#m-oracle"), (r) => r.margin.share_of_oracle, (v) => fmt.pct(v, 0), ".0%");
    bar($("#m-mae"), (r) => r.revenue.mae_usd, (v) => fmt.money(v, 2), "$.2f");
    bindLightbox(view);
  }

  // ------------------------------------------------------------------
  // Data
  // ------------------------------------------------------------------
  const DATA_LABELS = {
    date: "Date", campaign_number: "Campaign", user_engagement: "Engagement", banner: "Banner", placement: "Placement",
    displays: "Displays", clicks: "Clicks", cost: "Cost", revenue: "Revenue", firm_margin: "Margin",
    post_click_conversions: "Conversions", post_click_sales_amount: "Sales",
  };
  const DATA_FORMAT = {
    displays: fmt.int, clicks: fmt.int, post_click_conversions: fmt.int,
    cost: (v) => fmt.money(v, 2), revenue: (v) => fmt.money(v, 2), firm_margin: (v) => fmt.money(v, 2),
    post_click_sales_amount: (v) => fmt.money(v, 2),
  };

  async function renderData({ current }) {
    const ui = { page: 1, page_size: store.get("dt-size", 50), sort: "date", order: "asc" };
    view.innerHTML = `
      <div class="card">
        <div class="card-head">
          <div><h2>Rows</h2><p class="muted small" id="dt-sub"></p></div>
          <div class="controls">
            <label>Rows per page<select id="dt-size">${options([25, 50, 100, 250].map((n) => [String(n), String(n)]), String(ui.page_size))}</select></label>
            <a class="btn ghost" id="dt-csv" download><svg viewBox="0 0 24 24"><path d="M12 3v12M7 10l5 5 5-5M5 21h14"/></svg>Download CSV</a>
          </div>
        </div>
        <div class="table-wrap" id="dt-table">${skeleton(1)}</div>
        <div class="pager"><span class="muted small" id="dt-range"></span>
          <div class="controls"><button class="btn ghost small" id="dt-prev" type="button">← Previous</button><span class="small" id="dt-page"></span><button class="btn ghost small" id="dt-next" type="button">Next →</button></div>
        </div>
      </div>`;

    $("#dt-csv").href = buildUrl("/data.csv", activeFilters());
    const draw = async () => {
      const res = await getJson("/data", { ...activeFilters(), ...ui });
      if (!current()) return;
      ui.page = res.page;
      $("#dt-sub").textContent = `${fmt.int(res.total)} rows · ${filterSummary()}`;
      const head = res.columns.map((c) => {
        const numCol = c in DATA_FORMAT;
        const arrow = res.sort === c ? (res.descending ? " ↓" : " ↑") : "";
        return `<th class="sortable${numCol ? " n" : ""}${res.sort === c ? " sorted" : ""}" data-col="${c}">${DATA_LABELS[c] || c}${arrow}</th>`;
      }).join("");
      const body = res.rows.map((r) => `<tr>${res.columns.map((c) => c in DATA_FORMAT
        ? `<td class="n">${DATA_FORMAT[c](r[c])}</td>` : `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("");
      $("#dt-table").innerHTML = res.rows.length
        ? `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`
        : '<p class="empty">No rows match the current filters.</p>';
      const first = res.total ? (res.page - 1) * res.page_size + 1 : 0;
      $("#dt-range").textContent = `Rows ${fmt.int(first)}–${fmt.int(Math.min(res.total, res.page * res.page_size))} of ${fmt.int(res.total)}`;
      $("#dt-page").textContent = `Page ${res.page} of ${res.pages}`;
      $("#dt-prev").disabled = res.page <= 1;
      $("#dt-next").disabled = res.page >= res.pages;
      for (const th of $$("#dt-table th")) {
        th.addEventListener("click", () => {
          const c = th.dataset.col;
          if (ui.sort === c) ui.order = ui.order === "asc" ? "desc" : "asc";
          else { ui.sort = c; ui.order = c in DATA_FORMAT ? "desc" : "asc"; }
          ui.page = 1;
          draw().catch((e) => showError(view, e));
        });
      }
    };
    $("#dt-size").addEventListener("change", (e) => { ui.page_size = +e.target.value; ui.page = 1; store.set("dt-size", ui.page_size); draw(); });
    $("#dt-prev").addEventListener("click", () => { ui.page -= 1; draw(); });
    $("#dt-next").addEventListener("click", () => { ui.page += 1; draw(); });
    await draw();
  }

  // ------------------------------------------------------------------
  // API reference
  // ------------------------------------------------------------------
  const ENDPOINT_DOCS = {
    "/health": ["Liveness check, plus whether the dataset and trained model are available.", "/health"],
    "/api/info": ["Dataset summary, valid filter values and model metadata.", "/api/info"],
    "/metrics": ["Overall totals and KPIs. Filters: campaign_number, user_engagement, banner, placement, start, end.", "/metrics?campaign_number=camp 1"],
    "/metrics/<dimension>": ["Totals and KPIs per campaign_number, user_engagement, banner, placement, month, weekday or date.", "/metrics/banner"],
    "/pivot?rows=<dimension>&cols=<dimension>": ["Totals and KPIs for every pair of two dimensions.", "/pivot?rows=banner&cols=placement"],
    "/data": ["Row-level data, paginated: page, page_size (max 500), sort, order (asc | desc), plus the filters.", "/data?page=1&page_size=5&sort=revenue&order=desc"],
    "/data.csv": ["The filtered rows as a CSV download.", "/data.csv?placement=ghi"],
    "/insights": ["Every EDA finding as JSON (reports/eda_findings.json).", "/insights"],
    "/model": ["Held-out evaluation of the model and the baselines (reports/model_evaluation.json).", "/model"],
    "/report/<eda|model>": ["The written reports as Markdown.", "/report/eda"],
    "/predict": ["Predicted revenue, margin, ROI and a profitable flag for one planned buy (object) or up to 1,000 (list).", null],
  };

  async function renderApi() {
    const info = state.info;
    const rows = info.endpoints.map((e) => {
      const [method, path] = e.trim().split(/\s+/);
      const [desc, example] = ENDPOINT_DOCS[path] || ["", null];
      return `<div class="endpoint"><span class="method ${method === "POST" ? "post" : ""}">${method}</span><div>
        <code>${esc(path)}</code><div class="muted small">${esc(desc)}</div>
        ${example ? `<a class="small" href="${esc(example)}" target="_blank" rel="noopener">Open ${esc(example)}</a>` : ""}</div></div>`;
    }).join("");
    const body = JSON.stringify({ date: info.dataset.end, campaign_number: "camp 1", user_engagement: "High", banner: "240 x 400", placement: "ghi", displays: 30000, cost: 20 });
    view.innerHTML = `
      <div class="card"><div class="card-head"><div><h2>Endpoints</h2><p class="muted small">API version ${esc(info.api_version)} · all responses are JSON unless noted.</p></div></div>${rows}</div>
      <div class="card"><h2>Example: predict a buy</h2>
        <pre>curl -X POST ${esc(location.origin)}/predict \\
  -H "Content-Type: application/json" \\
  -d '${esc(body)}'</pre></div>`;
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------
  function initTheme() {
    $("#theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("finlatics-theme", root.dataset.theme); } catch (e) { /* ignore */ }
      render();
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (!document.documentElement.dataset.theme) render();
    });
  }

  async function boot() {
    initTheme();
    if (typeof Plotly === "undefined") {
      $("#status-banner").hidden = false;
      $("#status-banner").textContent = "Charts could not load: the plotly package is not installed. Run pip install -r requirements.txt.";
      window.Plotly = { react: (el) => { el.innerHTML = '<p class="empty">Chart unavailable</p>'; } };
    }
    try {
      const health = await getJson("/health");
      if (!health.data_available) {
        $("#status-banner").hidden = false;
        $("#status-banner").innerHTML = "The dataset was not found. Copy <code>online_advertising_performance_data.csv</code> into <code>data/raw/</code> and restart <code>python app.py</code>.";
        view.innerHTML = "";
        return;
      }
      state.info = await getJson("/api/info");
    } catch (err) {
      showError(view, err);
      return;
    }
    const d = state.info.dataset;
    $("#dataset-meta").textContent = `${fmt.int(d.rows)} rows · ${fmt.date(d.start)} – ${fmt.date(d.end)}`;
    for (const el of $$('#filters input[type="date"]')) { el.min = d.start; el.max = d.end; }
    initFilters(state.info.filters);
    window.addEventListener("hashchange", () => { window.scrollTo(0, 0); render(); });
    render();
  }

  boot();
})();
