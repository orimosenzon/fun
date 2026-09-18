/* עזרי גרפים (Chart.js) ושחזור משחק לדו"חות. כל גרף נבנה מטוקני הצבע של הדף ונבנה מחדש כשמצב התצוגה משתנה. */
(function () {
  const builders = [];

  function tokens() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    return {
      s1: g("--series-1"), s2: g("--series-2"), s3: g("--series-3"), s4: g("--series-4"), de: g("--de-emph"),
      ink: g("--ink"), ink2: g("--ink-2"), muted: g("--muted"), grid: g("--grid"), axis: g("--axis"),
      surface: g("--surface"),
      seq: [g("--seq-100"), g("--seq-250"), g("--seq-400"), g("--seq-550"), g("--seq-700")],
    };
  }
  const colorOf = (t, c) => (c && c.startsWith("seq") ? t.seq[+c[3]] : ({ s1: t.s1, s2: t.s2, s3: t.s3, s4: t.s4, de: t.de }[c] || c));
  const fmt = (v, d) => {
    if (v == null || isNaN(v)) return "";
    const n = Number(v);
    const digits = d != null ? d : Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 1 ? 1 : 4;
    return n.toLocaleString("he-IL", { maximumFractionDigits: digits });
  };

  // תווית ישירה בקצה הקו (רק לסדרות שביקשו)
  const directLabels = {
    id: "directLabels",
    afterDatasetsDraw(chart) {
      const t = tokens();
      const ctx = chart.ctx;
      ctx.save();
      ctx.font = "12px system-ui, -apple-system, 'Segoe UI', sans-serif";
      ctx.fillStyle = t.ink2;
      ctx.textBaseline = "middle";
      chart.data.datasets.forEach((ds, i) => {
        if (!ds.directLabel || !chart.isDatasetVisible(i)) return;
        const meta = chart.getDatasetMeta(i);
        const pts = meta.data.filter((p) => p && !isNaN(p.y));
        if (!pts.length) return;
        const last = pts[pts.length - 1];
        ctx.textAlign = "left";
        const x = Math.min(last.x + 6, chart.chartArea.right - 4);
        let y = last.y + (ds.directLabelOffset || 0);
        ctx.fillText(ds.label, x, y);
      });
      ctx.restore();
    },
  };

  function baseOptions(t, cfg) {
    const yFmt = cfg.yFmt || ((v) => fmt(v));
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: cfg.mode || "index", intersect: false },
      layout: { padding: { right: cfg.padRight == null ? 8 : cfg.padRight, top: 8 } },
      plugins: {
        legend: {
          display: cfg.legend !== false,
          position: "top",
          align: "end",
          rtl: true,
          labels: { boxWidth: 14, boxHeight: 3, color: t.ink2, usePointStyle: false, font: { size: 13 } },
        },
        tooltip: {
          rtl: true,
          textDirection: "rtl",
          backgroundColor: t.surface,
          titleColor: t.ink,
          bodyColor: t.ink2,
          borderColor: t.axis,
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: (items) => (cfg.tooltipTitle ? cfg.tooltipTitle(items[0]) : items[0].label),
            label: (item) => `${item.dataset.label}: ${cfg.tooltipValue ? cfg.tooltipValue(item) : yFmt(item.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          type: cfg.xLog ? "logarithmic" : cfg.xType || "linear",
          title: { display: !!cfg.xTitle, text: cfg.xTitle, color: t.muted, font: { size: 12 } },
          grid: { color: t.grid, drawTicks: false },
          border: { color: t.axis },
          ticks: { color: t.muted, maxTicksLimit: 9, callback: cfg.xTick || ((v) => fmt(v, 2)) },
        },
        y: {
          beginAtZero: cfg.yZero !== false,
          title: { display: !!cfg.yTitle, text: cfg.yTitle, color: t.muted, font: { size: 12 } },
          grid: { color: t.grid, drawTicks: false },
          border: { display: false },
          ticks: { color: t.muted, maxTicksLimit: 7, callback: (v) => yFmt(v) },
          max: cfg.yMax,
        },
      },
    };
  }

  function tableView(container, columns, rows) {
    const d = document.createElement("details");
    d.className = "table";
    const s = document.createElement("summary");
    s.textContent = "תצוגת טבלה";
    d.appendChild(s);
    const tbl = document.createElement("table");
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr>" + columns.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    tbl.appendChild(thead);
    const tb = document.createElement("tbody");
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = r.map((v, i) => `<td class="${i ? "num" : ""}">${typeof v === "number" ? fmt(v, 3) : v}</td>`).join("");
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    d.appendChild(tbl);
    container.appendChild(d);
  }

  function mount(id, build, table) {
    const wrap = document.getElementById(id);
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    let chart = null;
    const rebuild = () => {
      if (chart) chart.destroy();
      chart = build(canvas.getContext("2d"), tokens());
    };
    rebuild();
    builders.push(rebuild);
    if (table) tableView(wrap.parentElement, table.columns, table.rows);
  }

  // גרף קווים. cfg.series: [{label, x?, y, color, width, direct, dash}]
  function lineChart(id, cfg) {
    mount(id, (ctx, t) => {
      const datasets = cfg.series.map((s) => ({
        label: s.label,
        data: (s.x || cfg.x).map((x, i) => ({ x, y: s.y[i] })),
        borderColor: colorOf(t, s.color),
        backgroundColor: colorOf(t, s.color),
        borderWidth: s.width || 2,
        borderDash: s.dash || [],
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHitRadius: 12,
        tension: 0,
        spanGaps: true,
        directLabel: !!s.direct,
        directLabelOffset: s.directOffset || 0,
      }));
      (cfg.refs || []).forEach((r) => {
        const xs = cfg.x;
        datasets.push({
          label: r.label,
          data: [{ x: xs[0], y: r.value }, { x: xs[xs.length - 1], y: r.value }],
          borderColor: colorOf(t, r.color || "de"),
          backgroundColor: colorOf(t, r.color || "de"),
          borderWidth: 1.5,
          pointRadius: 0,
          pointHitRadius: 6,
          directLabel: true,
          directLabelOffset: r.offset || 0,
        });
      });
      const opts = baseOptions(t, Object.assign({ padRight: 90 }, cfg));
      return new Chart(ctx, { type: "line", data: { datasets }, options: opts, plugins: [directLabels] });
    }, cfg.table);
  }

  // גרף עמודות. cfg.labels, cfg.series: [{label, y, color}]
  function barChart(id, cfg) {
    mount(id, (ctx, t) => {
      const datasets = cfg.series.map((s) => ({
        label: s.label,
        data: s.y,
        backgroundColor: Array.isArray(s.color) ? s.color.map((c) => colorOf(t, c)) : colorOf(t, s.color),
        borderRadius: cfg.stacked ? 2 : 4,
        borderSkipped: "start",
        // רווח של 2 פיקסלים בצבע המשטח בין מקטעים בערימה
        borderColor: cfg.stacked ? t.surface : undefined,
        borderWidth: cfg.stacked ? { top: 2, left: 0, right: 0, bottom: 0 } : 0,
        barPercentage: cfg.series.length > 1 ? 0.85 : 0.6,
        categoryPercentage: 0.8,
        maxBarThickness: 48,
      }));
      const opts = baseOptions(t, Object.assign({ xType: "category", mode: "index", legend: cfg.series.length > 1 }, cfg));
      opts.scales.x.type = "category";
      opts.scales.x.grid.display = false;
      delete opts.scales.x.ticks.callback;
      opts.scales.x.ticks.maxRotation = 0;
      opts.scales.x.ticks.autoSkip = cfg.autoSkip !== false;
      if (cfg.stacked) { opts.scales.x.stacked = true; opts.scales.y.stacked = true; }
      return new Chart(ctx, { type: "bar", data: { labels: cfg.labels, datasets }, options: opts });
    }, cfg.table);
  }

  // פיזור (נקודות). cfg.series: [{label, points:[{x,y}], color}]
  function scatterChart(id, cfg) {
    mount(id, (ctx, t) => {
      const datasets = cfg.series.map((s) => ({
        label: s.label,
        data: s.points,
        backgroundColor: colorOf(t, s.color) + "aa",
        borderColor: t.surface,
        borderWidth: 1,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointHitRadius: 10,
      }));
      const opts = baseOptions(t, Object.assign({ mode: "nearest", legend: cfg.series.length > 1 }, cfg));
      opts.interaction.intersect = true;
      opts.plugins.tooltip.callbacks.title = () => "";
      opts.plugins.tooltip.callbacks.label = (item) => `${item.dataset.label}: ${cfg.xTitle} ${fmt(item.parsed.x)}, ${cfg.yTitle} ${fmt(item.parsed.y)}`;
      return new Chart(ctx, { type: "scatter", data: { datasets }, options: opts });
    }, cfg.table);
  }

  // שחזור משחק מוקלט
  function replay(id, game, opts) {
    const root = document.getElementById(id);
    root.classList.add("replay");
    const board = document.createElement("div");
    board.className = "board";
    const cells = [];
    for (let i = 0; i < 16; i++) {
      const c = document.createElement("div");
      c.className = "cell";
      board.appendChild(c);
      cells.push(c);
    }
    const ctl = document.createElement("div");
    ctl.className = "controls";
    const names = ["למעלה", "ימינה", "למטה", "שמאלה"];
    const arrows = ["↑", "→", "↓", "←"];
    ctl.innerHTML = `
      <div class="stat"><span>מהלך</span><b class="mv"></b></div>
      <div class="stat"><span>ניקוד</span><b class="sc"></b></div>
      <div class="stat"><span>הפעולה האחרונה</span><b class="ac"></b></div>
      <div class="stat"><span>אריח מקסימלי במשחק</span><b>${fmt(game.max_tile)}</b></div>
      <div class="stat"><span>ניקוד סופי</span><b>${fmt(game.score)}</b></div>
      <input type="range" min="0" max="${game.frames.length - 1}" value="0">
      <div><button class="play">▶ נגן</button><button class="step">צעד</button><button class="end">לסוף</button>
      <label style="margin-inline-start:8px">מהירות <select class="spd"><option value="250">איטי</option><option value="80" selected>רגיל</option><option value="20">מהיר</option></select></label></div>`;
    root.appendChild(board);
    root.appendChild(ctl);
    const range = ctl.querySelector("input");
    const mv = ctl.querySelector(".mv"), sc = ctl.querySelector(".sc"), ac = ctl.querySelector(".ac");
    const playBtn = ctl.querySelector(".play");
    let idx = 0, timer = null;
    function show(i) {
      idx = i;
      const f = game.frames[i];
      f.board.forEach((row, r) => row.forEach((v, c) => {
        const el = cells[r * 4 + c];
        const val = v ? 2 ** v : 0;
        el.textContent = val ? val : "";
        el.className = "cell" + (val ? (val > 2048 ? " tsuper" : " t" + val) : "");
      }));
      mv.textContent = `${fmt(i)} / ${fmt(game.frames.length - 1)}`;
      sc.textContent = fmt(f.score);
      ac.textContent = f.action == null ? "התחלה" : `${arrows[f.action]} ${names[f.action]}`;
      range.value = i;
    }
    function stop() { if (timer) clearInterval(timer); timer = null; playBtn.textContent = "▶ נגן"; }
    function play() {
      if (timer) return stop();
      if (idx >= game.frames.length - 1) idx = 0;
      playBtn.textContent = "⏸ עצור";
      timer = setInterval(() => { if (idx >= game.frames.length - 1) return stop(); show(idx + 1); }, +ctl.querySelector(".spd").value);
    }
    playBtn.onclick = play;
    ctl.querySelector(".step").onclick = () => { stop(); if (idx < game.frames.length - 1) show(idx + 1); };
    ctl.querySelector(".end").onclick = () => { stop(); show(game.frames.length - 1); };
    ctl.querySelector(".spd").onchange = () => { if (timer) { stop(); play(); } };
    range.oninput = () => { stop(); show(+range.value); };
    show(opts && opts.start === "end" ? game.frames.length - 1 : 0);
  }

  function init() {
    Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', Arial, sans-serif";
    Chart.defaults.font.size = 12;
    const mq = matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", () => builders.forEach((b) => b()));
  }

  window.R = { lineChart, barChart, scatterChart, replay, init, fmt };
})();
