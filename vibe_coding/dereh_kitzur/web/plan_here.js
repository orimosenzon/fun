/* "מה מתוכנן כאן?" - every plan that applies to one point on the ground.
 *
 * Why this exists
 * ---------------
 * Ori asked, on 24/9/2026, what is planned for the open space behind his house
 * and could not find out from any site: not the council's GIS, not govmap, not
 * קווים כחולים. The information was public the whole time. Three things hid it:
 *
 *   * the plan that governs his block is called "שינויים בתכנית מרכז המושבה".
 *     Its name carries no street, no neighbourhood, nothing anybody would type
 *     into a search box;
 *   * what applies to one point is the union of several plans of different
 *     dates, the later beating the earlier, and no public site works that out
 *     for you;
 *   * the boundary and the land use are different layers, and the boundary is
 *     the one that is on by default.
 *
 * So: tap a point, get every plan over it newest first, the designation in
 * force, the block and parcel, how many flats each plan adds, and - the only
 * sentence anybody can act on - whether objections are still open and until
 * when.
 *
 * Live, not built
 * ---------------
 * Everything here is asked at tap time. The two services allow it from a
 * browser, which was checked before any of this was written:
 *
 *     ags.iplan.gov.il      Access-Control-Allow-Origin: <the asking origin>
 *     open.govmap.gov.il    Access-Control-Allow-Origin: *
 *
 * That is worth more than a nightly build would be. A plan's objection window
 * is sixty days and the answer has to be today's, not the last run's; and the
 * parcels alone are ten thousand polygons, which is why build_cadastre.py
 * stops at blocks. Nothing here touches web/data or the data repo.
 *
 * What it cannot say
 * ------------------
 * Building permits. They are not in any national open service - data.gov.il has
 * no such dataset, and the local committee's own system is not open. A
 * developer building under an approved plan files no new plan, so this panel
 * stays silent while the diggers arrive. The panel says so out loud, because a
 * resident who reads "nothing planned here" and concludes "nothing will be
 * built here" has been misled by us rather than informed.
 *
 * Ownership, likewise: designation is not ownership (see build_public.py).
 */
const PlanHere = (() => {
  'use strict';

  const XPLAN = 'https://ags.iplan.gov.il/arcgisiplan/rest/services'
    + '/PlanningPublic/Xplan/MapServer';
  const PLANS = XPLAN + '/1/query';       // קוים כחולים: one polygon per plan
  const LANDUSE = XPLAN + '/4/query';     // יעודי קרקע: the cells inside them
  const WFS = 'https://open.govmap.gov.il/geoserver/opendata/ows';
  const XPLAN_SITE = 'https://ags.iplan.gov.il/xplan/';
  /* The beginner's guide to all of this, on the moshava's own wiki. Written
   * because "תכנית מתאר מקומית" and "שצ"פ" are the whole answer to somebody who
   * already knows what they mean and none of it to anybody else. */
  const GUIDE = 'https://pardespedia.info/wiki/' +
    encodeURIComponent('תכנון ובנייה בפרדס חנה-כרכור');

  const PLAN_FIELDS = [
    'pl_number', 'pl_name', 'pl_objectives', 'entity_subtype_desc',
    'internet_short_status', 'station_desc', 'plan_charactor_name',
    'pl_url', 'pl_area_dunam', 'pl_landuse_string', 'ja_concat',
    'pl_date_8', 'pl_date7', 'pl_date_advertise',
    'pl_last_deposit_date', 'pl_rejection_date',
    'quantity_delta_120', 'quantity_delta_125', 'quantity_delta_75',
    'quantity_delta_60', 'quantity_delta_80'
  ].join(',');

  const USE_FIELDS = 'mavat_name,num,pl_number,pl_name,shape_area,station_desc';

  /* The quantity columns are numbered by מבא"ת code; the names come from the
   * service's own Hebrew aliases (MapServer/1?f=pjson). build_plans.py carries
   * the same table and the note explaining how it was pinned down. */
  const QUANTITIES = [
    ['quantity_delta_120', (n) => (n === 1 ? 'יחידת דיור אחת' : fmt(n) + ' יחידות דיור')],
    ['quantity_delta_125', (n) => fmt(n) + ' מ"ר שטחי מגורים'],
    ['quantity_delta_75', (n) => fmt(n) + ' מ"ר מסחר'],
    ['quantity_delta_60', (n) => fmt(n) + ' מ"ר תעסוקה'],
    ['quantity_delta_80', (n) => fmt(n) + ' מ"ר מבני ציבור']
  ];

  /* ---------- small helpers ---------- */

  const fmt = (n) => Math.round(n).toLocaleString('he-IL');

  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const el = (id) => document.getElementById(id);

  /** An epoch-milliseconds field as a Hebrew date. */
  function date(stamp) {
    if (!stamp) return null;
    const d = new Date(stamp);
    if (isNaN(d)) return null;
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()}`;
  }

  /** Whole days from today. Negative is in the past. */
  function daysTo(stamp) {
    if (!stamp) return null;
    const day = 86400000;
    return Math.ceil((stamp - Date.now()) / day);
  }

  /** The day objections shut, whichever of the two fields carries it.
   *
   *  While a plan is on show only `pl_last_deposit_date` (תאריך אחרון להפקדה)
   *  is filled, sixty-three days after the newspaper notice. Once objections
   *  have been registered and the plan moves on, `pl_rejection_date` (תאריך
   *  אחרון להתנגדויות) holds the authoritative date. Prefer the recorded one. */
  const shutsOn = (a) => a.pl_rejection_date || a.pl_last_deposit_date || null;

  /** The date a plan became law, for ordering "what beats what". */
  const inForceOn = (a) => a.pl_date_8 || a.pl_date7 || null;

  /** What the plan changes, in words, or '' when it changes no quantity. */
  function adds(a) {
    const parts = [];
    QUANTITIES.forEach(([field, say]) => {
      const n = Number(a[field]);
      if (!n) return;
      // A minus in front of Hebrew reads as a dash, so a reduction is spelled
      // out. One plan here turned 6,936 m² of commerce into flats.
      parts.push(n > 0 ? say(n) : 'גריעה של ' + say(-n));
    });
    if (!parts.length) return '';
    if (parts.length === 1) return parts[0];
    return parts.slice(0, -1).join(', ') + ' ו' + parts[parts.length - 1];
  }

  /* ---------- asking ---------- */

  /** WGS84 to Web Mercator. The parcel layer's own CRS is EPSG:3857, and a CQL
   *  filter is read in the layer's CRS whatever `srsName` asks the answer to be
   *  - degrees there match nothing and come back empty. */
  function mercator(lng, lat) {
    const x = lng * 20037508.34 / 180;
    const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) * 20037508.34 / Math.PI;
    return x.toFixed(1) + ' ' + y.toFixed(1);
  }

  /* A government service behind a WAF, and it does go quiet: on the afternoon
   * this was written it answered every request for an hour and then stopped
   * answering this machine at all, for both GET and POST. `fetch` on its own
   * would wait for the browser's own timeout, which is measured in minutes and
   * looks to the user exactly like an app that has hung. Fifteen seconds and a
   * sentence saying what happened is the honest version. */
  const TIMEOUT = 15000;

  async function withTimeout(url, init) {
    const stop = new AbortController();
    const timer = setTimeout(() => stop.abort(), TIMEOUT);
    try {
      return await fetch(url, Object.assign({ signal: stop.signal }, init));
    } finally {
      clearTimeout(timer);
    }
  }

  async function ask(url, params) {
    const body = new URLSearchParams(params).toString();
    // POST: a point geometry plus twenty field names is longer than some
    // proxies like in a query string, and the service takes either. It stays a
    // CORS-simple request - form encoding needs no preflight - so the browser
    // asks once rather than twice.
    const res = await withTimeout(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body
    });
    if (!res.ok) throw new Error('שירות התכנון החזיר ' + res.status);
    const doc = await res.json();
    if (doc.error) throw new Error(doc.error.message || 'שגיאה בשירות התכנון');
    return doc.features || [];
  }

  const atPoint = (lngLat) => ({
    geometry: JSON.stringify({
      x: lngLat.lng, y: lngLat.lat, spatialReference: { wkid: 4326 }
    }),
    geometryType: 'esriGeometryPoint',
    inSR: '4326',
    spatialRel: 'esriSpatialRelIntersects',
    returnGeometry: 'false',
    f: 'json'
  });

  /** Block and parcel under the point, or null. Its own try/catch: the cadastre
   *  is a different service and a resident asking about plans should still get
   *  the plans when govmap is down. */
  async function parcel(lngLat) {
    try {
      const url = WFS + '?' + new URLSearchParams({
        service: 'WFS', version: '2.0.0', request: 'GetFeature',
        typeName: 'opendata:PARCEL_ALL', outputFormat: 'application/json',
        srsName: 'EPSG:4326', count: '1',
        CQL_FILTER: `INTERSECTS(the_geom,POINT(${mercator(lngLat.lng, lngLat.lat)}))`
      });
      const res = await withTimeout(url);
      if (!res.ok) return null;
      const doc = await res.json();
      const p = doc.features && doc.features[0] && doc.features[0].properties;
      return p ? { gush: p.GUSH_NUM, parcel: p.PARCEL, area: p.LEGAL_AREA,
                   locality: p.LOCALITY_N } : null;
    } catch (err) {
      return null;
    }
  }

  /** Everything about one point, in one go. */
  async function lookUp(lngLat) {
    const [plans, uses, plot] = await Promise.all([
      ask(PLANS, Object.assign(atPoint(lngLat), { outFields: PLAN_FIELDS })),
      ask(LANDUSE, Object.assign(atPoint(lngLat), { outFields: USE_FIELDS })),
      parcel(lngLat)
    ]);

    const byNumber = {};
    plans.forEach((f) => { byNumber[f.attributes.pl_number] = f.attributes; });

    // The designation in force: among the cells here, the one belonging to the
    // latest plan that is actually law. Same rule as build_public.py's
    // supersession, narrowed to a single point - which makes it exact rather
    // than the centre-point approximation that file has to settle for.
    let live = null;
    uses.forEach((f) => {
      const a = f.attributes;
      const plan = byNumber[a.pl_number];
      if (!plan || inForceOn(plan) == null) return;
      if (!live || inForceOn(plan) > inForceOn(byNumber[live.pl_number])) live = a;
    });

    const rows = plans.map((f) => f.attributes).sort((x, y) => {
      // Newest first, and anything still moving above anything finished: a
      // plan that can still be objected to is the news on this point.
      const open = (a) => (daysTo(shutsOn(a)) >= 0 ? 1 : 0);
      if (open(y) !== open(x)) return open(y) - open(x);
      return (inForceOn(y) || y.last_update_date || 0)
        - (inForceOn(x) || x.last_update_date || 0);
    });

    return { plans: rows, uses, live, plot, at: lngLat };
  }

  /* ---------- the sheet ---------- */

  function planRow(a) {
    const number = esc(a.pl_number || '');
    const shut = shutsOn(a);
    const left = daysTo(shut);
    const open = shut && left >= 0;
    const force = date(inForceOn(a));
    const change = adds(a);

    const facts = [];
    if (a.entity_subtype_desc) facts.push(esc(a.entity_subtype_desc));
    if (force) facts.push('בתוקף מ-' + force);
    else if (a.station_desc) facts.push(esc(a.station_desc));
    if (a.pl_area_dunam) {
      const d = a.pl_area_dunam;
      facts.push((d >= 10 ? Math.round(d) : Math.round(d * 10) / 10) + ' דונם');
    }

    const objectives = (a.pl_objectives || '').replace(/\^/g, ' ').trim();
    // A small plan's objectives are often its own name again.
    const blurb = objectives && objectives.replace(/[ .]+$/, '')
      !== (a.pl_name || '').replace(/[ .]+$/, '') ? objectives : '';

    return `<li class="ph-plan${open ? ' ph-open' : ''}">
      <div class="ph-plan-head">
        <b>${esc(a.pl_name || number)}</b>
        <span class="ph-num">${number}</span>
      </div>
      <p class="ph-facts">${facts.join(' · ')}</p>
      ${change ? `<p class="ph-adds">מוסיפה ${esc(change)}</p>` : ''}
      ${open ? `<p class="ph-deadline">אפשר להגיש התנגדות עד ${date(shut)},
        ${left === 0 ? 'כלומר היום' : left === 1 ? 'כלומר מחר' : 'בעוד ' + left + ' ימים'}</p>`
        : shut ? `<p class="ph-closed">חלון ההתנגדויות נסגר ב-${date(shut)}</p>` : ''}
      ${blurb ? `<p class="ph-obj">${esc(blurb)}</p>` : ''}
      <p class="ph-links">
        <a href="${esc(a.pl_url || XPLAN_SITE)}" target="_blank" rel="noopener">התכנית המלאה במבא"ת ↗</a>
        ${number ? `<a href="https://goinfo.co.il/hafkaot-map.html?plan=${encodeURIComponent(number)}"
           target="_blank" rel="noopener">מה נגרע מכל חלקה ↗</a>` : ''}
      </p>
    </li>`;
  }

  function render(found) {
    const { plans, live, plot } = found;
    const openOnes = plans.filter((a) => {
      const s = shutsOn(a);
      return s && daysTo(s) >= 0;
    });

    const where = plot
      ? `גוש ${plot.gush} חלקה ${plot.parcel}`
        + (plot.area ? ` · ${fmt(plot.area)} מ"ר` : '')
        + (plot.locality ? ` · ${esc(plot.locality)}` : '')
      : 'לא נמצאה חלקה בנקודה הזאת';

    const liveUse = live
      ? `<div class="ph-use">
           <b>${esc(live.mavat_name)}</b>
           <p>נקבע בתכנית ${esc(live.pl_number)}, ${esc(live.pl_name || '')}${
             live.num ? `, תא שטח ${esc(live.num)}` : ''}.</p>
         </div>`
      : `<div class="ph-use ph-unknown">
           <b>אין ייעוד דיגיטלי בנקודה הזאת</b>
           <p>כ-17,700 דונם במושבה מכוסים בתכניות מאושרות שקדמו למאגר הדיגיטלי,
              והייעוד שלהן רשום כ"ייעוד עפ"י תכנית מאושרת אחרת". לבן פירושו
              שאיננו יודעים, לא שהשטח פרטי.</p>
         </div>`;

    return `
      <header class="sheet-head">
        <h2>מה מתוכנן כאן?</h2>
        <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
      </header>
      <p class="sheet-lead">${where}</p>

      ${openOnes.length ? `<div class="ph-alert">
        ${openOnes.length === 1 ? 'תכנית אחת כאן פתוחה להתנגדות'
          : `${openOnes.length} תכניות כאן פתוחות להתנגדות`}.
        זה החלון היחיד שבו אפשר להשפיע.
      </div>` : ''}

      <h3>הייעוד שבתוקף</h3>
      ${liveUse}

      <h3>התכניות שחלות כאן (${plans.length})</h3>
      ${plans.length
        ? `<ul class="ph-plans">${plans.map(planRow).join('')}</ul>`
        : `<p class="ph-none">אין כאן תכנית מקוונת. המאגר מכיל תכניות שהוגשו
             דרך האינטרנט, בערך מ-2011. תכנית ישנה יותר קיימת במבא"ת אבל בלי
             גבול על המפה.</p>`}

      <h3>מה הדף הזה לא יודע</h3>
      <ul class="ph-limits">
        <li><b>היתרי בנייה.</b> אין מאגר ארצי פתוח. קבלן שבונה מכוח תכנית
            מאושרת קיימת לא מגיש תכנית חדשה, ולכן לא יופיע כאן דבר. אם נראה
            שמתחילים לבנות ואין כאן תכנית חדשה, זה המקום לשאול את הוועדה
            המקומית.</li>
        <li><b>בעלות.</b> ייעוד אינו בעלות ואינו רשות כניסה. אין שירות ציבורי
            שמפרסם בעלות על קרקע.</li>
        <li><b>מה נדון בוועדה.</b> הנתונים מתעדכנים כשהתכנית עוברת שלב, לא
            כשמישהו מדבר עליה.</li>
      </ul>

      <p class="sheet-credit">
        הנתונים מ<a href="${XPLAN_SITE}" target="_blank" rel="noopener">שירות המפה של מינהל התכנון</a>
        ומשכבת החלקות של govmap, בשאילתה חיה ברגע הלחיצה.
        <a href="${GUIDE}" target="_blank" rel="noopener">מה זה תכנית מתאר, מפורטת וכוללנית ↗</a>
      </p>`;
  }

  /* ---------- what is open for objection, town-wide ----------
   *
   * A plan on deposit is on show for sixty days and then the door shuts, and
   * the layer of plans has carried that date all along without saying it. On
   * 24/9/2026 four plans were open here and one - "מתחם אחוזת נעורים", 261
   * flats - had shut seven days earlier. Nobody in the app could have known.
   *
   * Asked live for the same reason the panel is: a count that is a week old is
   * worse than no count, because it is a number people would trust. Cached for
   * an hour in sessionStorage so that moving around the app is not forty
   * requests to a government service. */

  const OPEN_KEY = 'dk.objections.v1';
  const OPEN_TTL = 3600e3;

  /** Every plan in the moshava whose objection window has not shut. */
  async function openForObjection() {
    try {
      const kept = JSON.parse(sessionStorage.getItem(OPEN_KEY) || 'null');
      if (kept && Date.now() - kept.at < OPEN_TTL) return kept.rows;
    } catch (err) {
      /* private mode, or someone else's key: ask again, it is one request */
    }

    const feats = await ask(PLANS, {
      // Deposit is the only stage with an open window. The wider town filter is
      // the one build_plans.py settled on: three fields name the town and they
      // disagree, and this is the widest.
      where: "plan_area_name LIKE '%פרדס חנה%'"
        + " AND internet_short_status IN ('פרסום הפקדה', 'בתהליך הפקדה')",
      outFields: 'pl_number,pl_name,pl_url,pl_last_deposit_date,'
        + 'pl_rejection_date,pl_date_advertise,quantity_delta_120',
      returnGeometry: 'false',
      f: 'json'
    });

    const rows = feats
      .map((f) => f.attributes)
      .map((a) => ({
        num: a.pl_number, name: a.pl_name, url: a.pl_url,
        units: Number(a.quantity_delta_120) || 0,
        shuts: shutsOn(a), left: daysTo(shutsOn(a))
      }))
      .filter((r) => r.shuts && r.left >= 0)
      .sort((x, y) => x.left - y.left);

    try {
      sessionStorage.setItem(OPEN_KEY, JSON.stringify({ at: Date.now(), rows }));
    } catch (err) { /* nothing depends on it being kept */ }
    return rows;
  }

  /* ---------- the panel's own state ---------- */

  let armed = false;         // the next tap on the map asks
  let busy = false;

  const sheet = () => el('plan-sheet');
  const card = () => el('plan-card');

  function show(html) {
    card().innerHTML = html;
    sheet().hidden = false;
  }

  function close() {
    const s = sheet();
    if (s) s.hidden = true;
  }

  const isOpen = () => !!(sheet() && !sheet().hidden);

  /** Ask about one point and show the answer. */
  async function at(lngLat) {
    if (busy) return;
    busy = true;
    disarm();
    show(`<header class="sheet-head">
            <h2>מה מתוכנן כאן?</h2>
            <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
          </header>
          <p class="sheet-lead">שואל את מינהל התכנון...</p>
          <div class="ph-wait"></div>`);
    try {
      show(render(await lookUp(lngLat)));
    } catch (err) {
      console.error('דרך קיצור: שאילתת תכנון נכשלה', err);
      show(`<header class="sheet-head">
              <h2>מה מתוכנן כאן?</h2>
              <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
            </header>
            <p class="ph-none">לא הצלחתי להגיע לשירות של מינהל התכנון כרגע.
              השירות יושב מאחורי חומת אש ממשלתית ולפעמים אינו עונה.
              אפשר לנסות שוב בעוד רגע, או לפתוח את
              <a href="${XPLAN_SITE}" target="_blank" rel="noopener">האתר שלהם ↗</a>.</p>`);
    } finally {
      busy = false;
    }
  }

  /* Arming rather than a mode: one tap on the button, one tap on the map, and
   * the app goes back to being a trail browser. A mode you have to remember to
   * leave is a mode that swallows the next tap you meant for a trail. */
  function arm() {
    armed = true;
    document.body.classList.add('asking-plan');
    const btn = el('plan-ask');
    if (btn) btn.classList.add('on');
  }

  function disarm() {
    armed = false;
    document.body.classList.remove('asking-plan');
    const btn = el('plan-ask');
    if (btn) btn.classList.remove('on');
  }

  const isArmed = () => armed;

  /** The list behind the banner: what is open, soonest first.
   *
   *  Its own sheet rather than a filter over the plans layer, because that
   *  layer is a file built by build_plans.py and this is live. A banner whose
   *  count comes from today and whose list comes from the last build would
   *  disagree with itself on the day it mattered. */
  async function showOpen() {
    show(`<header class="sheet-head">
            <h2>פתוח להתנגדות</h2>
            <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
          </header>
          <p class="sheet-lead">שואל את מינהל התכנון...</p>
          <div class="ph-wait"></div>`);
    let rows;
    try {
      rows = await openForObjection();
    } catch (err) {
      show(`<header class="sheet-head">
              <h2>פתוח להתנגדות</h2>
              <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
            </header>
            <p class="ph-none">לא הצלחתי להגיע לשירות של מינהל התכנון כרגע.</p>`);
      return;
    }
    show(`
      <header class="sheet-head">
        <h2>פתוח להתנגדות</h2>
        <button class="sheet-x" data-act="close" aria-label="סגירה">&times;</button>
      </header>
      <p class="sheet-lead">תכניות שהופקדו במושבה ושחלון ההתנגדויות שלהן עוד לא נסגר.</p>
      ${rows.length ? `<ul class="ph-plans">${rows.map((r) => `
        <li class="ph-plan ph-open">
          <div class="ph-plan-head">
            <b>${esc(r.name)}</b><span class="ph-num">${esc(r.num)}</span>
          </div>
          ${r.units > 0 ? `<p class="ph-adds">מוסיפה ${r.units === 1 ? 'יחידת דיור אחת'
            : fmt(r.units) + ' יחידות דיור'}</p>` : ''}
          <p class="ph-deadline">עד ${date(r.shuts)},
            ${r.left === 0 ? 'כלומר היום' : r.left === 1 ? 'כלומר מחר' : 'בעוד ' + r.left + ' ימים'}</p>
          <p class="ph-links">
            <a href="${esc(r.url || XPLAN_SITE)}" target="_blank" rel="noopener">התכנית, ושם גם מגישים התנגדות ↗</a>
          </p>
        </li>`).join('')}</ul>`
        : '<p class="ph-none">אין כרגע תכנית פתוחה להתנגדות במושבה.</p>'}
      <h3>מה זה אומר</h3>
      <p class="ph-none">הפקדה היא השלב היחיד בחיי תכנית שבו הציבור יכול להגיש
        התנגדות, והוא נמשך שישים יום מהפרסום בעיתונים. אחרי שהחלון נסגר אי אפשר
        עוד להתנגד, גם אם מדובר במאות יחידות דיור.</p>
      <p class="sheet-credit">
        נשאל עכשיו מ<a href="${XPLAN_SITE}" target="_blank" rel="noopener">שירות המפה של מינהל התכנון</a>.
        <a href="${GUIDE}" target="_blank" rel="noopener">איך עובד הליך התכנון ↗</a>
      </p>`);
  }

  /** Fill the panel's banner, and say nothing at all when nothing is open. */
  async function paintBanner() {
    const btn = el('plan-banner');
    if (!btn) return [];
    let rows = [];
    try {
      rows = await openForObjection();
    } catch (err) {
      console.info('דרך קיצור: לא הצלחתי לבדוק תכניות פתוחות להתנגדות', err);
      return [];                       // a banner that says "unknown" is noise
    }
    if (!rows.length) { btn.hidden = true; return rows; }

    const soon = rows[0];
    const units = rows.reduce((n, r) => n + Math.max(0, r.units), 0);
    const when = soon.left === 0 ? 'היום'
      : soon.left === 1 ? 'מחר' : `בעוד ${soon.left} ימים`;
    btn.innerHTML =
      `<b>${rows.length === 1 ? 'תכנית אחת פתוחה להתנגדות'
        : `${rows.length} תכניות פתוחות להתנגדות`} במושבה</b>` +
      `<span>הקרובה נסגרת ${when}, ב-${date(soon.shuts)}.` +
      (units ? ` יחד הן מוסיפות ${fmt(units)} יחידות דיור.` : '') +
      ' לחץ כדי לראות אותן ברשימה.</span>';
    btn.hidden = false;
    return rows;
  }

  return { at, arm, disarm, isArmed, close, isOpen, lookUp, paintBanner, showOpen,
           openForObjection, shutsOn, daysTo, date, GUIDE };
})();
