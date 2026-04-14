/**
 * api-guard.js — API Kill Switch
 *
 * Tracks API call counts in localStorage per day and per minute.
 * Call ApiGuard.check('name') before every API fetch — it throws
 * ApiGuardError (and shows a hard-stop banner) when a limit is exceeded.
 *
 * Usage:
 *   ApiGuard.configure({ dayLimits: { places: 50 } }); // optional override
 *   ApiGuard.check('places');   // throws + shows banner if over limit
 *   ApiGuard.check('gemini');
 *   ApiGuard.getStats();        // { day, counts, minuteCounts, limits }
 *   ApiGuard.reset();           // clear all counters (dev use)
 *   ApiGuard.reset('places');   // clear one counter
 *
 * Works as a plain <script> tag (window.ApiGuard) and as an ES module.
 */

(function (global) {
    'use strict';

    const STORAGE_KEY = 'api_guard_v1';

    // ── Default limits ──────────────────────────────────────────────────────
    // Tune these to match your expected usage and budget.
    // Places Text Search: ~$0.017/call → 50/day ≈ $0.85/day max
    // Gemini Flash: negligible cost, generous limit
    // Maps JS: loaded once per page, limit is a sanity check
    let config = {
        dayLimits: {
            places: 50,
            gemini: 150,
            maps:   30,
        },
        minuteLimits: {
            places: 8,
            gemini: 20,
            maps:   5,
        }
    };

    // ── Error class ─────────────────────────────────────────────────────────
    class ApiGuardError extends Error {
        constructor(message, limitType, api, count, limit) {
            super(message);
            this.name      = 'ApiGuardError';
            this.limitType = limitType; // 'day' | 'minute'
            this.api       = api;
            this.count     = count;
            this.limit     = limit;
        }
    }

    // ── Storage helpers ─────────────────────────────────────────────────────
    function _getStore() {
        let store = {};
        try { store = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch (_) {}

        const today      = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
        const thisMinute = new Date().toISOString().slice(0, 16); // YYYY-MM-DDTHH:MM

        if (store.day !== today) {
            store.day       = today;
            store.dayCounts = {};
        }
        if (store.minute !== thisMinute) {
            store.minute       = thisMinute;
            store.minuteCounts = {};
        }
        store.dayCounts    = store.dayCounts    || {};
        store.minuteCounts = store.minuteCounts || {};
        return store;
    }

    function _saveStore(store) {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(store)); } catch (_) {}
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /** Override default limits. Only keys you pass are changed. */
    function configure(overrides) {
        if (overrides.dayLimits)    Object.assign(config.dayLimits,    overrides.dayLimits);
        if (overrides.minuteLimits) Object.assign(config.minuteLimits, overrides.minuteLimits);
    }

    /**
     * Call before every API fetch. Increments counters.
     * Throws ApiGuardError (+ shows hard-stop banner) if any limit is exceeded.
     * @param {string} apiName  e.g. 'places', 'gemini', 'maps'
     * @returns {{ dayCount: number, dayLimit: number }}
     */
    function check(apiName) {
        const store    = _getStore();
        const dayCount = store.dayCounts[apiName]    || 0;
        const minCount = store.minuteCounts[apiName] || 0;
        const dayLimit = config.dayLimits[apiName];
        const minLimit = config.minuteLimits[apiName];

        if (dayLimit !== undefined && dayCount >= dayLimit) {
            const err = new ApiGuardError(
                `Daily limit reached for "${apiName}": ${dayCount}/${dayLimit} calls today. App paused. Resets tomorrow.`,
                'day', apiName, dayCount, dayLimit
            );
            showBanner(err);
            throw err;
        }

        if (minLimit !== undefined && minCount >= minLimit) {
            const err = new ApiGuardError(
                `Rate spike for "${apiName}": ${minCount}/${minLimit} calls/min. Slow down.`,
                'minute', apiName, minCount, minLimit
            );
            showBanner(err);
            throw err;
        }

        store.dayCounts[apiName]    = dayCount + 1;
        store.minuteCounts[apiName] = minCount + 1;
        _saveStore(store);

        console.debug(`[ApiGuard] ${apiName}: ${dayCount + 1}/${dayLimit} today`);
        return { dayCount: dayCount + 1, dayLimit };
    }

    /** Returns current counters for all tracked APIs. */
    function getStats() {
        const store = _getStore();
        return {
            day:          store.day,
            counts:       { ...store.dayCounts },
            minuteCounts: { ...store.minuteCounts },
            limits:       { day: { ...config.dayLimits }, minute: { ...config.minuteLimits } }
        };
    }

    /**
     * Reset counters. Useful in dev/testing.
     * @param {string} [apiName]  If omitted, resets everything.
     */
    function reset(apiName) {
        const store = _getStore();
        if (apiName) {
            delete store.dayCounts[apiName];
            delete store.minuteCounts[apiName];
        } else {
            store.dayCounts    = {};
            store.minuteCounts = {};
        }
        _saveStore(store);
        console.info('[ApiGuard] counters reset', apiName || '(all)');
    }

    /**
     * Show a hard-stop banner at the top of the page.
     * Called automatically by check() on limit breach.
     * Can also be called manually for testing: ApiGuard.showBanner(err)
     */
    function showBanner(err) {
        // Remove existing banner
        const old = document.getElementById('api-guard-banner');
        if (old) old.remove();

        const isDay    = err.limitType === 'day';
        const bgColor  = isDay ? '#c62828' : '#e65100';
        const icon     = isDay ? '🛑' : '⚠️';
        const title    = isDay ? 'API Hard Stop' : 'Rate Spike Detected';
        const when     = isDay ? 'today' : 'this minute';
        const resetMsg = isDay ? 'Resets tomorrow at midnight.' : 'Auto-resumes next minute.';

        const banner = document.createElement('div');
        banner.id = 'api-guard-banner';
        banner.setAttribute('role', 'alert');
        banner.style.cssText = [
            'position:fixed', 'top:0', 'left:0', 'right:0',
            `background:${bgColor}`, 'color:#fff',
            'padding:14px 20px', 'z-index:2147483647',
            'font-family:system-ui,sans-serif', 'font-size:14px',
            'display:flex', 'align-items:center', 'gap:12px',
            'box-shadow:0 3px 12px rgba(0,0,0,.4)',
            'line-height:1.4'
        ].join(';');

        banner.innerHTML = `
            <span style="font-size:24px;flex-shrink:0">${icon}</span>
            <span style="flex:1">
                <strong>${title}:</strong>
                "<em>${err.api}</em>" hit ${err.count}/${err.limit} calls ${when}.
                All API calls are paused to prevent runaway costs.
                ${resetMsg}
            </span>
            <button
                onclick="document.getElementById('api-guard-banner').remove()"
                style="background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);
                       color:#fff;padding:6px 14px;border-radius:6px;cursor:pointer;
                       font-size:13px;white-space:nowrap;flex-shrink:0">
                Dismiss
            </button>
        `;

        // Insert at very top of body (before any existing content)
        if (document.body) {
            document.body.insertBefore(banner, document.body.firstChild);
        } else {
            // DOM not ready yet — wait
            document.addEventListener('DOMContentLoaded', () =>
                document.body.insertBefore(banner, document.body.firstChild)
            );
        }
    }

    // ── Export ───────────────────────────────────────────────────────────────
    const ApiGuard = { configure, check, getStats, reset, showBanner, ApiGuardError };

    // ES module export (when bundled / imported with type="module")
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = ApiGuard;
    }

    // Always attach to global for plain <script> usage
    global.ApiGuard = ApiGuard;

})(typeof window !== 'undefined' ? window : (typeof global !== 'undefined' ? global : this));
