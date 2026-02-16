import { describe, it, expect } from 'vitest';
import {
    haversine,
    sortResults,
    filterResults,
    isFavorite,
    addToHistory,
    removeFromHistory,
    closestSnap,
    buildWhatsAppUrl,
    buildMapsUrl,
    parseGeminiResponse
} from './search-utils.js';

// ── haversine ──────────────────────────────────────────────

describe('haversine', () => {
    it('returns 0 for the same point', () => {
        const p = { lat: 32.0853, lng: 34.7818 };
        expect(haversine(p, p)).toBe(0);
    });

    it('calculates Tel Aviv → Jerusalem (~54 km)', () => {
        const telAviv = { lat: 32.0853, lng: 34.7818 };
        const jerusalem = { lat: 31.7683, lng: 35.2137 };
        const dist = haversine(telAviv, jerusalem);
        expect(dist).toBeGreaterThan(50000);
        expect(dist).toBeLessThan(58000);
    });

    it('calculates a short distance (< 1 km)', () => {
        const a = { lat: 32.0853, lng: 34.7818 };
        const b = { lat: 32.0860, lng: 34.7825 };
        const dist = haversine(a, b);
        expect(dist).toBeGreaterThan(0);
        expect(dist).toBeLessThan(1000);
    });

    it('handles antipodal points (~20,000 km)', () => {
        const north = { lat: 0, lng: 0 };
        const south = { lat: 0, lng: 180 };
        const dist = haversine(north, south);
        expect(dist).toBeGreaterThan(20_000_000);
        expect(dist).toBeLessThan(20_100_000);
    });
});

// ── sortResults ────────────────────────────────────────────

describe('sortResults', () => {
    const places = [
        { _dist: 500, rating: 3.5, userRatingCount: 10 },
        { _dist: 100, rating: 4.8, userRatingCount: 200 },
        { _dist: 300, rating: 4.0, userRatingCount: 50 },
    ];

    it('sorts by distance (default)', () => {
        const sorted = sortResults(places, 'distance');
        expect(sorted[0]._dist).toBe(100);
        expect(sorted[1]._dist).toBe(300);
        expect(sorted[2]._dist).toBe(500);
    });

    it('sorts by rating descending', () => {
        const sorted = sortResults(places, 'rating');
        expect(sorted[0].rating).toBe(4.8);
        expect(sorted[1].rating).toBe(4.0);
        expect(sorted[2].rating).toBe(3.5);
    });

    it('sorts by review count descending', () => {
        const sorted = sortResults(places, 'reviews');
        expect(sorted[0].userRatingCount).toBe(200);
        expect(sorted[1].userRatingCount).toBe(50);
        expect(sorted[2].userRatingCount).toBe(10);
    });

    it('does not mutate the original array', () => {
        const original = [...places];
        sortResults(places, 'rating');
        expect(places).toEqual(original);
    });

    it('handles empty array', () => {
        expect(sortResults([], 'distance')).toEqual([]);
    });

    it('handles missing rating/review fields', () => {
        const items = [{ _dist: 1 }, { _dist: 2, rating: 4.0 }];
        const sorted = sortResults(items, 'rating');
        expect(sorted[0].rating).toBe(4.0);
        expect(sorted[1].rating).toBeUndefined();
    });
});

// ── filterResults ──────────────────────────────────────────

describe('filterResults', () => {
    const center = { lat: 32.0853, lng: 34.7818 }; // Tel Aviv

    const makePlaces = (coords) => coords.map(([lat, lng, rating], i) => ({
        location: { latitude: lat, longitude: lng },
        rating,
        displayName: { text: `Place ${i}` },
    }));

    it('filters by radius', () => {
        const places = makePlaces([
            [32.0854, 34.7819, 4.0], // very close
            [32.09, 34.79, 4.0],     // ~500m
            [32.20, 34.90, 4.0],     // ~16 km away
        ]);
        const result = filterResults(places, {}, center, 2000, 'distance');
        expect(result.length).toBe(2);
    });

    it('filters by minimum rating', () => {
        const places = makePlaces([
            [32.0854, 34.7819, 3.0],
            [32.0855, 34.7820, 4.5],
        ]);
        const result = filterResults(places, { minRating: 4.0 }, center, 5000, 'distance');
        expect(result.length).toBe(1);
        expect(result[0].rating).toBe(4.5);
    });

    it('returns empty when nothing matches', () => {
        const places = makePlaces([
            [33.0, 35.0, 4.0], // far away
        ]);
        const result = filterResults(places, {}, center, 1000, 'distance');
        expect(result).toEqual([]);
    });

    it('adds _loc and _dist to each result', () => {
        const places = makePlaces([[32.0854, 34.7819, 4.0]]);
        const result = filterResults(places, {}, center, 5000, 'distance');
        expect(result[0]._loc).toEqual({ lat: 32.0854, lng: 34.7819 });
        expect(result[0]._dist).toBeGreaterThan(0);
    });

    it('returns results sorted by chosen criterion', () => {
        const places = makePlaces([
            [32.0854, 34.7819, 3.0],
            [32.0855, 34.7820, 5.0],
            [32.0856, 34.7821, 4.0],
        ]);
        const result = filterResults(places, {}, center, 5000, 'rating');
        expect(result[0].rating).toBe(5.0);
        expect(result[1].rating).toBe(4.0);
        expect(result[2].rating).toBe(3.0);
    });
});

// ── isFavorite ─────────────────────────────────────────────

describe('isFavorite', () => {
    const favorites = [{ id: 'abc' }, { id: 'xyz' }];

    it('returns true for a favorite', () => {
        expect(isFavorite('abc', favorites)).toBe(true);
    });

    it('returns false for a non-favorite', () => {
        expect(isFavorite('nope', favorites)).toBe(false);
    });

    it('returns false for empty favorites', () => {
        expect(isFavorite('abc', [])).toBe(false);
    });
});

// ── addToHistory ───────────────────────────────────────────

describe('addToHistory', () => {
    it('adds a new query to the front', () => {
        const result = addToHistory('pizza', ['sushi', 'burgers']);
        expect(result[0]).toBe('pizza');
        expect(result.length).toBe(3);
    });

    it('moves duplicate to the front', () => {
        const result = addToHistory('sushi', ['pizza', 'sushi', 'burgers']);
        expect(result[0]).toBe('sushi');
        expect(result.length).toBe(3);
    });

    it('caps at 10 items', () => {
        const history = Array.from({ length: 10 }, (_, i) => `q${i}`);
        const result = addToHistory('new', history);
        expect(result.length).toBe(10);
        expect(result[0]).toBe('new');
        expect(result).not.toContain('q9');
    });

    it('does not mutate input array', () => {
        const history = ['a', 'b'];
        addToHistory('c', history);
        expect(history).toEqual(['a', 'b']);
    });

    it('handles empty history', () => {
        expect(addToHistory('first', [])).toEqual(['first']);
    });
});

// ── removeFromHistory ──────────────────────────────────────

describe('removeFromHistory', () => {
    it('removes an existing entry', () => {
        const result = removeFromHistory('pizza', ['pizza', 'sushi']);
        expect(result).toEqual(['sushi']);
    });

    it('returns same items if query not found', () => {
        const result = removeFromHistory('tacos', ['pizza', 'sushi']);
        expect(result).toEqual(['pizza', 'sushi']);
    });

    it('handles empty history', () => {
        expect(removeFromHistory('pizza', [])).toEqual([]);
    });
});

// ── closestSnap ────────────────────────────────────────────

describe('closestSnap', () => {
    const peek = 110;
    const half = 400;
    const full = 800;

    it('snaps to peek when closest', () => {
        expect(closestSnap(120, peek, half, full)).toBe(peek);
    });

    it('snaps to half when closest', () => {
        expect(closestSnap(380, peek, half, full)).toBe(half);
    });

    it('snaps to full when closest', () => {
        expect(closestSnap(750, peek, half, full)).toBe(full);
    });

    it('snaps to exact value', () => {
        expect(closestSnap(110, peek, half, full)).toBe(peek);
        expect(closestSnap(400, peek, half, full)).toBe(half);
        expect(closestSnap(800, peek, half, full)).toBe(full);
    });

    it('handles midpoint — ties go to first found', () => {
        // midpoint between peek(110) and half(400) = 255
        // |255-110| = 145, |255-400| = 145 — equal, so peek stays as best (first)
        expect(closestSnap(255, peek, half, full)).toBe(peek);
    });
});

// ── buildWhatsAppUrl ───────────────────────────────────────

describe('buildWhatsAppUrl', () => {
    it('uses googleMapsUri when available', () => {
        const place = {
            displayName: { text: 'Pizza Place' },
            formattedAddress: '123 Main St',
            googleMapsUri: 'https://maps.google.com/place123',
            _loc: { lat: 32.08, lng: 34.78 }
        };
        const url = buildWhatsAppUrl(place);
        expect(url).toContain('https://wa.me/?text=');
        expect(url).toContain('Pizza%20Place');
        expect(url).toContain('maps.google.com');
    });

    it('falls back to coordinate URL', () => {
        const place = {
            displayName: { text: 'Cafe' },
            formattedAddress: '',
            _loc: { lat: 32.08, lng: 34.78 }
        };
        const url = buildWhatsAppUrl(place);
        expect(url).toContain('32.08');
        expect(url).toContain('34.78');
    });

    it('handles missing displayName', () => {
        const place = { _loc: { lat: 32.08, lng: 34.78 } };
        const url = buildWhatsAppUrl(place);
        expect(url).toContain('https://wa.me/?text=');
    });
});

// ── buildMapsUrl ───────────────────────────────────────────

describe('buildMapsUrl', () => {
    it('returns googleMapsUri when available', () => {
        const place = { googleMapsUri: 'https://maps.google.com/xyz', _loc: { lat: 0, lng: 0 } };
        expect(buildMapsUrl(place)).toBe('https://maps.google.com/xyz');
    });

    it('falls back to coordinate-based URL', () => {
        const place = { _loc: { lat: 32.08, lng: 34.78 } };
        expect(buildMapsUrl(place)).toContain('query=32.08,34.78');
    });
});

// ── parseGeminiResponse ────────────────────────────────────

describe('parseGeminiResponse', () => {
    it('extracts JSON from a text with surrounding content', () => {
        const text = 'Here is the result:\n{"textQuery":"pizza","openNow":true,"minRating":4,"keywords":["italian"]}\nDone.';
        const result = parseGeminiResponse(text, 'pizza');
        expect(result.textQuery).toBe('pizza');
        expect(result.openNow).toBe(true);
        expect(result.minRating).toBe(4);
        expect(result.keywords).toEqual(['italian']);
    });

    it('extracts JSON from clean response', () => {
        const text = '{"textQuery":"sushi","openNow":false,"minRating":null,"keywords":[]}';
        const result = parseGeminiResponse(text, 'sushi');
        expect(result.textQuery).toBe('sushi');
    });

    it('returns fallback for invalid JSON', () => {
        const result = parseGeminiResponse('not json at all', 'pizza');
        expect(result.textQuery).toBe('pizza');
        expect(result.openNow).toBe(false);
        expect(result.minRating).toBeNull();
        expect(result.keywords).toEqual([]);
    });

    it('returns fallback for broken JSON', () => {
        const result = parseGeminiResponse('{"textQuery": broken}', 'burgers');
        expect(result.textQuery).toBe('burgers');
    });

    it('returns fallback for empty string', () => {
        const result = parseGeminiResponse('', 'tacos');
        expect(result.textQuery).toBe('tacos');
    });
});
