// Extracted pure functions from index.html for testing and reuse.
// Each function accepts all its dependencies as parameters (no globals).

/**
 * Haversine formula — great-circle distance between two points in metres.
 * @param {{lat:number, lng:number}} p1
 * @param {{lat:number, lng:number}} p2
 * @returns {number} distance in metres
 */
export function haversine(p1, p2) {
    const R = 6371000;
    const dLat = (p2.lat - p1.lat) * Math.PI / 180;
    const dLng = (p2.lng - p1.lng) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(p1.lat * Math.PI / 180) * Math.cos(p2.lat * Math.PI / 180) *
              Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Sort an array of place results by the given criterion.
 * @param {Array} results - each item should have `rating`, `userRatingCount`, `_dist`
 * @param {'distance'|'rating'|'reviews'} sortBy
 * @returns {Array} new sorted array (original is not mutated)
 */
export function sortResults(results, sortBy) {
    const sorted = [...results];
    switch (sortBy) {
        case 'rating':
            sorted.sort((a, b) => (b.rating || 0) - (a.rating || 0));
            break;
        case 'reviews':
            sorted.sort((a, b) => (b.userRatingCount || 0) - (a.userRatingCount || 0));
            break;
        default: // 'distance'
            sorted.sort((a, b) => a._dist - b._dist);
    }
    return sorted;
}

/**
 * Filter places by radius and minimum rating, compute distance, then sort.
 * @param {Array} places - raw Places API results (each has location.latitude/longitude)
 * @param {{minRating?:number}} parsedQuery
 * @param {{lat:number, lng:number}} center
 * @param {number} radius - in metres
 * @param {'distance'|'rating'|'reviews'} sortBy
 * @returns {Array} filtered and sorted results with `_loc` and `_dist` added
 */
export function filterResults(places, parsedQuery, center, radius, sortBy) {
    const filtered = places
        .map(p => {
            const loc = { lat: p.location.latitude, lng: p.location.longitude };
            const dist = haversine(center, loc);
            return { ...p, _loc: loc, _dist: dist };
        })
        .filter(p => p._dist <= radius)
        .filter(p => !parsedQuery.minRating || (p.rating && p.rating >= parsedQuery.minRating));
    return sortResults(filtered, sortBy);
}

/**
 * Check if a placeId is in the favorites list.
 * @param {string} placeId
 * @param {Array<{id:string}>} favorites
 * @returns {boolean}
 */
export function isFavorite(placeId, favorites) {
    return favorites.some(f => f.id === placeId);
}

/**
 * Add a query to search history (front of list, max 10, no duplicates).
 * Returns a new array — does not mutate the input.
 * @param {string} query
 * @param {string[]} history
 * @returns {string[]}
 */
export function addToHistory(query, history) {
    const filtered = history.filter(h => h !== query);
    filtered.unshift(query);
    if (filtered.length > 10) filtered.pop();
    return filtered;
}

/**
 * Remove a query from search history. Returns a new array.
 * @param {string} query
 * @param {string[]} history
 * @returns {string[]}
 */
export function removeFromHistory(query, history) {
    return history.filter(h => h !== query);
}

/**
 * Find the closest snap point for the mobile bottom sheet.
 * @param {number} h - current height in pixels
 * @param {number} peekHeight - e.g. 110
 * @param {number} halfHeight - e.g. window.innerHeight * 0.44
 * @param {number} fullHeight - e.g. window.innerHeight * 0.9
 * @returns {number} the snap height closest to h
 */
export function closestSnap(h, peekHeight, halfHeight, fullHeight) {
    const points = [peekHeight, halfHeight, fullHeight];
    let best = points[0], bestDist = Math.abs(h - best);
    for (const p of points) {
        const d = Math.abs(h - p);
        if (d < bestDist) { best = p; bestDist = d; }
    }
    return best;
}

/**
 * Build a WhatsApp share URL for a place.
 * @param {{displayName?:{text:string}, formattedAddress?:string, googleMapsUri?:string, _loc:{lat:number,lng:number}}} place
 * @returns {string} WhatsApp URL
 */
export function buildWhatsAppUrl(place) {
    const name = place.displayName?.text || '';
    const url = place.googleMapsUri || `https://www.google.com/maps/search/?api=1&query=${place._loc.lat},${place._loc.lng}`;
    const text = encodeURIComponent(`${name}\n${place.formattedAddress || ''}\n${url}`);
    return `https://wa.me/?text=${text}`;
}

/**
 * Build a Google Maps URL for a place.
 * @param {{googleMapsUri?:string, _loc:{lat:number,lng:number}}} place
 * @returns {string}
 */
export function buildMapsUrl(place) {
    return place.googleMapsUri || `https://www.google.com/maps/search/?api=1&query=${place._loc.lat},${place._loc.lng}`;
}

/**
 * Extract a JSON object from a Gemini API text response.
 * Falls back to a default ParsedQuery on failure.
 * @param {string} text - raw text from Gemini response
 * @param {string} originalQuery - the user's original query (used in fallback)
 * @returns {{textQuery:string, openNow:boolean, minRating:number|null, keywords:string[]}}
 */
export function parseGeminiResponse(text, originalQuery) {
    try {
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) return JSON.parse(jsonMatch[0]);
    } catch (err) {
        // fall through to default
    }
    return { textQuery: originalQuery, openNow: false, minRating: null, keywords: [] };
}
