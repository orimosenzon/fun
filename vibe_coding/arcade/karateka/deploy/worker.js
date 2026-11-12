// A private copy of the game for friends. The code is public anyway; only the disk image is gated:
// it is served only to a browser that came in through the link with the secret key (?k=...),
// which leaves a cookie behind. Anyone else gets the public page, which asks for their own disk file.
export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const cookie = req.headers.get('Cookie') || '';
    const keyed = url.searchParams.get('k') === env.KEY;
    const allowed = keyed || cookie.split(/;\s*/).includes('k=' + env.KEY);

    if (url.pathname === '/data/karateka.dsk' && !allowed) return new Response('Not found', { status: 404 });

    const res = new Response(...(r => [r.body, r])(await env.ASSETS.fetch(req)));
    res.headers.set('X-Robots-Tag', 'noindex, nofollow');
    if (keyed) res.headers.append('Set-Cookie', `k=${env.KEY}; Path=/; Max-Age=31536000; Secure; HttpOnly; SameSite=Lax`);
    return res;
  },
};
