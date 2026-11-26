// Web Push: a short message to a member's phone or computer while the network is closed there.
// It travels through the push service of their browser (Google, Mozilla, Apple, Microsoft),
// sealed so that only the browser can open it (RFC 8291), and signed so the service knows it
// comes from this network (VAPID, RFC 8292). The service learns that something was sent, and
// when, and nothing of what it says.

const encoder = new TextEncoder();
const RECORD_SIZE = 4096;
const PASS_SECONDS = 12 * 60 * 60;

const b64 = (bytes) => btoa(String.fromCharCode(...new Uint8Array(bytes))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const unb64 = (text) => Uint8Array.from(atob(text.replace(/-/g, '+').replace(/_/g, '/')), (ch) => ch.charCodeAt(0));

function join(...parts) {
  const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

// A device is what the browser hands over when it subscribes. The address must belong to one of
// the push services, or the network could be made to knock on any door someone names.
export function pushDevice(input, { local = false } = {}) {
  const { endpoint, keys } = input ?? {};
  if (typeof endpoint !== 'string' || endpoint.length > 1000) return null;
  let url;
  try {
    url = new URL(endpoint);
  } catch {
    return null;
  }
  const host = url.hostname;
  const known = url.protocol === 'https:' && (host === 'fcm.googleapis.com' || host.endsWith('.push.services.mozilla.com')
    || host.endsWith('.push.apple.com') || host.endsWith('.notify.windows.com'));
  if (!known && !(local && host === 'localhost')) return null;
  try {
    if (unb64(keys.p256dh).length !== 65 || unb64(keys.auth).length !== 16) return null;
  } catch {
    return null;
  }
  return { endpoint, p256dh: keys.p256dh, auth: keys.auth };
}

async function hkdf(salt, secret, info, length) {
  const key = await crypto.subtle.importKey('raw', secret, 'HKDF', false, ['deriveBits']);
  return new Uint8Array(await crypto.subtle.deriveBits({ name: 'HKDF', hash: 'SHA-256', salt, info }, key, length * 8));
}

// Seals the text for one device: a fresh key pair for this message alone, a secret agreed with
// the device's public key, and one AES-GCM record.
export async function sealPush(device, text) {
  const theirs = unb64(device.p256dh);
  const pair = await crypto.subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits']);
  const ours = new Uint8Array(await crypto.subtle.exportKey('raw', pair.publicKey));
  const theirKey = await crypto.subtle.importKey('raw', theirs, { name: 'ECDH', namedCurve: 'P-256' }, false, []);
  const shared = new Uint8Array(await crypto.subtle.deriveBits({ name: 'ECDH', public: theirKey }, pair.privateKey, 256));
  const secret = await hkdf(unb64(device.auth), shared, join(encoder.encode('WebPush: info\0'), theirs, ours), 32);
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const key = await crypto.subtle.importKey('raw', await hkdf(salt, secret, encoder.encode('Content-Encoding: aes128gcm\0'), 16), 'AES-GCM', false, ['encrypt']);
  const nonce = await hkdf(salt, secret, encoder.encode('Content-Encoding: nonce\0'), 12);
  // The byte 2 after the text marks the last (and only) record.
  const sealed = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv: nonce }, key, join(encoder.encode(text), Uint8Array.of(2))));
  const size = new Uint8Array(4);
  new DataView(size.buffer).setUint32(0, RECORD_SIZE);
  return join(salt, size, Uint8Array.of(ours.length), ours, sealed);
}

// The signed pass that tells a push service who is sending. One pass serves a service for hours.
const passes = new Map();

async function pass(env, service, sender) {
  const now = Math.floor(Date.now() / 1000);
  const name = `${service} ${sender}`;
  let kept = passes.get(name);
  if (!kept || kept.until < now + 600) {
    const point = unb64(env.VAPID_PUBLIC);
    const key = await crypto.subtle.importKey('jwk',
      { kty: 'EC', crv: 'P-256', d: env.VAPID_PRIVATE, x: b64(point.slice(1, 33)), y: b64(point.slice(33, 65)) },
      { name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign']);
    const until = now + PASS_SECONDS;
    const part = (value) => b64(encoder.encode(JSON.stringify(value)));
    const claims = `${part({ typ: 'JWT', alg: 'ES256' })}.${part({ aud: service, exp: until, sub: sender })}`;
    const signature = await crypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' }, key, encoder.encode(claims));
    kept = { until, token: `${claims}.${b64(signature)}` };
    passes.set(name, kept);
  }
  return `vapid t=${kept.token}, k=${env.VAPID_PUBLIC}`;
}

export const pushReady = (env) => !!env.VAPID_PUBLIC && !!env.VAPID_PRIVATE;

// Sends one message to one device and returns the answer of its push service: 201 is accepted,
// 404 and 410 mean the device has unsubscribed. `sender` is the address of the network. The
// service keeps the message for three days if the device is off.
export async function sendPush(env, device, message, sender) {
  const response = await fetch(device.endpoint, {
    method: 'POST',
    headers: {
      Authorization: await pass(env, new URL(device.endpoint).origin, sender),
      'Content-Encoding': 'aes128gcm',
      'Content-Type': 'application/octet-stream',
      TTL: String(3 * 24 * 60 * 60),
      Urgency: 'high',
    },
    body: await sealPush(device, JSON.stringify(message)),
  });
  return response.status;
}
