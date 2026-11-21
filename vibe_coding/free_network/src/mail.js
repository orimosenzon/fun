// Invitations by email. There is no mail service: a letter goes out from the Gmail of the
// person who invites, after they allowed the network to send in their name, and nothing else.

export const SEND_SCOPE = 'https://www.googleapis.com/auth/gmail.send';
export const MAX_NOTE = 1500;
export const DEFAULT_NOTE = 'אני בונה רשת חברתית חדשה. אין בה פרסומות, ואין בה אלגוריתם שמחליט בשבילנו מה נראה. היא עדיין קטנה ונמצאת בניסוי בין חברים, ואשמח לראות אותך בין הראשונים.';
const SUBJECT = 'הזמנה אישית לרשת החופשית';
const CLOSING = 'אשמח לשמוע מה חסר ומה מפריע. בשביל זה הניסוי.';

const encoder = new TextEncoder();

function base64(bytes) {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

const unbase64 = (text) => Uint8Array.from(atob(text), (ch) => ch.charCodeAt(0));
const urlSafe = (text) => text.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

async function sealKey(secret) {
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(`mailer:${secret}`));
  return crypto.subtle.importKey('raw', digest, 'AES-GCM', false, ['encrypt', 'decrypt']);
}

// The key to someone's Gmail is stored sealed with a secret the database does not hold.
export async function seal(secret, text) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const sealed = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await sealKey(secret), encoder.encode(text));
  return `${base64(iv)}.${base64(new Uint8Array(sealed))}`;
}

export async function unseal(secret, sealed) {
  const [iv, data] = sealed.split('.').map(unbase64);
  return new TextDecoder().decode(await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, await sealKey(secret), data));
}

const escapeHtml = (text) => text.replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
const firstName = (name) => (name ?? '').trim().split(/\s+/)[0] ?? '';

// The letter, as plain text and as a page that reads from right to left.
export function inviteLetter({ name, sender, note, link, site, expiresAt }) {
  const until = new Intl.DateTimeFormat('he-IL', { day: 'numeric', month: 'long', timeZone: 'Asia/Jerusalem' }).format(expiresAt);
  const hello = `היי${firstName(name) ? ` ${firstName(name)}` : ''},`;
  const film = `${site}/intro`;
  const paragraphs = (note || DEFAULT_NOTE).split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
  const text = [hello, '', ...paragraphs.flatMap((part) => [part, '']),
    'סרטון קצר שמסביר מה זה ולמה:', film, '',
    `הקישור האישי שלך לכניסה (עובד פעם אחת, עד ${until}):`, link, '',
    CLOSING, firstName(sender)].join('\n');
  const p = 'margin:0 0 16px';
  const html = `<div dir="rtl" lang="he" style="direction:rtl;text-align:right;max-width:560px;font-family:Arial,Helvetica,sans-serif;font-size:16px;line-height:1.6;color:#0f2a38">
<p style="${p}">${escapeHtml(hello)}</p>
${paragraphs.map((part) => `<p style="${p}">${escapeHtml(part).replace(/\n/g, '<br>')}</p>`).join('\n')}
<p style="${p}">סרטון קצר שמסביר מה זה ולמה:</p>
<p style="${p}"><a href="${film}"><img src="${site}/video/intro.jpg" width="480" alt="לצפייה בסרטון" style="display:block;width:100%;max-width:480px;height:auto;border:0;border-radius:12px"></a></p>
<p style="${p}"><a href="${film}" style="color:#1e4a63">לצפייה בסרטון (שתי דקות)</a></p>
<p style="margin:24px 0 8px"><a href="${escapeHtml(link)}" style="display:inline-block;padding:11px 28px;border-radius:999px;background:#1e4a63;color:#ffffff;font-weight:bold;text-decoration:none">כניסה לרשת החופשית</a></p>
<p style="margin:0 0 20px;font-size:14px;color:#5d6b70">הקישור אישי. הוא עובד פעם אחת, עד ${until}.</p>
<p style="${p}">${escapeHtml(CLOSING)}<br>${escapeHtml(firstName(sender))}</p>
</div>`;
  return { subject: SUBJECT, text, html };
}

// Text in a mail header or body travels as base64, in lines a mail server accepts.
const encoded = (text) => base64(encoder.encode(text));
const wrapped = (text) => encoded(text).match(/.{1,76}/g).join('\r\n');

function mime({ to, subject, text, html }) {
  const boundary = `part-${crypto.randomUUID()}`;
  const part = (type, body) => `--${boundary}\r\nContent-Type: ${type}; charset=UTF-8\r\nContent-Transfer-Encoding: base64\r\n\r\n${wrapped(body)}\r\n`;
  return [`To: ${to}`, `Subject: =?UTF-8?B?${encoded(subject)}?=`, 'MIME-Version: 1.0',
    `Content-Type: multipart/alternative; boundary="${boundary}"`, '',
    part('text/plain', text) + part('text/html', html) + `--${boundary}--`].join('\r\n');
}

// Sends one letter from the Gmail that the sealed key belongs to. Says 'expired' when Google
// no longer honours the key, so the admin knows to connect again.
export async function sendFromGmail(env, sealedKey, letter) {
  try {
    const grant = await fetch(env.GOOGLE_TOKEN_URL ?? 'https://oauth2.googleapis.com/token', {
      method: 'POST',
      body: new URLSearchParams({
        client_id: env.GOOGLE_CLIENT_ID,
        client_secret: env.GOOGLE_CLIENT_SECRET,
        refresh_token: await unseal(env.GOOGLE_CLIENT_SECRET, sealedKey),
        grant_type: 'refresh_token',
      }),
    });
    const tokens = await grant.json().catch(() => ({}));
    if (!grant.ok) return tokens.error === 'invalid_grant' ? 'expired' : 'failed';
    const sent = await fetch(`${env.GMAIL_ORIGIN ?? 'https://gmail.googleapis.com'}/gmail/v1/users/me/messages/send`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${tokens.access_token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ raw: urlSafe(encoded(mime(letter))) }),
    });
    if (sent.ok) return 'sent';
    return sent.status === 401 ? 'expired' : 'failed';
  } catch {
    return 'failed';
  }
}
