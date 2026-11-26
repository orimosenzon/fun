// The part of the network that stays awake on a device after the page was closed, for one
// purpose: to show a notification when the server pushes one, and to open the right page when
// it is pressed. It keeps no copy of the site and stands in the way of no request.

// Safari takes the permission away from a site that receives a push and shows nothing, so
// there a notification is shown always. Elsewhere a page that is being looked at tells it itself.
const MUST_SHOW = /^((?!chrome|chromium|android|crios|fxios|firefox).)*safari/i.test(self.navigator.userAgent);

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

const openPages = () => self.clients.matchAll({ type: 'window', includeUncontrolled: true });

self.addEventListener('push', (event) => {
  let message = {};
  try {
    message = event.data.json();
  } catch {
    // A push with nothing in it still says that something is waiting.
  }
  event.waitUntil((async () => {
    const pages = await openPages();
    // A page that is being looked at tells the news itself, unless the push asks to be shown.
    const shown = MUST_SHOW || !!message.show || !pages.some((page) => page.visibilityState === 'visible' && page.focused);
    for (const page of pages) page.postMessage({ type: 'push', message, shown });
    if (Number.isInteger(message.count)) await self.navigator.setAppBadge?.(message.count).catch(() => {});
    if (!shown) return;
    await self.registration.showNotification(message.title ?? 'הרשת החופשית', {
      body: message.body ?? 'משהו חדש מחכה לך',
      icon: '/icon-192.png',
      badge: '/notice-badge.png',
      dir: 'rtl',
      lang: 'he',
      tag: message.tag,
      renotify: !!message.tag,
      data: { path: message.path ?? '/notices' },
    });
  })());
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const path = event.notification.data?.path ?? '/notices';
  event.waitUntil((async () => {
    const page = (await openPages())[0];
    if (!page) {
      await self.clients.openWindow(path);
      return;
    }
    // An open page moves there by itself, without loading the network again.
    page.postMessage({ type: 'open', path });
    await page.focus().catch(() => {});
  })());
});

// The browser replaced the address it is reached at: the new one is handed to the server.
self.addEventListener('pushsubscriptionchange', (event) => {
  event.waitUntil((async () => {
    const options = event.oldSubscription?.options;
    const fresh = event.newSubscription ?? (options && await self.registration.pushManager.subscribe(options));
    if (event.oldSubscription) {
      await fetch('/api/push', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ endpoint: event.oldSubscription.endpoint }) });
    }
    if (fresh) await fetch('/api/push', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fresh) });
  })());
});
