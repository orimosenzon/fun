/* The service worker that makes the app installable, and nothing more.
 *
 * Chrome on Android wants a registered worker before it offers "add to home
 * screen" with the app's own icon and a window of its own; the manifest alone
 * gets a plain bookmark. This one deliberately caches nothing: the app is the
 * initiative's database, and a cached trails.json would show somebody the map
 * as it was, after they were just told a trail had been added. Every request
 * goes to the network exactly as it did before, and the app's own offline
 * copies (localStorage, the bundled data/) do what they always did. */
'use strict';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
// A fetch listener has to exist for the install criteria; one that does not
// answer leaves the request to the browser.
self.addEventListener('fetch', () => {});
