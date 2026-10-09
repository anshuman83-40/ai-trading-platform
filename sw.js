// TradePulse service worker — makes the website installable as a phone app.
// Pages and live data ALWAYS come from the server (never from a saved copy), so an old version of the
// site can never be shown. Only the app icons are kept offline.
const CACHE = 'tradepulse-v3';   // a new name deletes every older saved copy
const ICONS = ['/icon-192.png', '/icon-512.png', '/apple-touch-icon.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ICONS)).catch(() => {}).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || !ICONS.includes(url.pathname)) return;
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request)));   // icons: saved copy first
});
