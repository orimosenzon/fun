// The flights this player has flown, kept in the browser (IndexedDB) so any of them can be
// watched again or saved as a film later. Two stores: a small card per flight for the list,
// and the recording itself, which is only read when that flight is opened.
const DB = 'foehn-flights', KEEP = 60;       // a recording is about a megabyte per minute of flight

let dbp = null;
function open() {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    if (!('indexedDB' in self)) return rej(new Error('no IndexedDB'));
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => { r.result.createObjectStore('meta', { keyPath: 'id' }); r.result.createObjectStore('data'); };
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
  dbp.catch(() => { dbp = null; });
  return dbp;
}
const done = (tx) => new Promise((res, rej) => { tx.oncomplete = () => res(); tx.onerror = tx.onabort = () => rej(tx.error); });
const req = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

export const flights = {
  // newest first
  async list() { const db = await open(); const all = await req(db.transaction('meta').objectStore('meta').getAll()); return all.sort((a, b) => b.id - a.id); },
  async get(id) { const db = await open(); return req(db.transaction('data').objectStore('data').get(id)); },
  async put(meta, data) {
    const db = await open(), tx = db.transaction(['meta', 'data'], 'readwrite');
    tx.objectStore('meta').put(meta); tx.objectStore('data').put(data, meta.id);
    await done(tx);
    const all = await this.list();
    for (const m of all.slice(KEEP)) await this.del(m.id);      // the oldest make room
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  },
  async del(id) { const db = await open(), tx = db.transaction(['meta', 'data'], 'readwrite'); tx.objectStore('meta').delete(id); tx.objectStore('data').delete(id); await done(tx); },
};
