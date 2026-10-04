/* store.js: שמירת המיתוג וההגדרות ב-IndexedDB של הדפדפן.
 * הלוגו והסרטונים הם Blob-ים, ולכן IndexedDB ולא localStorage.
 * אם הדפדפן חוסם אחסון (חלון פרטי וכו') הכול ממשיך לעבוד, רק בלי זיכרון.
 */
window.C = window.C || {};

C.store = (() => {
  const DB = 'clipper';
  const OS = 'kv';
  let dbp = null;

  function db() {
    if (!dbp) {
      dbp = new Promise((res, rej) => {
        let req;
        try { req = indexedDB.open(DB, 1); } catch (e) { rej(e); return; }
        req.onupgradeneeded = () => req.result.createObjectStore(OS);
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    }
    return dbp;
  }

  async function get(key) {
    try {
      const d = await db();
      return await new Promise((res, rej) => {
        const r = d.transaction(OS).objectStore(OS).get(key);
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
    } catch (e) { console.warn('store.get', key, e); return undefined; }
  }

  async function set(key, val) {
    try {
      const d = await db();
      await new Promise((res, rej) => {
        const tx = d.transaction(OS, 'readwrite');
        if (val === undefined || val === null) tx.objectStore(OS).delete(key);
        else tx.objectStore(OS).put(val, key);
        tx.oncomplete = res;
        tx.onerror = () => rej(tx.error);
      });
    } catch (e) { console.warn('store.set', key, e); }
  }

  return { get, set };
})();
