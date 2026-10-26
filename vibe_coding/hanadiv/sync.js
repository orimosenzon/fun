/* דרך הנדיב · the personal plan on every device · (C) 2026 Ori Mosenzon · GPL-3.0-or-later

   Signing in with Google (or Facebook, once its app is live) keeps the plan in one
   Firestore document, plans/{uid}, that only its owner can read or write: event ids and a
   time, nothing else. The browser keeps its own copy as before, so the app works the same
   signed out or offline. The first sign-in on a device joins the two plans; after that the
   account is the plan, and a change on one device shows on the others within a second.

   Firebase is loaded only for someone who signs in, so the map stays as quick as it was.
   The web config below is public by design; the security is in firebase/firestore.rules. */

const FB = {
  apiKey: 'AIzaSyBoRa0LenTz39r2vX7DiGKEahP7-Ln8DuE',
  authDomain: 'hanadiv-2026.firebaseapp.com',
  projectId: 'hanadiv-2026',
  appId: '1:403203200136:web:975e7e98e32929130fbb3d',
};
const FB_SDK = 'https://www.gstatic.com/firebasejs/10.14.1/';
// Each joins once it is switched on in the Firebase console (Facebook also needs an app on
// Meta's side). With none, nothing about accounts shows.
const PROVIDERS = ['google'];
const SIGNED = 'hanadiv-signed';  // this browser has a session to restore

let fb = null, unsub = null, pushTimer = null, applying = false;
S.user = null; S.synced = false;

async function fbLoad() {
  if (fb) return fb;
  const [app, au, fs] = await Promise.all(['app', 'auth', 'firestore'].map(m => import(`${FB_SDK}firebase-${m}.js`)));
  const a = app.initializeApp(FB);
  fb = { au, fs, auth: au.getAuth(a), db: fs.getFirestore(a) };
  fb.auth.languageCode = 'he';
  au.onAuthStateChanged(fb.auth, onUser);
  au.getRedirectResult(fb.auth).catch(signInError);
  return fb;
}

function onUser(u) {
  if (unsub) { unsub(); unsub = null; }
  S.synced = false;
  if (!u) {
    S.user = null;
    try { localStorage.removeItem(SIGNED); } catch (e) { /* private window */ }
    drawAcct();
    return render();
  }
  S.user = { uid: u.uid, name: u.displayName || u.email || 'החשבון שלך', photo: u.photoURL };
  try { localStorage.setItem(SIGNED, '1'); } catch (e) { /* private window */ }
  const ref = fb.fs.doc(fb.db, 'plans', u.uid);
  unsub = fb.fs.onSnapshot(ref, snap => {
    if (snap.metadata.hasPendingWrites) return;  // our own write coming back
    const remote = (snap.exists() ? snap.data().ids || [] : []).filter(id => S.byId.has(id));
    if (!S.synced) {
      // first contact on this device: whatever was starred here joins what the account had
      const merged = [...new Set([...remote, ...S.plan])];
      S.synced = true;
      setPlan(merged);
      if (merged.length !== remote.length || !snap.exists()) pushPlan(true);
      const added = merged.length - remote.length;
      if (added && remote.length) toast(`${added} אירועים מהמכשיר הזה נוספו לתוכנית שבחשבון`);
      return;
    }
    setPlan(remote);
  }, () => toast('לא הצלחנו לקרוא את התוכנית מהחשבון. היא נשמרת במכשיר הזה בינתיים.'));
  drawAcct();
  render();
}

// the plan from the account replaces the one on screen
function setPlan(ids) {
  const same = ids.length === S.plan.size && ids.every(id => S.plan.has(id));
  if (same) return render();
  applying = true;
  S.plan = new Set(ids);
  savePlan();
  applying = false;
  redrawAll(); render();
  if (S.detail && typeof redrawDetail === 'function') redrawDetail();
}

// savePlan() calls this after every change made on this device
function planChanged() { if (S.user && S.synced && !applying) pushPlan(); }
function pushPlan(now) {
  clearTimeout(pushTimer);
  pushTimer = setTimeout(() => {
    if (!fb || !S.user) return;
    fb.fs.setDoc(fb.fs.doc(fb.db, 'plans', S.user.uid), { ids: [...S.plan], updated: fb.fs.serverTimestamp() })
      .catch(() => toast('השינוי נשמר במכשיר, אבל לא הגיע לחשבון. ננסה שוב בשינוי הבא.'));
  }, now ? 0 : 500);
}

/* ---------- signing in and out ---------- */

// Google refuses to sign in inside the browsers built into Facebook, Instagram and friends
const inApp = /FBAN|FBAV|FB_IAB|Instagram|Line\/|TikTok|; wv\)/.test(navigator.userAgent);

async function signIn(kind) {
  let f;
  try { f = await fbLoad(); } catch (e) { return toast('לא הצלחנו להתחבר לשירות. כדאי לבדוק את החיבור לאינטרנט.'); }
  const p = kind === 'facebook' ? new f.au.FacebookAuthProvider() : new f.au.GoogleAuthProvider();
  if (kind === 'google') p.setCustomParameters({ prompt: 'select_account' });
  try {
    await f.au.signInWithPopup(f.auth, p);
    closeLogin();
    toast('התחברתם. התוכנית שלכם תופיע בכל מכשיר שתתחברו ממנו');
  } catch (e) {
    if (['auth/popup-blocked', 'auth/operation-not-supported-in-this-environment'].includes(e.code)) {
      try { localStorage.setItem(SIGNED, '1'); } catch (x) { /* private window */ }
      return f.au.signInWithRedirect(f.auth, p);
    }
    signInError(e);
  }
}
function signInError(e) {
  if (!e || ['auth/popup-closed-by-user', 'auth/cancelled-popup-request', 'auth/user-cancelled'].includes(e.code)) return;
  if (e.code === 'auth/account-exists-with-different-credential') return toast('המייל הזה כבר מחובר דרך הספק השני. כדאי להיכנס דרכו.');
  if (e.code === 'auth/unauthorized-domain') return toast('ההתחברות עוד לא פתוחה בכתובת הזאת.');
  toast('ההתחברות לא הצליחה. אפשר לנסות שוב.');
}
async function signOut() {
  if (!fb) return;
  await fb.au.signOut(fb.auth);
  toast('יצאתם מהחשבון. התוכנית נשארת גם במכשיר הזה.');
}

// the account's copy goes; the one in this browser stays
async function forget() {
  if (!fb || !S.user || !confirm('למחוק את התוכנית מהחשבון ולצאת? במכשיר הזה היא תישאר.')) return;
  try {
    if (unsub) { unsub(); unsub = null; }
    await fb.fs.deleteDoc(fb.fs.doc(fb.db, 'plans', S.user.uid));
    await fb.au.signOut(fb.auth);
    toast('התוכנית נמחקה מהחשבון');
  } catch (e) { toast('המחיקה לא הצליחה. אפשר לנסות שוב.'); }
}

const BRAND = {
  google: ['Google', '<svg viewBox="0 0 48 48"><path fill="#4285F4" d="M45.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h11.8a10 10 0 0 1-4.4 6.6v5.5h7.1c4.2-3.8 6.6-9.5 6.6-16.1z"/><path fill="#34A853" d="M24 46c5.9 0 10.9-2 14.5-5.3l-7.1-5.5c-2 1.3-4.5 2.1-7.4 2.1-5.7 0-10.6-3.9-12.3-9.1H4.4v5.7A22 22 0 0 0 24 46z"/><path fill="#FBBC05" d="M11.7 28.2a13.2 13.2 0 0 1 0-8.4v-5.7H4.4a22 22 0 0 0 0 19.8z"/><path fill="#EA4335" d="M24 10.7c3.2 0 6.1 1.1 8.4 3.3l6.3-6.3A22 22 0 0 0 4.4 14.1l7.3 5.7c1.7-5.2 6.6-9.1 12.3-9.1z"/></svg>'],
  facebook: ['Facebook', '<svg viewBox="0 0 24 24"><path fill="#fff" d="M13.5 21v-7.5h2.5l.4-3h-2.9V8.6c0-.9.3-1.5 1.5-1.5h1.5V4.4a20 20 0 0 0-2.2-.1c-2.2 0-3.7 1.3-3.7 3.8v2.4H8v3h2.6V21z"/></svg>'],
};
function openLogin() {
  const box = $('#loginPanel');
  box.hidden = false;
  if (S.user) {
    box.innerHTML = `<div class="sphead"><b>התוכנית שלי בכל מכשיר</b><button class="iconbtn" data-loginclose aria-label="סגירה">${ICON.x}</button></div>
      ${syncBarHTML()}
      <p class="shnote">כל מכשיר שמתחברים ממנו לאותו חשבון מציג את אותה תוכנית.</p>`;
    return;
  }
  box.innerHTML = `<div class="sphead"><b>התוכנית שלי בכל מכשיר</b><button class="iconbtn" data-loginclose aria-label="סגירה">${ICON.x}</button></div>
    <p class="shnote">מתחברים פעם אחת בכל מכשיר, בטלפון ובמחשב, והתוכנית אותה תוכנית: כוכב שמסמנים באחד מופיע מיד בשני. מה שכבר סימנתם כאן מצטרף לחשבון.</p>
    ${inApp ? '<p class="loginwarn">גוגל לא מאפשרת להתחבר מתוך הדפדפן של פייסבוק, אינסטגרם או טיקטוק. כדאי לפתוח את הקישור בכרום או בספארי (בתפריט ⋮ או ···).</p>' : ''}
    ${PROVIDERS.map(k => `<button class="btn login ${k}" data-signin="${k}">${BRAND[k][1]}<span>כניסה עם ${BRAND[k][0]}</span></button>`).join('')}
    <p class="loginfine">נשמרים רק מספרי האירועים שסימנתם, בלי פרסום ובלי מיילים. <a href="privacy.html" target="_blank" rel="noopener">פרטיות</a></p>`;
}
function closeLogin() { $('#loginPanel').hidden = true; }

// the round button in the toolbar: a person when signed out, the account's picture when in
const PERSON = '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>';
function drawAcct() {
  const b = $('#acctBtn');
  if (!b) return;
  b.hidden = !PROVIDERS.length;
  b.classList.toggle('in', !!S.user);
  b.innerHTML = S.user && S.user.photo ? `<img src="${esc(S.user.photo)}" alt="" referrerpolicy="no-referrer">` : PERSON;
  b.title = b.ariaLabel = S.user ? `מחובר: ${S.user.name}` : 'התוכנית שלי בכל מכשיר';
}

// the line in the plan view that says where the plan lives
function syncBarHTML() {
  if (!PROVIDERS.length) return '';
  if (!S.user) {
    // with a single way in, the button signs in right away; otherwise it opens the choice
    const one = PROVIDERS.length === 1 && !inApp ? PROVIDERS[0] : null;
    const act = one ? `data-signin="${one}"` : 'data-login';
    const label = one ? `${BRAND[one][1]}<span>כניסה עם ${BRAND[one][0]}</span>` : '<span>כניסה לחשבון</span>';
    return `<div class="syncbar off"><span>${ICON.cloud}<span><b>התוכנית שמורה רק במכשיר הזה.</b> מתחברים כדי לראות אותה גם בטלפון וגם במחשב.</span></span><button class="btn login ${one || ''}" ${act}>${label}</button></div>`;
  }
  const pic = S.user.photo ? `<img src="${esc(S.user.photo)}" alt="" referrerpolicy="no-referrer">` : ICON.cloud;
  return `<div class="syncbar on"><span>${pic}${S.synced ? 'מסונכרן עם החשבון של' : 'מתחבר לחשבון של'} ${esc(S.user.name)}</span><span class="syncact"><button class="more" data-logout>יציאה</button><button class="more" data-forget>מחיקה מהחשבון</button></span></div>`;
}

document.addEventListener('click', ev => {
  const t = ev.target;
  if (t.closest('[data-login]')) return openLogin();
  if (t.closest('[data-loginclose]')) return closeLogin();
  const si = t.closest('[data-signin]'); if (si) return signIn(si.dataset.signin);
  if (t.closest('[data-logout]')) return signOut();
  if (t.closest('[data-forget]')) return forget();
  if (!$('#loginPanel').hidden && !t.closest('#loginPanel')) closeLogin();
});
document.addEventListener('keydown', ev => { if (ev.key === 'Escape' && !$('#loginPanel').hidden) closeLogin(); });

// a session from an earlier visit (or a sign-in that went by redirect) comes back by itself;
// main() calls this once the programme is loaded, so the account's ids have events to match
function syncBoot() {
  drawAcct();
  try { if (localStorage.getItem(SIGNED)) fbLoad().catch(() => {}); } catch (e) { /* private window */ }
}
