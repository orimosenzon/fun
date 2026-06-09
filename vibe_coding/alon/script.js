// ===== עזרי נתיב לתמונות =====
const imgFull = (gid, n) => `images/${gid}/${n}.jpg`;
const imgThumb = (gid, n) => `images/${gid}/thumbs/${n}.jpg`;

// ===== בניית קישור פנייה (וואטסאפ אם יש טלפון, אחרת מייל) =====
function buildContactLink(extraText) {
  const text = encodeURIComponent(extraText || CONTACT.whatsappText);
  if (CONTACT.phone) return `https://wa.me/${CONTACT.phone}?text=${text}`;
  if (CONTACT.email) return `mailto:${CONTACT.email}?subject=${encodeURIComponent("בקשה לגבי הגיטרות של אלון")}&body=${text}`;
  return "#contact";
}

// ===== מילוי תוכן הנצחה / hero =====
function renderStatic() {
  document.getElementById("bio").textContent = LUTHIER.bio;
  if (LUTHIER.years) document.getElementById("hero-years").textContent = LUTHIER.years;

  // ציטוט מפתח ב-hero ובהנצחה
  if (LUTHIER.quote) {
    document.getElementById("hero-quote").textContent = `״${LUTHIER.quote}״`;
    document.getElementById("memorial-quote").textContent = `״${LUTHIER.quote}״`;
  }

  // פסקאות נוספות בהנצחה
  if (LUTHIER.paragraphs) {
    document.getElementById("bio-extra").innerHTML = LUTHIER.paragraphs
      .map((p) => `<p>${p}</p>`)
      .join("");
  }
  if (LUTHIER.invitation) document.getElementById("invitation").textContent = LUTHIER.invitation;

  // סרטון
  if (typeof VIDEO !== "undefined") {
    const v = document.getElementById("isaac-video");
    v.src = VIDEO.src;
    v.poster = VIDEO.poster;
    document.getElementById("video-title").textContent = VIDEO.title;
    document.getElementById("video-caption").textContent = VIDEO.caption || "";
  }

  // כפתורי יצירת קשר
  const btns = document.getElementById("contact-btns");
  const note = document.getElementById("contact-note");
  let html = "";
  if (CONTACT.phone) {
    html += `<a class="btn wa" href="https://wa.me/${CONTACT.phone}?text=${encodeURIComponent(CONTACT.whatsappText)}" target="_blank" rel="noopener">💬 וואטסאפ</a>`;
  }
  if (CONTACT.email) {
    html += `<a class="btn mail" href="mailto:${CONTACT.email}">✉️ אימייל</a>`;
  }
  if (!CONTACT.phone && !CONTACT.email) {
    html = `<span class="todo-badge">⚠ יש להוסיף טלפון/אימייל בקובץ data.js</span>`;
    note.textContent = "כרגע לא הוגדרו פרטי יצירת קשר. ערכו את האובייקט CONTACT בקובץ data.js.";
  }
  btns.innerHTML = html;
}

// ===== כרטיסי הקטלוג =====
function renderGrid() {
  const grid = document.getElementById("grid");
  grid.innerHTML = GUITARS.map((g) => `
    <article class="card" data-id="${g.id}">
      <div class="photo"><img loading="lazy" src="${imgThumb(g.id, g.images[0])}" alt="${g.name}" /></div>
      <div class="body">
        <h3>${g.name}</h3>
        <div class="sub">${g.subtitle}</div>
        <div class="desc">${g.short}</div>
        <div class="foot">
          <span class="price">${g.price}</span>
          <span class="more">פרטים ותמונות</span>
        </div>
      </div>
    </article>
  `).join("");

  grid.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => openModal(card.dataset.id));
  });
}

// ===== מודאל גיטרה =====
let currentGuitar = null;
let currentIdx = 0;

function openModal(id) {
  const g = GUITARS.find((x) => x.id === id);
  if (!g) return;
  currentGuitar = g;
  currentIdx = 0;

  document.getElementById("m-name").textContent = g.name;
  document.getElementById("m-sub").textContent = g.subtitle;
  document.getElementById("m-story").textContent = g.story;
  document.getElementById("m-price").textContent = g.price;

  // מפרט
  document.getElementById("m-specs").innerHTML = Object.entries(g.specs)
    .map(([k, v]) => `<li><span class="k">${k}</span><span class="v">${v}</span></li>`)
    .join("");

  // כפתור פנייה
  document.getElementById("m-buy").href = buildContactLink(`שלום, אשמח לפרטים על הגיטרה "${g.name}" של אלון`);

  // גלריה
  setMainImage(0);
  document.getElementById("m-thumbs").innerHTML = g.images
    .map((n, i) => `<img data-i="${i}" src="${imgThumb(g.id, n)}" alt="${g.name} ${i + 1}" class="${i === 0 ? "active" : ""}" />`)
    .join("");
  document.querySelectorAll("#m-thumbs img").forEach((t) => {
    t.addEventListener("click", () => setMainImage(parseInt(t.dataset.i, 10)));
  });

  document.getElementById("modal").classList.add("open");
  document.body.style.overflow = "hidden";
}

function setMainImage(i) {
  currentIdx = i;
  const g = currentGuitar;
  document.getElementById("m-main").src = imgFull(g.id, g.images[i]);
  document.querySelectorAll("#m-thumbs img").forEach((t, j) => t.classList.toggle("active", j === i));
}

function closeModal() {
  document.getElementById("modal").classList.remove("open");
  document.body.style.overflow = "";
}

// ===== Lightbox מסך מלא =====
function openLightbox() {
  document.getElementById("lb-img").src = imgFull(currentGuitar.id, currentGuitar.images[currentIdx]);
  document.getElementById("lightbox").classList.add("open");
}
function lbNav(dir) {
  const len = currentGuitar.images.length;
  currentIdx = (currentIdx + dir + len) % len;
  setMainImage(currentIdx);
  document.getElementById("lb-img").src = imgFull(currentGuitar.id, currentGuitar.images[currentIdx]);
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
}

// ===== חיווט אירועים =====
function wireEvents() {
  document.getElementById("m-close").addEventListener("click", closeModal);
  document.getElementById("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") closeModal();
  });
  document.getElementById("m-main").addEventListener("click", openLightbox);

  document.getElementById("lb-close").addEventListener("click", closeLightbox);
  document.getElementById("lb-prev").addEventListener("click", () => lbNav(1));   // RTL: הקודם = ימינה
  document.getElementById("lb-next").addEventListener("click", () => lbNav(-1));
  document.getElementById("lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox") closeLightbox();
  });

  document.addEventListener("keydown", (e) => {
    if (document.getElementById("lightbox").classList.contains("open")) {
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowRight") lbNav(1);
      if (e.key === "ArrowLeft") lbNav(-1);
    } else if (document.getElementById("modal").classList.contains("open")) {
      if (e.key === "Escape") closeModal();
    }
  });
}

// ===== אתחול =====
renderStatic();
renderGrid();
wireEvents();
