# surfaces

הדגמה אינטראקטיבית של המושג **משטח (surface)** במתמטיקה — העתקה
פרמטרית מ-R² ל-R³. בוחרים מתוך קטלוג של פרמטריזציות קלאסיות וצופים
בכל אחת ב-3D עם תאורה, סיבוב חופשי, וטאב הסבר עם דיאגרמה תלת-ממדית
המראה איך מגיעים לנוסחה.

## מה כלול

המשטחים בקטלוג:

| משטח | פרמטריזציה | הערה |
|------|------------|------|
| Sphere | `(sin u·cos v, sin u·sin v, cos u)` | קואורדינטות כדוריות |
| Torus | `((R + r cos v)·cos u, (R + r cos v)·sin u, r sin v)` | סיבוב מעגל סביב ציר |
| 3D Helix tube | צינור סביב ספירלה הליקואידית | בנוי על מסגרת (T,N,B) |
| Conical spiral | ספירלה שרדיוסה מתכווץ עם הגובה | קליפת חילזון |
| Möbius strip | `((1+v cos(u/2))·cos u, (1+v cos(u/2))·sin u, v sin(u/2))` | משטח חד-צדדי |
| Klein bottle | אימוס קליין סטנדרטי | משטח סגור, חד-צדדי, ללא שפה |
| Saddle | `(u, v, u·v)` | פרבולואיד היפרבולי |

לכל משטח: טאב **Definition** עם הנוסחה, וטאב **How it's derived** עם
הסבר אינטואיטיבי + דיאגרמה תלת-ממדית אינטראקטיבית שניתן לסובב כדי
לראות את הזוויות והמרחקים מכל זווית מבט.

## בנייה והרצה

### תלויות

- C++17
- Qt 6.2 או חדש יותר (Core, Gui, Widgets, OpenGL, OpenGLWidgets)
- CMake ≥ 3.16
- OpenGL 3.3 Core (כרטיס מסך + דרייבר תקני)

### Ubuntu / Debian

```bash
sudo apt-get install -y cmake qt6-base-dev libgl1-mesa-dev
```

### Fedora

```bash
sudo dnf install -y cmake qt6-qtbase-devel mesa-libGL-devel
```

### Arch

```bash
sudo pacman -S cmake qt6-base
```

### macOS (Homebrew)

```bash
brew install cmake qt
```

### בנייה

```bash
git clone https://github.com/orimosenzon/fun.git
cd fun/vibe_coding/surfaces

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/surfaces
```

## שימוש

- **בחירת משטח** — לחיצה על שם ברשימה משמאל.
- **חלונית 3D ראשית (ימין)**:
  - גרירה בעכבר — סיבוב המבט.
  - גלגל — זום.
  - צ'קבוקס "Auto-rotate" — סיבוב אוטומטי.
  - "Reset view" — איפוס המצלמה.
- **טאב "Definition"** — הנוסחה ותחומי הפרמטרים.
- **טאב "How it's derived"** — דיאגרמה תלת-ממדית של הזוויות והמרחקים
  (גרירה לסיבוב, גלגל לזום), והסבר אינטואיטיבי איך מגיעים לנוסחה.

## מבנה הקוד

```
surfaces/
├── CMakeLists.txt
├── shaders.qrc
├── shaders/
│   ├── surface.vert       — Phong vertex shader
│   └── surface.frag       — Phong + צביעה לפי הפרמטרים + grid
└── src/
    ├── main.cpp           — נקודת כניסה, יצירת QSurfaceFormat ברירת מחדל
    ├── MainWindow.{h,cpp} — UI: רשימה, טאבים, חלוניות
    ├── SurfaceView.{h,cpp}      — QOpenGLWidget של המשטח הראשי
    ├── Surface.{h,cpp}          — מחלקה אבסטרקטית + 7 משטחים קונקרטיים
    ├── Derivation.{h,cpp}       — טקסט ההסבר לכל משטח
    ├── DiagramScene.{h,cpp}     — תיאור מופשט של דיאגרמה תלת-ממדית
    └── DerivationView3D.{h,cpp} — QOpenGLWidget שמרנדר את הדיאגרמה
```

## הוספת משטח חדש

1. בקובץ [src/Surface.h](src/Surface.h) — הוסף תת-מחלקה של `Surface`.
2. בקובץ [src/Surface.cpp](src/Surface.cpp) — מימוש `evaluate(u,v)` ותחומי u, v;
   הוסף מופע ל-`allSurfaces()`.
3. בקובץ [src/MainWindow.cpp](src/MainWindow.cpp) → `descriptionFor` — הוסף תיאור.
4. בקובץ [src/Derivation.cpp](src/Derivation.cpp) — הוסף טקסט הסבר ב-`derivationTextFor`.
5. בקובץ [src/DiagramScene.cpp](src/DiagramScene.cpp) — בנה סצנה ב-`sceneFor`.

ה-mesh נבנה אוטומטית ב-`Surface::tessellate()` עם נורמלים בהפרשים סופיים.

## רישיון

MIT
