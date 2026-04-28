# surfaces — ידע כללי על הפרויקט

## מטרה
דוגמה אינטראקטיבית למושג **משטח (surface)** במתמטיקה — העתקה רציפה
f : R² → R³. בוחרים מתוך קטלוג של פרמטריזציות קלאסיות וצופים בכל אחת
ב-3D עם תאורה, סיבוב אוטומטי וסיבוב ידני בעכבר.

## סטאק
- C++17
- Qt 6.2 (Widgets + OpenGL + OpenGLWidgets)
- OpenGL 3.3 Core
- CMake ≥ 3.16

## מבנה
- `src/Surface.{h,cpp}` — מחלקת Surface אבסטרקטית + הקטלוג:
  Sphere, Torus, HelixTube, ConicalSpiral, Mobius, Klein, Saddle.
  כל מחלקה מממשת `evaluate(u,v) → R³` ותחום הפרמטרים. ה-mesh נבנה
  אוטומטית ע"י `Surface::tessellate()` ונורמלים מחושבים בהפרשים סופיים.
- `src/SurfaceView.{h,cpp}` — `QOpenGLWidget` שמרנדר את ה-mesh, מנהל מצלמה
  אורביטלית ושולח uniforms לשיידרים.
- `src/MainWindow.{h,cpp}` — UI: רשימת משטחים משמאל, פאנל הסבר, צ'קבוקס
  סיבוב אוטומטי ו-reset.
- `shaders/surface.{vert,frag}` — Phong + צביעה לפי תחום הפרמטרים + grid עדין.
- `shaders.qrc` — מארז את השיידרים לבינארי.

## בנייה והרצה
```bash
cd ~/fun/vibe_coding/surfaces
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/surfaces
```

## תלויות מערכת
- Ubuntu 22.04: `sudo apt-get install -y cmake qt6-base-dev libgl1-mesa-dev`

## עניינים טכניים שכדאי לזכור
- **VAO/IBO על Qt6+NVIDIA**: חובה לשחרר את ה-VAO לפני ה-IBO. שחרור IBO
  בזמן שה-VAO עוד מקושר מבטל את הקישור בתוך ה-VAO וגורם ל-segfault
  ב-`glDrawElements` (drv NVIDIA). הסדר ב-`uploadMesh()`: vao.release()
  → vbo.release() → ibo.release().
- **תאורה דו-צדדית**: משטחים כמו רצועת מביוס/קליין דורשים `gl_FrontFacing`
  כדי להפוך את הנורמל בצד האחורי. הוטמע ב-fragment shader.
- **נירמול mesh**: כל משטח מנורמל למסגרת [-1,1]³ ב-`SurfaceView::normalizeMesh`
  כדי שכל הדגמים ייראו באותו גודל ובמיקום מרכזי, בלי קשר לפרמטרים שלהם.
- **ניבוט (frame) לאורך עקומה**: `HelixTube` ו-`ConicalSpiral` בונים פריים
  Frenet-משופר (אורתוגונליזציה של Z העולמי על המישור הניצב ל-T) — זה יציב
  יותר מ-Frenet מלא כשהעקמומיות זניחה.

## רעיונות להמשך
- שליטה אינטראקטיבית בפרמטרים (R, r, מספר סיבובים) דרך QSpinBox.
- מצב wireframe / נקודות.
- ייצוא ל-OBJ.
- הצגת שדות מתמטיים נוספים: עיקול גאוסי בצבע, וקטורים נורמליים.
