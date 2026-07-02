# בדיקות פיזיקה headless

הרצה (מתוך תיקיית tests):
```bash
echo '{"type":"module"}' > package.json
npm install three@0.160.0
cp ../js/{physics,world,noise}.js .
node simtest.js
```
15 בדיקות: מנוחה על המשטח, ריחוף, מהירות שיא בטיסה מאוזנת, פניות בבנק,
גלגולי Z/X, בוסט S, התרסקות בנפילה חופשית, ריספון, ויציבות נומרית ב-60 שניות טיסה.
