# cloudrun-hello — שרת בדיקה ללימוד מסלול הפריסה

שירות Flask מינימלי שנפרס ל-Cloud Run בפרויקט ה-sandbox (`oris-sandbox-501014`),
כדי לוודא שכל צינור הפריסה עובד לפני שנשים שם את הבודק האמיתי.

## מבנה
- `main.py` — אפליקציית Flask (`/` דף חי, `/healthz` בדיקת בריאות).
- `requirements.txt` — Flask + gunicorn.
- `Procfile` — פקודת ההרצה בענן (gunicorn). Cloud Run buildpacks קורא אותו.
- `deploy.sh` — פקודת הפריסה.

## דרישה חד-פעמית: אימות
gcloud מותקן ב-`~/google-cloud-sdk/bin/`. הוסף ל-PATH (או השתמש בנתיב מלא):

```bash
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
gcloud auth login                       # ייפתח דפדפן / ייתן URL+קוד
gcloud config set project oris-sandbox-501014
```

## פריסה
```bash
cd ~/fun/vibe_coding/haskala/infra/cloudrun-hello
./deploy.sh
```

בפריסה הראשונה gcloud יציע להפעיל את ה-APIs הדרושים
(`run`, `cloudbuild`, `artifactregistry`) — אשר. בסיום יודפס **Service URL**.

## הרצה מקומית (לא חובה)
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
PORT=8080 ./.venv/bin/gunicorn --bind :8080 main:app
# → http://localhost:8080
```

## מלכודת אפשרית
בארגוני Workspace יש לפעמים org policy (`iam.allowedPolicyMemberDomains`) שחוסמת
`--allow-unauthenticated`. אם הפריסה נכשלת על כך — או שמסירים את הדגל (וניגשים
עם token), או שמתירים את המדיניות לפרויקט זה ב-IAM.
