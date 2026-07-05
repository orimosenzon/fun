"""שירות Cloud Run מינימלי — נועד ללמוד את מסלול הפריסה מקצה לקצה.
מאזין על PORT (Cloud Run מזריק אותו כמשתנה סביבה) ומחזיר עמוד 'חי' פשוט."""
import os

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return "<h1>השכלה — שרת בדיקה חי ✅</h1><p>Cloud Run, פרויקט oris-sandbox.</p>"


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", service="cloudrun-hello")


if __name__ == "__main__":
    # ריצה מקומית בלבד. בענן gunicorn מריץ את השירות (ראה Procfile).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
