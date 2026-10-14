# פרדס חנה תלת-ממד
#   make setup   סביבת פייתון
#   make data    משיכת נתונים גולמיים (OSM, גבהים)
#   make extra   מקורות נוספים: מיקרוסופט, GHSL, לוויין, Commons, פרדספדיה
#   make world   בניית העולם ל-game/world
#   make web     ייצוא המודל לדרך קיצור (web/data/model3d)
#   make run     הרצת המשחק
#   make shot    צילום מסך לבדיקה (shot.png)

PY = .venv/bin/python
GODOT ?= godot

.PHONY: setup data extra world web run shot editor

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

data:
	cd pipeline && ../$(PY) fetch_osm.py && ../$(PY) fetch_dem.py

extra:
	cd pipeline && ../$(PY) fetch_extra.py

world:
	cd pipeline && ../$(PY) build_world.py

web:
	cd pipeline && ../$(PY) export_web.py

run:
	$(GODOT) --path game

editor:
	$(GODOT) --path game --editor

shot:
	$(GODOT) --path game -- --shot=$(CURDIR)/shot.png $(ARGS)
