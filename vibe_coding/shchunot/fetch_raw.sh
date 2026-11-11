#!/bin/bash
# מוריד מחדש את נתוני המקור ל-raw/ (לא נשמרים בגיט). אחרי זה: python3 build_data.py
set -e
cd "$(dirname "$0")"; mkdir -p raw
op() { # op <file> <overpass query>, עם ניסיונות חוזרים על שרתים שונים
  for i in 1 2 3 4; do
    for srv in https://overpass-api.de/api/interpreter https://overpass.kumi.systems/api/interpreter; do
      curl -s --max-time 300 -A "shchunot-planner" "$srv" --data-urlencode "data=$2" -o "raw/$1"
      head -c 1 "raw/$1" | grep -q '{' && { echo "ok $1"; return 0; }
    done; sleep $((i*15))
  done; echo "FAIL $1"; return 1
}
op boundary.json '[out:json][timeout:240];rel(1392849);out geom;'
op buildings.json '[out:json][timeout:180];way["building"](32.444,34.941,32.506,35.015);out tags geom;'
op streets.json '[out:json][timeout:120];(way["highway"]["name"](32.444,34.941,32.506,35.015);way["railway"="rail"](32.444,34.941,32.506,35.015););out tags geom;'
# מפקד 2022 לפי אזורים סטטיסטיים (סמל יישוב 7800)
curl -s -G --max-time 60 'https://services.arcgis.com/b6zSqPB0TQc3mBgU/arcgis/rest/services/מפקד_2022/FeatureServer/0/query' \
  --data-urlencode f=geojson --data-urlencode "where=SEMEL_YISHUV=7800" --data-urlencode outSR=4326 \
  --data-urlencode "outFields=STAT_2022,YISHUV_STAT_2022,Stat2022_Unite,Main_Function_Txt,ROVA,TAT_ROVA,pop_approx,hh_total_approx,pop_density,age0_19_pcnt,age65_pcnt,AcadmCert_pcnt,employeesAnnual_medWage,change_pcnt" \
  -o raw/stat2022.geojson && echo ok stat2022.geojson
