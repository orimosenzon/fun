#!/bin/bash
# מוריד את נתוני מפקד 2022 לפי אזורים סטטיסטיים (כל הארץ) ל-raw/. אחרי זה: python3 build_data.py
set -e
cd "$(dirname "$0")"; mkdir -p raw
B='https://services.arcgis.com/b6zSqPB0TQc3mBgU/arcgis/rest/services/מפקד_2022/FeatureServer/0/query'
for o in 0 2000 4000 6000; do
  curl -s -G --max-time 120 "$B" --data-urlencode f=json --data-urlencode where=1=1 \
    --data-urlencode returnGeometry=false --data-urlencode resultOffset=$o --data-urlencode resultRecordCount=2000 \
    --data-urlencode "outFields=SHEM_YISHUV_HEB,SEMEL_YISHUV,pop_approx,age0_19_pcnt,AcadmCert_pcnt" \
    -o raw/census_$o.json
  n=$(python3 -c "import json;print(len(json.load(open('raw/census_$o.json')).get('features',[])))")
  echo "offset $o: $n"
  [ "$n" -lt 2000 ] && break
done
