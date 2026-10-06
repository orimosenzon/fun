import sys, json
sys.path.insert(0, "/home/ori/fun/vibe_coding/dereh_kitzur")
import build_plans as bp
url = bp.XPLAN.replace("/1/query", "/4/query")
env = {"xmin":34.9718,"ymin":32.4718,"xmax":34.9765,"ymax":32.4760,"spatialReference":{"wkid":4326}}
r = bp.get(url, {"f":"json","geometry":json.dumps(env),"geometryType":"esriGeometryEnvelope","inSR":4326,"outSR":4326,
  "spatialRel":"esriSpatialRelIntersects","outFields":"*","returnGeometry":"true"})
json.dump(r, open("xplan_l4.json","w"), ensure_ascii=False)
print(len(r.get("features",[])), list(r["features"][0]["attributes"].keys()))
from collections import Counter
print(Counter((f["attributes"].get("pl_number"), f["attributes"].get("mavat_name") or f["attributes"].get("landuse_name")) for f in r["features"]).most_common(80))
