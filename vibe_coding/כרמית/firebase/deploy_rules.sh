#!/bin/bash
# Publish firestore.rules to the carmit-riddles project (no firebase CLI login needed, uses gcloud).
set -e
P=carmit-riddles
T=$(~/google-cloud-sdk/bin/gcloud auth print-access-token --account=orimosenzon@gmail.com)
H=(-H "Authorization: Bearer $T" -H "x-goog-user-project: $P" -H "Content-Type: application/json")
BODY=$(python3 -c "import json,sys;print(json.dumps({'source':{'files':[{'name':'firestore.rules','content':open(sys.argv[1]).read()}]}}))" "$(dirname "$0")/firestore.rules")
NAME=$(curl -s -X POST "${H[@]}" "https://firebaserules.googleapis.com/v1/projects/$P/rulesets" -d "$BODY" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('name') or d)")
echo "ruleset: $NAME"
REL="projects/$P/releases/cloud.firestore"
curl -s -X PATCH "${H[@]}" "https://firebaserules.googleapis.com/v1/$REL" -d "{\"release\":{\"name\":\"$REL\",\"rulesetName\":\"$NAME\"}}" | grep -q rulesetName \
  || curl -s -X POST "${H[@]}" "https://firebaserules.googleapis.com/v1/projects/$P/releases" -d "{\"name\":\"$REL\",\"rulesetName\":\"$NAME\"}"
echo "released"
