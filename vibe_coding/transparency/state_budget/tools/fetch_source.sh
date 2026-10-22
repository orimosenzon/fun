#!/bin/bash
# מוריד את קובץ התקציב של משרד האוצר ברמת תקנה, כפי שמפתח התקציב מפרסם אותו (מתעדכן כמה פעמים בשנה).
set -e
cd "$(dirname "$0")/.."
mkdir -p src
curl -fsS -o src/national-budget.csv https://next.obudget.org/datapackages/budget/national/original/data/national-budget.csv
ls -la src/national-budget.csv
