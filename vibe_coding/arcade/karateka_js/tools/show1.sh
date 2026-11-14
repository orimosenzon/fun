#!/bin/bash
# compact one-line view of a range of a study listing: tools/show1.sh A B [scene]
node "$(dirname "$0")/show.mjs" "$1" "$2" "${3:-game}" | paste -sd'|' | sed 's/|/ | /g; s/ | L_/\n L_/g; s/ | sub_/\n\nsub_/g; s/ | ---/\n ---/g'
