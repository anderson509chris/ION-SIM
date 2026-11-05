#!/bin/sh
# Rebuild ../viewer.js from viewer.js and three.js (needs Node.js): sh build.sh
set -e
cd "$(dirname "$0")"
[ -d node_modules ] || npm install --no-audit --no-fund
npx esbuild viewer.js --bundle --minify --format=iife --legal-comments=eof --outfile=../viewer.js
