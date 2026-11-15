// Builds what the friends globe needs into public/, so the browser never asks another site for
// anything: the projection code as one module, and the coastlines of the world.
// Run again only after upgrading d3-geo, topojson-client or world-atlas: node tools/build_globe.mjs
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const entry = `
export { geoOrthographic, geoPath, geoGraticule10, geoDistance, geoInterpolate } from 'd3-geo';
export { feature } from 'topojson-client';`;

mkdirSync('public/vendor', { recursive: true });
mkdirSync('public/geo', { recursive: true });
execFileSync('npx', ['esbuild', '--bundle', '--format=esm', '--minify', '--outfile=public/vendor/geo.js'], { input: entry, stdio: ['pipe', 'inherit', 'inherit'] });
for (const size of ['110m', '50m']) copyFileSync(`node_modules/world-atlas/land-${size}.json`, `public/geo/land-${size}.json`);

const licence = (name) => `--- ${name} ---\n${readFileSync(`node_modules/${name}/LICENSE`, 'utf8').trim()}\n`;
writeFileSync('public/vendor/LICENSES.txt', [
  'geo.js bundles d3-geo (with d3-array) and topojson-client. geo/land-*.json comes from world-atlas,',
  'which redistributes the public domain Natural Earth coastlines.\n',
  ...['d3-geo', 'd3-array', 'topojson-client', 'world-atlas'].map(licence),
].join('\n'));
