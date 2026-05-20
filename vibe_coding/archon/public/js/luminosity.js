// Luminosity cycle — squares shift through 4 brightness levels over time.
// In the original Archon, the cycle drifts continuously; pieces are stronger
// on squares matching their alignment (light pieces on bright squares, dark on dark).

const LUMINOSITY_LEVELS = 4;
const CYCLE_DURATION_MS = 60_000; // 1 minute for a full cycle (placeholder)

// Each square has a fixed "phase offset" derived from (row+col), creating a checker-like wave.
function squarePhaseOffset(row, col) {
  return ((row + col) % LUMINOSITY_LEVELS) / LUMINOSITY_LEVELS;
}

// Returns the current luminosity index (0..3) for a given square at time t (ms).
// 0 = darkest, 3 = brightest.
function squareLuminosity(row, col, tMs) {
  const offset = squarePhaseOffset(row, col);
  const cyclePos = ((tMs / CYCLE_DURATION_MS) + offset) % 1;
  // Triangle wave: 0 -> 1 -> 0 mapped to 0..LUMINOSITY_LEVELS-1
  const tri = cyclePos < 0.5 ? cyclePos * 2 : (1 - cyclePos) * 2;
  return Math.floor(tri * (LUMINOSITY_LEVELS - 0.001));
}

// Colors for each luminosity level — darkest to brightest.
const LUMINOSITY_COLORS = [
  '#1a1a24',  // 0 — pure dark
  '#3a3a48',  // 1
  '#7a7a8c',  // 2
  '#c0c0d0',  // 3 — pure light
];

function luminosityColor(level) {
  return LUMINOSITY_COLORS[level] || LUMINOSITY_COLORS[0];
}

// Returns the global cycle phase as a label for the HUD.
function globalCyclePhaseLabel(tMs) {
  const pos = (tMs / CYCLE_DURATION_MS) % 1;
  if (pos < 0.25) return 'Dawn';
  if (pos < 0.50) return 'Day';
  if (pos < 0.75) return 'Dusk';
  return 'Night';
}

// Strength bonus for a piece on this square: +1 if matches alignment, -1 if opposite, 0 if neutral.
// Levels 0-1 = dark territory, 2-3 = light territory.
function alignmentBonus(side, luminosityLevel) {
  const isDarkSquare = luminosityLevel <= 1;
  const isLightSquare = luminosityLevel >= 2;
  if (side === 'light' && isLightSquare) return 1;
  if (side === 'light' && isDarkSquare) return -1;
  if (side === 'dark'  && isDarkSquare) return 1;
  if (side === 'dark'  && isLightSquare) return -1;
  return 0;
}
