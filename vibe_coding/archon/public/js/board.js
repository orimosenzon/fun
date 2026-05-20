// Board rendering — 9×9 grid with luminosity cycle and piece sprites.

const BOARD_SIZE = 9;
const POWER_POINTS = [
  [0, 0], [0, 8], [8, 0], [8, 8], [4, 4],
];
const MOVE_TWEEN_MS = 350;
const DEATH_FADE_MS = 600;

function isPowerPoint(row, col) {
  return POWER_POINTS.some(([r, c]) => r === row && c === col);
}

function drawBoard(ctx, canvas, gameState, tMs) {
  const cellSize = canvas.width / BOARD_SIZE;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Tiles
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const lum = squareLuminosity(row, col, tMs);
      ctx.fillStyle = luminosityColor(lum);
      ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);

      ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.lineWidth = 1;
      ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
    }
  }

  // Power points (gold pulsing border)
  const ppPulse = 0.7 + Math.sin(tMs / 500) * 0.3;
  for (const [r, c] of POWER_POINTS) {
    ctx.strokeStyle = `rgba(255, 215, 0, ${ppPulse})`;
    ctx.lineWidth = 3;
    ctx.strokeRect(c * cellSize + 4, r * cellSize + 4, cellSize - 8, cellSize - 8);
  }

  // Legal-move highlights
  if (gameState.legalMoves) {
    for (const { row, col } of gameState.legalMoves) {
      const flash = 0.25 + Math.sin(tMs / 250) * 0.1;
      const occupant = gameState.pieces.find(p => p.row === row && p.col === col);
      ctx.fillStyle = occupant
        ? `rgba(220, 80, 80, ${flash + 0.2})`   // red flash if attack
        : `rgba(120, 220, 120, ${flash})`;       // green if empty
      ctx.fillRect(col * cellSize + 3, row * cellSize + 3, cellSize - 6, cellSize - 6);
    }
  }

  // Selected piece highlight
  if (gameState.selectedPiece) {
    const p = gameState.selectedPiece;
    const pulse = 0.6 + Math.sin(tMs / 200) * 0.4;
    ctx.strokeStyle = `rgba(255, 240, 100, ${pulse})`;
    ctx.lineWidth = 4;
    ctx.strokeRect(p.col * cellSize + 2, p.row * cellSize + 2, cellSize - 4, cellSize - 4);
  }

  // Living pieces
  for (const piece of gameState.pieces) {
    drawPieceWithAnimation(ctx, piece, cellSize, tMs, gameState.selectedPiece);
  }

  // Dying pieces (combat fade-out)
  if (gameState.dyingPieces) {
    for (let i = gameState.dyingPieces.length - 1; i >= 0; i--) {
      const d = gameState.dyingPieces[i];
      const age = tMs - d.startMs;
      if (age >= DEATH_FADE_MS) {
        gameState.dyingPieces.splice(i, 1);
        continue;
      }
      const t = age / DEATH_FADE_MS;
      ctx.save();
      ctx.globalAlpha = 1 - t;
      const cx = d.piece.col * cellSize + cellSize / 2;
      const cy = d.piece.row * cellSize + cellSize / 2 - t * cellSize * 0.4;
      drawPieceSprite(ctx, d.piece, cx, cy, cellSize * (1 - t * 0.3), tMs);
      ctx.restore();
    }
  }

  // Combat flash overlay
  if (gameState.combatFlash) {
    const age = tMs - gameState.combatFlash.startMs;
    if (age < 250) {
      const a = 1 - age / 250;
      const { row, col } = gameState.combatFlash;
      ctx.fillStyle = `rgba(255, 80, 60, ${a * 0.55})`;
      ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    } else {
      gameState.combatFlash = null;
    }
  }
}

function drawPieceWithAnimation(ctx, piece, cellSize, tMs, selectedPiece) {
  // Compute drawn position (tween if moving)
  let row = piece.row;
  let col = piece.col;
  if (piece.tween) {
    const t = (tMs - piece.tween.startMs) / MOVE_TWEEN_MS;
    if (t >= 1) {
      piece.tween = null;
    } else {
      const ease = easeOutCubic(t);
      row = piece.tween.fromRow + (piece.row - piece.tween.fromRow) * ease;
      col = piece.tween.fromCol + (piece.col - piece.tween.fromCol) * ease;
    }
  }

  // Idle bob (each piece has its own phase based on id)
  const bobAmp = cellSize * 0.04;
  const bobFreq = 0.0018 + (piece.id % 7) * 0.00015;
  const bob = Math.sin(tMs * bobFreq + piece.id * 0.8) * bobAmp;

  const cx = col * cellSize + cellSize / 2;
  const cy = row * cellSize + cellSize / 2 + bob;

  // Selected scale pulse
  let scale = 1;
  if (selectedPiece && selectedPiece.id === piece.id) {
    scale = 1.08 + Math.sin(tMs / 150) * 0.04;
  }

  ctx.save();
  if (scale !== 1) {
    ctx.translate(cx, cy);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);
  }
  drawPieceSprite(ctx, piece, cx, cy, cellSize, tMs);
  ctx.restore();
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function pixelToCell(x, y, canvas) {
  const cellSize = canvas.width / BOARD_SIZE;
  const col = Math.floor(x / cellSize);
  const row = Math.floor(y / cellSize);
  if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return null;
  return { row, col };
}
