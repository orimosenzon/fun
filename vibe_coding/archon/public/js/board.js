// Board rendering — 9×9 grid with luminosity cycle and piece sprites (placeholder rectangles).

const BOARD_SIZE = 9;
const POWER_POINTS = [
  [0, 0], [0, 8], [8, 0], [8, 8], [4, 4],
];

function isPowerPoint(row, col) {
  return POWER_POINTS.some(([r, c]) => r === row && c === col);
}

function drawBoard(ctx, canvas, gameState, tMs) {
  const cellSize = canvas.width / BOARD_SIZE;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      const lum = squareLuminosity(row, col, tMs);
      ctx.fillStyle = luminosityColor(lum);
      ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);

      if (isPowerPoint(row, col)) {
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 3;
        ctx.strokeRect(col * cellSize + 4, row * cellSize + 4, cellSize - 8, cellSize - 8);
      }

      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
    }
  }

  if (gameState.legalMoves) {
    for (const { row, col } of gameState.legalMoves) {
      ctx.fillStyle = 'rgba(120, 220, 120, 0.35)';
      ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    }
  }

  if (gameState.selectedPiece) {
    const p = gameState.selectedPiece;
    ctx.strokeStyle = '#ffff66';
    ctx.lineWidth = 3;
    ctx.strokeRect(p.col * cellSize + 2, p.row * cellSize + 2, cellSize - 4, cellSize - 4);
  }

  for (const piece of gameState.pieces) {
    drawPiece(ctx, piece, cellSize);
  }
}

function drawPiece(ctx, piece, cellSize) {
  const def = PIECE_TYPES[piece.type];
  const cx = piece.col * cellSize + cellSize / 2;
  const cy = piece.row * cellSize + cellSize / 2;
  const r = cellSize * 0.35;

  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = def.color;
  ctx.fill();
  ctx.strokeStyle = def.side === 'light' ? '#fff' : '#000';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = def.side === 'light' ? '#000' : '#fff';
  ctx.font = `bold ${Math.floor(cellSize * 0.28)}px monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(def.name[0].toUpperCase(), cx, cy);

  if (def.isSovereign) {
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function pixelToCell(x, y, canvas) {
  const cellSize = canvas.width / BOARD_SIZE;
  const col = Math.floor(x / cellSize);
  const row = Math.floor(y / cellSize);
  if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return null;
  return { row, col };
}
