// Client-side game state + turn logic. Hot-seat for skeleton stage.
// Server-authoritative validation will replace this in the multiplayer milestone.

function createInitialGameState() {
  return {
    pieces: createStartingPieces(),
    turn: 'light',
    selectedPiece: null,
    legalMoves: null,
    pendingCombat: null,
    dyingPieces: [],
    combatFlash: null,
    winner: null,
  };
}

function pieceAt(gameState, row, col) {
  return gameState.pieces.find(p => p.row === row && p.col === col);
}

// Legal moves: any empty square within `range` (Chebyshev distance, like a king with longer step).
// Flying pieces can pass over other pieces; ground pieces must have a clear line in 8 directions.
function computeLegalMoves(gameState, piece) {
  const def = PIECE_TYPES[piece.type];
  const moves = [];

  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      if (row === piece.row && col === piece.col) continue;
      const dRow = row - piece.row;
      const dCol = col - piece.col;
      const dist = Math.max(Math.abs(dRow), Math.abs(dCol));
      if (dist > def.range) continue;

      if (!def.flies) {
        if (!isPathClear(gameState, piece, row, col)) continue;
      }

      const occupant = pieceAt(gameState, row, col);
      if (occupant) {
        if (PIECE_TYPES[occupant.type].side === def.side) continue; // can't land on ally
      }
      moves.push({ row, col });
    }
  }
  return moves;
}

function isPathClear(gameState, piece, targetRow, targetCol) {
  const dRow = Math.sign(targetRow - piece.row);
  const dCol = Math.sign(targetCol - piece.col);
  const steps = Math.max(Math.abs(targetRow - piece.row), Math.abs(targetCol - piece.col));
  for (let i = 1; i < steps; i++) {
    const r = piece.row + dRow * i;
    const c = piece.col + dCol * i;
    if (pieceAt(gameState, r, c)) return false;
  }
  return true;
}

function handleCellClick(gameState, row, col) {
  if (gameState.winner) return;

  const clicked = pieceAt(gameState, row, col);

  // If a piece is already selected, attempt move
  if (gameState.selectedPiece) {
    const legal = gameState.legalMoves || [];
    const isLegal = legal.some(m => m.row === row && m.col === col);
    if (isLegal) {
      return executeMove(gameState, gameState.selectedPiece, row, col);
    }
    // Clicked invalid square — deselect or switch selection
    if (clicked && PIECE_TYPES[clicked.type].side === gameState.turn) {
      gameState.selectedPiece = clicked;
      gameState.legalMoves = computeLegalMoves(gameState, clicked);
      return { kind: 'select', piece: clicked };
    }
    gameState.selectedPiece = null;
    gameState.legalMoves = null;
    return { kind: 'deselect' };
  }

  // No selection yet — try to select
  if (clicked && PIECE_TYPES[clicked.type].side === gameState.turn) {
    gameState.selectedPiece = clicked;
    gameState.legalMoves = computeLegalMoves(gameState, clicked);
    return { kind: 'select', piece: clicked };
  }
  return { kind: 'noop' };
}

function executeMove(gameState, piece, row, col) {
  const occupant = pieceAt(gameState, row, col);
  const fromRow = piece.row;
  const fromCol = piece.col;
  if (occupant) {
    gameState.pendingCombat = { attacker: piece, defender: occupant, row, col, fromRow, fromCol };
    resolveCombatPlaceholder(gameState);
    gameState.selectedPiece = null;
    gameState.legalMoves = null;
    endTurn(gameState);
    return { kind: 'combat', attacker: piece, defender: occupant };
  }
  piece.row = row;
  piece.col = col;
  piece.tween = { fromRow, fromCol, startMs: performance.now() };
  gameState.selectedPiece = null;
  gameState.legalMoves = null;
  endTurn(gameState);
  return { kind: 'move', piece, row, col };
}

// Skeleton combat: instant resolve by comparing attack*alignment vs defender HP, no action arena yet.
function resolveCombatPlaceholder(gameState) {
  const { attacker, defender, row, col, fromRow, fromCol } = gameState.pendingCombat;
  const aDef = PIECE_TYPES[attacker.type];
  const dDef = PIECE_TYPES[defender.type];
  const tMs = performance.now();
  const lum = squareLuminosity(row, col, tMs);
  const aBonus = alignmentBonus(aDef.side, lum);
  const dBonus = alignmentBonus(dDef.side, lum);
  const aScore = aDef.attack + aBonus + Math.random() * 4;
  const dScore = dDef.attack + dBonus + Math.random() * 4;
  const attackerWins = aScore >= dScore;

  gameState.combatFlash = { row, col, startMs: tMs };

  if (attackerWins) {
    gameState.dyingPieces.push({ piece: { ...defender }, startMs: tMs });
    gameState.pieces = gameState.pieces.filter(p => p.id !== defender.id);
    attacker.tween = { fromRow, fromCol, startMs: tMs };
    attacker.row = row;
    attacker.col = col;
  } else {
    gameState.dyingPieces.push({ piece: { ...attacker, row, col }, startMs: tMs });
    gameState.pieces = gameState.pieces.filter(p => p.id !== attacker.id);
  }
  gameState.pendingCombat = null;
  checkVictory(gameState);
}

function endTurn(gameState) {
  gameState.turn = gameState.turn === 'light' ? 'dark' : 'light';
}

function checkVictory(gameState) {
  const lightAlive = gameState.pieces.some(p => PIECE_TYPES[p.type].side === 'light');
  const darkAlive  = gameState.pieces.some(p => PIECE_TYPES[p.type].side === 'dark');
  if (!lightAlive) { gameState.winner = 'dark'; return; }
  if (!darkAlive)  { gameState.winner = 'light'; return; }

  // Power points victory: one side occupies all 5
  const owners = POWER_POINTS.map(([r, c]) => {
    const p = pieceAt(gameState, r, c);
    return p ? PIECE_TYPES[p.type].side : null;
  });
  if (owners.every(o => o === 'light')) gameState.winner = 'light';
  if (owners.every(o => o === 'dark'))  gameState.winner = 'dark';
}
