// Entry point — wires up canvas, game state, render loop, input.

(function main() {
  const canvas = document.getElementById('board');
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;

  const gameState = createInitialGameState();
  const startMs = performance.now();

  const turnSideEl = document.getElementById('turn-side');
  const statusEl = document.getElementById('status');
  const luminosityEl = document.getElementById('luminosity-phase');
  const selectedEl = document.getElementById('selected-piece');

  Network.init();

  canvas.addEventListener('click', (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top)  * (canvas.height / rect.height);
    const cell = pixelToCell(x, y, canvas);
    if (!cell) return;
    const result = handleCellClick(gameState, cell.row, cell.col);
    updateUI(result);
  });

  function updateUI(actionResult) {
    turnSideEl.textContent = gameState.turn.toUpperCase();
    turnSideEl.className = gameState.turn === 'dark' ? 'dark' : '';

    if (gameState.winner) {
      statusEl.textContent = `${gameState.winner.toUpperCase()} WINS!`;
      selectedEl.textContent = '—';
      return;
    }

    if (gameState.selectedPiece) {
      const p = gameState.selectedPiece;
      const def = PIECE_TYPES[p.type];
      selectedEl.textContent = `${def.name} (${def.side}) — range ${def.range}${def.flies ? ', flies' : ''}, HP ${def.hp}, ATK ${def.attack}`;
    } else {
      selectedEl.textContent = 'No piece selected';
    }

    if (actionResult && actionResult.kind === 'combat') {
      const a = PIECE_TYPES[actionResult.attacker.type].name;
      const d = PIECE_TYPES[actionResult.defender.type].name;
      const winner = gameState.pieces.find(p => p.id === actionResult.attacker.id) ? a : d;
      statusEl.textContent = `Combat: ${a} vs ${d} → ${winner} wins`;
    } else if (actionResult && actionResult.kind === 'move') {
      statusEl.textContent = `Moved to (${actionResult.row}, ${actionResult.col})`;
    } else if (actionResult && actionResult.kind === 'select') {
      statusEl.textContent = 'Click highlighted square to move';
    } else if (actionResult && actionResult.kind === 'deselect') {
      statusEl.textContent = 'Click a piece to see legal moves';
    }
  }

  function loop() {
    const tMs = performance.now() - startMs;
    drawBoard(ctx, canvas, gameState, tMs);
    luminosityEl.textContent = globalCyclePhaseLabel(tMs);
    requestAnimationFrame(loop);
  }

  updateUI(null);
  loop();
})();
