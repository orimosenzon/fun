// Wait till the browser is ready to render the game (avoids glitches)
window.requestAnimationFrame(() => {
  // Exposed on window so that scripts (or a trained agent) can drive the game:
  //   game.move(0..3)            0 = up, 1 = right, 2 = down, 3 = left
  //   game.grid.serialize()      current board
  window.game = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
});
