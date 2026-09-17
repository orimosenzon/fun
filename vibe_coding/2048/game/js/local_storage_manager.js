// LocalStorageManager: persists the best score and the current game state,
// so a page reload (or closing the tab) does not lose the game.
// Falls back to an in-memory store when localStorage is unavailable (e.g. private mode).
class LocalStorageManager {
  constructor() {
    this.bestScoreKey = "bestScore";
    this.gameStateKey = "gameState";
    this.storage = this.localStorageSupported() ? window.localStorage : this.fakeStorage();
  }

  localStorageSupported() {
    const testKey = "test";
    try {
      const storage = window.localStorage;
      storage.setItem(testKey, "1");
      storage.removeItem(testKey);
      return true;
    } catch (error) {
      return false;
    }
  }

  fakeStorage() {
    const data = {};
    return {
      setItem: (id, val) => (data[id] = String(val)),
      getItem: (id) => (Object.prototype.hasOwnProperty.call(data, id) ? data[id] : undefined),
      removeItem: (id) => delete data[id],
      clear: () => Object.keys(data).forEach((k) => delete data[k]),
    };
  }

  // Best score getters/setters
  getBestScore() {
    return this.storage.getItem(this.bestScoreKey) || 0;
  }

  setBestScore(score) {
    this.storage.setItem(this.bestScoreKey, score);
  }

  // Game state getters/setters and clearing
  getGameState() {
    const stateJSON = this.storage.getItem(this.gameStateKey);
    return stateJSON ? JSON.parse(stateJSON) : null;
  }

  setGameState(gameState) {
    this.storage.setItem(this.gameStateKey, JSON.stringify(gameState));
  }

  clearGameState() {
    this.storage.removeItem(this.gameStateKey);
  }
}
