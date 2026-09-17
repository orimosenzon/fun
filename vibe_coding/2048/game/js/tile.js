// Tile: a single numbered tile with its position and animation bookkeeping.
class Tile {
  constructor(position, value) {
    this.x = position.x;
    this.y = position.y;
    this.value = value || 2;

    this.previousPosition = null; // where the tile was before the last move (for the slide animation)
    this.mergedFrom = null;       // the two tiles that merged into this one (for the pop animation)
  }

  savePosition() {
    this.previousPosition = { x: this.x, y: this.y };
  }

  updatePosition(position) {
    this.x = position.x;
    this.y = position.y;
  }

  serialize() {
    return {
      position: { x: this.x, y: this.y },
      value: this.value,
    };
  }
}
