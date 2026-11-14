// The fighters' records. The game keeps one record of 16 bytes per fighter ($60 the player,
// $70 the opponent) and works on a copy at $20; these routines move the records about.
import { extend } from './core.js';

extend({
  // $7071: the working copy becomes the opponent's record
  r7071() { const m = this.m; for (let i = 0; i < 16; i++) m[0x70 + i] = m[0x20 + i]; this.x = 16; },
  // $707D: work on the opponent, and look up the class of the player's pose
  r707D() { const m = this.m; for (let i = 0; i < 16; i++) m[0x20 + i] = m[0x70 + i]; this.a = m[0x60]; this.r70A6(); },
  // $708D: the working copy becomes the player's record
  r708D() { const m = this.m; for (let i = 0; i < 16; i++) m[0x60 + i] = m[0x20 + i]; this.x = 16; },
  // $7099: work on the player, and look up the class of the opponent's pose
  r7099() { const m = this.m; for (let i = 0; i < 16; i++) m[0x20 + i] = m[0x60 + i]; this.a = m[0x70]; this.r70A6(); },
  // $70A6 (also entered as $7012): $32 = the class of pose A, $33 = the distance between the two fighters
  r70A6() {
    const m = this.m;
    this.x = this.a; m[0x32] = m[0x70c6 + this.x];
    let d;
    if (m[0x4b] !== m[0x4c] && (m[0x52] >= 0x80 || m[0x52] < 0x31)) d = 0xff;
    else d = (m[0x72] - m[0x62]) & 0xff;
    m[0x33] = d; this.a = d;
  },
  r7012() { this.r70A6(); },
});
