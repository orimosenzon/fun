// Setting a level up and running it frame by frame.
import { extend, u8 } from './core.js';
import { GameOver, NextLevel } from './main.js';
import { Restart, StartOver } from './frame.js';

extend({
  // $B700: the first scene from the top: the player at the cliff, full strength, the first opponent waiting
  rB700() {
    const m = this.m;
    this.r7000();
    m[0xb8] = 0x1e; m[0xb9] = 0x1e; m[0xba] = 0x1e;
    for (const a of [0x10, 0x50, 0xad, 0xae, 0x5d, 0x5e]) m[a] = 1;
    m[0x52] = 0x30; m[0xe0] = 0x30; m[0xe1] = 0x30;
    for (const a of [0x5a, 0x57, 0xaf, 0x53, 0x35, 0x40, 0x41, 0x42, 0x43, 0x44, 0x4c, 0x82, 0x83]) m[a] = 0;
    m[0x91] = 0x50; m[0x94] = 0x50; m[0x51] = 0xfe; m[0xab] = 0xfe; m[0xac] = 0xfe; m[0x5f] = 2; m[0x54] = 2; m[0xb0] = 0x0e; m[0xb1] = 0x0e;
    this.rBB41(); this.rBB52(); m[0x45] = 0;
    this.a = 0; this.rB763();
  },
  // $B763: the part of the set-up that is repeated when the fighting starts over
  rB763() {
    const m = this.m, a = this.a;
    m[0x4e] = a; m[0x9a] = a; m[0x3e] = a; m[0x97] = a; this.rBB63();
    m[0xa0] = 5; m[0x56] = 5; m[0x90] = 5; m[0x92] = 0x10; m[0x84] = 7; m[0x89] = 7; m[0x98] = 7;
    for (const z of [0x35, 0x4d, 0x42, 0x30, 0x87, 0x88, 0x8f, 0x36, 0xa1, 0xa2]) m[z] = 0;
    m[0xa5] = 2; m[0xa4] = 2; m[0x99] = 0x0b;
    this.r7003(); m[0xe2] = m[0x62]; m[0xe3] = m[0x62];
    this.phase = 'opening';
  },

  // $0203: the end of the game: the closing picture, shown on both pages
  gameOver() {
    const m = this.m, picture = () => { m[3] = 0x74; m[4] = 0x9a; m[5] = 0x0f; m[6] = 0x5a; m[0x0f] = 1; this.r1903(); };
    this.r1900(); picture(); this.r7609(); this.r1900();
    if (m[7] !== 0x20) { picture(); this.r7609(); this.r1900(); }
  },
  // ... and a key starts the scene again
  again() { const m = this.m; this.r7609(); m[0xdc] = 1; m[0xd2] = m[0xd0]; m[7] = 0x40; this.rB700(); },

  // One pass of the game's main loop. Returns what happened: 'frame', or how the scene ended.
  step() {
    const m = this.m;
    try {
      this.rB7CA(); this.r7600();
      if (this.phase === 'opening') {                    // the title of the scene is on the screen
        this.r7603();
        if (m[0x99] === 0) { this.r1009(); m[0x46] = 0; m[0x47] = 0; this.phase = 'play'; }
      }
      return 'frame';
    } catch (e) {
      if (e instanceof GameOver) return 'over';
      if (e instanceof NextLevel) return 'next';
      if (e instanceof Restart) return 'restart';
      if (e instanceof StartOver) { this.a = 0; this.rB763(); return 'frame'; }
      throw e;
    }
  },
});
