# Karateka in JavaScript: working notes

## Method (decided 14/11/2026)
- New, separate project. No emulator at run time, no original code executed. Every routine of the
  original is re-written by hand as a JS method on `Game` (src/*.js), named r<ADDR> after the
  routine it replaces, working on the same 64K memory layout `m` (so tables and pictures can be
  read from the player's disk and every routine can be compared with the original).
- Measuring instrument: the sibling project's machine (`tools/oracle.mjs`), development only.
- `node tools/dis.mjs` writes the study listing re/game.lst (not committed). `tools/show1.sh A B` prints a range.
- `node tools/check.mjs all --secs 90` plays the original with random keys and compares every
  ported routine's memory effect with the original's. Add `--v` for per-routine counts.
- Registers a/x/y/c on the Game object carry values between routines only where callers read them.

## Memory map of the running game (level 1)
- $02-$1F, $60-$BF: exact copies of disk sectors (see tools/diskmap.mjs). $20-$5F hi-res pages.
- code: $0200-03 misc, $0A-$0D status bar + sound, $10 jump table, $19-$1C drawing (ported: src/gfx.js),
  $64-$7E fighters, AI, scenes, $AC-$AF loader, $B0-$BC main loop, scenery, level logic.

## Variables (zero page)
- $60-$6F player record, $70-$7F opponent record, $20-$2F working copy. Fields: +0 legs pose, +1 upper pose,
  +2 column, +4 kicks in a row, +5/+6 script index legs/upper, +7/+8 script direction, +9 wish to move,
  +A attack height (-1 high, 0 mid, 1 low), +B attack landing this frame, +C facing, +F 0 = fighting stance.
- $32 pose class of the defender, $33 distance, $40/$41 result of opponent's/player's attack,
  $45 swish sound, $46/$47 "player began a move" flags, $5D/$5E alive flags, $B6/$B7 health, $B0/$B1 max health.

## Status
See the bottom of this file; update after every batch.

### Ported and identical to the original
- fighter_state.js: 7071 707D 708D 7099 70A6/7012
- fighter_anim.js: 6400 6540 6741 674C
- draw_calls.js: wrappers 1900 1903 1906 1909 190C 0A00 0A03 over gfx.js (pixels identical; the drawing
  routines' scratch bytes $00 $01 $0A $14-$1D and self-modified code are not reproduced and are excluded from comparison)
- fighter_draw.js: 6780 6783 6786 6789 678C 693F 6945 6959 6971 697C 6987 69BD 69DC 6A6C 6AC5 (rBC00 still missing: special opponents)
- opponent.js: 6B00 (AI) 6C71 6C95 (random: seed*17+$A1 at $48) 6D03
- sound.js: 0C40 0C43 0C46 0C49 0C4C 0C4F 0C52 0DC0, click times and duration cycle-exact (this.clicks, this.cycles)
- status.js: 0B00 0B03 0B06 0B09 0B0C 0B0F
- frame.js: 7600 7603 7609 6E00 (keyboard; joystick stick r6D00 is a stub). Commands: $C5 kick, $D7 punch, $9B stance, $C2 bow.

## Design decisions
- No generators. One call = one iteration of the main loop; r7609 pushes {at, show} to this.flips, sounds push
  cycle stamps to this.clicks and advance this.cycles exactly. The page plays flips and clicks back in real time.
- Non-local jumps of the original (restart, next level, death screen) are exceptions (frame.js Restart...).
- Esc pause: sets this.paused; the page resumes on the next key and swallows it.
- Still to do for timing: estimate cycles of the non-sound work (per draw call), calibrate against the original.
- combat.js: 7000 7003 7006-701E (jump table) 70F0 710D 7207 7145 72BE(+fall $7131) 7366 (hit judge) 74C0 (hit flash)
- main.js: B7CA (frame) B874 (frame proper) BA91 (player's turn) BB1D BB31 BB41 BB52 BB63 B600 BC00 BC03 BC4A (gate)
- scenery1.js (level-1 overlay $AC-$B1): AC30 AC52 AD03 AD06 AD09 AD0C AD0F AD12 AD15 AD18 AD1F AE3F AE65 B0C0 B0C3 B1AE
- view.js: B260 (camera) B34A B3DF. arena.js: B400 B403 (next opponent cut scenes) B406 B409 B562 B569 B58A B5A3 B5E9
- check.mjs has --cheat (player full strength, opponent 1 point at each frame start) to get deep into the game.
- 106 routines identical as of this point. Exceptions GameOver/NextLevel/Restart replace JMP $0203/$0200/$0206.
- cutscene.js: Akuma's hall 79A0 (whole scene) 7800 7803 780C 78B5 78D8 78F9 791A 7A57 7A62 7A6D 7AB1 7AB3 7AC1 7ADC 7C53 7CDC 7D03 7D6A, 7606 (column wipe; pushes a flip with a page copy per column)
- music.js: 0D00 two-voice tune player, cycle-exact (clicks and duration verified); tunes 1000 1003 1006 1009 100C at $10E4 $108B $10BF $1040 $10F4. this.deaf = [from,to] cycle spans where keys are swallowed.
- flips now carry a copy of the shown page: {at, show, page}.
- 130+ routines identical. NOT yet done: game set-up B700/B763, game over $0203 (r02CB...), whole-game driver, timing of non-sound work,
  the page (index.html), other levels (overlays differ: $AC-$B1 scenery, $BC00, tunes), title/intro, ending.

## MILESTONE (14/11/2026): level 1 is byte-for-byte identical
`node tools/compare.mjs --frames 8000 --seed 2 --cheat` → 3408 frames identical (memory and both screens) from the
level's set-up to the moment the original loads the next scene; seed 3: 4006 frames; without cheat seed 1: 1198 frames to game over.
compare.mjs: both start from the original's memory at $B700; same keys pressed between passes of the main loop
(boundaries: pc $B7AB / $B7C1); `--timing` prints cycle statistics (g.work is reserved for the cost model).
- game.js: B700 B763 set-up, step() = one pass, returns 'frame' | 'over' | 'next' | 'restart'.

## Next
1. Stand-alone start: build memory from the disk (page map) + baked zero page; index.html shell (display from flips, clicks to audio, keys).
2. Cost model for non-sound work (cycles per pass) so the pace is right.
3. Game over screen ($0203: r02CB etc.), restart.
4. Levels 2, 3 (overlays), title, ending. Use the compare bot to reach them in the original and dump listings.
5. Wipe the original's code bytes from memory after loading (only tables and pictures are needed).

## State at the end of the first long session (14/11/2026)
- Scene 1 is complete, stand-alone and playable: `index.html` (serve arcade/ on 8932 → /karateka_js/). Shell: src/shell/player.js
  (real-time playback of flips/clicks, keys by physical position, Esc pause, game over + restart), display.js, speaker.js.
- loader.js builds memory from the disk with data/scene1.js (made by `node tools/bake.mjs scene1`) and wipes the original's code.
  `node tools/compare.mjs --frames 8000 --seed 2 --cheat --standalone` → 3408 frames identical.
- Timing: data/timing.js (SHAPE/FILL per-call models from tools/measure_draw.mjs, PASS = 6300 cycles of thinking per frame).
  `--timing`: total within 0.2-2%, median error per pass about 1%.
- Scenes: $D2 = scene number. Scene 2 replaces pages $10-$11 (tunes), $15-$17, $1E-$1F (pictures), $A0-$BF (scenery AND main loop).
  `node tools/reach.mjs 1 scene2 4000` saves re/scene2.ram/.cov/.start.ram; `node tools/dis.mjs re/scene2.lst scene2`; `tools/show1.sh A B scene2`.
  Plan: scene-2 routines as methods that are assigned onto the Game instance when the scene loads (same address = same role
  for jump-table entries such as $BC00, $AD03.., $1000..), with their own check/compare runs.

## 14/11/2026, second session: no disk at all (Ori: "all the pictures and everything ours")
- The game no longer reads a disk. `src/loader.js buildMemory(scene)` builds the memory from this project's own data:
  computed tables ($0800 rows, $0980 flip, $1A34 masks), `src/data/rules1.js` (rule tables as numbers, made once by
  tools/transcribe.mjs from the ranges tools/needs.mjs saw the game read), `src/data/art_fighters.js` and
  `src/data/art_scenery1.js` (drawn by tools/art/*.mjs), `src/data/tunes.js` (our tunes, written as notes, and the blow sound).
- The reference (original memory from the disk) is now a development instrument only: `tools/lib.mjs memoryFor(false)`,
  page map in tools/ref/scene1.map.js. compare.mjs/check.mjs still prove the ENGINE identical to the original.
  `node tools/test_own.mjs` proves the game on OUR data makes the same decisions pass for pass (all variables $20-$F5).
- Fighters (tools/art/fighters.mjs): a jointed figure posed by hip/feet/hands (IK), cut into legs / body / head / floor mask.
  The pose tables $6000-$62FF (+$6AEA) are OURS (generated), shapes live at $C000 up. Only drawing reads those tables.
  How $6783 composes a fighter: legs (OR) at col $6054 row $607E; anchor = 7*$6150+$617A (per legs pose) = our hip x;
  body (OR) at anchor + 7*$6272, row $6298 (per body pose); head: $61A4 = 0..6 fine shift (kind 0) or 7/8 = kind 1/2 (lying),
  col $61DA row $6200; opponents' heads are drawn 2 rows higher and one byte further back (box 3 bytes wide).
  "Shadow" tables $60A8.. are AND masks that clear the floor (rows 153-163) behind the feet. Nothing may be drawn below
  local row 49 (screen 163): the game repaints only rows 114-163 of an 11-byte strip.
- Scenery (tools/art/scenery1.mjs): our drawings in the original boxes at the original addresses (positions are literals
  in the engine and in rules1.js).
- Dev tools added: catalog.mjs (every shape drawn, + contact sheet), needs.mjs (table bytes read, by routine), poses.mjs
  (pose sheets, --ours / --left / --who N / --list "L/U ..."), shots.mjs (screenshots, --both, --log = what was drawn),
  measure_poses.mjs, transcribe.mjs.
- For scenes 2 and 3: port the routines (as planned above), then per scene: catalog → needs → rules<N>.js → scenery<N>.mjs
  → tunes. Fighters are shared (scene 2/3 may add poses or opponents' heads; the eagle and the gate are their own pictures).
