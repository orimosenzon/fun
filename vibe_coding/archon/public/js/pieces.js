// Piece type definitions and starting layout.
// NOTE: HP/attack/range values are tentative — to be verified against original Archon (1983).

const PIECE_TYPES = {
  knight:       { name: 'Knight',       side: 'light', range: 3, flies: false, hp: 7,  attack: 6, color: '#e8d090' },
  archer:       { name: 'Archer',       side: 'light', range: 3, flies: false, hp: 6,  attack: 5, color: '#c8d088' },
  valkyrie:     { name: 'Valkyrie',     side: 'light', range: 3, flies: true,  hp: 7,  attack: 6, color: '#f0c050' },
  golem:        { name: 'Golem',        side: 'light', range: 2, flies: false, hp: 16, attack: 9, color: '#a8a890' },
  unicorn:      { name: 'Unicorn',      side: 'light', range: 4, flies: false, hp: 9,  attack: 7, color: '#fff0d0' },
  djinni:       { name: 'Djinni',       side: 'light', range: 4, flies: true,  hp: 9,  attack: 7, color: '#88c8ff' },
  phoenix:      { name: 'Phoenix',      side: 'light', range: 4, flies: true,  hp: 11, attack: 9, color: '#ff8848' },
  wizard:       { name: 'Wizard',       side: 'light', range: 2, flies: false, hp: 8,  attack: 5, color: '#ffd966', isSovereign: true },

  goblin:       { name: 'Goblin',       side: 'dark',  range: 3, flies: false, hp: 7,  attack: 6, color: '#807060' },
  manticore:    { name: 'Manticore',    side: 'dark',  range: 3, flies: false, hp: 6,  attack: 5, color: '#806878' },
  harpy:        { name: 'Harpy',        side: 'dark',  range: 3, flies: true,  hp: 7,  attack: 6, color: '#a06880' },
  troll:        { name: 'Troll',        side: 'dark',  range: 2, flies: false, hp: 16, attack: 9, color: '#605830' },
  basilisk:     { name: 'Basilisk',     side: 'dark',  range: 3, flies: false, hp: 9,  attack: 7, color: '#506848' },
  dragon:       { name: 'Dragon',       side: 'dark',  range: 4, flies: true,  hp: 11, attack: 9, color: '#a04848' },
  shapeshifter: { name: 'Shapeshifter', side: 'dark',  range: 3, flies: false, hp: 8,  attack: 6, color: '#7060a0' },
  sorceress:    { name: 'Sorceress',    side: 'dark',  range: 2, flies: false, hp: 8,  attack: 5, color: '#c878ff', isSovereign: true },
};

// Starting layout — TENTATIVE, needs verification against original.
// Board: rows 0..8, cols 0..8. Row 0 is light's back row; row 8 is dark's back row.
// Row 0 (light back):  KN GO VA PH WI PH VA GO KN
// Row 1 (light front): KN KN AR UN DJ UN AR KN KN
// Row 7 (dark front):  GB GB MA BA SH BA MA GB GB
// Row 8 (dark back):   GB TR HA DR SO DR HA TR GB
const STARTING_LAYOUT = [
  // Row 0 — light back rank
  ['knight','golem','valkyrie','phoenix','wizard','phoenix','valkyrie','golem','knight'],
  // Row 1 — light front rank
  ['knight','knight','archer','unicorn','djinni','unicorn','archer','knight','knight'],
  // Rows 2..6 empty
  [null,null,null,null,null,null,null,null,null],
  [null,null,null,null,null,null,null,null,null],
  [null,null,null,null,null,null,null,null,null],
  [null,null,null,null,null,null,null,null,null],
  [null,null,null,null,null,null,null,null,null],
  // Row 7 — dark front rank
  ['goblin','goblin','manticore','basilisk','shapeshifter','basilisk','manticore','goblin','goblin'],
  // Row 8 — dark back rank
  ['goblin','troll','harpy','dragon','sorceress','dragon','harpy','troll','goblin'],
];

function createStartingPieces() {
  const pieces = [];
  let nextId = 1;
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      const typeKey = STARTING_LAYOUT[row][col];
      if (typeKey) {
        pieces.push({
          id: nextId++,
          type: typeKey,
          row,
          col,
          hp: PIECE_TYPES[typeKey].hp,
        });
      }
    }
  }
  return pieces;
}
