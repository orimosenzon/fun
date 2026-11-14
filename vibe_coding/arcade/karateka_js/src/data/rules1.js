// The rule tables of the first scene: [address, bytes in hexadecimal]. Numbers that say how the game
// plays (see tools/transcribe.mjs); the pictures and the tunes are our own and live in art1.js and tunes.js.
export default [
  [0x6300, '0001800181ff008205830014151606ffff0708090a06ff0b0c0deeeeee0c0b07ff0e0f10111213ffffff061718191a1b1c1d1e1f2006ff061718211f2006ff07272822232425222324262907ff2907ff1f2006ff062017191bff061718ff07ff000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000020101000101000000000000000200000000000001000102030001020304000000fe0000000000000000020200000000020001dd01ddff00dd05dd1414158406ffff0708090a06ff0b0c0deeeeee0c0b07ff0e0f10111213ffffff0723241e1e20211f1f20222507ff158606ff'],   // how every move unfolds: the pose of each step, for the legs and for the body
  [0x6c6d, '0d'],   // the opponent: how often he thinks
  [0x70c6, '000102030405060707060607070706060606060606060608090a0b0c0d0e0f1011121314151616171819'],   // which class every pose belongs to when it is hit
  [0x7300, '070707070707070607070706070706070707070707060706070600010203040506070608090709090709090909060008000600060000000000030300030000000300000303000300020200000002020200020202020000020202000200000000000000000200'],   // the hit tables: for each attack and distance, which poses it reaches and how hard
  [0x7446, '07070707070707060907000200ff0000ff00000000'],   // what a blow does to the one who takes it
  [0x74f7, '727783727986727b88'],   // where the flash of a blow is drawn, by height
  [0x7815, '004ace1d4380'],   // the hall: places of the figures
  [0x7820, '969696979799'],
  [0x782b, '202124242022'],
  [0x7836, '535f5fad995b'],
  [0x7d0c, 'b8'],   // the hall: its pieces, where they stand
  [0x7d12, '62d308564b4f8cc4d8585c9e'],
  [0x7d23, '9a98999997989f9f9f98980d'],
  [0x7d34, '0f1111110d140d0d0c120c79'],
  [0x7d45, '717c7c7c777da3706a796a'],
  [0xac33, '004eb9d30707060484979ba7'],   // the cliff: its pieces
  [0xadf7, '848a7ba6c0d4ef030763d12b575f651a0077a6a6a8a6a6a6a6a7a7a7a7a8a8a8a8ac80a8070201030406090907010204060202090002426f999fa5ab99421e4237302a546868006f'],   // the road: fence posts and the pieces of the background, where they stand
  [0xb19a, '7d4a7c8e94aaabababab06000a0a0a9870687570'],   // the mountain: its pieces
  [0xb60f, 'c5e90b255aa4d2f24872ccdc'],   // the gate house: its pieces
  [0xb62f, 'a3a3a4a4a4a4a4a4a5a5a5a5'],
  [0xb64f, '0a090b0a0c0a0b0a0b0a0c0b'],
  [0xb66f, '8d9e8c948b8f898f838d787f'],
  [0xbb75, 'e0e830fff8ffe0e0ff30e8e8ffe0c0a08050'],   // the opponents of this scene: strength and character of each
  [0xbc06, '000638a846b4427496a4b0b8de0e234d9ca0a0a0a0a1a1a2a2a2a2a2a2a2a3a3a3a30a0404000204080a0cfe04090a0c0604046364594d44362e26645170ac9599a4998c'],   // the gate: its pieces
  [0xbfe8, 'ffffffa0a0c0808080800000408080'],   // the opponent: how keen he is to attack, to retreat, to block, by opponent
  [0xbffb, '000001'],
];
