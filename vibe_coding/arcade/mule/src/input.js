// M.U.L.E. controls on top of the shared C64 input layer.
// One player: arrows + Space, or WASD + F, all drive the joystick in port 1 (the game wants at
// least one joystick, and takes its first human player from there). Two players: WASD + F is
// port 1, arrows + Space port 2.
import { Input } from '../../archon/src/input.js';

export class MuleInput extends Input {
  apply() {
    if (!this.machine) return;
    const a = this.sticks.a | this.touch | this.pads[0], b = this.sticks.b | this.pads[1];
    if (this.twoPlayers) { this.machine.joy[0] = b; this.machine.joy[1] = a; }
    else { this.machine.joy[0] = a | b; this.machine.joy[1] = 0; }
  }
}
