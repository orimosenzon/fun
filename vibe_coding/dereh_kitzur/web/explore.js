/* מצב תעופה - flying over the moshava to find its shortcuts.
 *
 * The rest of the app answers "where is the shortcut I already know about".
 * This answers the other question: what is out there. You take off over the
 * satellite image, steer with the mouse, and trails light up as you come near
 * them - nothing is drawn until you are close enough to have found it. The
 * photos that are attached to a trail hang in the air above it and grow as
 * you close in, so a picture is a thing you fly towards.
 *
 * Built on the map that is already here rather than on a 3D engine. MapLibre
 * already holds the satellite raster, the terrain, and a camera that tilts to
 * 80 degrees; what it does not have is a flight model. So the whole of this
 * file is really three things: a camera that thinks in metres above the
 * ground, a reveal rule that decides what you have discovered, and an engine
 * you can hear.
 *
 * The camera is honest. `alt` is a real height in metres, and the zoom handed
 * to the map is derived from it through MapLibre's own projection geometry
 * (see `zoomFor`). That is what makes the photos grow correctly on approach -
 * they are sized in world metres and projected, not scaled by a fudge factor.
 *
 * The controls are an aircraft's, loosely. The mouse is the stick: where the
 * cursor sits left or right of the middle of the screen is how hard you turn,
 * no button held. The button is the throttle - held, the engine spools up;
 * released, it winds down. W and S are the elevator, nose down and nose up.
 * A and D roll, all the way round if you keep them down, and a bank turns you
 * the way a bank does. The point of all of it is to be able to see a photo in
 * the distance and simply go there.
 *
 * Since 16/9/2026 there are two aircraft on the button. The jet is the one
 * described above. The balloon takes the same hands - mouse, button, arrows -
 * and answers them the way a balloon would: the up arrow is a burner that
 * heats air which lifts you some seconds later and goes on lifting after you
 * let go, the down arrow is the valve that lets that air out, and nothing
 * but the burner makes a sound. Its numbers come from a real one; see the
 * tuning block for the balloon and `balloonControls` for the flight.
 *
 * Since 18/9/2026 there is a third: the jet bike from the fable project,
 * brought over whole (bike.js). It is the one aircraft here you can see -
 * a chase camera sits behind it, and the bike, its rider and its flames are
 * drawn by Three.js on a transparent canvas over the map, with the camera
 * of that scene rebuilt every frame from the map's own, so the two never
 * disagree about where the ground is. Its hands are fable's: the arrows
 * lean the rider, W and S are the throttle grip, E and D climb and descend
 * with the flight computer flying the lift nozzles, Space is the burner,
 * Shift and Z and X are the reaction thrusters; and the mouse is the lean
 * too, both axes, with the button on the throttle, so one hand can fly it.
 * It can land, and it can crash. See `bikeControls`.
 */
'use strict';

const Explore = (() => {

  /* ---------- tuning ----------
   *
   * Speeds and ranges scale with altitude, all of them. Ground covered per
   * second has to grow with height or climbing feels like slowing down, and
   * the reveal radius has to grow with it or climbing shows you a wider view
   * of nothing. Height is the one control that changes the game: low is a
   * walk-through, high is a survey. */

  const ALT_MIN = 45;
  const ALT_START = 260;

  const PITCH_LOW = 80;      // skimming: a lot of horizon, dramatic
  const PITCH_HIGH = 74;     // high up: looking down, a sliver of sky
  const PITCH_MIN = 58;      // limits for the nose-up / nose-down trim
  const PITCH_MAX = 80;
  const TRIM_DOWN = 18;      // degrees of nose-down the mouse can ask for
  const TRIM_UP = 8;         // and of nose-up; the auto pitch already sits near the top

  /* ---------- the jet: an F-16 ----------
   *
   * Since 16/9/2026 (evening) the jet is a named aircraft and its limits are
   * the F-16's, the public figures: a service ceiling of 50,000 ft (15,240 m);
   * Mach 1.2 at sea level, where the airframe's dynamic-pressure limit is what
   * stops it, and Mach 2.05 up high, where the intake does; 50,000 ft/min
   * (254 m/s) of climb; and a 9 g limit, which at these speeds is what sets
   * the turn rate, not the stick. The engine (F110-GE-129) gives 76 kN dry
   * and 129 kN in afterburner for some 12 t of aeroplane, 6 and 11 m/s² of
   * push; here they are a third more, so that going supersonic from a
   * standstill is half a minute of held button rather than a full one.
   *
   * The speed of sound falls with height (ISA: 340 m/s at sea level, 295
   * above 11 km), so Mach 1 is 1,225 km/h over the rooftops and 1,062 at the
   * ceiling; the Mach number on the HUD is against the local value, and the
   * boom is where that reads one.
   *
   * What is not the F-16's is the speed floor. A real one stalls below about
   * 250 km/h; this one slows to a hover when the button is let go, because
   * the mode exists to find shortcuts on the ground and a thing that cannot
   * stop cannot look. So it is an F-16 from the speed of a car upwards and a
   * helicopter below that. The braking is compressed too: chopped to idle at
   * Mach 1.2 a real F-16 takes over a minute to get subsonic, this one about
   * twenty seconds, which is already a long time to be going somewhere you
   * did not mean to.
   *
   * The first version of this mode had a cruise of 421 km/h at 260 m and it
   * was three times too fast: the whole four kilometres of the moshava in ten
   * seconds, over open fields before anything had a chance to light up. It
   * survived testing because of the clamp on `dt` below - a headless browser
   * drawing five frames a second ran every test flight at a third of real
   * time. A flight model has to be checked against the clock, not against how
   * far it got in a test. The F-16 is faster still, on purpose this time, and
   * the throttle is squared below the detent so that the bottom of its travel
   * is still a walking pace over the rooftops. */
  const ALT_MAX = 15240;     // 50,000 ft, the service ceiling
  const ALT_SURVEY = 1400;   // the view is fully tilted down by here; above it, the picture only shrinks
  const ALT_TROPO = 11000;   // ISA tropopause: the speed of sound stops falling here
  const ALT_FAST = 12000;    // where the top speed reaches MACH_HI
  const MACH_SL = 1.2;       // top speed at sea level, in afterburner
  const MACH_HI = 2.05;      // and at ALT_FAST and above
  const MACH_MIL_SL = 0.95;  // the most the dry engine gives at sea level
  const MACH_MIL_HI = 1.1;   // and up high
  const MIL = 0.66;          // throttle position of the military-power detent: past it is afterburner
  const A_MIL = 8;           // m/s² of push, dry (real: 6)
  const A_AB = 15;           // and in afterburner (real: 11)
  const BRAKE_0 = 12;        // m/s² of slowing at idle, plus
  const BRAKE_V = 10;        // this much more times (speed / top)²: the drag
  const SPEED_K = 2.2;       // how the speed settles on what the throttle asks, per second, inside the push and brake limits
  const THR_UP = 2.0;        // seconds of button held from idle to full throttle
  const THR_DOWN = 3.5;      // seconds from full throttle back to idle once released
  const TRANS_W = 0.05;      // the transonic band is Mach 1 ± this: buffet, and drag the burner has to push through
  const TRANS_DRAG = 0.5;    // the share of the push lost in the middle of the band
  const CLIMB_MAX = 254;     // m/s: 50,000 ft/min
  const DIVE_MAX = 400;      // a dive is not a climb
  const G_MAX = 9;

  const YAW_RATE = 70;       // degrees per second with the mouse at the edge
  const YAW_ACCEL = 5.0;
  const DEAD = 0.09;         // the middle of the screen where the mouse asks nothing
  const PIXEL_CAP = 1.5;     // device pixels per CSS pixel the map is allowed while flying

  const ROLL_RATE = 150;     // degrees per second with A or D held: a full roll in 2.4 s
  const ROLL_ACCEL = 6;
  const BANK_MAX = 22;       // how far a plain mouse turn banks the picture
  const BANK_TURN = 48;      // degrees per second of turn that a 90-degree bank buys
  const LEVEL_K = 2.6;       // how quickly the wings level once A and D are released
  const SINK_K = 0.08;       // banked over, the wings hold less: fraction of altitude lost per second, inverted

  const CLIMB_K = 0.85;      // fraction of current altitude gained per second, up to CLIMB_MAX

  /* ---------- the balloon ----------
   *
   * Numbers from a real one, then compressed. Daidzic (Aviation 25:3, 2021)
   * works a 3,000 m³ AX8-class balloon through: 800 kg gross, 18 m across,
   * 254 m² of drag area, C_D 0.5 going up and 0.9 coming down mouth-first,
   * 7.5 kg of lift for every degree above 90 °C, 3.4 m/s the best climb and
   * 8 m/s the terminal fall of a cold envelope. The FAA Balloon Flying
   * Handbook (ch. 7) gives the feel: a standard burn is four seconds, level
   * flight is one of those every 25 to 40 seconds, and the balloon answers a
   * burn 6 to 15 seconds after it, because the air inside has to circulate
   * before it is lift. The whole craft of a pilot is burning ahead of what
   * they want and knowing when to stop so the momentum carries them there.
   *
   * What is kept is the shape of all of that: heat goes through a plume
   * before it is envelope heat, lift is heat above the level-flight excess,
   * the envelope has inertia and drag, and the burner is a rhythm and not a
   * lever. What is compressed, about threefold, is the heating, the cooling
   * and the vertical speeds, so the rooftop-to-survey range is a minute and
   * not seven. The wind is the one lie. A real balloon at 260 m makes
   * 15 km/h; this one makes 68, because four kilometres of moshava at a
   * walking pace is a nap. It keeps the one rule of wind that matters here,
   * that higher is faster, which is how a balloon pilot steers at all. */

  const BAL = {
    ALT_MIN: 40,
    ALT_MAX: 1400,         // where the air is too thin to hold more, for this envelope
    ALT_START: 90,
    PITCH_LOW: 76,         // near the ground: horizon, and a lot of it
    PITCH_HIGH: 68,        // high up: more of the ground beneath you
    PITCH_MIN: 50,
    TRIM_DOWN: 22,         // W leans you out over the edge of the basket
    TRIM_UP: 6,
    DT_EQ: 80,             // K above ambient that just carries the load: level flight
    DT_MAX: 130,           // the fabric's limit; past it the burner adds nothing
    DT_FLOOR: 62,          // the least a grounded envelope cools to, so a relaunch is a long burn and not an afternoon
    BURN_K: 6,             // K per second with the burner open
    VENT_K: 7,             // K per second with the parachute valve open
    COOL_T: 220,           // seconds for the heat to fall to 1/e of itself
    MIX_T: 3,              // the plume becomes envelope heat over this long: the lag
    LIFT_K: 73.6,          // N per K: 7.5 kg per degree
    MASS: 3000,            // kg, inertial: structure, the hot air, and the outside air dragged along
    DRAG_UP: 22,           // N per (m/s)², envelope rising
    DRAG_DOWN: 40,         // and falling, mouth first
    CEIL_FADE: 150,        // m below the ceiling over which the lift thins out
    WIND_BASE: 6,          // m/s: the wind is this plus WIND_K per metre of height,
    WIND_K: 0.05,          // held between WIND_MIN and WIND_MAX
    WIND_MIN: 7,
    WIND_MAX: 28,
    RIDE_IDLE: 0.32,       // the wind never lets go: the share of it you drift at with nothing pressed
    RIDE_UP: 2,            // seconds to catch the full wind with the button down
    RIDE_DOWN: 10,         // and to lose it again once released
    SPEED_T: 2,            // the basket's own lag behind the wind
    DRIFT_T: 2.0,          // how long the drift takes to come round behind the basket
    YAW_RATE: 26,          // degrees per second with the mouse at the edge: rotation vents, not a rudder
    YAW_T: 1.2,
    SWAY_W: 2 * Math.PI / 7,   // a seven-second pendulum: the basket under the envelope's centre
    SWAY_Z: 0.12,          // lightly damped, three or four swings before it settles
    SWAY_GAIN: 0.6,        // degrees of lean per m/s² of the wind pulling on the envelope
    SWAY_MAX: 6,
    LAUNCH: 2.5            // seconds past the take-off that the burner is held for you
  };

  /* ---------- the jet bike ----------
   *
   * The flight model and its numbers are fable's and live in bike.js
   * (Bike.TUNE). What is here is the seam between that world and this one:
   * the camera, and what the map can and cannot do for it.
   *
   * The map cannot look up. MapLibre's pitch runs from straight down to a
   * few degrees under the horizon, so a camera that follows a climbing bike
   * is raised to keep its target beneath it (bike.js, camDip) and its pitch
   * is clamped here as a last resort. And the map cannot roll, so the roll
   * is CSS, as for the jet - but a chase camera does not roll with what it
   * chases; it leans a little, and the bike is seen banking in front of it.
   * The lean is a share of the bank, capped, and the map hangs past the
   * screen by exactly enough to cover that cap (fitFrame). The rider's-eye
   * camera (C) rolls fully, and gets the jet's diagonal square instead.
   *
   * The field of view is wider than the jet's 32 degrees, because a chase
   * camera ten metres behind a three-metre motorcycle needs the room and
   * the speed is felt at the edges. fable uses 68 and more; MapLibre 4 caps
   * the container's field at 60, and the container is larger than the
   * window, so 42 is what the window can be given (see fitFrame). */
  const BIKE = {
    VFOV: 42,              // degrees, vertical, in the window
    ROLL_K: 0.3,           // the picture leans by this share of the bike's bank
    ROLL_MAX: 16,          // and never more than this, degrees; fitFrame covers exactly this
    ROLL_T: 0.18,          // seconds for the lean to follow the bank
    PITCH_MIN: 12,         // the map's camera, degrees from straight down
    PITCH_MAX: 82,
    ZOOM_MAX: 24,          // the ground under a parked bike is zoom 22 and a half
    CARD_W: 12,            // metres: the floor for a photo's width at the bike's heights
    CARD_H: 8,             // and for how high it hangs
    RUSH_V: 45,            // m/s at which the speed streaks are fully there
    MSG_MS: 2400           // how long a message stays on the screen
  };

  const REVEAL_K = 5.0;      // reveal radius = altitude * this
  const REVEAL_MIN = 420;
  const REVEAL_MAX = 3000;
  const PHOTO_K = 4.0;
  const PHOTO_MIN = 340;
  const PHOTO_MAX = 2400;

  const MAX_CARDS = 8;       // floating photos on screen at once
  const MAX_CHIPS = 5;       // trail names on screen at once
  const MAX_CHIPS_TRAILS = 12;   // and in the trails view, where the names are the point
  const MAX_LINES = 40;      // trails fed to the glow source

  const CARD_MIN_PX = 44;
  const CARD_MAX_VH = 0.42;
  const CHIP_MIN_PX = 11;
  const CHIP_MAX_PX = 30;

  const HOVER_MS = 120;      // how long the cursor rests on a photo before the stick lets go
  const ARM_T = 1.4;         // seconds for a freshly taken stick to reach full authority
  const INTRO_MOVE = 60;     // pixels of mouse travel that put the take-off card away
  const BAND_H = 600;        // height of the sky gradient, in css; must match .fly-band

  /* The vertical field of view the viewer actually sees, in degrees. The map
   * container is larger than the screen (see `fitFrame`), so the field of
   * view handed to MapLibre is wider than this; what is held constant is the
   * picture in the window. */
  const VFOV = 32;

  const RAD = Math.PI / 180;
  const M_PER_DEG_LAT = 111320;

  /* ---------- state ---------- */

  let on = false;
  let flying = false;        // false during the take-off ease, true after
  let raf = null;
  let last = 0;
  let tick = 0;              // seconds until the next reveal recompute
  let frame = 0;

  const keys = new Set();
  let pos = { lat: 0, lng: 0 };
  let alt = ALT_START;
  let bearing = 0;
  let speed = 0;
  let throttle = 0;          // 0..1, driven by the mouse button
  let hold = false;          // the mouse button is down over open ground
  let yaw = 0;               // current turn rate, degrees/second
  let bank = 0;              // right wing down is positive; the picture is rotated by minus this
  let bankRate = 0;
  let pitchTrim = 0;         // nose up or down, from the mouse, on top of the auto pitch
  let burn = 0;              // afterburner, 0..1, eased for the sound and the glow
  let mach = 0;              // speed over the local speed of sound
  let gee = 1;               // the load in the turn, for the HUD
  let buffet = 0;            // 0..1 through the transonic band: the shake
  let sonic = false;         // past Mach 1, with a little hysteresis so it does not flicker
  let booms = 0;             // how many times the barrier has been crossed this flight, for the tests
  let shook = false;         // last frame shook the picture, so this one has to set it straight

  const mouse = { nx: 0, ny: 0, px: 0, py: 0, in: false };   // stick position, -1..1 from the middle, and the last pixel
  const hover = { hud: false, card: null, since: 0 };
  /* The stick is taken, not simply held. Whenever the mouse has been busy
   * being a mouse - reading the help, closing a picture, coming back into the
   * window - it is wherever that left it, and a stick that engages there at
   * full strength throws the aircraft into a turn nobody asked for. So it
   * engages on the first movement after that, and its authority comes in
   * over ARM_T seconds from nothing: a cursor left at the edge gives a turn
   * that grows, which is seen and corrected, rather than a lurch.
   *
   * Until 16/9/2026 (night) it engaged only when the cursor passed through
   * the ring in the middle of the screen, and the take-off card blocked it
   * for its nine seconds. Ori flew it and reported "the mouse does not turn
   * at first, then after a while it does, and it is not clear what makes
   * the difference". Both were deliberate and neither explained itself,
   * which for a control is the same as a bug. Now the card goes on the
   * first real movement of the mouse (it is one press of ? away), and the
   * ring is only the dead zone drawn. */
  let armed = false;
  let stickGain = 0;         // 0..1, the authority of a freshly taken stick
  let introMoved = 0;        // pixels of mouse travel since the take-off card appeared

  let canRoll = false;       // fine pointer: a keyboard and a mouse are here
  let ox = 0, oy = 0;        // how far the map hangs past the screen, pixels
  let focal = 1.5;           // camera distance in container heights, see fitFrame

  let world = [];            // every item near enough to ever matter
  let shots = [];            // every photo, placed on the ground
  let seen = new Map();      // item id -> when it came into range, for the pulse
  let near = [];             // last reveal result, reused between recomputes
  let nearest = null;        // the closest revealed item, for Enter
  let paused = false;        // a photo is open full-size
  let gal = null;            // the open gallery: { owner, list, i }

  let touchHold = false;     // a finger is on the screen: the touch throttle
  let touchBurn = false;     // balloon, touch: the finger has been dragged down, the burner is open
  let touchVent = false;     // and up, the valve is open

  /* Which aircraft. Chosen outside the mode, on the small button beside the
   * flight button, and kept between visits; read at take-off and fixed for
   * the flight. The balloon is the one a first visit gets (Ori's call,
   * 16/9/2026): it is the gentler introduction to the place, and the jet is
   * there on the small button for whoever wants it. index.html and its
   * titles start from the same choice. */
  const KEY_CRAFT = 'dk.fly.craft';
  const CRAFTS = ['balloon', 'jet', 'bike'];   // the order the small button cycles them in
  let craft = 'balloon';
  try { const c = localStorage.getItem(KEY_CRAFT); if (CRAFTS.includes(c)) craft = c; } catch (_) { /* private mode */ }

  /* The bike's state outside the physics: the scene it is drawn in, the
   * camera's lean, the pose the map was last asked for, the messages. The
   * rig is made on the first bike flight and kept. */
  const bk = {
    rig: null, canvas: null,
    cam: 0,                  // 0 chase, 1 the rider's eyes; kept between flights
    lever: 0,                // the throttle lever, 0..1; the engine gets its square
    grip: false,             // the lever was last raised by the mouse button, which lets it go
    camRoll: 0,              // the picture's lean, degrees, eased towards its share of the bank
    want: { x: 0, y: 0, z: 0, dx: 0, dy: 0, dz: 1, roll: 0 },   // where the chase camera wants to be
    pose: null,              // { center, zoom, bearing, pitch } last handed to the map
    crashes: 0,              // this flight, for the tests
    msg: '', msgUntil: 0     // what the message line says, and until when
  };
  const tb = {               // the touch controls, fable's: a joystick and hold buttons
    steer: 0, pitch: 0,
    throttleUp: false, throttleDown: false, collUp: false, collDown: false, boost: false
  };

  /* The balloon's own state. Heat is kelvin above the air outside; the
   * plume is the burner's heat on its way to being the envelope's; the
   * pendulum angles are degrees and their rates degrees per second. */
  const bal = {
    dT: 0, plume: 0, vz: 0,
    drift: 0,                // the bearing the wind carries you on, behind `bearing`
    ride: 0,                 // share of the wind caught, RIDE_IDLE..1
    roll: 0, rollV: 0,       // the basket's swing, side to side
    pit: 0, pitV: 0,         // and fore-and-aft
    pvx: 0, pvy: 0,          // last frame's velocity, for the acceleration the basket feels
    launch: 0,               // seconds of lift-off burn still to go
    burning: false,          // last frame's burner, for the moment it lights
    t: 0,                    // flight seconds, the clock the turbulence runs on
    seed: []                 // phases for the turbulence channels, drawn at take-off
  };

  let restore = null;        // what to put back on the way out

  /* What is drawn over the ground, on the button at the top right (Ori,
   * 17/9/2026): the photos are the point of the flight and also the thing
   * that hides the land, and sometimes the land, or the trails through it,
   * is what you came to look at.
   *
   *   normal   the photos, the names of the trails, the trails lit as found
   *   clean    the trails alone, nothing hanging over the ground
   *   trails   no photos, more names, and the trails lit hard
   *
   * 'trails' is there only while there is a trail in the world to light; with
   * the shortcuts layer off the button has two positions. Kept between
   * flights like the aircraft is, and like the sound. */
  const VIEWS = ['normal', 'clean', 'trails'];
  const VIEW_NAME = { normal: 'רגיל', clean: 'נקי', trails: 'שבילים' };
  const VIEW_WHAT = {
    normal: 'תמונות, שמות ושבילים',
    clean: 'רק השבילים, בלי תמונות ושמות',
    trails: 'השבילים מוארים חזק, עם השמות, בלי תמונות'
  };
  const KEY_VIEW = 'dk.fly.view';
  let view = 'normal';
  try { const v = localStorage.getItem(KEY_VIEW); if (VIEWS.includes(v)) view = v; } catch (_) { /* private mode */ }
  let hasTrails = false;     // a line in the world, decided by buildWorld

  const cards = new Map();   // photo key -> element, pooled across frames
  const chips = new Map();   // item id -> element

  let root = null, sky = null, band = null, hazeEl = null, worldEl = null;
  let elAlt = null, elSpeed = null, elName = null, elHint = null, rushEl = null, elThr = null;
  let burnerEl = null, elBurn = null, sndBtn = null, elVsi = null, elVsiG = null;
  let elMach = null, elMachG = null, elG = null, elGG = null, coneEl = null;
  let intro = null, viewer = null, lookBtn = null, elLook = null;
  let elThrMark = null, elLift = null, elLiftFill = null, elLiftMark = null;
  let elFc = null, elFcG = null, elMsg = null, touchEl = null;

  const el = (id) => document.getElementById(id);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;

  /* ---------- the engine ----------
   *
   * Synthesised, not sampled. A jet is noise shaped by resonance, which is
   * what a filter is, and building it from parts means every knob is live:
   * the body deepens and the hiss rises with speed, and the afterburner is a
   * separate roar that comes in under the throttle key. No file to fetch,
   * nothing to license, and the sound is the flight model's, not a loop
   * played over it.
   *
   * The balloon has its own voice, built the same way and kept on its own
   * bus: a propane burner, which is the only sound a balloon makes. The rest
   * of a balloon flight is silent, famously so - you move with the air, so
   * there is not even wind - and that silence is half of what the burner's
   * roar means when it comes.
   *
   * The bike's voice is fable's (audio.js there), on a third bus: white
   * noise through a low-pass for the rumble and a sawtooth for the turbine
   * whine, both following the total jet power, plus the chuff of a
   * reaction-thruster pulse and the thump of a crash. */

  const Engine = (() => {
    const KEY = 'dk.fly.sound';
    let ctx = null, master = null, n = null, bv = null, kv = null;
    let jetBus = null, balBus = null, bikeBus = null;
    let mode = 'jet';
    let muted = false;
    try { muted = localStorage.getItem(KEY) === '0'; } catch (_) { /* private mode */ }

    function noise(seconds, brown) {
      const len = Math.floor(ctx.sampleRate * seconds);
      const buf = ctx.createBuffer(1, len, ctx.sampleRate);
      const d = buf.getChannelData(0);
      let b = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        // Brown noise is white noise integrated with a leak, which is the
        // rumble under everything; white on its own is only the hiss.
        if (brown) { b = (b + 0.02 * w) / 1.02; d[i] = b * 3.5; } else d[i] = w;
      }
      return buf;
    }
    const loop = (buf) => {
      const s = ctx.createBufferSource();
      s.buffer = buf; s.loop = true; s.start();
      return s;
    };
    const gain = (v) => { const g = ctx.createGain(); g.gain.value = v; return g; };
    const filter = (type, f, q) => {
      const x = ctx.createBiquadFilter();
      x.type = type; x.frequency.value = f; if (q) x.Q.value = q;
      return x;
    };

    function build() {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      ctx = new AC();
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18;
      comp.ratio.value = 6;
      master = gain(0);
      master.connect(comp);
      comp.connect(ctx.destination);
      return true;
    }

    function buildJet() {
      const brown = noise(3, true), white = noise(2, false);
      jetBus = gain(1);
      jetBus.connect(master);
      // The engine's voices go through two more gains. `eng` is the boom's:
      // it pulls them all down at once for the thump. `hush` is the sound
      // barrier's: past Mach 1 the engine is behind you and its sound goes
      // through air that is streaming backwards faster than sound travels,
      // so nothing of it can reach you; what a real pilot still hears is
      // the airframe carrying a remnant, and pilots do say the cockpit goes
      // quiet. The air over the airframe is local and stays, so the hiss
      // is fed past both.
      const eng = gain(1), hush = gain(1);
      eng.connect(hush); hush.connect(jetBus);

      // The body of the turbine: rumble through a low-pass whose cutoff climbs
      // with speed, so spooling up is heard as the sound opening.
      const coreLP = filter('lowpass', 160, 0.8);
      const coreG = gain(0.2);
      loop(brown).connect(coreLP); coreLP.connect(coreG); coreG.connect(eng);

      // Air over the airframe: a band of white noise that is barely there at
      // a hover and most of the sound at top speed.
      const hissBP = filter('bandpass', 2600, 0.5);
      const hissG = gain(0.01);
      loop(white).connect(hissBP); hissBP.connect(hissG); hissG.connect(jetBus);

      // The whine: a sawtooth and a sine an octave up, slightly off, so they
      // beat against each other the way real blades do. Pitch follows speed.
      const whineLP = filter('lowpass', 1500, 1.5);
      const whineG = gain(0.02);
      const o1 = ctx.createOscillator(); o1.type = 'sawtooth'; o1.frequency.value = 70;
      const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = 141;
      o1.connect(whineLP); o2.connect(whineLP);
      whineLP.connect(whineG); whineG.connect(eng);
      o1.start(); o2.start();

      // The afterburner: brown noise driven hard into a soft clipper, kept low,
      // with a slow tremolo so it crackles rather than hums. Silent until W.
      const shaper = ctx.createWaveShaper();
      const curve = new Float32Array(256);
      for (let i = 0; i < 256; i++) curve[i] = Math.tanh(3.2 * (i / 127.5 - 1));
      shaper.curve = curve;
      const burnLP = filter('lowpass', 240, 1.2);
      const burnG = gain(0);
      const trem = gain(1);
      const lfo = ctx.createOscillator(); lfo.type = 'sine'; lfo.frequency.value = 9.5;
      const lfoG = gain(0.35);
      lfo.connect(lfoG); lfoG.connect(trem.gain); lfo.start();
      loop(brown).connect(shaper); shaper.connect(burnLP); burnLP.connect(burnG);
      burnG.connect(trem); trem.connect(eng);

      n = { coreLP, coreG, hissG, o1, o2, whineG, burnG, brown, eng, hush };
    }

    /** The sonic boom. What reaches the ground is an N-wave: a step up in
     *  pressure at the nose shock, a step down at the tail, and for an
     *  aircraft the length of an F-16 the two are a tenth of a second apart,
     *  which is why a boom is heard as two - the "ba-boom" of the films, and
     *  of the coast when the air force is out over the sea. Each step is a
     *  thump with almost nothing above a hundred hertz, and it leaves a
     *  rumble behind it. Built from a burst of the brown noise through a low
     *  pass with a snapped attack, and under it a sine falling through the
     *  bottom of hearing, which is the pressure step itself. Loud, on
     *  purpose, and the engine is pulled down under it for half a second,
     *  because a boom is louder than everything else and the compressor
     *  alone would flatten the two into each other - measured: with the
     *  burner roaring the thump added a fifth to the level and nothing to
     *  the ear.
     *
     *  Strictly the pilot never hears their own: the boom is left behind
     *  with the shock cone. But there is no cockpit around this camera, the
     *  moshava is beneath it, and a barrier crossed in silence would be no
     *  barrier at all. */
    function boom() {
      if (!ctx || muted || !n || mode !== 'jet') return;
      const t0 = ctx.currentTime;
      n.eng.gain.cancelScheduledValues(t0);
      n.eng.gain.setValueAtTime(n.eng.gain.value, t0);
      n.eng.gain.linearRampToValueAtTime(0.08, t0 + 0.015);
      n.eng.gain.setTargetAtTime(1, t0 + 0.3, 0.35);
      for (const [at, amp] of [[0, 1], [0.11, 0.8]]) {
        const t = t0 + at;
        const src = ctx.createBufferSource();
        src.buffer = n.brown;
        src.playbackRate.value = 0.5;
        const lp = filter('lowpass', 110, 0.7);
        const g = gain(0);
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(4 * amp, t + 0.012);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.8);
        src.connect(lp); lp.connect(g); g.connect(jetBus);
        src.start(t); src.stop(t + 0.9);

        const o = ctx.createOscillator();
        o.type = 'sine';
        o.frequency.setValueAtTime(72, t);
        o.frequency.exponentialRampToValueAtTime(26, t + 0.32);
        const og = gain(0);
        og.gain.setValueAtTime(0.0001, t);
        og.gain.exponentialRampToValueAtTime(2.2 * amp, t + 0.01);
        og.gain.exponentialRampToValueAtTime(0.0001, t + 0.55);
        o.connect(og); og.connect(jetBus);
        o.start(t); o.stop(t + 0.6);
      }
    }

    /* The burner. A propane burner is a flame the size of a room, and what
     * it sounds like is a roar with most of its weight a couple of hundred
     * hertz up, a tearing edge above that, and the hiss of the jets over
     * everything; and it flutters, several times a second, which is what
     * says fire and not wind. Loud on purpose. It is the loudest thing in
     * ballooning and the only thing you hear, and pilots time their burns
     * partly by ear. */
    function buildBalloon() {
      const brown = noise(3, true), white = noise(2, false);
      balBus = gain(1);
      balBus.connect(master);

      // The flutter: a gain the three voices pass through, swung by noise
      // low-passed to a few hertz. Its resting value is under one so the
      // swing has room both ways.
      const flick = gain(0.72);
      const flLP = filter('lowpass', 11, 1.2);
      const flG = gain(14);
      loop(white).connect(flLP); flLP.connect(flG); flG.connect(flick.gain);

      // The body of the roar.
      const roarBP = filter('bandpass', 240, 0.5);
      const roarG = gain(3.2);
      loop(brown).connect(roarBP); roarBP.connect(roarG); roarG.connect(flick);

      // The edge, and the jets.
      const midBP = filter('bandpass', 950, 0.6);
      const midG = gain(0.3);
      loop(white).connect(midBP); midBP.connect(midG); midG.connect(flick);
      const hissHP = filter('highpass', 2600, 0.7);
      const hissG = gain(0.07);
      loop(white).connect(hissHP); hissHP.connect(hissG); hissG.connect(flick);

      // The valve: shut until the key opens it, a fast attack and a quick
      // close, and nothing at all in between.
      const out = gain(0);
      flick.connect(out); out.connect(balBus);

      bv = { out };
    }

    /** The whoomp of the valve opening: a pitch that falls through the
     *  bottom of the roar in a third of a second, once per lighting. */
    function ignite() {
      if (!ctx || muted || !bv || mode !== 'balloon') return;
      const t = ctx.currentTime;
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.setValueAtTime(150, t);
      o.frequency.exponentialRampToValueAtTime(36, t + 0.35);
      const g = gain(0);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.6, t + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
      o.connect(g); g.connect(balBus);
      o.start(t); o.stop(t + 0.55);
    }

    /* The bike: a turbojet the size of a barrel, a metre under the saddle.
     * fable's two voices - looped white noise through a low-pass whose
     * cutoff opens with the power, and a sawtooth whine whose pitch climbs
     * with it - with the rumble given a little more weight here, because it
     * arrives through the master and the compressor the other two share. */
    function buildBikeVoice() {
      const white = noise(2, false);
      bikeBus = gain(1);
      bikeBus.connect(master);
      const lp = filter('lowpass', 300);
      const g = gain(0);
      loop(white).connect(lp); lp.connect(g); g.connect(bikeBus);
      const osc = ctx.createOscillator(); osc.type = 'sawtooth'; osc.frequency.value = 90;
      const wg = gain(0);
      osc.connect(wg); wg.connect(bikeBus); osc.start();
      kv = { lp, g, osc, wg };
    }

    /** RCS pulse: a short pressurized-gas chuff (fable). */
    function pulse() {
      if (!ctx || muted || !bikeBus || mode !== 'bike') return;
      const t = ctx.currentTime, dur = 0.16;
      const buf = ctx.createBuffer(1, Math.floor(dur * ctx.sampleRate), ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const f = filter('bandpass', 1900, 0.8);
      const g = gain(0);
      g.gain.setValueAtTime(0.5, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + dur);
      src.connect(f); f.connect(g); g.connect(bikeBus);
      src.start(t);
    }

    /** The crash: a burst of noise swept down through a low-pass, and a
     *  sub-bass thump under it (fable). */
    function crash() {
      if (!ctx || muted || !bikeBus || mode !== 'bike') return;
      const t = ctx.currentTime, dur = 1.3;
      const buf = ctx.createBuffer(1, Math.floor(dur * ctx.sampleRate), ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const f = filter('lowpass', 950);
      f.frequency.setValueAtTime(950, t);
      f.frequency.exponentialRampToValueAtTime(55, t + dur);
      const g = gain(0);
      g.gain.setValueAtTime(1.2, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + dur);
      src.connect(f); f.connect(g); g.connect(bikeBus);
      src.start(t);
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.setValueAtTime(72, t);
      o.frequency.exponentialRampToValueAtTime(34, t + 0.5);
      const og = gain(0);
      og.gain.setValueAtTime(0.8, t);
      og.gain.exponentialRampToValueAtTime(0.001, t + 0.65);
      o.connect(og); og.connect(bikeBus);
      o.start(t); o.stop(t + 0.7);
    }

    /** Every frame of the bike: `power` is the jets' total as a fraction of
     *  full, roughly 0 to 1.6, and `speed` is in m/s. */
    function setBike(power, speed) {
      if (!ctx || muted || !kv) return;
      const t = ctx.currentTime;
      kv.g.gain.setTargetAtTime(0.22 + power * 0.7, t, 0.08);
      kv.lp.frequency.setTargetAtTime(220 + power * 1500 + speed * 8, t, 0.1);
      kv.wg.gain.setTargetAtTime(0.025 + power * 0.1, t, 0.1);
      kv.osc.frequency.setTargetAtTime(85 + power * 150 + speed * 1.2, t, 0.15);
    }

    /** Which aircraft the next start() is for. */
    function setMode(m) { mode = m; }

    /** Start, or resume. Must be called from a user gesture the first time:
     *  browsers refuse to make a sound a page asked for on its own. */
    function start() {
      if (muted) return;
      if (!ctx && !build()) return;
      // Each aircraft's voices are built the first time it is flown and kept;
      // the buses decide which one is heard.
      if (mode === 'jet' && !n) buildJet();
      if (mode === 'balloon' && !bv) buildBalloon();
      if (mode === 'bike' && !kv) buildBikeVoice();
      if (jetBus) jetBus.gain.value = mode === 'jet' ? 1 : 0;
      if (balBus) balBus.gain.value = mode === 'balloon' ? 1 : 0;
      if (bikeBus) bikeBus.gain.value = mode === 'bike' ? 1 : 0;
      if (bv) bv.out.gain.value = 0;
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});
      master.gain.cancelScheduledValues(ctx.currentTime);
      master.gain.setTargetAtTime(0.55, ctx.currentTime, 0.4);
    }

    function poke() {
      if (ctx && ctx.state === 'suspended' && !muted) ctx.resume().catch(() => {});
    }

    function stop() {
      if (!ctx) return;
      master.gain.cancelScheduledValues(ctx.currentTime);
      master.gain.setTargetAtTime(0, ctx.currentTime, 0.25);
      // Suspend once the fade is over, so a page with the mode closed is not
      // still running a synthesiser into silence.
      setTimeout(() => { if (ctx && master.gain.value < 0.01) ctx.suspend().catch(() => {}); }, 900);
    }

    /** Every frame. For the jet, `s` is speed as a fraction of top speed,
     *  `b` the afterburner and `quiet` whether the sound barrier is behind
     *  you; for the balloon only `b` matters, the burner, open or shut. */
    function set(s, b, quiet) {
      if (!ctx || muted) return;
      const t = ctx.currentTime;
      if (mode === 'balloon') {
        if (bv) bv.out.gain.setTargetAtTime(b > 0.5 ? 1 : 0, t, b > 0.5 ? 0.035 : 0.09);
        return;
      }
      if (!n) return;
      s = clamp(s, 0, 1);
      n.coreLP.frequency.setTargetAtTime(150 + 950 * s, t, 0.1);
      n.coreG.gain.setTargetAtTime(0.2 + 0.4 * s, t, 0.1);
      n.hissG.gain.setTargetAtTime(0.01 + 0.2 * s * s, t, 0.1);
      const f = 70 + 330 * s;
      n.o1.frequency.setTargetAtTime(f, t, 0.15);
      n.o2.frequency.setTargetAtTime(f * 2.02, t, 0.15);
      n.whineG.gain.setTargetAtTime(0.015 + 0.05 * s, t, 0.1);
      n.burnG.gain.setTargetAtTime(0.65 * b, t, b ? 0.12 : 0.3);
      // Out through the barrier the engine is left behind over half a
      // second; back through it, it catches up a little faster.
      n.hush.gain.setTargetAtTime(quiet ? 0.07 : 1, t, quiet ? 0.5 : 0.35);
    }

    function toggle() {
      muted = !muted;
      try { localStorage.setItem(KEY, muted ? '0' : '1'); } catch (_) { /* fine */ }
      if (muted) stop(); else start();
      return !muted;
    }

    return { start, stop, set, setBike, pulse, crash, poke, toggle, setMode, ignite, boom, isOn: () => !muted };
  })();

  /* ---------- geometry ---------- */

  /** Where you end up going `m` metres on `brg` from a point. Equirectangular,
   *  which over the few kilometres a flight covers is exact to centimetres. */
  function destination(lat, lng, brg, m) {
    const dN = m * Math.cos(brg * RAD);
    const dE = m * Math.sin(brg * RAD);
    return {
      lat: lat + dN / M_PER_DEG_LAT,
      lng: lng + dE / (M_PER_DEG_LAT * Math.cos(lat * RAD))
    };
  }

  const angleDiff = (a, b) => {
    let d = (a - b) % 360;
    if (d > 180) d -= 360;
    if (d < -180) d += 360;
    return d;
  };

  /** The zoom that puts the camera exactly `alt` metres above the ground.
   *
   *  MapLibre holds the camera `focal * height` pixels from the centre point,
   *  along the view axis; the vertical leg of that is `cos(pitch)` of it. So
   *  the metres-per-pixel we need is alt / (that leg), and the zoom is what
   *  gives that scale at this latitude. Inverting the projection like this,
   *  rather than picking zooms by eye, is what lets the altitude readout mean
   *  something and the photos size themselves in real metres. */
  function zoomFor(a, pitch, lat) {
    const h = map.getContainer().clientHeight || 800;
    const mpp = a / (focal * h * Math.cos(pitch * RAD));
    const worldPx = 40075016.686 * Math.cos(lat * RAD) / mpp;
    // The bike parks three metres over the ground, which is past zoom 22;
    // enter() raises the map's ceiling to match.
    return clamp(Math.log2(worldPx / 512), 1, craft === 'bike' ? BIKE.ZOOM_MAX : 22);
  }

  /** The inverse: where the map's camera is, in the bike's frame of the
   *  world (x east, y up, z south, metres), and the unit direction it looks
   *  along, from the map's own centre, zoom, pitch and bearing. Exact to
   *  the pixel, because it undoes zoomFor and the centre-ahead step with the
   *  same focal and the same container height. */
  function mapCamera() {
    const c = map.getCenter();
    const pitch = map.getPitch(), brg = map.getBearing();
    const ch = map.getContainer().clientHeight || 800;
    const mpp = 40075016.686 * Math.cos(c.lat * RAD) / (512 * Math.pow(2, map.getZoom()));
    const dist = focal * ch * mpp;                       // camera to the centre point, metres
    const sp = Math.sin(pitch * RAD), cp = Math.cos(pitch * RAD);
    const sb = Math.sin(brg * RAD), cb = Math.cos(brg * RAD);
    const [cx, cn] = toLocal(c.lat, c.lng);              // east, north
    return {
      x: cx - sb * dist * sp,
      y: dist * cp,
      z: -(cn - cb * dist * sp),
      dx: sb * sp, dy: -cp, dz: -cb * sp,
      vfov: 2 * Math.atan(innerHeight / (2 * focal * ch)) / RAD
    };
  }

  /** Screen y of the horizon, in map-container pixels. Points at infinity sit
   *  `focal / tan(pitch)` heights above the centre of the view. */
  function horizonY(pitch) {
    const h = map.getContainer().clientHeight || 800;
    return h * (0.5 - focal / Math.tan(pitch * RAD));
  }

  /** Screen pixels per world metre at a given place on the ground.
   *
   *  Measured rather than derived: project the point and a point 40 m to its
   *  side, and take the distance between them. That gets perspective, terrain
   *  and whatever else the transform is doing for free, and it is two matrix
   *  multiplies. The side-step is perpendicular to the heading so it is never
   *  foreshortened by the tilt. */
  function scaleAt(lngLat, screen) {
    const q = destination(lngLat[1], lngLat[0], bearing + 90, 40);
    const p2 = map.project([q.lng, q.lat]);
    return Math.hypot(p2.x - screen.x, p2.y - screen.y) / 40;
  }

  /** Nearest point of a polyline to (px, py), all in local metres. */
  function nearestOn(xy, px, py) {
    if (xy.length === 1) {
      return { d: Math.hypot(px - xy[0][0], py - xy[0][1]), x: xy[0][0], y: xy[0][1] };
    }
    let best = Infinity, bx = xy[0][0], by = xy[0][1];
    for (let i = 0; i < xy.length - 1; i++) {
      const ax = xy[i][0], ay = xy[i][1];
      const dx = xy[i + 1][0] - ax, dy = xy[i + 1][1] - ay;
      const len2 = dx * dx + dy * dy;
      let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = ax + t * dx, qy = ay + t * dy;
      const d2 = (px - qx) ** 2 + (py - qy) ** 2;
      if (d2 < best) { best = d2; bx = qx; by = qy; }
    }
    return { d: Math.sqrt(best), x: bx, y: by };
  }

  /* ---------- the frame ----------
   *
   * Rolling the picture is done in CSS, because MapLibre 4 has no roll axis,
   * and a rotated rectangle only keeps covering the screen if it is big
   * enough: for any angle at all, a square whose side is the screen's
   * diagonal. So on a desktop the map is grown to that square, hanging well
   * past every edge, and the window shows its middle.
   *
   * That would narrow the view - the window is now a small part of the
   * container, and MapLibre sets its camera distance from the container's
   * height - so the field of view is widened by exactly the amount that puts
   * the same picture back in the window. It is the same pinhole camera with a
   * larger sensor behind it; nothing in the visible frame changes, there is
   * just more of it painted off-screen for the roll to bring in. */

  function getFov() {
    if (typeof map.getVerticalFieldOfView === 'function') return map.getVerticalFieldOfView();
    return map.transform ? map.transform.fov : 36.87;
  }

  function setFov(deg) {
    if (typeof map.setVerticalFieldOfView === 'function') map.setVerticalFieldOfView(deg);
    else if (map.transform) map.transform.fov = deg;
  }

  function fitFrame() {
    const w = innerWidth, h = innerHeight;
    canRoll = matchMedia('(pointer: fine)').matches;
    if (craft === 'bike' && bk.cam === 0) {
      // The chase camera leans by ROLL_MAX at most, on any device: a corner
      // of the screen turned by that angle reaches half the other side times
      // its sine past the edge, and that is all the map needs to hang over.
      const s = Math.sin(BIKE.ROLL_MAX * RAD);
      ox = Math.ceil(h * 0.5 * s) + 2;
      oy = Math.ceil(w * 0.5 * s) + 2;
    } else if (canRoll) {
      const d = Math.ceil(Math.hypot(w, h));
      ox = Math.ceil((d - w) / 2);
      oy = Math.ceil((d - h) / 2);
    } else if (craft === 'balloon') {
      // No roll on touch, but the basket still sways, and a picture tilted
      // by a few degrees needs a little material past each edge: a corner
      // of the screen rotated by `a` reaches about half the other side
      // times sin(a) beyond it. Eight per cent covers nine degrees.
      ox = Math.ceil(h * 0.08);
      oy = Math.ceil(w * 0.08);
    } else {
      ox = 0;
      oy = 0;
    }
    const rs = document.documentElement.style;
    rs.setProperty('--fly-ox', `${ox}px`);
    rs.setProperty('--fly-oy', `${oy}px`);
    // A retina screen would have the map fill four device pixels for every
    // css pixel of a canvas that is already twice the screen. Satellite tiles
    // at flying speed do not repay that, and the frame time does.
    if (map.setPixelRatio) map.setPixelRatio(Math.min(devicePixelRatio || 1, PIXEL_CAP));
    map.resize();

    const ch = map.getContainer().clientHeight || (h + 2 * oy);
    const dist = 0.5 * h / Math.tan(((craft === 'bike' ? BIKE.VFOV : VFOV) / 2) * RAD);
    // MapLibre caps the field of view at 60 degrees. An ultrawide screen asks
    // for more and gets a slightly narrower view instead of a broken one;
    // `focal` is taken from what was actually set, so the geometry stays true.
    const fov = clamp((2 * Math.atan((0.5 * ch) / dist)) / RAD, 10, 60);
    setFov(fov);
    focal = 0.5 / Math.tan((fov / 2) * RAD);

    // The bike's canvas is the window, not the container: the lean is put
    // into its camera rather than into CSS, so it never needs material past
    // the edges, and it is a quarter the pixels of the map's square.
    if (bk.rig) bk.rig.resize(w, h, Math.min(devicePixelRatio || 1, PIXEL_CAP));
  }

  /* ---------- the world, precomputed once per flight ----------
   *
   * Everything the reveal test needs, in a flat local metric frame so that a
   * frame costs arithmetic and no trigonometry. Rebuilt on entry rather than
   * kept live: layers can be switched while flying only by leaving first. */

  let refLat = 0, refLng = 0, mPerLng = 0;

  const toLocal = (lat, lng) => [(lng - refLng) * mPerLng, (lat - refLat) * M_PER_DEG_LAT];
  const toLngLat = (x, y) => [refLng + x / mPerLng, refLat + y / M_PER_DEG_LAT];

  /** A point a given fraction along a path, by arc length. Photos of one trail
   *  are spread along it instead of stacked on its first vertex - a trail with
   *  four pictures should read as four places, which is what it is. */
  function alongPath(path, frac) {
    if (path.length === 1) return path[0];
    const segs = [];
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const d = Math.hypot(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0]);
      segs.push(d);
      total += d;
    }
    let want = total * frac, acc = 0;
    for (let i = 0; i < segs.length; i++) {
      if (acc + segs[i] >= want) {
        const t = segs[i] ? (want - acc) / segs[i] : 0;
        return [path[i][0] + (path[i + 1][0] - path[i][0]) * t,
                path[i][1] + (path[i + 1][1] - path[i][1]) * t];
      }
      acc += segs[i];
    }
    return path[path.length - 1];
  }

  /** The pictures and videos of an item, in the order the app shows them. A
   *  video is a photo entry with `yt`; it floats as its thumbnail and plays
   *  when opened. */
  const mediaOf = (entry) =>
    (entry.item.photos || []).filter((p) => p && (p.yt || p.thumb || p.full));

  function buildWorld() {
    refLat = pos.lat;
    refLng = pos.lng;
    mPerLng = M_PER_DEG_LAT * Math.cos(refLat * RAD);

    world = [];
    shots = [];
    near = [];
    nearest = null;
    seen.clear();

    const items = [...Layers.visibleSegments(), ...Layers.visibleWaypoints()];
    const REACH = 30000;   // Houten and Curitiba are on the same map and are not here

    for (const it of items) {
      const path = it.path && it.path.length ? it.path
        : (it.lat != null && it.lng != null ? [[it.lat, it.lng]] : null);
      if (!path) continue;

      const xy = path.map(([lat, lng]) => toLocal(lat, lng));
      if (Math.hypot(xy[0][0], xy[0][1]) > REACH) continue;

      const entry = { id: it.id, name: it.name || '', item: it, xy, line: path.length > 1 };
      world.push(entry);

      const photos = mediaOf(entry).filter((p) => p.thumb || p.full);
      photos.forEach((p, i) => {
        // A waypoint has one coordinate and possibly several pictures, so they
        // are fanned onto a small ring rather than left in one pile.
        let at;
        if (path.length > 1) {
          at = alongPath(path, (i + 0.5) / photos.length);
        } else {
          const a = (i / Math.max(photos.length, 1)) * 360 + 40;
          const d = destination(path[0][0], path[0][1], a, photos.length > 1 ? 26 : 0);
          at = [d.lat, d.lng];
        }
        const [x, y] = toLocal(at[0], at[1]);
        shots.push({
          key: `${it.id}:${i}`,
          owner: entry,
          name: it.name || '',
          cap: p.cap || '',
          src: p.thumb || p.full,
          full: p.full || p.thumb,
          yt: p.yt || null,
          lngLat: [at[1], at[0]],
          x, y
        });
      });
    }
    hasTrails = world.some((e) => e.line);
  }

  /* ---------- reveal ----------
   *
   * Measured from a point ahead of the flyer rather than from underneath: at
   * this tilt, what is directly below is at the very bottom of the screen and
   * mostly out of it, and revealing things you cannot see is the same as not
   * revealing them. */

  function focusPoint(pitch) {
    const ahead = alt * Math.tan(pitch * RAD) * 0.55;
    return destination(pos.lat, pos.lng, bearing, ahead);
  }

  const EMPTY = { type: 'FeatureCollection', features: [] };

  function recompute(pitch, now) {
    const f = focusPoint(pitch);
    const [fx, fy] = toLocal(f.lat, f.lng);
    const reach = clamp(alt * REVEAL_K, REVEAL_MIN, REVEAL_MAX);

    near = [];
    for (const e of world) {
      const hit = nearestOn(e.xy, fx, fy);
      if (hit.d > reach) { seen.delete(e.id); continue; }
      if (!seen.has(e.id)) seen.set(e.id, now);
      // Eased so a trail arrives as a glow that swells rather than a line that
      // switches on, and a pulse for the first second it is in range: finding
      // something should be an event.
      const t = 1 - hit.d / reach;
      const pulse = Math.exp(-(now - seen.get(e.id)) / 0.85) * 0.75;
      let g = t * t * (3 - 2 * t);
      // The trails view lights a trail hard almost as soon as it is in range,
      // still from nothing at the edge so it swells rather than pops: a fifth
      // of the way in it is already half lit, halfway in it is nearly full.
      if (view === 'trails') g = 1 - Math.pow(1 - g, 3);
      e.g = clamp(g + pulse, 0, 1);
      e.d = hit.d;
      e.at = toLngLat(hit.x, hit.y);
      near.push(e);
    }
    near.sort((a, b) => a.d - b.d);
    nearest = near[0] || null;

    const src = map.getSource('fly-trails');
    if (src) {
      const halo = view === 'trails' ? 1 : 0.7;
      src.setData({
        type: 'FeatureCollection',
        features: near.filter((e) => e.line).slice(0, MAX_LINES).map((e) => ({
          type: 'Feature',
          properties: { g: e.g, h: e.g * halo },
          geometry: {
            type: 'LineString',
            coordinates: e.item.path.map(([lat, lng]) => [lng, lat])
          }
        }))
      });
    }

    paintChips();
    return { fx, fy };
  }

  /* ---------- floating photos ----------
   *
   * Sized in world metres and projected, so approaching one really does make
   * it bigger - the growth is perspective, not a distance curve. The card
   * hangs above its point on a tether, because a picture lying flat on a
   * trail at this tilt is a smear and a picture with nothing under it belongs
   * nowhere. */

  function cardFor(shot) {
    let node = cards.get(shot.key);
    if (node) return node;
    node = document.createElement('div');
    node.className = 'fly-card' + (shot.yt ? ' video' : '');
    // The halo is the click target, wider than the card by a finger's width:
    // the card is moving while you aim at it, and a target that has to be hit
    // exactly is not a target you can hit from a moving aircraft.
    node.innerHTML =
      `<i class="fly-card-hit"></i>` +
      `<div class="fly-card-img"><img alt="" decoding="async" referrerpolicy="no-referrer"></div>` +
      `<div class="fly-card-cap"></div>`;
    const img = node.querySelector('img');
    // A picture that will not load is a black rectangle hanging over a trail,
    // which reads as a fault in the trail rather than in the file. Mark it and
    // it stops being a place you can fly to.
    img.addEventListener('error', () => { shot.dead = true; node.hidden = true; });
    img.src = shot.src;
    node.querySelector('.fly-card-cap').textContent = shot.name;
    node.addEventListener('click', (e) => { e.stopPropagation(); openShot(shot); });
    worldEl.appendChild(node);
    cards.set(shot.key, node);
    return node;
  }

  function paintCards(pitch, fx, fy) {
    // Only the normal view hangs photos; applyView put the existing ones
    // away, and nothing here makes a new one until the view is back.
    if (view !== 'normal') return;
    const reach = clamp(alt * PHOTO_K, PHOTO_MIN, PHOTO_MAX);
    const vh = map.getContainer().clientHeight || 800;
    const sinP = Math.sin(pitch * RAD);

    // World size of a card, and how high above the ground it floats. Both
    // follow altitude so that a card is a similar size on screen whether you
    // are skimming or surveying - what changes with distance, and only that,
    // is how much bigger it gets as you approach it.
    // The bike flies at rooftop height, where a 22 m card a hundred metres
    // off would fill a third of the screen; its floors are lower.
    const isBike = craft === 'bike';
    const wMetres = clamp(alt * 0.20, isBike ? BIKE.CARD_W : 22, 90);
    const hMetres = clamp(alt * 0.18, isBike ? BIKE.CARD_H : 20, 90);

    // The window the viewer can actually see, in the map's own pixels. The map
    // hangs past the screen on every side, so its centre and the screen's
    // centre coincide but its edges are outside. Once the picture is rolled
    // the window is no longer axis-aligned in these pixels, and the whole
    // container is used instead: a few cards painted where nobody sees them
    // are cheaper than a card missing from a corner that has just rolled in.
    const cw = map.getContainer().clientWidth;
    const rolled = Math.abs(bank) > 6;
    const x0 = rolled ? -40 : ox - 40;
    const x1 = rolled ? cw + 40 : ox + innerWidth + 40;
    const y1 = rolled ? vh + 40 : oy + innerHeight + 40;

    const [mx, my] = toLocal(pos.lat, pos.lng);
    const live = [];
    for (const s of shots) {
      if (s.dead) continue;
      const d = Math.hypot(s.x - fx, s.y - fy);
      if (d > reach) continue;
      // Behind you is not a view. The projection folds points behind the
      // camera plane back onto the screen, so this cull is correctness and
      // not only economy.
      const brgTo = Math.atan2(s.x - mx, s.y - my) / RAD;
      if (Math.abs(angleDiff(brgTo, bearing)) > 88) continue;
      s.d = d;
      live.push(s);
    }
    live.sort((a, b) => a.d - b.d);

    const skyY = horizonY(pitch);
    const keep = new Set();
    const placed = [];

    for (const s of live) {
      if (placed.length >= MAX_CARDS) break;
      const p = map.project(s.lngLat);
      if (!isFinite(p.x) || !isFinite(p.y)) continue;
      // Right on the horizon a card is a smudge at the very top of the screen,
      // and it is the tether reaching up out of the frame that you notice, not
      // the picture. Those distances are the reveal radius doing its job; they
      // do not also need a photograph.
      if (p.y < skyY + 26) continue;

      const ppm = scaleAt(s.lngLat, p);
      if (!isFinite(ppm) || ppm <= 0) continue;

      const w = clamp(wMetres * ppm, CARD_MIN_PX, innerHeight * CARD_MAX_VH);
      const tether = clamp(hMetres * ppm * sinP, 6, innerHeight * 0.22);
      const cardH = w * 0.75 + 26;

      // Off the screen is off the screen. The heading test above only rejects
      // what is behind the camera; a photo a little to the side and close by
      // passes it and still lands two thousand pixels below the bottom edge,
      // because at this tilt the ground under you is not in the picture. Those
      // were counting against the ten slots and showing nothing.
      if (p.x + w / 2 < x0 || p.x - w / 2 > x1) continue;
      if (p.y - tether - cardH > y1) continue;

      // Nearest wins the spot. Cards are placed closest-first, so a photo that
      // would land on top of one already placed is the further of the two and
      // is the one to drop - otherwise a trail with eight pictures becomes one
      // illegible pile and hides the trail beside it.
      let clash = false;
      for (const q of placed) {
        if (Math.abs(p.x - q.x) < (w + q.w) * 0.5 &&
            Math.abs(p.y - q.y) < (w + q.w) * 0.4) { clash = true; break; }
      }
      if (clash) continue;
      placed.push({ x: p.x, y: p.y, w });
      // Fades in over the outer fifth of the range, so cards arrive rather
      // than pop, and never fully vanishes while in range.
      const fade = clamp((1 - s.d / reach) * 5, 0, 1);

      const node = cardFor(s);
      node.style.transform =
        `translate3d(${p.x.toFixed(1)}px, ${(p.y - tether).toFixed(1)}px, 0) translate(-50%, -100%)`;
      node.style.width = `${w}px`;
      node.style.setProperty('--tether', `${tether}px`);
      node.style.opacity = fade;
      node.style.zIndex = String(4000 - Math.round(s.d));
      node.classList.toggle('small', w < 88);
      node.hidden = false;
      keep.add(s.key);
    }

    for (const [key, node] of cards) {
      if (!keep.has(key)) node.hidden = true;
    }
  }

  function paintChips() {
    if (view === 'clean') return;   // applyView hid them; nothing to paint
    const keep = new Set();
    // Five names fit across a laptop and pile on top of each other across a
    // phone, where they are also the widest thing on the screen. In the
    // trails view the names are what is being looked at, so more of them.
    const phone = innerWidth < 560;
    const cap = view === 'trails' ? (phone ? 4 : MAX_CHIPS_TRAILS) : (phone ? 2 : MAX_CHIPS);
    const pick = near.filter((e) => e.g > 0.42 && e.name).slice(0, cap);
    const vh = map.getContainer().clientHeight || 800;
    // The bottom strip belongs to the instruments, and the nearest trail's name
    // is already printed there. A chip that lands under it prints the same
    // words twice, half of each behind the other.
    const floor = (vh - innerHeight) / 2 + innerHeight - 96;
    // And the map hangs past the sides of the screen: a chip centred out
    // there is a word cut in half at the edge. Once the picture is rolled
    // the window is not axis-aligned in these pixels, and the whole
    // container is allowed, as with the cards.
    const rolled = Math.abs(bank) > 6;
    const x0 = rolled ? -40 : ox + 16;
    const x1 = rolled ? (map.getContainer().clientWidth || innerWidth) + 40 : ox + innerWidth - 16;

    for (const e of pick) {
      const p = map.project([e.at[0], e.at[1]]);
      if (!isFinite(p.x) || !isFinite(p.y)) continue;
      let node = chips.get(e.id);
      if (!node) {
        node = document.createElement('div');
        node.className = 'fly-chip';
        worldEl.appendChild(node);
        chips.set(e.id, node);
      }
      node.textContent = e.name;
      node.style.transform =
        `translate3d(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px, 0) translate(-50%, -175%)`;
      node.style.fontSize =
        `${clamp(CHIP_MIN_PX + (e.g - 0.42) * 34, CHIP_MIN_PX, CHIP_MAX_PX)}px`;
      node.style.opacity = clamp((e.g - 0.42) * 3.4, 0, 1);
      node.style.zIndex = String(3000 - Math.round(e.d));
      node.hidden = p.y < -80 || p.y > floor || p.x < x0 || p.x > x1;
      keep.add(e.id);
    }
    for (const [id, node] of chips) if (!keep.has(id)) node.hidden = true;
  }

  /* ---------- the flight ---------- */

  /** The stick's response: nothing inside the dead zone, then a curve that is
   *  gentle near the middle and firm at the edge, so small corrections are
   *  possible and a hard turn is still there when it is wanted. */
  function stick(v) {
    const a = Math.abs(v);
    if (a <= DEAD) return 0;
    const s = (a - DEAD) / (1 - DEAD);
    return Math.sign(v) * Math.pow(s, 1.6);
  }

  const altMin = () => (craft === 'balloon' ? BAL.ALT_MIN : ALT_MIN);

  /** The tilt the altitude asks for, before trim and sway. */
  function basePitch() {
    const isBal = craft === 'balloon';
    const lo = isBal ? BAL.PITCH_LOW : PITCH_LOW;
    const hi = isBal ? BAL.PITCH_HIGH : PITCH_HIGH;
    const t = clamp((alt - altMin()) / ((isBal ? BAL.ALT_MAX : ALT_SURVEY) - altMin()), 0, 1);
    return lerp(lo, hi, Math.sqrt(t));
  }

  function viewPitch() {
    const isBal = craft === 'balloon';
    return clamp(basePitch() + pitchTrim - (isBal ? bal.pit : 0),
                 isBal ? BAL.PITCH_MIN : PITCH_MIN, PITCH_MAX);
  }

  /** Where the stick is, or nothing if the mouse is busy being a mouse: on a
   *  photo, on a button, on the help card, outside the window. */
  function readStick(now) {
    // A card that slid out of range while the cursor rested on it would
    // otherwise hold the stick until the mouse next moved.
    if (hover.card && hover.card.hidden) hover.card = null;
    const held = !!hover.card && now - hover.since > HOVER_MS;   // the cursor is on a photo
    const steer = canRoll && mouse.in && armed && intro.hidden && !hover.hud && !held;
    const g = steer ? stickGain * stickGain : 0;
    const sx = stick(mouse.nx) * g;
    // The other axis, for the bike: the cursor below the middle is the
    // rider leaning back, nose up, the way a stick pulled back is; above it
    // is leaning forward and down. The jet and the balloon do not read it.
    const sy = stick(mouse.ny) * g;
    const keyTurn = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
    const keyPitch = (keys.has('ArrowDown') ? 1 : 0) - (keys.has('ArrowUp') ? 1 : 0);
    return { held, turnIn: clamp(sx + keyTurn, -1, 1), pitchIn: clamp(sy + keyPitch, -1, 1) };
  }

  /** ISA speed of sound at a height: 340 m/s at sea level, falling 4 m/s per
   *  kilometre with the temperature until the tropopause, level above it. */
  const soundSpeed = (h) => (h < ALT_TROPO ? 340.3 - 0.00411 * h : 295.1);

  /** The F-16's limits at a height: the local speed of sound, the most the
   *  afterburner gives, and the most the dry engine gives. */
  function envelope(h) {
    const t = clamp(h / ALT_FAST, 0, 1);
    const a = soundSpeed(h);
    return { a, top: lerp(MACH_SL, MACH_HI, t) * a, mil: lerp(MACH_MIL_SL, MACH_MIL_HI, t) * a };
  }

  /** The jet, one frame. Returns how much afterburner it wants heard. */
  function jetControls(dt, now) {
    const env = envelope(alt);

    // Throttle. The button held opens it, the button released lets it wind
    // down, and the speed follows the throttle rather than the button, so a
    // press is a push rather than a switch. Rise is quicker than fall on
    // purpose: a rhythm of short presses holds a speed without effort, and
    // letting go altogether is a coast to a stop, not a brake. Below the
    // detent the throttle asks for a speed that grows as its square, so the
    // bottom of the travel is a walking pace and the detent is the dry
    // engine's most; past the detent it is the afterburner, and the rest of
    // the travel runs to the F-16's limit at this height.
    const open = hold || touchHold;
    throttle = open ? Math.min(1, throttle + dt / THR_UP)
                    : Math.max(0, throttle - dt / THR_DOWN);
    const ab = throttle > MIL;
    const want = ab ? lerp(env.mil, env.top, (throttle - MIL) / (1 - MIL))
                    : env.mil * Math.pow(throttle / MIL, 2);

    // How fast the speed may follow: the push is the engine's, and it is
    // halved in the middle of the transonic band, where the drag rises and
    // the burner is felt working for it; the brake is drag, and grows with
    // the square of the speed.
    mach = Math.abs(speed) / env.a;
    buffet = clamp(1 - Math.abs(mach - 1) / TRANS_W, 0, 1);
    const push = (ab ? A_AB : A_MIL) * (1 - TRANS_DRAG * buffet);
    const brake = (BRAKE_0 + BRAKE_V * Math.pow(speed / env.top, 2)) * (1 + 0.5 * buffet);
    speed += clamp((want - speed) * SPEED_K, -brake, push) * dt;

    // The barrier. Crossed upwards it is the boom and the cone; crossed
    // back it is the cone alone. A little hysteresis, so a speed that sits
    // on Mach 1 is one crossing and not a drum roll.
    mach = Math.abs(speed) / env.a;
    if (!sonic && mach >= 1) {
      sonic = true;
      booms++;
      Engine.boom();
      shockCone();
    } else if (sonic && mach < 0.985) {
      sonic = false;
      shockCone();
    }

    // The burner is the top of the throttle, past the detent. Below it, with
    // the button down, the engine is heard spooling, so the press is
    // answered the moment it lands.
    const burnWant = ab ? 1 : open ? 0.45 : 0;

    const { held, turnIn } = readStick(now);

    // Roll. A and D roll at a fixed rate for as long as they are down, past
    // the vertical and over the top if you like. Released, the wings level
    // themselves - the short way round, so a roll let go past inverted
    // finishes rather than unwinds - onto the bank a plain turn would show.
    const rollIn = canRoll ? (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0) : 0;
    if (rollIn) {
      bankRate += (rollIn * ROLL_RATE - bankRate) * clamp(ROLL_ACCEL * dt, 0, 1);
      bank += bankRate * dt;
    } else {
      bankRate = 0;
      const want = canRoll ? turnIn * BANK_MAX : 0;
      bank += angleDiff(want, bank) * clamp(LEVEL_K * dt, 0, 1);
    }
    bank = angleDiff(bank, 0);

    // Turn: what the stick asks, plus what the bank gives. A bank turns the
    // nose the way lift does, so a rolled-in turn is tighter than a flat one
    // and an inverted aircraft goes straight - which is also what makes a
    // full roll come out pointing where it went in. Over all of it, the
    // airframe: a turn is a centripetal acceleration of speed times rate,
    // and past 9 g the F-16's own computer refuses the stick. Slow, the cap
    // is far above anything the mouse asks; at Mach 1.2 it is twelve degrees
    // a second and a half-circle two kilometres across, which is why a
    // fighter turns back for a target with such patience.
    const v = Math.abs(speed);
    const yawCap = Math.min(YAW_RATE + BANK_TURN, G_MAX * 9.81 / Math.max(v, 1) / RAD);
    const wantYaw = clamp(turnIn * YAW_RATE + Math.sin(bank * RAD) * BANK_TURN, -yawCap, yawCap);
    // The turn dies quickly under a held photo, so it stops sliding away
    // from the cursor that is trying to click it.
    yaw += (wantYaw - yaw) * clamp((held ? 12 : YAW_ACCEL) * dt, 0, 1);
    bearing = (bearing + yaw * dt + 360) % 360;
    const ac = v * yaw * RAD / 9.81;
    gee = Math.sqrt(1 + ac * ac);

    // Elevator. W pushes the nose down and dives, S pulls it up and climbs,
    // the way a stick pushed forward does; the arrows climb and sink without
    // moving the view. Low down the climb is a fraction of the height you
    // are at, so rooftop to survey height is a handful of seconds either
    // way; from a few hundred metres up it is the F-16's own 254 m/s, and
    // the ceiling is a minute's climb, which is what a ceiling should be.
    const elev = (keys.has('KeyS') ? 1 : 0) - (keys.has('KeyW') ? 1 : 0);
    const trimWant = elev > 0 ? TRIM_UP : elev < 0 ? -TRIM_DOWN : 0;
    pitchTrim += (trimWant - pitchTrim) * clamp(4 * dt, 0, 1);
    const climb = (keys.has('ArrowUp') ? 1 : 0) - (keys.has('ArrowDown') ? 1 : 0) + elev;
    if (climb) {
      const rate = Math.min(alt * CLIMB_K + 6, climb > 0 ? CLIMB_MAX : DIVE_MAX);
      alt = clamp(alt + climb * rate * dt, ALT_MIN, ALT_MAX);
    }
    // Banked over, the wings hold less. Gentle enough that a mouse turn's
    // bank costs nothing you would notice and a knife-edge costs a little.
    const lift = Math.cos(bank * RAD);
    if (lift < 0.9) alt = clamp(alt * (1 - (1 - lift) * SINK_K * dt), ALT_MIN, ALT_MAX);

    if (speed) pos = destination(pos.lat, pos.lng, bearing, speed * dt);
    return burnWant;
  }

  /* ---------- the balloon's flight ----------
   *
   * Same hands, different machine. The mouse still says where to face, the
   * button still says go, the arrows still say up and down; what changes is
   * what is on the other end of each. Up is the burner: it heats air, and
   * the air lifts you a few seconds later and goes on lifting after you let
   * go. Down is the parachute valve at the crown, and it lets that air out.
   * The button is the wind: hold it and the balloon is carried at the full
   * wind of its height, let go and it eases back to a drift, never to a
   * stop, because a balloon is never still. Facing somewhere is a request
   * the drift takes a couple of seconds to grant.
   *
   * The basket is a pendulum under the envelope, and everything the envelope
   * does reaches you through it. So every horizontal acceleration becomes a
   * lean the basket swings towards and past, and the picture rolls and tips
   * with the swing. That, and the silence, is what makes it a balloon. */

  /** Slow, smooth, never-repeating noise in about -1..1: three sines that
   *  share no period. `ch` picks a channel with phases of its own, `rate`
   *  sets its tempo. */
  function turb(ch, rate) {
    const s = bal.seed[ch];
    const t = bal.t * rate;
    return 0.5 * Math.sin(t * 0.31 + s[0]) + 0.3 * Math.sin(t * 0.83 + s[1])
         + 0.2 * Math.sin(t * 1.93 + s[2]);
  }

  /** The balloon, one frame. Returns whether the burner is lit. */
  function balloonControls(dt, now) {
    bal.t += dt;

    // The launch burn holds the burner for you for the first seconds, so the
    // roar and the lift arrive together and say what the key does.
    if (bal.launch > 0) bal.launch -= dt;
    const burnerOn = (keys.has('ArrowUp') || touchBurn || bal.launch > 0) && bal.dT < BAL.DT_MAX;
    const ventOpen = keys.has('ArrowDown') || touchVent;

    // Heat. The burner feeds a plume, the plume becomes envelope heat over
    // MIX_T, the envelope loses heat to the air outside in proportion to
    // what it holds, and the valve dumps it. Lift is heat above what level
    // flight needs, so the envelope that lifted you sinks you once it has
    // cooled past that mark: the rhythm of short burns is not optional.
    bal.plume += ((burnerOn ? BAL.BURN_K : 0) - bal.plume) * clamp(dt / BAL.MIX_T, 0, 1);
    bal.dT += (bal.plume - bal.dT / BAL.COOL_T - (ventOpen ? BAL.VENT_K : 0)) * dt;
    if (alt <= BAL.ALT_MIN + 1) bal.dT = Math.max(bal.dT, BAL.DT_FLOOR);
    bal.dT = clamp(bal.dT, 0, BAL.DT_MAX + 8);

    // Lift against weight, drag against motion, a slow breath of thermal
    // luck on top, and a ceiling where the air is too thin to hold more.
    let lift = BAL.LIFT_K * (bal.dT - BAL.DT_EQ) + 220 * turb(3, 0.25);
    if (lift > 0) lift *= clamp((BAL.ALT_MAX - alt) / BAL.CEIL_FADE, 0, 1);
    const drag = (bal.vz > 0 ? BAL.DRAG_UP : BAL.DRAG_DOWN) * bal.vz * Math.abs(bal.vz);
    bal.vz += (lift - drag) / BAL.MASS * dt;
    alt += bal.vz * dt;
    if (alt <= BAL.ALT_MIN) {
      alt = BAL.ALT_MIN;
      // The basket touches down: a bump through the pendulum, harder the
      // faster you came.
      if (bal.vz < -0.4) {
        bal.pitV += bal.vz * 1.5;
        bal.rollV += bal.vz * (Math.random() - 0.5) * 2;
      }
      bal.vz = 0;
    }
    if (alt >= BAL.ALT_MAX) { alt = BAL.ALT_MAX; bal.vz = Math.min(bal.vz, 0); }

    // Facing. The stick turns the basket, slowly and with weight - these are
    // the rotation vents, not a rudder - and A and D do the same. The basket
    // also wanders a little on its own, the way one does.
    const { held, turnIn } = readStick(now);
    const rollIn = canRoll ? (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0) : 0;
    const want = clamp(turnIn + rollIn, -1, 1) * BAL.YAW_RATE;
    yaw += (want - yaw) * clamp(dt / (held ? 0.15 : BAL.YAW_T), 0, 1);
    bearing = (bearing + (yaw + 0.7 * turb(2, 1)) * dt + 360) % 360;

    // The wind. Its strength is the altitude's, its share is the button's,
    // and the basket follows both with a lag, so a press is felt as a pull
    // that builds and a release as a long glide.
    const open = hold || touchHold;
    bal.ride += ((open ? 1 : BAL.RIDE_IDLE) - bal.ride)
              * clamp(dt / (open ? BAL.RIDE_UP : BAL.RIDE_DOWN), 0, 1);
    const wind = clamp(BAL.WIND_BASE + alt * BAL.WIND_K, BAL.WIND_MIN, BAL.WIND_MAX) * bal.ride;
    speed += (wind - speed) * clamp(dt / BAL.SPEED_T, 0, 1);
    bal.drift = (bal.drift + angleDiff(bearing, bal.drift) * clamp(dt / BAL.DRIFT_T, 0, 1) + 360) % 360;
    if (speed) pos = destination(pos.lat, pos.lng, bal.drift, speed * dt);

    // The pendulum. Whatever accelerates the envelope reaches the basket as
    // a lean it swings towards and past: the wind taking hold tips you, a
    // turn leans you outward. Lightly damped, so it takes a few swings to
    // settle, and the turbulence channels see that it never quite does.
    const sb = Math.sin(bearing * RAD), cb = Math.cos(bearing * RAD);
    const vx = speed * Math.sin(bal.drift * RAD), vy = speed * Math.cos(bal.drift * RAD);
    let fwd = 0, lat = 0;
    if (dt > 0) {
      const ax = (vx - bal.pvx) / dt, ay = (vy - bal.pvy) / dt;
      fwd = ax * sb + ay * cb;
      lat = ax * cb - ay * sb;
    }
    bal.pvx = vx;
    bal.pvy = vy;
    const rollEq = clamp(lat * BAL.SWAY_GAIN, -BAL.SWAY_MAX, BAL.SWAY_MAX)
                 + 0.9 * turb(0, 1) + (burnerOn ? 0.5 * turb(4, 4) : 0);
    const pitEq = clamp(fwd * BAL.SWAY_GAIN, -BAL.SWAY_MAX, BAL.SWAY_MAX) + 0.6 * turb(1, 1);
    const w = BAL.SWAY_W, z = BAL.SWAY_Z;
    bal.rollV += (w * w * (rollEq - bal.roll) - 2 * z * w * bal.rollV) * dt;
    bal.roll += bal.rollV * dt;
    bal.pitV += (w * w * (pitEq - bal.pit) - 2 * z * w * bal.pitV) * dt;
    bal.pit += bal.pitV * dt;
    // The burner lighting is a small shove: the flame is never quite centred
    // and the envelope answers before the basket does.
    if (burnerOn && !bal.burning) {
      bal.pitV -= 1.4;
      bal.rollV += (Math.random() - 0.5) * 2;
      Engine.ignite();
    }
    bal.burning = burnerOn;
    bank = bal.roll;

    // Looking over the edge. W leans out and down, S back and up, and the
    // view follows slowly, as a body does.
    const elev = (keys.has('KeyS') ? 1 : 0) - (keys.has('KeyW') ? 1 : 0);
    const trimWant = elev > 0 ? BAL.TRIM_UP : elev < 0 ? -BAL.TRIM_DOWN : 0;
    pitchTrim += (trimWant - pitchTrim) * clamp(dt / 0.7, 0, 1);

    // The bar under the gauges shows heat here, not throttle; see paintHud.
    throttle = bal.dT / BAL.DT_MAX;
    return burnerOn ? 1 : 0;
  }

  /* ---------- the bike's flight ----------
   *
   * fable's main.js, the part of it that reads the hands and moves the
   * world: the levers are ramped, the physics is stepped at its own 120 Hz
   * under the frame rate, the mesh and the flames follow, and the chase
   * camera decides where the map should look. The map is then asked for
   * exactly that (bikePose), and when it has drawn it, the scene is drawn
   * over it from the same place (onMapRender).
   *
   * The mouse is fable's touch joystick in the other hand: both axes lean
   * the rider, the button is the throttle grip held open, and the arrows do
   * the same for whoever prefers them. Everything else is a key, and the
   * keys are fable's. */

  /** A line across the middle of the screen for a couple of seconds: what
   *  just happened, in words. */
  function say(text, ms = BIKE.MSG_MS, cls = '') {
    if (!elMsg) return;
    bk.msg = text;
    bk.msgUntil = performance.now() + ms;
    elMsg.textContent = text;
    elMsg.className = 'fly-msg show ' + cls;
  }

  const CRASH_TEXT = {
    ground: 'ריסוק! פגיעה חזקה מדי בקרקע',
    flip: 'התהפכות!'
  };

  /** One reaction-thruster pulse: a fixed bit of impulse, and its chuff. */
  function bikePulse(which) {
    if (!bk.rig || paused) return;
    if (bk.rig.phys.firePulse(which)) Engine.pulse();
  }

  function bikeAssist() {
    if (!bk.rig) return;
    const p = bk.rig.phys;
    p.assist = !p.assist;
    say(p.assist ? 'מחשב טיסה: פועל. הגובה נשמר אוטומטית'
                 : 'מחשב טיסה: כבוי. שליטה ידנית מלאה, E/D הם עוצמת העילוי');
  }

  /** R: level the bike where it is. fable resets to its pad; here there is
   *  no pad, and the thing you want back after a tumble is the horizon, not
   *  the take-off point. */
  function bikeLevel() {
    if (!bk.rig || paused) return;
    bk.rig.phys.level();
    say('יישור', 900);
  }

  /** C: the chase camera, or the rider's eyes. The frame changes with it -
   *  the rider's-eye camera rolls all the way and gets the jet's square. */
  function bikeCamera() {
    if (!bk.rig) return;
    bk.cam = bk.cam ? 0 : 1;
    bk.rig.setCamMode(bk.cam);
    bk.camRoll = 0;
    fitFrame();
    say(bk.cam ? 'מהאוכף' : 'מצלמת מעקב', 900);
  }

  /** The bike, one frame. Sets pos, alt, bearing and speed from the
   *  physics for the reveal and the instruments, and leaves the camera it
   *  wants in bk.want. Returns the afterburner, for the glow. */
  function bikeControls(dt, now) {
    const rig = bk.rig, p = rig.phys;
    const { held, turnIn, pitchIn } = readStick(now);

    // fable's applyInput, with the lever between the hand and the engine.
    // W winds the lever up while held and S winds it down, and it stays
    // where it is let go, so a cruise is set once: fable's grip. The mouse
    // button is the jet's: held it winds the lever up, released the lever
    // winds itself down, because that is what the button means in this
    // mode and a bike that ran on after the hand let go would be the one
    // aircraft here that did. A photo under the cursor takes the mouse off
    // the button, or aiming at it would be a shove. S with the lever at
    // idle is the brake: the rider sits up into the wind.
    //
    // The lever's law is squared. fable's bike does 400 km/h flat out and
    // the moshava is four kilometres across; a linear lever put a walking
    // pace in its first twentieth and a highway in the rest. Squared, the
    // first half of the travel is 0 to 210 km/h and the top is still
    // fable's top, so a tap of W is a cruise and a held W is the whole
    // thing. Same reasoning as the F-16's throttle below its detent.
    const keyUp = keys.has('KeyW') || tb.throttleUp;
    const btnUp = hold && !held;
    if (keyUp) { bk.grip = false; bk.lever = clamp(bk.lever + dt * 1.1, 0, 1); }
    else if (btnUp) { bk.grip = true; bk.lever = clamp(bk.lever + dt * 1.1, 0, 1); }
    else if (bk.grip) bk.lever = clamp(bk.lever - dt * 1.5, 0, 1);
    const keyDown = keys.has('KeyS') || tb.throttleDown;
    if (keyDown) bk.lever = clamp(bk.lever - dt * 1.5, 0, 1);
    p.throttle = bk.lever * bk.lever;
    p.brake = keyDown && bk.lever <= 0.001;
    // E/D: with the flight computer on they command climb/descend (the
    // computer flies the lift nozzles); in manual mode they move the
    // collective directly.
    const upDown = (keys.has('KeyE') || tb.collUp ? 1 : 0) - (keys.has('KeyD') || tb.collDown ? 1 : 0);
    if (p.assist && p.autoLift) {
      p.vert += (upDown - p.vert) * Math.min(1, dt * 6);
    } else {
      p.vert = 0;
      if (upDown) p.collective = clamp(p.collective + upDown * dt * 0.55, 0, 1.25);
    }
    p.steer += (clamp(turnIn + tb.steer, -1, 1) - p.steer) * Math.min(1, dt * 8);
    p.pitch += (clamp(pitchIn + tb.pitch, -1, 1) - p.pitch) * Math.min(1, dt * 6);
    p.input.boostRear = keys.has('Space') || tb.boost;

    const crashReason = rig.advance(dt);
    if (crashReason) {
      bk.crashes++;
      bk.lever = 0;      // the wreck's throttle is not the next bike's
      bk.grip = false;
      Engine.crash();
      say(`${CRASH_TEXT[crashReason] || 'ריסוק'} · מתחילים מחדש כאן`, 2400, 'crash');
    }

    // Where the flyer is, for everything that is not the camera: the
    // reveal, the photos, the instruments. Bike frame z is south.
    const [lng, lat] = toLngLat(p.pos.x, -p.pos.z);
    pos = { lat, lng };
    alt = Math.max(0, p.pos.y);
    bearing = rig.heading();
    speed = p.vel.length();
    throttle = p.throttle;

    // The camera it wants; the map is asked for it in step, and the lean
    // follows the bank with a short lag so a wobble is not a shake.
    rig.wantCamera(dt, bk.want);
    const leanWant = bk.cam ? bk.want.roll : clamp(bk.want.roll * BIKE.ROLL_K, -BIKE.ROLL_MAX, BIKE.ROLL_MAX);
    bk.camRoll += angleDiff(leanWant, bk.camRoll) * clamp(dt / BIKE.ROLL_T, 0, 1);
    bk.camRoll = angleDiff(bk.camRoll, 0);
    return p.ab;
  }

  /** The map's view of the camera the bike wants: the point on the ground
   *  under the middle of the screen, and the zoom that puts the camera at
   *  its height. The look is clamped to what the map can do - it cannot
   *  look up, nor quite straight down - and the scene is later drawn from
   *  wherever the map actually went (mapCamera), so a clamp shows as the
   *  bike a little off centre and never as a bike floating off the ground. */
  function bikePose(w) {
    const brg = (Math.atan2(w.dx, -w.dz) / RAD + 360) % 360;
    const dip = Math.asin(clamp(-w.dy, -1, 1)) / RAD;          // degrees below the horizon
    const pitch = clamp(90 - dip, BIKE.PITCH_MIN, BIKE.PITCH_MAX);
    const camAlt = Math.max(w.y, 0.6);
    const [lng, lat] = toLngLat(w.x, -w.z);
    const c = destination(lat, lng, brg, camAlt * Math.tan(pitch * RAD));
    return { center: [c.lng, c.lat], zoom: zoomFor(camAlt, pitch, c.lat), bearing: brg, pitch };
  }

  /** After the map has drawn: the scene, from where the map's camera is,
   *  and the lean of the picture, both in the same frame as the tiles they
   *  have to agree with. Reading the camera back from the map rather than
   *  reusing what was asked for is what makes the bike sit on the ground
   *  through the take-off ease, when the map is between two views, and
   *  through any clamp the map applied on the way. */
  function onMapRender() {
    if (!on || !bk.rig || craft !== 'bike') return;
    const c = mapCamera();
    bk.rig.setCamera(c.x, c.y, c.z, c.dx, c.dy, c.dz, bk.camRoll, c.vfov);
    bk.rig.render();
    const tf = `rotate(${(-bk.camRoll).toFixed(2)}deg)`;
    map.getContainer().style.transform = tf;
    worldEl.style.transform = tf;
    sky.style.transform = tf;
  }

  function step(now) {
    raf = requestAnimationFrame(step);
    const dt = clamp((now - last) / 1000, 0, 0.06);
    last = now;
    const isBal = craft === 'balloon';
    const isBike = craft === 'bike';
    // During the take-off ease the camera belongs to MapLibre, but the sky
    // still has to follow the horizon it is climbing towards - and the
    // balloon's burner is already lit, because that is what lifts it. The
    // bike sits on the ground with its nozzles idling while the camera
    // arrives, so its physics runs, hands off.
    if (!flying) {
      if (isBal) {
        burn += (1 - burn) * clamp(8 * dt, 0, 1);
        if ((frame++ & 1) === 0) Engine.set(0, 1);
      } else if (isBike && bk.rig) {
        bk.rig.advance(dt);
        if ((frame++ & 1) === 0) Engine.setBike(bikePower(), 0);
      }
      paintHud(map.getPitch());
      return;
    }

    // The stick's authority, coming in after it is taken.
    stickGain = armed ? Math.min(1, stickGain + dt / ARM_T) : 0;

    // A photo open full-size holds everything where it is. The speed you had
    // is the speed you get back, because closing a picture is not landing.
    let burnWant = 0;
    if (!paused) burnWant = isBal ? balloonControls(dt, now) : isBike ? bikeControls(dt, now) : jetControls(dt, now);
    const pitch = isBike ? (bk.pose ? bk.pose.pitch : map.getPitch()) : viewPitch();

    burn += (burnWant - burn) * clamp((burnWant > burn ? (isBal ? 14 : 6) : (isBal ? 8 : 3)) * dt, 0, 1);

    // Bank the picture. The map, the sky and the floating cards are rotated
    // together in CSS - they have to move as one or the photos slide off
    // their trails. Right wing down is a counter-clockwise picture. The
    // balloon's basket sways on a phone too, where the jet does not roll;
    // and the jet shakes through the transonic band on any device, which
    // is the buffet of the shocks forming and walking back over the wing.
    // The bike's lean is applied when the map has drawn (onMapRender), so
    // it lands in the same frame as the tiles it leans.
    if (!isBike) {
      let tf = `rotate(${(-bank).toFixed(2)}deg)`;
      if (!isBal && buffet > 0) {
        const a = buffet * buffet * 5;
        tf = `translate(${((Math.random() * 2 - 1) * a).toFixed(1)}px, ${((Math.random() * 2 - 1) * a).toFixed(1)}px) ` + tf;
      }
      if (canRoll || isBal || buffet > 0 || shook) {
        map.getContainer().style.transform = tf;
        worldEl.style.transform = tf;
        sky.style.transform = tf;
        shook = buffet > 0;   // one more frame after the band, to put the picture back
      }
    }

    if (isBike) {
      if (!paused) {
        bk.pose = bikePose(bk.want);
        map.jumpTo(bk.pose);
      }
    } else {
      const ahead = destination(pos.lat, pos.lng, bearing, alt * Math.tan(pitch * RAD));
      map.jumpTo({
        center: [ahead.lng, ahead.lat],
        zoom: zoomFor(alt, pitch, ahead.lat),
        bearing,
        pitch
      });
    }

    // The reveal test is the expensive half and does not need every frame; the
    // positions of what it revealed do, or the cards swim behind the camera.
    tick -= dt;
    let fx, fy;
    if (tick <= 0) {
      tick = 0.11;
      ({ fx, fy } = recompute(pitch, now / 1000));
    } else {
      const f = focusPoint(pitch);
      [fx, fy] = toLocal(f.lat, f.lng);
      paintChips();   // on last frame's reveal: what was found has not changed,
    }                 // but where it is on the screen has
    paintCards(pitch, fx, fy);
    paintHud(pitch);

    // Every other frame is plenty for an automation timeline; the ramps are
    // smoothed on the audio thread anyway. The balloon's burner takes the
    // key and not the eased glow: a valve is open or it is not.
    if ((frame++ & 1) === 0) {
      if (isBal) Engine.set(0, paused ? 0 : burnWant);
      else if (isBike) Engine.setBike(paused ? 0.1 : bikePower(), paused ? 0 : speed);
      else Engine.set(paused ? 0.12 : Math.abs(speed) / topSpeed(), paused ? 0 : burn, sonic && !paused);
    }
  }

  const topSpeed = () => envelope(alt).top;

  /** The bike's jets as a share of full power, for its voice: fable's sum
   *  of the rear jet and the lift nozzles, weighted as fable weights them. */
  function bikePower() {
    const p = bk.rig.phys;
    return clamp(p.jetRear / 8400 * 1.2 + p.jetLift / 7800 * 0.8, 0, 1.6);
  }

  /** The vapour cone. In the low pressure behind a shock the air's water
   *  condenses, and for the second or two an aircraft spends going through
   *  Mach 1 in damp air it wears a disc of cloud, the famous photograph off
   *  the deck of a carrier. Here the disc forms around the camera and the
   *  shock sweeps back over it: a white ring that blooms from the middle of
   *  the screen and passes out of the frame. The animation is restarted by
   *  taking the class off and putting it back, with a reflow between, which
   *  is the one way CSS lets an animation be played twice. */
  function shockCone() {
    if (!coneEl) return;
    coneEl.classList.remove('go');
    void coneEl.offsetWidth;
    coneEl.classList.add('go');
  }

  function paintHud(pitch) {
    // The sky is a band that ends exactly at the horizon, so the pale end of
    // its gradient always meets the ground rather than landing wherever a
    // fixed gradient happened to put it. It sits behind the map, so the seam
    // itself is never seen; the haze in front of the map covers where the far
    // tiles stop.
    //
    // Both are moved with a transform and never resized or repositioned: a
    // painted layer the size of the map that changes every frame has to be
    // rasterised again every frame, and when Chrome falls behind on that it
    // shows the unpainted parts as nothing at all. That was the flicker.
    const y = horizonY(pitch);
    hazeEl.style.transform = `translate3d(0, ${y.toFixed(1)}px, 0)`;
    band.style.transform = `translate3d(0, ${(y + 6 - BAND_H).toFixed(1)}px, 0)`;

    elAlt.textContent = `${Math.round(alt)} מ׳`;
    elSpeed.textContent = `${Math.round(Math.abs(speed) * 3.6)} קמ״ש`;
    if (craft !== 'bike') elThr.style.width = `${(throttle * 100).toFixed(0)}%`;
    paintCompass();

    if (bk.msg && performance.now() > bk.msgUntil) {
      bk.msg = '';
      elMsg.className = 'fly-msg';
    }

    if (craft === 'bike') {
      // fable's instruments: the vertical speed; the throttle and the lift
      // as two-layer bars, the fill being what the turbines are actually
      // doing and the mark what the lever asks, so the spool lag is seen;
      // the flight computer's state; and the boost.
      const p = bk.rig.phys;
      const v = p.vel.y;
      elVsi.textContent = `${v > 0.05 ? '▲' : v < -0.05 ? '▼' : '•'} ${Math.abs(v).toFixed(1)}`;
      // Both in the lever's units, so the fill arrives at the mark: the
      // engine's share is the lever squared, and here it is unsquared.
      elThr.style.width = `${(Math.sqrt(Math.min(1, p.spoolRear + p.ab * 0.2)) * 100).toFixed(0)}%`;
      elThrMark.style.insetInlineStart = `${(bk.lever * 100).toFixed(0)}%`;
      elLiftFill.style.width = `${(p.spoolLift / 1.25 * 100).toFixed(0)}%`;
      elLiftMark.style.insetInlineStart = `${(p.collective / 1.25 * 100).toFixed(0)}%`;
      elFc.textContent = p.assist ? 'פועל' : 'ידני';
      elFcG.classList.toggle('off', !p.assist);
      rushEl.style.opacity = clamp((speed / BIKE.RUSH_V - 0.35) * 0.55, 0, 0.3);
      // The burner is on the screen, in three dimensions, two metres behind
      // the saddle; the glow at the bottom is only its light on the ground.
      burnerEl.style.opacity = burn * 0.45;
    } else if (craft === 'balloon') {
      // A balloon pilot's instrument is the variometer: not where you are
      // but which way you are going, because by the time the altimeter
      // shows it the burn that fixes it is late. The bar is the envelope's
      // heat, cold blue below the mark that holds you and warm above it, so
      // the sink you are about to start is visible before it starts.
      const v = bal.vz;
      elVsi.textContent = `${v > 0.05 ? '▲' : v < -0.05 ? '▼' : '•'} ${Math.abs(v).toFixed(1)}`;
      elThr.style.background = bal.dT >= BAL.DT_EQ ? '#ffb066' : '#8ec3ff';
      rushEl.style.opacity = 0;
      // The flame is not steady, and neither is its light on the envelope.
      const flick = 0.78 + 0.22 * Math.abs(Math.sin(bal.t * 23) * Math.sin(bal.t * 7.3));
      burnerEl.style.opacity = burn * 0.9 * flick;
    } else {
      rushEl.style.opacity = clamp((Math.abs(speed) / topSpeed() - 0.34) * 0.62, 0, 0.34);
      burnerEl.style.opacity = burn * 0.85;
      // The Mach number is the F-16's own instrument, and the one that says
      // where the boom is; the g is why the turn is slow when it is slow.
      elMach.textContent = mach.toFixed(2);
      elG.textContent = gee.toFixed(1);
      document.body.classList.toggle('fly-super', sonic);
    }
    document.body.classList.toggle('fly-burn', burn > 0.5);
    elBurn.hidden = burn < 0.5;

    if (nearest && nearest.name) {
      elName.textContent = nearest.name;
      elName.hidden = false;
      elHint.hidden = false;
      elHint.classList.toggle('open', !mediaOf(nearest).length);
    } else {
      elName.hidden = true;
      elHint.hidden = true;
    }
  }

  /* ---------- looking at one photo, or all of them ----------
   *
   * The viewer is a gallery of the item the card belongs to, not the one
   * picture: a trail with six photos and a video is one place, and having
   * flown to it you should be able to see all of it without landing. Videos
   * play in place. The flight is held, not stopped, while it is open. */

  function openShot(shot) {
    const list = mediaOf(shot.owner);
    let i = list.findIndex((p) => (shot.yt ? p.yt === shot.yt : (p.full || p.thumb) === shot.full));
    openGallery(shot.owner, list, i < 0 ? 0 : i);
  }

  function openGallery(owner, list, i) {
    if (!list.length) return;
    gal = { owner, list, i };
    paused = true;
    // The instruments go while a picture is being looked at. They report a
    // flight that is standing still, and their close button sits in the same
    // corner as the viewer's own - two crosses on top of each other, neither
    // of them obviously the one that closes what is in front of you.
    document.body.classList.add('fly-paused');
    viewer.dataset.item = owner.id;
    viewer.hidden = false;
    paintGallery();
  }

  /** Same lesson the app's lightbox learned: `src = ''` on an iframe resolves
   *  to this page and loads a second copy of the app inside the frame.
   *  about:blank is a real navigation away from YouTube, which is what
   *  actually stops the sound. */
  function blankVideo(vid) {
    if (vid.getAttribute('src')) vid.src = 'about:blank';
    vid.removeAttribute('src');
    vid.hidden = true;
  }

  function paintGallery() {
    const p = gal.list[gal.i];
    const img = viewer.querySelector('.fly-view-img');
    const vid = viewer.querySelector('.fly-view-video');
    if (p.yt) {
      img.hidden = true;
      img.removeAttribute('src');
      vid.hidden = false;
      // autoplay, because getting here took a deliberate press; nocookie so
      // that looking does not set a tracking cookie for somebody who never
      // pressed play.
      vid.src = 'https://www.youtube-nocookie.com/embed/'
        + encodeURIComponent(p.yt) + '?autoplay=1&rel=0';
    } else {
      blankVideo(vid);
      img.hidden = false;
      img.src = p.full || p.thumb;
    }
    viewer.querySelector('.fly-view-name').textContent = gal.owner.name;
    const cap = viewer.querySelector('.fly-view-cap');
    cap.textContent = p.cap || '';
    cap.hidden = !p.cap;
    const many = gal.list.length > 1;
    viewer.querySelector('.fly-view-count').textContent =
      many ? `${gal.i + 1} / ${gal.list.length}` : '';
    viewer.querySelector('.fly-view-prev').hidden = !many;
    viewer.querySelector('.fly-view-next').hidden = !many;
  }

  function stepGallery(d) {
    if (!gal) return;
    gal.i = (gal.i + d + gal.list.length) % gal.list.length;
    paintGallery();
  }

  function closeShot() {
    document.body.classList.remove('fly-paused');
    viewer.hidden = true;
    viewer.querySelector('.fly-view-img').removeAttribute('src');
    blankVideo(viewer.querySelector('.fly-view-video'));
    gal = null;
    paused = false;
    disarm();   // the cursor is on the close button, not on the stick
  }

  /** Enter: the pictures of what you are over, without landing. */
  function openNearest() {
    if (!nearest) return;
    const list = mediaOf(nearest);
    if (list.length) openGallery(nearest, list, 0);
    else leaveTo(nearest.id);
  }

  /** Land on an item: leave the mode and open its page. */
  function leaveTo(id) {
    if (!id) return;
    if (viewer && !viewer.hidden) closeShot();
    exit({ keepPlace: true });
    setTimeout(() => select(id), 320);
  }

  /* ---------- map layers ---------- */

  /** How a lit trail is drawn: a wide soft glow under a thin bright core. The
   *  trails view is the same pair with the glow nearly twice as wide and a
   *  deeper amber, and the core wider and white, so that a trail reads from
   *  across the screen and not only when you are on top of it. */
  function trailPaint(v) {
    const hard = v === 'trails';
    return {
      glow: {
        'line-color': hard ? '#ffb340' : '#ffd166',
        'line-width': ['interpolate', ['linear'], ['zoom'], 13, hard ? 16 : 9, 18, hard ? 64 : 36],
        'line-blur': ['interpolate', ['linear'], ['zoom'], 13, hard ? 8 : 6, 18, hard ? 26 : 20],
        'line-opacity': ['get', 'h']
      },
      core: {
        'line-color': hard ? '#ffffff' : '#fffaea',
        'line-width': ['interpolate', ['linear'], ['zoom'], 13, hard ? 4 : 2.4, 18, hard ? 13 : 8],
        'line-opacity': ['get', 'g']
      }
    };
  }

  function addLayers() {
    if (!map.getSource('fly-trails')) {
      map.addSource('fly-trails', { type: 'geojson', data: EMPTY });
    }
    const paint = trailPaint(view);
    if (!map.getLayer('fly-trail-glow')) {
      map.addLayer({
        id: 'fly-trail-glow',
        type: 'line',
        source: 'fly-trails',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: paint.glow
      });
    }
    if (!map.getLayer('fly-trail-core')) {
      map.addLayer({
        id: 'fly-trail-core',
        type: 'line',
        source: 'fly-trails',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: paint.core
      });
    }
  }

  /* ---------- the view: what hangs over the ground ---------- */

  /** The views this flight can offer: without a trail in the world there is
   *  nothing for the trails view to light, and a position on the button
   *  that changes nothing is a broken button. */
  const viewsNow = () => (hasTrails ? VIEWS : VIEWS.slice(0, 2));

  function setView(v) {
    if (!VIEWS.includes(v)) return view;
    if (on && !viewsNow().includes(v)) v = 'clean';   // the nearest thing to what was asked
    view = v;
    try { localStorage.setItem(KEY_VIEW, v); } catch (_) { /* fine */ }
    if (on) applyView();
    return view;
  }

  function cycleView() {
    const opts = viewsNow();
    const i = opts.indexOf(view);
    return setView(opts[(i + 1) % opts.length]);
  }

  /** Put the flight into the current view: the photos and names that are
   *  not part of it go away, the trail layers take its paint, and the reveal
   *  is redone on the next frame so the glow changes now and not a tenth of
   *  a second later. The button shows where it is and what the next press
   *  brings. */
  function applyView() {
    if (view !== 'normal') for (const [, node] of cards) node.hidden = true;
    if (view === 'clean') for (const [, node] of chips) node.hidden = true;
    if (map && map.getLayer('fly-trail-core')) {
      const paint = trailPaint(view);
      for (const [k, val] of Object.entries(paint.glow)) map.setPaintProperty('fly-trail-glow', k, val);
      for (const [k, val] of Object.entries(paint.core)) map.setPaintProperty('fly-trail-core', k, val);
    }
    tick = 0;
    document.body.dataset.flyView = view;
    if (lookBtn) {
      const opts = viewsNow();
      const next = opts[(opts.indexOf(view) + 1) % opts.length];
      lookBtn.dataset.view = view;
      elLook.textContent = VIEW_NAME[view];
      lookBtn.setAttribute('aria-label', `תצוגה: ${VIEW_NAME[view]}. לחיצה: ${VIEW_NAME[next]}`);
      lookBtn.title = `${VIEW_NAME[view]}: ${VIEW_WHAT[view]}. לחיצה או V: ${VIEW_NAME[next]}`;
    }
  }

  function removeLayers() {
    for (const id of ['fly-trail-core', 'fly-trail-glow']) {
      if (map.getLayer(id)) map.removeLayer(id);
    }
    if (map.getSource('fly-trails')) map.removeSource('fly-trails');
  }

  /** Hide everything the app normally draws, and remember what was hidden so
   *  the way out puts back exactly what was there. Nothing is drawn until you
   *  find it - that is the whole mechanic, and a map with all 62 trails
   *  already on it has nothing left to discover. */
  function hideOverlays() {
    const was = [];
    for (const layer of map.getStyle().layers) {
      if (layer.id.startsWith('fly-') || layer.source === 'sat') continue;
      const vis = map.getLayoutProperty(layer.id, 'visibility');
      was.push([layer.id, vis]);
      map.setLayoutProperty(layer.id, 'visibility', 'none');
    }
    return was;
  }

  function showOverlays(was) {
    for (const [id, vis] of was) {
      if (!map.getLayer(id)) continue;
      map.setLayoutProperty(id, 'visibility', vis === undefined ? 'visible' : vis);
    }
  }

  /* ---------- input ---------- */

  /* Keyed on `code` and never on `key`: the layout here is Hebrew half the
   * time, and on a Hebrew layout `key` for the W position is 'ט'. `code` is
   * the physical key and is the same either way. */

  function onKeyDown(e) {
    if (!on) return;
    if (e.code === 'Escape') {
      e.preventDefault();
      if (!viewer.hidden) closeShot(); else exit();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!viewer.hidden) {
      // The app's convention, RTL: left is onward.
      if (e.code === 'ArrowLeft') { e.preventDefault(); stepGallery(1); }
      else if (e.code === 'ArrowRight') { e.preventDefault(); stepGallery(-1); }
      return;
    }
    if (e.code === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) leaveTo(nearest && nearest.id); else openNearest();
      return;
    }
    if (e.code === 'KeyM') { e.preventDefault(); setSound(Engine.toggle()); return; }
    if (e.code === 'KeyV') { e.preventDefault(); cycleView(); return; }
    if (e.code === 'KeyH' || e.code === 'Slash') { e.preventDefault(); toggleIntro(); return; }
    if (craft === 'bike' && bk.rig) {
      // fable's one-shot keys. Not on repeat: a held Shift is one pulse, not
      // a stream of them, and the nozzle's own refractory gap agrees.
      const once = !e.repeat;
      switch (e.code) {
        case 'KeyZ': if (once) bikePulse('L'); e.preventDefault(); dismissIntro(); return;
        case 'KeyX': if (once) bikePulse('R'); e.preventDefault(); dismissIntro(); return;
        case 'ShiftLeft':
        case 'ShiftRight': if (once) bikePulse('C'); e.preventDefault(); dismissIntro(); return;
        case 'KeyT': if (once) bikeAssist(); e.preventDefault(); return;
        case 'KeyR': if (once) bikeLevel(); e.preventDefault(); return;
        case 'KeyC': if (once) bikeCamera(); e.preventDefault(); return;
      }
    }
    if (/^(Key[WASDE]|Space|Arrow(Up|Down|Left|Right))$/.test(e.code)) {
      e.preventDefault();
      keys.add(e.code);
      dismissIntro();
      Engine.poke();   // a key is a gesture, for a context the browser left suspended
    }
  }

  function onKeyUp(e) {
    keys.delete(e.code);
  }

  /* The mouse is the stick: where it sits is where the nose goes, no button
   * held. It is read on every move and turned into a position relative to the
   * middle of the screen; what it does with that position is decided in
   * `step`, which also knows when the cursor is busy being a cursor. */

  function onPointerMove(e) {
    if (!on || e.pointerType === 'touch') return;
    const hw = innerWidth / 2, hh = innerHeight / 2;
    // The take-off card goes on the first real movement of the mouse once
    // the aircraft is flying: a hand that moves the mouse wants to fly. Only
    // that card - the help reopened with ? was asked for, and stays until
    // it is put away.
    if (!intro.hidden && introTimer && flying) {
      if (mouse.in) introMoved += Math.abs(e.clientX - mouse.px) + Math.abs(e.clientY - mouse.py);
      if (introMoved > INTRO_MOVE) dismissIntro();
    }
    mouse.px = e.clientX; mouse.py = e.clientY;
    mouse.nx = clamp((e.clientX - hw) / hw, -1, 1);
    mouse.ny = clamp((e.clientY - hh) / hh, -1, 1);
    mouse.in = true;
    if (!armed && intro.hidden && !paused) {
      armed = true;
      document.body.classList.remove('fly-unarmed');
    }
    const t = e.target;
    hover.hud = !!(t && t.closest && t.closest('.fly-hud button, .fly-view, .fly-intro, .fly-touch'));
    const card = t && t.closest ? t.closest('.fly-card') : null;
    if (card !== hover.card) {
      hover.card = card;
      hover.since = performance.now();
    }
  }

  // Out of the window is hands off the stick, and so is the window losing
  // focus; a cursor parked at the edge of the screen while you read something
  // else would otherwise fly you in circles.
  function onPointerLeave() {
    mouse.in = false;
    hover.card = null;
    hover.hud = false;
    hold = false;
    disarm();
  }

  /* The button is the throttle. Only the main button, and only over open
   * ground: a press on a photo is a click, a press on a button is a button.
   * The release is taken from anywhere, because a finger that slid off the
   * window is still a finger that let go. */
  function onPointerDown(e) {
    if (!on || e.pointerType === 'touch' || e.button !== 0) return;
    const t = e.target;
    if (t && t.closest && t.closest('.fly-card, .fly-hud, .fly-view, .fly-intro, .fly-touch')) return;
    hold = true;
    Engine.poke();
  }

  function onPointerUp() { hold = false; }

  function disarm() {
    if (!canRoll) return;
    armed = false;
    document.body.classList.add('fly-unarmed');
  }

  function onBlur() {
    keys.clear();
    touchHold = false;
    touchBurn = false;
    touchVent = false;
    clearTouch();
    onPointerLeave();
  }

  function onClick(e) {
    const t = e.target;
    if (t && t.closest && t.closest('.fly-card, .fly-hud, .fly-view, .fly-intro')) return;
    dismissIntro();
    Engine.poke();
  }

  /* Touch: no keyboard to fly with, so a drag steers and holding the screen
   * is the throttle. Deliberately small - this mode is a desktop pleasure and
   * a phone should get something that works rather than a second control
   * scheme to learn. */
  let touch = null;

  function onTouchStart(e) {
    const tgt = e.target;
    if (tgt && tgt.closest && tgt.closest('.fly-card, .fly-hud, .fly-view, .fly-touch')) return;
    // The bike has fable's joystick and buttons (buildTouch); a finger on
    // open ground is not a lever there.
    if (craft === 'bike') { dismissIntro(); Engine.poke(); return; }
    const t = e.touches[0];
    touch = { x: t.clientX, y: t.clientY, y0: t.clientY };
    touchHold = true;
    dismissIntro();
    Engine.poke();
  }

  function onTouchMove(e) {
    if (!touch) return;
    const t = e.touches[0];
    const dx = t.clientX - touch.x, dy = t.clientY - touch.y;
    touch.x = t.clientX; touch.y = t.clientY;
    bearing = (bearing - dx * 0.28 + 360) % 360;
    if (craft === 'balloon') {
      // The balloon's height is not a thing a finger can set; what it can do
      // is open the burner or the valve. A finger pulled well below where
      // it landed is the burner, pushed well above it the valve, and back
      // in the middle neither.
      const off = t.clientY - touch.y0;
      touchBurn = off > 40;
      touchVent = off < -40;
    } else if (Math.abs(dy) > 1) {
      alt = clamp(alt * (1 + dy * 0.004), ALT_MIN, ALT_MAX);
    }
  }

  function onTouchEnd() { touch = null; touchHold = false; touchBurn = false; touchVent = false; }

  /* fable's touch controls (touch.js there), for the bike on a phone: a
   * virtual joystick on the left for the lean - pull the knob down to
   * raise the nose, the yoke convention the arrows use - and on the right
   * hold-buttons for the throttle and for climb and descend, three pulse
   * buttons, the boost, and the camera, the flight computer and the
   * levelling as taps. Pointer Events, so it also works with a mouse for
   * testing. Built once with the rest of the DOM; CSS shows it only for
   * the bike on a coarse pointer. */
  function buildTouch(root) {
    const el = (cls, parent, tag = 'div') => {
      const d = document.createElement(tag);
      d.className = cls;
      parent.appendChild(d);
      return d;
    };
    touchEl = el('fly-touch', root);

    const joyBase = el('tjoy', touchEl);
    const joyKnob = el('tjoy-knob', joyBase);
    const JOY_R = 46;
    let joyId = null, joyCX = 0, joyCY = 0;
    const expo = (v) => 0.4 * v + 0.6 * v * Math.abs(v);   // fine authority near the centre
    function joyMove(e) {
      let dx = e.clientX - joyCX, dy = e.clientY - joyCY;
      const d = Math.hypot(dx, dy);
      if (d > JOY_R) { dx *= JOY_R / d; dy *= JOY_R / d; }
      joyKnob.style.transform = `translate(${dx}px, ${dy}px)`;
      tb.steer = expo(clamp(dx / JOY_R, -1, 1));
      tb.pitch = expo(clamp(dy / JOY_R, -1, 1));
    }
    function joyReset() {
      tb.steer = 0; tb.pitch = 0;
      joyKnob.style.transform = 'translate(0,0)';
    }
    joyBase.addEventListener('pointerdown', (e) => {
      joyId = e.pointerId;
      const r = joyBase.getBoundingClientRect();
      joyCX = r.left + r.width / 2; joyCY = r.top + r.height / 2;
      try { joyBase.setPointerCapture(joyId); } catch (_) { /* synthetic pointer */ }
      joyMove(e);
      dismissIntro();
      Engine.poke();
      e.preventDefault();
    });
    joyBase.addEventListener('pointermove', (e) => { if (e.pointerId === joyId) { joyMove(e); e.preventDefault(); } });
    const joyEnd = (e) => { if (e.pointerId === joyId) { joyId = null; joyReset(); } };
    joyBase.addEventListener('pointerup', joyEnd);
    joyBase.addEventListener('pointercancel', joyEnd);

    function holdBtn(cls, parent, label, onDown, onUp) {
      const b = el(cls, parent, 'button');
      b.type = 'button';
      b.textContent = label;
      b.addEventListener('pointerdown', (e) => {
        try { b.setPointerCapture(e.pointerId); } catch (_) { /* synthetic pointer */ }
        onDown();
        b.classList.add('active');
        dismissIntro();
        Engine.poke();
        e.preventDefault();
      });
      const release = () => { onUp(); b.classList.remove('active'); };
      b.addEventListener('pointerup', release);
      b.addEventListener('pointercancel', release);
      return b;
    }
    function tapBtn(cls, parent, label, fn) {
      const b = el(cls, parent, 'button');
      b.type = 'button';
      b.textContent = label;
      b.addEventListener('pointerdown', (e) => {
        fn();
        b.classList.add('active');
        setTimeout(() => b.classList.remove('active'), 160);
        e.preventDefault();
      });
      return b;
    }

    const side = el('tside', touchEl);
    const pair = el('tvert-pair', side);
    const thrBox = el('tvert', pair);
    el('tlabel', thrBox).textContent = 'מצערת';
    holdBtn('tbtn', thrBox, '▲', () => { tb.throttleUp = true; }, () => { tb.throttleUp = false; });
    holdBtn('tbtn', thrBox, '▼', () => { tb.throttleDown = true; }, () => { tb.throttleDown = false; });
    const colBox = el('tvert', pair);
    el('tlabel', colBox).textContent = 'גובה';
    holdBtn('tbtn', colBox, '▲', () => { tb.collUp = true; }, () => { tb.collUp = false; });
    holdBtn('tbtn', colBox, '▼', () => { tb.collDown = true; }, () => { tb.collDown = false; });
    const rollRow = el('troll', side);
    tapBtn('tbtn tsmall', rollRow, '↺', () => bikePulse('L'));
    tapBtn('tbtn tsmall', rollRow, '⬆', () => bikePulse('C'));
    tapBtn('tbtn tsmall', rollRow, '↻', () => bikePulse('R'));
    holdBtn('tboost', side, '🔥', () => { tb.boost = true; }, () => { tb.boost = false; });

    const menu = el('tmenu', touchEl);
    tapBtn('tbtn tsmall', menu, '📷', bikeCamera);
    tapBtn('tbtn tsmall', menu, 'T', bikeAssist);
    tapBtn('tbtn tsmall', menu, '⟲', bikeLevel);
  }

  function clearTouch() {
    tb.steer = 0; tb.pitch = 0;
    tb.throttleUp = tb.throttleDown = tb.collUp = tb.collDown = tb.boost = false;
  }

  function onVisibility() {
    if (!on) return;
    if (document.hidden) { Engine.stop(); keys.clear(); } else Engine.start();
  }

  // A right-click over the map is a slip of the hand, and a context menu
  // over a flight is the browser stepping in front of it.
  function onContextMenu(e) {
    if (on && !e.target.closest('.fly-view')) e.preventDefault();
  }

  function setSound(isOn) {
    if (!sndBtn) return;
    sndBtn.classList.toggle('off', !isOn);
    const what = craft === 'balloon' ? 'המבער' : 'המנוע';   // the bike's is an engine too
    sndBtn.setAttribute('aria-label', isOn ? `השתקת ${what}` : 'הפעלת הצליל');
    sndBtn.title = isOn ? 'השתקה (M)' : 'צליל (M)';
  }

  /* ---------- intro ---------- */

  /* Set only for the take-off card, and cleared with it: onPointerMove reads
   * it to tell that card, which a moving mouse puts away, from the help
   * reopened with ?, which stays until it is closed. */
  let introTimer = null;

  function dismissIntro() {
    if (!intro || intro.hidden) return;
    intro.hidden = true;
    clearTimeout(introTimer);
    introTimer = null;
    disarm();
  }

  function toggleIntro() {
    intro.hidden = !intro.hidden;
    clearTimeout(introTimer);
    introTimer = null;
    if (intro.hidden) disarm();
  }

  /* ---------- the compass ----------
   *
   * A needle compass, the kind in a pocket: the dial is fixed with the top
   * being where you look, and the needle swings so that its red end points
   * north. (The first version was a card compass, the rose turning under a
   * fixed mark with letters on it; Ori asked for the needle and no big
   * letters.) Under it the heading in words, because "צפון־מערב" is read
   * at a glance and 315° is not. The balloon gets one thing more, a marker
   * on the rim for the way the wind is carrying you, since in a balloon
   * where you face and where you go are two different questions. */

  const WINDS = ['צפון', 'צפון־מזרח', 'מזרח', 'דרום־מזרח', 'דרום', 'דרום־מערב', 'מערב', 'צפון־מערב'];
  let elNeedle = null, elDrift = null, elHeading = null;
  let headingShown = -1;

  /** The tick marks of the dial: every ten degrees, longer every thirty,
   *  longest at the four quarters. */
  function compassTicks() {
    let out = '';
    for (let a = 0; a < 360; a += 10) {
      const k = a % 90 === 0 ? 'rose-tick quarter' : a % 30 === 0 ? 'rose-tick long' : 'rose-tick';
      const len = a % 90 === 0 ? -31 : a % 30 === 0 ? -34 : -37;
      out += `<line x1="0" y1="-41" x2="0" y2="${len}" class="${k}" transform="rotate(${a})"/>`;
    }
    return out;
  }

  function paintCompass() {
    elNeedle.setAttribute('transform', `rotate(${(-bearing).toFixed(1)})`);
    if (craft === 'balloon') elDrift.setAttribute('transform', `rotate(${angleDiff(bal.drift, bearing).toFixed(1)})`);
    // The bike's marker is the wind: where it blows to, relative to the
    // nose, because a hover drifts with it and the nose vanes into it.
    if (craft === 'bike' && bk.rig) {
      const W = bk.rig.phys.wind.W;
      const wb = Math.atan2(W.x, -W.z) / RAD;
      elDrift.setAttribute('transform', `rotate(${angleDiff(wb, bearing).toFixed(1)})`);
    }
    const deg = Math.round(bearing) % 360;
    if (deg !== headingShown) {
      headingShown = deg;
      elHeading.textContent = `${WINDS[Math.round(deg / 45) % 8]} ${deg}°`;
    }
  }

  /* ---------- entering and leaving ---------- */

  function buildDom() {
    if (root) return;
    root = document.createElement('div');
    root.className = 'fly-root';
    root.innerHTML = `
      <div class="fly-sky" id="fly-sky"><div class="fly-band" id="fly-band"></div></div>
      <div class="fly-world" id="fly-world"><div class="fly-haze" id="fly-haze"></div></div>
      <canvas class="fly-bike" id="fly-bike" hidden aria-hidden="true"></canvas>
      <div class="fly-rush" id="fly-rush"></div>
      <div class="fly-burner" id="fly-burner"></div>
      <div class="fly-cone" id="fly-cone" aria-hidden="true"></div>
      <div class="fly-vig"></div>
      <div class="fly-basket" aria-hidden="true"></div>
      <div class="fly-hud">
        <div class="fly-reticle" aria-hidden="true"></div>
        <div class="fly-readout">
          <span class="fly-gauge"><b id="fly-alt">—</b><i>גובה</i></span>
          <span class="fly-gauge" id="fly-vsi-g" hidden><b id="fly-vsi">—</b><i>מ׳/שנ׳ אנכי</i></span>
          <span class="fly-gauge"><b id="fly-speed">—</b><i>מהירות</i></span>
          <span class="fly-gauge mach" id="fly-mach-g"><b id="fly-mach">—</b><i>מאך</i></span>
          <span class="fly-gauge gee" id="fly-g-g"><b id="fly-g">—</b><i>g</i></span>
          <span class="fly-gauge fc" id="fly-fc-g" hidden><b id="fly-fc">פועל</b><i>מחשב טיסה</i></span>
          <span class="fly-gauge burn" id="fly-burn" hidden><b>מבער</b><i>אחורי</i></span>
        </div>
        <div class="fly-thr" aria-hidden="true"><i id="fly-thr-fill"></i><b id="fly-thr-mark" hidden></b></div>
        <div class="fly-thr fly-lift" id="fly-lift" aria-hidden="true" hidden><i id="fly-lift-fill"></i><b id="fly-lift-mark"></b></div>
        <p class="fly-msg" id="fly-msg" aria-live="polite"></p>
        <div class="fly-compass" id="fly-compass">
          <svg viewBox="-50 -50 100 100" aria-hidden="true">
            <circle r="41" class="rose-ring"/>
            ${compassTicks()}
            <g id="fly-needle">
              <path class="needle-n" d="M0,-35 L5.5,0 L-5.5,0 Z"/>
              <path class="needle-s" d="M0,35 L5.5,0 L-5.5,0 Z"/>
              <circle r="3.2" class="needle-pin"/>
            </g>
            <path id="fly-drift" class="rose-drift" d="M0,-45 l-5,-9 h10 z"/>
            <rect class="rose-lubber" x="-1.6" y="-45" width="3.2" height="10" rx="1"/>
          </svg>
          <b id="fly-heading" aria-live="off">—</b>
        </div>
        <button class="fly-x" id="fly-x" aria-label="יציאה ממצב תעופה">&times;</button>
        <button class="fly-help" id="fly-help" aria-label="מקשים">?</button>
        <button class="fly-snd" id="fly-snd" aria-label="השתקת המנוע">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path class="snd-body" d="M4 9v6h4l5 4V5L8 9H4z"/>
            <path class="snd-wave" d="M16 8.5a4.5 4.5 0 010 7M18.5 5.5a8.5 8.5 0 010 13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            <path class="snd-off" d="M16.5 9.5l5 5m0-5l-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
        <button class="fly-look" id="fly-look" aria-label="תצוגה">
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <g class="look-normal">
              <rect x="3" y="5" width="18" height="14" rx="2.2"/>
              <circle cx="8.6" cy="9.6" r="1.6" fill="currentColor" stroke="none"/>
              <path d="M3.6 17.2l4.9-4.9 3.6 3.6 2.6-2.6 5.6 5.6"/>
            </g>
            <g class="look-clean">
              <path d="M8.2 5H19a2 2 0 012 2v9.2M16.2 19H5a2 2 0 01-2-2V7.6"/>
              <path d="M3.5 3.5l17 17" stroke-width="2"/>
            </g>
            <g class="look-trails">
              <path class="look-glow" d="M3.5 18.5C7 8 11 22 15 11.5S20 7.5 20.5 5.5" stroke-width="6" opacity=".38"/>
              <path d="M3.5 18.5C7 8 11 22 15 11.5S20 7.5 20.5 5.5" stroke-width="2.2"/>
            </g>
          </svg>
          <b id="fly-look-name">רגיל</b>
        </button>
        <div class="fly-found">
          <strong id="fly-name" hidden></strong>
          <span id="fly-hint" class="fly-hint" hidden>
            <span class="hint-media"><kbd>Enter</kbd> <span>לתמונות</span></span>
            <span class="hint-open"><kbd>⇧ Enter</kbd> <span>לפתיחה</span></span>
          </span>
        </div>
        <p class="fly-credit">Esri · Maxar · Earthstar Geographics · גובה: Mapzen / AWS</p>
      </div>
      <div class="fly-intro" id="fly-intro">
        <div class="fly-intro-card jet">
          <h2>מצב תעופה: F-16</h2>
          <p>אתה בתא הטייס של F-16 (בחיל האוויר: נץ, ברק וסופה) מעל פרדס חנה־כרכור.
             תקרת הטיסה 15,240 מ׳, שהם 50,000 רגל. המהירות המרבית מאך 1.2 קרוב לקרקע,
             כ־1,470 קמ״ש, ומאך 2 בגובה. שני שלישים מהמצערת הם המנוע היבש; מעבר להם נדלק
             המבער האחורי, ורק איתו עוברים את מהירות הקול. המעבר מלווה בטלטלה, בענן הלם
             ובבום על־קולי, ומעבר לו המנוע נשאר מאחור: שומעים רק את הרוח. הפנייה מוגבלת
             ל־9 g, אז במהירות גבוהה המטוס פונה לאט. דרכי הקיצור נדלקות כשמתקרבים אליהן, והתמונות שלהן
             תלויות באוויר מעל השביל. טסים אל תמונה, ולוחצים עליה.</p>
          <ul class="fly-keys">
            <li><kbd class="wide">עכבר</kbd><span>ההגה: ימינה ושמאלה פונים</span></li>
            <li><kbd class="wide">לחיצה</kbd><span>המצערת: כל עוד הכפתור לחוץ המנוע מגביר, וכשמשחררים הוא דועך. להחזיק כחצי דקה כדי לעבור את מאך 1</span></li>
            <li><kbd>W</kbd><kbd>S</kbd><span>אף למטה, אף למעלה</span></li>
            <li><kbd>A</kbd><kbd>D</kbd><span>גלגול. להחזיק לגלגול שלם</span></li>
            <li><kbd>↑</kbd><kbd>↓</kbd><span>גובה. עד התקרה זו דקה של טיפוס</span></li>
            <li><kbd>Enter</kbd><span>התמונות של השביל הקרוב</span></li>
            <li><kbd>V</kbd><span>תצוגה: רגיל, נקי (בלי תמונות, כדי לראות את השטח), שבילים (מוארים חזק)</span></li>
            <li><kbd>M</kbd><span>המנוע</span></li>
            <li><kbd>Esc</kbd><span>יציאה</span></li>
          </ul>
          <p class="fly-intro-foot">הזזת העכבר סוגרת את הכרטיס הזה ותופסת את ההגה; <kbd>?</kbd> מחזיר אותו.
             כשהסמן נח על תמונה ההגה משתחרר, כדי שאפשר יהיה ללחוץ עליה.
             בניגוד ל־F-16 אמיתי, כשמרפים מהמצערת המטוס נעצר באוויר: זה מטוס שנועד להסתכל.</p>
          <p class="fly-intro-touch">אצבע על המסך היא המצערת: מחזיקים וטסים, משחררים ונעצרים.
             גרירה לצדדים פונה, למעלה ולמטה משנה גובה. נגיעה בתמונה פותחת אותה.</p>
        </div>
        <div class="fly-intro-card balloon">
          <h2>כדור פורח</h2>
          <p>אתה נישא בשקט מעל פרדס חנה־כרכור. דרכי הקיצור נדלקות כשמתקרבים אליהן,
             והתמונות שלהן תלויות באוויר מעל השביל. הכדור מגיב באיחור של כמה שניות
             לכל דבר, אז מבעירים מעט, ומחכים.</p>
          <ul class="fly-keys">
            <li><kbd>↑</kbd><span>המבער. מחמם את האוויר, והכדור עולה אחרי כמה שניות וממשיך גם כשמרפים</span></li>
            <li><kbd>↓</kbd><span>פתח האוורור בקודקוד: משחרר אוויר חם, ויורדים</span></li>
            <li><kbd class="wide">לחיצה</kbd><span>הרוח: כל עוד הכפתור לחוץ נסחפים במלוא הרוח, וכשמשחררים היא נרגעת</span></li>
            <li><kbd class="wide">עכבר</kbd><span>לאן הסל פונה. הרוח מתיישרת אחריו, לאט</span></li>
            <li><kbd>W</kbd><kbd>S</kbd><span>להביט מטה מעבר לדופן, ובחזרה למעלה</span></li>
            <li><kbd>Enter</kbd><span>התמונות של השביל הקרוב</span></li>
            <li><kbd>V</kbd><span>תצוגה: רגיל, נקי (בלי תמונות, כדי לראות את השטח), שבילים (מוארים חזק)</span></li>
            <li><kbd>M</kbd><span>המבער</span></li>
            <li><kbd>Esc</kbd><span>יציאה</span></li>
          </ul>
          <p class="fly-intro-foot">הזזת העכבר סוגרת את הכרטיס הזה; <kbd>?</kbd> מחזיר אותו.
             הבערה קצרה כל עשרים שניות מחזיקה גובה. ככל שגבוה יותר, הרוח חזקה יותר.</p>
          <p class="fly-intro-touch">אצבע על המסך תופסת את הרוח. גרירה לצדדים פונה, גרירה למטה מבעירה,
             גרירה למעלה מאווררת. הכדור מגיב באיחור, אז מבעירים מעט ומחכים.</p>
        </div>
        <div class="fly-intro-card bike">
          <h2>אופנוע סילון: FLYING HOG</h2>
          <p>אתה על האוכף של אופנוע סילון מעופף, על הקרקע, בפרדס חנה־כרכור. אתה מטה את הגוף,
             ומחשב הטיסה מטיס בשבילך את צינורות העילוי: האופנוע טס לאן שהאף מצביע, ואף ישר
             שומר גובה לבד. אפשר לנחות על כל שביל, ואפשר להתרסק. דרכי הקיצור נדלקות
             כשמתקרבים אליהן, והתמונות שלהן תלויות באוויר מעל השביל.</p>
          <ul class="fly-keys">
            <li><kbd class="wide">עכבר</kbd><span>הטיית הגוף: לצדדים לפנייה, למטה להרמת האף, למעלה לצלילה</span></li>
            <li><kbd class="wide">לחיצה</kbd><span>המצערת, כל עוד הכפתור לחוץ; כשמשחררים היא נסגרת</span></li>
            <li><kbd>W</kbd><kbd>S</kbd><span>מנוף המצערת: נשאר איפה שעזבו. S כשהמצערת סגורה הוא בלם</span></li>
            <li><kbd>E</kbd><kbd>D</kbd><span>עלייה וירידה. מהקרקע: E מרים, ואז מצערת</span></li>
            <li><kbd>←</kbd><kbd>→</kbd><kbd>↑</kbd><kbd>↓</kbd><span>הטיית הגוף במקשים; חץ למטה מרים את האף</span></li>
            <li><kbd class="wide">רווח</kbd><span>מבער אחורי</span></li>
            <li><kbd class="wide">Shift</kbd><kbd>Z</kbd><kbd>X</kbd><span>פולסים: ניתור, גלגול שמאלה, גלגול ימינה</span></li>
            <li><kbd>C</kbd><kbd>T</kbd><kbd>R</kbd><span>מצלמה, מחשב טיסה, יישור</span></li>
            <li><kbd>Enter</kbd><span>התמונות של השביל הקרוב</span></li>
            <li><kbd>V</kbd><span>תצוגה: רגיל, נקי, שבילים</span></li>
            <li><kbd>M</kbd><span>המנוע</span></li>
            <li><kbd>Esc</kbd><span>יציאה</span></li>
          </ul>
          <p class="fly-intro-foot">הזזת העכבר סוגרת את הכרטיס הזה ותופסת את ההגה; <kbd>?</kbd> מחזיר אותו.
             הסמן במרכז המסך הוא גוף ישר. פגיעה בקרקע מעל 40 קמ״ש היא ריסוק, ואחריו מתחילים
             מאותו מקום.</p>
          <p class="fly-intro-touch">ג'ויסטיק משמאל מטה את הגוף (למטה מרים את האף). מימין: מצערת,
             עלייה וירידה, שלושה פולסים, ומבער. הגובה נשמר אוטומטית.</p>
        </div>
      </div>
      <div class="fly-view" id="fly-view" hidden>
        <button class="fly-view-x" aria-label="סגירה">&times;</button>
        <button class="fly-view-nav fly-view-prev" aria-label="הקודמת">&rsaquo;</button>
        <button class="fly-view-nav fly-view-next" aria-label="הבאה">&lsaquo;</button>
        <div class="fly-view-stage">
          <img class="fly-view-img" alt="" referrerpolicy="no-referrer">
          <iframe class="fly-view-video" hidden allow="autoplay; encrypted-media; picture-in-picture"
                  allowfullscreen referrerpolicy="strict-origin-when-cross-origin" title="סרטון"></iframe>
        </div>
        <div class="fly-view-foot">
          <strong class="fly-view-name"></strong>
          <span class="fly-view-cap"></span>
          <span class="fly-view-count"></span>
          <button class="fly-view-open">פתח את השביל</button>
        </div>
      </div>`;
    document.body.appendChild(root);

    sky = el('fly-sky');
    band = el('fly-band');
    worldEl = el('fly-world');
    hazeEl = el('fly-haze');
    rushEl = el('fly-rush');
    burnerEl = el('fly-burner');
    elAlt = el('fly-alt');
    elSpeed = el('fly-speed');
    elVsi = el('fly-vsi');
    elVsiG = el('fly-vsi-g');
    elMach = el('fly-mach');
    elMachG = el('fly-mach-g');
    elG = el('fly-g');
    elGG = el('fly-g-g');
    coneEl = el('fly-cone');
    elNeedle = el('fly-needle');
    elDrift = el('fly-drift');
    elHeading = el('fly-heading');
    elBurn = el('fly-burn');
    elThr = el('fly-thr-fill');
    elName = el('fly-name');
    elHint = el('fly-hint');
    sndBtn = el('fly-snd');
    lookBtn = el('fly-look');
    elLook = el('fly-look-name');
    intro = el('fly-intro');
    viewer = el('fly-view');
    elThrMark = el('fly-thr-mark');
    elLift = el('fly-lift');
    elLiftFill = el('fly-lift-fill');
    elLiftMark = el('fly-lift-mark');
    elFc = el('fly-fc');
    elFcG = el('fly-fc-g');
    elMsg = el('fly-msg');
    bk.canvas = el('fly-bike');
    buildTouch(root);

    el('fly-x').addEventListener('click', () => exit());
    el('fly-help').addEventListener('click', toggleIntro);
    sndBtn.addEventListener('click', () => setSound(Engine.toggle()));
    lookBtn.addEventListener('click', cycleView);
    intro.addEventListener('click', dismissIntro);
    viewer.querySelector('.fly-view-x').addEventListener('click', closeShot);
    viewer.querySelector('.fly-view-prev').addEventListener('click', () => stepGallery(-1));
    viewer.querySelector('.fly-view-next').addEventListener('click', () => stepGallery(1));
    viewer.querySelector('.fly-view-open').addEventListener('click', () => leaveTo(viewer.dataset.item));
    setSound(Engine.isOn());
  }

  function enter() {
    if (on || !map) return;
    buildDom();

    const c = map.getCenter();
    restore = {
      center: [c.lng, c.lat],
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch(),
      fov: getFov(),
      base: baseIndex,
      dragPan: map.dragPan.isEnabled()
    };

    on = true;
    flying = false;
    paused = false;
    pos = { lat: c.lat, lng: c.lng };
    bearing = (map.getBearing() + 360) % 360;
    const isBal = craft === 'balloon';
    const isBike = craft === 'bike';
    alt = isBal ? BAL.ALT_START : isBike ? 0.6 : ALT_START;
    speed = 0;
    throttle = 0;
    hold = false;
    yaw = 0;
    bank = 0;
    bankRate = 0;
    burn = 0;
    pitchTrim = 0;
    mach = 0;
    gee = 1;
    buffet = 0;
    sonic = false;
    booms = 0;
    shook = false;
    keys.clear();
    mouse.in = false;
    hover.card = null;
    hover.hud = false;
    touchHold = false;
    touchBurn = false;
    touchVent = false;
    armed = false;
    stickGain = 0;
    introMoved = 0;

    if (isBal) {
      // On the ground with a warm envelope that will not quite carry you:
      // the launch burn is what lifts you off, and it is already lit through
      // the take-off ease.
      bal.dT = BAL.DT_EQ - 6;
      bal.plume = 0;
      bal.vz = 0.8;
      bal.drift = bearing;
      bal.ride = BAL.RIDE_IDLE;
      bal.roll = 0; bal.rollV = 0;
      bal.pit = 0; bal.pitV = 0;
      bal.pvx = 0; bal.pvy = 0;
      bal.launch = 1.75 + BAL.LAUNCH;
      bal.burning = false;
      bal.t = 0;
      bal.seed = [];
      for (let i = 0; i < 5; i++) {
        bal.seed.push([Math.random() * 6.3, Math.random() * 6.3, Math.random() * 6.3]);
      }
    }
    elBurn.querySelector('i').textContent = isBal ? 'דולק' : 'אחורי';
    elVsiG.hidden = !isBal && !isBike;
    elMachG.hidden = isBal || isBike;
    elGG.hidden = isBal || isBike;
    elFcG.hidden = !isBike;
    elLift.hidden = !isBike;
    elThrMark.hidden = !isBike;
    elMsg.className = 'fly-msg';
    bk.msg = '';
    bk.crashes = 0;
    bk.camRoll = 0;
    bk.pose = null;
    bk.lever = 0;
    bk.grip = false;
    clearTouch();
    coneEl.classList.remove('go');
    headingShown = -1;
    if (!isBal) elThr.style.background = '';
    setSound(Engine.isOn());

    document.body.classList.add('flying');
    document.body.classList.toggle('fly-balloon', isBal);
    document.body.classList.toggle('fly-bike', isBike);

    // Full screen is asked for, never depended on: iOS refuses it outright and
    // the mode is perfectly good without it.
    if (document.documentElement.requestFullscreen && !document.fullscreenElement) {
      document.documentElement.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
    }

    map.dragPan.disable();
    map.scrollZoom.disable();
    map.doubleClickZoom.disable();
    map.keyboard.disable();
    map.touchZoomRotate.disable();
    map.dragRotate.disable();

    // The engine starts here, inside the click that opened the mode, because
    // that is the gesture the browser wants before it will let a page make a
    // sound. Started later, from a timer, it would stay silent.
    Engine.setMode(craft);
    Engine.start();

    const start = () => {
      fitFrame();
      disarm();
      restore.hidden = hideOverlays();

      // Terrain off, and this was measured rather than assumed. At a pitch in
      // the seventies the DEM mesh is seen almost edge-on, and since the tiles
      // stop at zoom 14 while the flight sits around 16 to 18, every triangle
      // is stretched over five zoom levels: the near half of the picture
      // becomes vertical smears of colour. Turning it off gave a clean
      // photograph of the moshava all the way to the horizon.
      //
      // Nothing is lost. Real relief here is 46 m across the whole moshava,
      // which at flying height is invisible; the sense of three dimensions
      // comes from the perspective, the motion, and the photos standing up out
      // of the ground - none of which need a height field.
      map.setTerrain(null);

      addLayers();
      buildWorld();
      // A trails view chosen on a flight that had trails, on one that has
      // none, becomes the clean view: the nearest thing to what was asked.
      if (view === 'trails' && !hasTrails) setView('clean'); else applyView();

      let target;
      if (isBike) {
        // The bike starts on the ground at the middle of the map, facing the
        // way the map faced, and the camera comes down to it. The map's
        // pitch ceiling goes up to the chase camera's, and its zoom ceiling
        // to a parked bike's; both go back on the way out.
        if (!bk.rig) bk.rig = Bike.makeRig(bk.canvas);
        bk.canvas.hidden = false;
        restore.maxPitch = map.getMaxPitch();
        restore.maxZoom = map.getMaxZoom();
        map.setMaxPitch(85);
        map.setMaxZoom(BIKE.ZOOM_MAX);
        fitFrame();   // again, now that there is a rig to size
        bk.rig.setCamMode(bk.cam);
        bk.rig.place(0, 0, Bike.yawFor(bearing));
        bk.rig.wantCamera(1 / 60, bk.want);
        bk.pose = bikePose(bk.want);
        target = bk.pose;
        map.on('render', onMapRender);
      } else {
        const pitch = basePitch();
        const ahead = destination(pos.lat, pos.lng, bearing, alt * Math.tan(pitch * RAD));
        target = { center: [ahead.lng, ahead.lat], zoom: zoomFor(alt, pitch, ahead.lat), bearing, pitch };
      }

      // A take-off rather than a cut. The mode is a change of place as much as
      // a change of controls, and arriving at altitude in one frame reads as a
      // glitch where a rise reads as leaving the ground. The balloon's burner
      // lights as the rise begins.
      if (isBal) Engine.ignite();
      map.easeTo({ ...target, duration: 1700, essential: true });
      setTimeout(() => {
        if (!on) return;
        flying = true;
        last = performance.now();
        tick = 0;
      }, 1750);

      intro.hidden = false;
      introTimer = setTimeout(dismissIntro, 9000);
      last = performance.now();
      raf = requestAnimationFrame(step);
    };

    // Satellite is not a preference here, it is the material: the mode is a
    // flight over a photograph of the place. The bike also waits for its
    // engine, Three.js, fetched on the first ride and kept; a fetch that
    // fails leaves the mode the way it came.
    const ready = isBike ? Bike.load() : Promise.resolve();
    ready.then(() => {
      if (!on) return;
      if (baseIndex !== 1) {
        map.once('style.load', () => setTimeout(start, 60));
        setBasemap(1);
      } else {
        setTimeout(start, 30);
      }
    }).catch((err) => {
      console.error('bike: three.js did not load', err);
      exit();
      alert('האופנוע צריך ספרייה תלת־ממדית שלא נטענה. בדוק את החיבור ונסה שוב.');
    });

    addEventListener('keydown', onKeyDown);
    addEventListener('keyup', onKeyUp);
    addEventListener('blur', onBlur);
    // Bound to the window and not to our own overlay: the map canvas sits on
    // top of everything we draw except the cards and the HUD, so a pointer
    // over open ground never reaches an element of ours. The handlers filter
    // by target instead, which is also what keeps a tap on a photo from
    // steering.
    addEventListener('pointermove', onPointerMove);
    addEventListener('pointerdown', onPointerDown);
    addEventListener('pointerup', onPointerUp);
    addEventListener('pointercancel', onPointerUp);
    addEventListener('click', onClick);
    addEventListener('contextmenu', onContextMenu);
    document.documentElement.addEventListener('mouseleave', onPointerLeave);
    addEventListener('touchstart', onTouchStart, { passive: true });
    addEventListener('touchmove', onTouchMove, { passive: true });
    addEventListener('touchend', onTouchEnd);
    addEventListener('resize', onResize);
    document.addEventListener('fullscreenchange', onFullscreen);
    document.addEventListener('visibilitychange', onVisibility);
  }

  function onResize() {
    if (!on) return;
    fitFrame();
  }

  function onFullscreen() {
    // Leaving full screen with F11 or the browser's own gesture means leaving
    // the mode; staying in a hidden-chrome flight the user just dismissed
    // would be the app arguing with them.
    if (on && !document.fullscreenElement) exit();
  }

  function exit(opts = {}) {
    if (!on) return;
    on = false;
    flying = false;
    cancelAnimationFrame(raf);
    raf = null;
    clearTimeout(introTimer);
    Engine.stop();

    removeEventListener('keydown', onKeyDown);
    removeEventListener('keyup', onKeyUp);
    removeEventListener('blur', onBlur);
    removeEventListener('pointermove', onPointerMove);
    removeEventListener('pointerdown', onPointerDown);
    removeEventListener('pointerup', onPointerUp);
    removeEventListener('pointercancel', onPointerUp);
    removeEventListener('click', onClick);
    removeEventListener('contextmenu', onContextMenu);
    document.documentElement.removeEventListener('mouseleave', onPointerLeave);
    removeEventListener('touchstart', onTouchStart);
    removeEventListener('touchmove', onTouchMove);
    removeEventListener('touchend', onTouchEnd);
    removeEventListener('resize', onResize);
    document.removeEventListener('fullscreenchange', onFullscreen);
    document.removeEventListener('visibilitychange', onVisibility);

    for (const [, node] of cards) node.remove();
    cards.clear();
    for (const [, node] of chips) node.remove();
    chips.clear();
    if (viewer) closeShot();

    // The bike's scene is kept for the next ride; its canvas goes, and the
    // map gets its ceilings back.
    map.off('render', onMapRender);
    if (bk.canvas) bk.canvas.hidden = true;
    clearTouch();
    if (restore && restore.maxPitch !== undefined) {
      map.setMaxPitch(restore.maxPitch);
      map.setMaxZoom(restore.maxZoom);
    }

    map.getContainer().style.transform = '';
    worldEl.style.transform = '';
    sky.style.transform = '';
    elThr.style.background = '';
    document.body.classList.remove('flying', 'fly-paused', 'fly-burn', 'fly-unarmed', 'fly-balloon', 'fly-bike', 'fly-super');
    delete document.body.dataset.flyView;
    coneEl.classList.remove('go');
    document.documentElement.style.removeProperty('--fly-ox');
    document.documentElement.style.removeProperty('--fly-oy');

    removeLayers();
    if (restore && restore.hidden) showOverlays(restore.hidden);

    map.dragPan.enable();
    map.scrollZoom.enable();
    map.doubleClickZoom.enable();
    map.keyboard.enable();
    map.touchZoomRotate.enable();
    map.dragRotate.enable();

    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});

    const back = restore;
    restore = null;
    setFov(back.fov);
    if (map.setPixelRatio) map.setPixelRatio(undefined);   // back to the device's own
    map.resize();

    // What you flew to is the find, so the position stays; everything else -
    // the base map, the tilt, the zoom - goes back to how it was, because
    // those were choices made before the flight and not by it.
    const land = () => map.easeTo({
      center: [pos.lng, pos.lat],
      zoom: Math.max(back.zoom, 15.5),
      bearing: back.bearing,
      pitch: back.pitch,
      duration: 900
    });

    if (back.base !== 1) {
      // The style swap rebuilds everything through `applyOverlays`, terrain
      // included. Restoring it here as well would start a round of DEM tile
      // requests that setStyle then aborts a moment later, which MapLibre
      // reports to the console as a bare AbortError with no explanation.
      map.once('style.load', () => setTimeout(land, 60));
      setBasemap(back.base);
    } else {
      if (map.getSource('dem')) {
        map.setTerrain({ source: 'dem', exaggeration: TERRAIN_X });
      }
      land();
    }

    if (craftNext) { craft = craftNext; craftNext = null; }

    if (opts.keepPlace) return;
  }

  const isOn = () => on;
  const toggle = () => (on ? exit() : enter());

  /** Which aircraft the next take-off uses. Kept across visits; a flight
   *  already in the air keeps the one it took off in. */
  let craftNext = null;
  function setCraft(c) {
    if (!CRAFTS.includes(c)) c = 'jet';
    try { localStorage.setItem(KEY_CRAFT, c); } catch (_) { /* fine */ }
    if (on) craftNext = c; else craft = c;
    return c;
  }
  const getCraft = () => craftNext || craft;
  /** The aircraft after this one on the small button: the order is the
   *  balloon, the jet, the bike, and round again. */
  const nextCraft = (c) => CRAFTS[(CRAFTS.indexOf(c || getCraft()) + 1) % CRAFTS.length];

  /** The flight model's state, for the tests. */
  const debug = () => {
    const p = bk.rig && bk.rig.phys;
    return {
      on, flying, craft, alt, speed, bearing, bank, pitch: map ? map.getPitch() : 0,
      burn, throttle, mach, gee, buffet, sonic, booms, yaw, armed, stickGain, intro: !!(intro && !intro.hidden),
      view, views: viewsNow(), hasTrails,
      cards: [...cards.values()].filter((n) => !n.hidden).length,
      chips: [...chips.values()].filter((n) => !n.hidden).length,
      lit: near.filter((e) => e.line).map((e) => e.g),
      envelope: envelope(alt),
      balloon: { dT: bal.dT, plume: bal.plume, vz: bal.vz, drift: bal.drift, ride: bal.ride,
                 roll: bal.roll, pit: bal.pit, launch: bal.launch },
      bike: p ? {
        x: p.pos.x, y: p.pos.y, z: p.pos.z, vx: p.vel.x, vy: p.vel.y, vz: p.vel.z,
        throttle: p.throttle, spoolRear: p.spoolRear, spoolLift: p.spoolLift, collective: p.collective,
        ab: p.ab, vert: p.vert, steer: p.steer, pitchCmd: p.pitch, grounded: p.grounded, crashed: p.crashed,
        assist: p.assist, heading: bk.rig.heading(), bank: bk.rig.bank(), nose: bk.rig.nose(),
        w: [p.angVel.x, p.angVel.y, p.angVel.z].map((v) => +v.toFixed(2)),
        cam: bk.cam, camRoll: bk.camRoll, want: { ...bk.want }, pose: bk.pose, crashes: bk.crashes,
        visible: bk.rig.bike.group.visible, canvas: !!(bk.canvas && !bk.canvas.hidden),
        mapCam: on && craft === 'bike' ? mapCamera() : null, msg: bk.msg,
        wind: { x: p.wind.W.x, z: p.wind.W.z, speed: p.wind.speed }
      } : null
    };
  };

  return { enter, exit, toggle, isOn, setCraft, getCraft, nextCraft, setView, cycleView, debug };
})();
