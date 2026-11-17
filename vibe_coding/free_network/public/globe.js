// The friends globe: every friend who shares a place stands on the earth, and the rest sit on
// a cloud above it. The coastlines and the projection code are served from here, so looking
// at the globe tells no other site anything.
import { feature, geoDistance, geoGraticule10, geoOrthographic, geoPath } from '/vendor/geo.js';

const HOME = { lng: 34.97, lat: 32.47 };
const MAX_ZOOM = 800;
const DETAIL_ZOOM = 4;
const MAX_LAT = 85;
const PIN_GAP = 30;
const CLOUD_SEATS = 7;
const RADIANS = Math.PI / 180;
const COLORS = { oceanLight: '#dcebe8', ocean: '#a9cacd', land: '#f3e2bd', coast: '#9b7344', grid: 'rgba(30, 74, 99, 0.13)', rim: '#1e4a63' };

const land = {};
let viewMemory = null;

async function loadLand(size) {
  land[size] ??= fetch(`/geo/land-${size}.json`)
    .then((response) => response.json())
    .then((world) => feature(world, world.objects.land));
  return land[size];
}

// Close up, almost all of the world is out of view. Each piece of land gets a centre and a
// reach, so a frame draws only the pieces that can be seen.
async function loadDetail() {
  land.pieces ??= loadLand('50m').then((shape) => shape.geometry.coordinates.map((polygon) => {
    const ring = polygon[0];
    const centre = ring[Math.floor(ring.length / 2)];
    let reach = 0;
    for (let i = 0; i < ring.length; i += 3) reach = Math.max(reach, geoDistance(centre, ring[i]));
    return { polygon, centre, reach };
  }));
  return land.pieces;
}

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const wrap = (lng) => ((lng + 540) % 360) - 180;

// Rounded in the browser already, so nothing more exact than a neighbourhood ever leaves it.
const roundPlace = ({ lat, lng }) => ({ lat: Number(lat.toFixed(2)), lng: Number(lng.toFixed(2)) });

function lastPlace(userId, value) {
  try {
    const key = `place:${userId}`;
    if (value) localStorage.setItem(key, JSON.stringify(value));
    return JSON.parse(localStorage.getItem(key));
  } catch {
    return null;
  }
}

// Starts where most of the people are, so the first thing in view is the crowd.
function startingPoint(people, me) {
  if (me.place) return me.place;
  let best = null;
  let most = 0;
  for (const person of people) {
    const near = people.filter((other) => geoDistance([person.place.lng, person.place.lat], [other.place.lng, other.place.lat]) < 0.05).length;
    if (near > most) [best, most] = [person.place, near];
  }
  return best ?? HOME;
}

export function friendsGlobe({ me, friends, ui }) {
  const { el, icon, avatar, go, api, tell, demo } = ui;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const canvas = el('canvas', { className: 'globe-canvas', ariaLabel: 'כדור הארץ עם החברים שלך. גוררים כדי לסובב, גלגלת או צביטה כדי להתקרב.', role: 'img' });
  const pins = el('div', { className: 'globe-pins' });
  const stage = el('div', { className: 'globe-stage' }, canvas, pins);
  const cloud = el('div', { className: 'globe-cloud' });
  const list = el('div', { className: 'globe-list', hidden: true });
  const controls = el('div', { className: 'globe-controls' });
  const zoomBar = el('div', { className: 'globe-zoom' },
    el('button', { type: 'button', ariaLabel: 'התקרבות', title: 'התקרבות', onclick: () => flyTo({ zoom: view.zoom * 2.5 }) }, icon('plus')),
    el('button', { type: 'button', ariaLabel: 'התרחקות', title: 'התרחקות', onclick: () => flyTo({ zoom: view.zoom / 2.5 }) }, icon('minus')),
    el('button', { type: 'button', ariaLabel: 'חזרה לתצוגת כל העולם', title: 'כל העולם', onclick: () => flyTo({ ...startingPoint(located(), me), zoom: 1 }) }, icon('globe')));
  const root = el('section', { className: 'globe' }, cloud, el('div', { className: 'globe-frame' }, stage, zoomBar), list, controls);

  const projection = geoOrthographic().clipAngle(90).precision(0.4);
  const context = canvas.getContext('2d');
  const path = geoPath(projection, context);
  const graticule = geoGraticule10();
  const start = startingPoint(friends.filter((friend) => friend.place), me);
  const view = viewMemory ?? { lng: wrap(start.lng - (reduceMotion ? 0 : 70)), lat: 12, zoom: 1 };
  let size = 0;
  let coarse = null;
  let fine = null;
  let queued = false;
  let flight = null;
  let spin = null;
  let picking = false;
  let pinKey = '';
  let pinNodes = [];
  let arrived = !!viewMemory;
  // While the reader's own picture travels between the cloud and the earth, its pin or its
  // seat on the cloud waits empty, and the traveler stands in for it.
  let travelling = false;

  function located() {
    const people = friends.filter((friend) => friend.place);
    return me.place ? [{ ...me, mine: true }, ...people] : people;
  }

  function redraw() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      if (root.isConnected) draw();
    });
  }

  function aim() {
    const radius = (size / 2 - 5) * view.zoom;
    projection.translate([size / 2, size / 2]).scale(radius).rotate([-view.lng, -view.lat]).clipExtent([[-2, -2], [size + 2, size + 2]]);
    return radius;
  }

  function draw() {
    if (!size) return;
    viewMemory = view;
    const radius = aim();
    const half = size / 2;
    const ratio = canvas.width / size;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, size, size);

    const ocean = context.createRadialGradient(half - radius * 0.35, half - radius * 0.4, radius * 0.05, half, half, radius);
    ocean.addColorStop(0, COLORS.oceanLight);
    ocean.addColorStop(1, COLORS.ocean);
    context.beginPath();
    path({ type: 'Sphere' });
    context.fillStyle = ocean;
    context.fill();

    if (view.zoom < 40) {
      context.beginPath();
      path(graticule);
      context.strokeStyle = COLORS.grid;
      context.lineWidth = 1;
      context.stroke();
    }

    const sight = Math.asin(Math.min(1, (size * 0.75) / radius));
    const centre = [view.lng, view.lat];
    const shape = view.zoom >= DETAIL_ZOOM && fine
      ? { type: 'MultiPolygon', coordinates: fine.filter((piece) => geoDistance(centre, piece.centre) < sight + piece.reach).map((piece) => piece.polygon) }
      : coarse;
    if (shape) {
      context.beginPath();
      path(shape);
      context.fillStyle = COLORS.land;
      context.fill();
      context.strokeStyle = COLORS.coast;
      context.lineWidth = 1.1;
      context.lineJoin = 'round';
      context.stroke();
    }

    // A soft shadow on the far side gives the disc its roundness.
    const shade = context.createRadialGradient(half - radius * 0.3, half - radius * 0.35, radius * 0.4, half, half, radius);
    shade.addColorStop(0, 'rgba(15, 42, 56, 0)');
    shade.addColorStop(1, 'rgba(15, 42, 56, 0.2)');
    context.beginPath();
    path({ type: 'Sphere' });
    context.fillStyle = shade;
    context.fill();
    context.strokeStyle = COLORS.rim;
    context.lineWidth = 2;
    context.stroke();

    placePins();
    if (view.zoom >= DETAIL_ZOOM && !fine) loadDetail().then((pieces) => { fine = pieces; redraw(); }).catch(() => {});
  }

  // People who would overlap on screen share one pin with a count. Turning or zooming
  // regroups them, so a crowd opens up as the view gets closer.
  function groups() {
    const centre = [view.lng, view.lat];
    const found = [];
    for (const person of located()) {
      const point = [person.place.lng, person.place.lat];
      if (geoDistance(centre, point) > 1.5) continue;
      const [x, y] = projection(point);
      if (Math.hypot(x - size / 2, y - size / 2) > size / 2 + 4) continue;
      const near = found.find((group) => Math.hypot(group.x - x, group.y - y) < PIN_GAP);
      if (near) near.people.push(person);
      else found.push({ x, y, people: [person] });
    }
    return found;
  }

  function placePins() {
    const found = picking ? [] : groups();
    const key = found.map((group) => group.people.map((person) => person.id).join(',')).join('|');
    if (key !== pinKey) {
      pinKey = key;
      pinNodes = found.map(pin);
      pins.replaceChildren(...pinNodes);
    }
    found.forEach((group, index) => {
      pinNodes[index].style.transform = `translate(${group.x.toFixed(1)}px, ${group.y.toFixed(1)}px)`;
    });
  }

  function pin(group) {
    const [first] = group.people;
    const many = group.people.length > 1;
    const names = group.people.map((person) => (person.mine ? 'אני' : person.name));
    const node = el('a', {
      className: `globe-pin${group.people.some((person) => person.mine) ? ` mine${travelling ? ' waiting' : ''}` : ''}`,
      href: `/u/${first.id}`,
      ariaLabel: names.join(', '),
      onclick: (e) => {
        e.preventDefault();
        if (!many) go(`/u/${first.id}`);
        else openGroup(group.people);
      },
    }, avatar(first, { small: true }),
    many ? el('span', { className: 'pin-count' }, String(group.people.length)) : null,
    el('span', { className: `pin-name${!many && first.status ? ' with-status' : ''}`, dir: 'auto' },
      many ? `${names.slice(0, 3).join(', ')}${names.length > 3 ? '…' : ''}` : names[0],
      !many && first.status ? el('span', { className: 'pin-status' }, first.status) : null));
    return node;
  }

  // A crowd opens by flying closer. When its people stand on the same spot, or the view is
  // as close as it gets, their names are listed instead.
  function openGroup(people) {
    const lats = people.map((person) => person.place.lat);
    const lngs = people.map((person) => person.place.lng);
    const spread = Math.max(Math.max(...lats) - Math.min(...lats), Math.max(...lngs) - Math.min(...lngs));
    showList(people, 'כאן נמצאים');
    if (spread === 0 || view.zoom >= MAX_ZOOM) return;
    const centre = { lat: (Math.max(...lats) + Math.min(...lats)) / 2, lng: (Math.max(...lngs) + Math.min(...lngs)) / 2 };
    const wanted = (size * 0.45) / (spread * RADIANS * (size / 2));
    flyTo({ ...centre, zoom: clamp(wanted, view.zoom * 2, MAX_ZOOM) });
  }

  function showList(people, title) {
    list.hidden = false;
    list.replaceChildren(
      el('div', { className: 'globe-list-head' }, el('strong', {}, title),
        el('button', { className: 'plain', type: 'button', ariaLabel: 'סגירת הרשימה', onclick: () => { list.hidden = true; } }, icon('close'))),
      ...people.map((person) => el('a', {
        className: 'globe-person',
        href: `/u/${person.id}`,
        onclick: (e) => { e.preventDefault(); go(`/u/${person.id}`); },
      }, avatar(person, { small: true }), el('span', { dir: 'auto' }, person.mine || person.id === me.id ? `${person.name} (אני)` : person.name))));
  }

  function renderCloud() {
    const riders = friends.filter((friend) => !friend.place);
    if (!me.place) riders.unshift({ ...me, mine: true });
    cloud.hidden = !riders.length;
    if (!riders.length) return;
    const seated = riders.length > CLOUD_SEATS ? riders.slice(0, CLOUD_SEATS - 1) : riders;
    const rest = riders.length - seated.length;
    cloud.replaceChildren(
      el('div', { className: 'cloud-riders' },
        ...seated.map((person) => el('a', {
          className: `cloud-rider${person.mine ? ` mine${travelling ? ' waiting' : ''}` : ''}`,
          href: `/u/${person.id}`,
          title: person.mine ? 'אני' : person.name,
          ariaLabel: person.mine ? 'אני' : person.name,
          onclick: (e) => { e.preventDefault(); go(`/u/${person.id}`); },
        }, avatar(person, { small: true }))),
        rest ? el('button', { className: 'cloud-more', type: 'button', title: 'כל מי שעל הענן', onclick: () => showList(riders, 'על הענן') }, `+${rest}`) : null),
      el('img', { className: 'cloud-art', src: '/art/cloud.svg', alt: '' }),
      el('p', { className: 'cloud-caption' }, riders.length === 1 && !me.place ? 'המקום שלך לא משותף, אז הענן שלך' : 'על הענן: מי שלא משתף את המקום שלו'));
  }

  function animate(step) {
    const token = {};
    flight = token;
    spin = null;
    const tick = (now) => {
      if (flight !== token || !root.isConnected) return;
      if (step(now)) requestAnimationFrame(tick);
      else flight = null;
      draw();
    };
    requestAnimationFrame(tick);
  }

  function flyTo({ lng = view.lng, lat = view.lat, zoom = view.zoom }, ms = 700) {
    const from = { ...view };
    const to = { lng: from.lng + wrap(lng - from.lng), lat: clamp(lat, -MAX_LAT, MAX_LAT), zoom: clamp(zoom, 1, MAX_ZOOM) };
    if (reduceMotion) {
      Object.assign(view, to, { lng: wrap(to.lng) });
      redraw();
      return;
    }
    let began = null;
    animate((now) => {
      began ??= now;
      const t = Math.min(1, (now - began) / ms);
      const eased = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
      view.lng = wrap(from.lng + (to.lng - from.lng) * eased);
      view.lat = from.lat + (to.lat - from.lat) * eased;
      view.zoom = from.zoom * (to.zoom / from.zoom) ** eased;
      return t < 1;
    });
  }

  // Keeps the globe turning for a moment after the hand lets go, like a real one.
  function coast(speedLng, speedLat) {
    if (reduceMotion || Math.hypot(speedLng, speedLat) < 0.01) return;
    let last = null;
    animate((now) => {
      const dt = Math.min(40, now - (last ?? now));
      last = now;
      view.lng = wrap(view.lng + speedLng * dt);
      view.lat = clamp(view.lat + speedLat * dt, -MAX_LAT, MAX_LAT);
      const fade = 0.995 ** dt;
      speedLng *= fade;
      speedLat *= fade;
      return Math.hypot(speedLng, speedLat) > 0.004;
    });
  }

  function local(event) {
    const box = canvas.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top];
  }

  function pointAt(xy) {
    aim();
    if (Math.hypot(xy[0] - size / 2, xy[1] - size / 2) >= (size / 2 - 5) * view.zoom) return null;
    const spot = projection.invert(xy);
    return spot && Number.isFinite(spot[0]) && Number.isFinite(spot[1]) ? { lng: spot[0], lat: spot[1] } : null;
  }

  // Zooms around the spot under the pointer: whatever was under it stays under it.
  function zoomAt(xy, factor) {
    const before = pointAt(xy);
    view.zoom = clamp(view.zoom * factor, 1, MAX_ZOOM);
    const after = before && pointAt(xy);
    if (after) {
      view.lng = wrap(view.lng + before.lng - after.lng);
      view.lat = clamp(view.lat + before.lat - after.lat, -MAX_LAT, MAX_LAT);
    }
    flight = null;
    redraw();
  }

  const touches = new Map();
  let drag = null;
  let dragged = false;

  stage.addEventListener('pointerdown', (e) => {
    if (e.button) return;
    touches.set(e.pointerId, local(e));
    flight = null;
    dragged = false;
    drag = { x: e.clientX, y: e.clientY, at: e.timeStamp, speedLng: 0, speedLat: 0, moved: 0 };
  });

  stage.addEventListener('pointermove', (e) => {
    if (!touches.has(e.pointerId) || !drag) return;
    const before = [...touches.values()];
    touches.set(e.pointerId, local(e));
    const after = [...touches.values()];
    if (after.length === 2) {
      const gap = (points) => Math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]);
      const middle = [(after[0][0] + after[1][0]) / 2, (after[0][1] + after[1][1]) / 2];
      dragged = true;
      if (gap(before) > 0) zoomAt(middle, gap(after) / gap(before));
      drag.x = null;
      return;
    }
    if (drag.x === null) {
      Object.assign(drag, { x: e.clientX, y: e.clientY, at: e.timeStamp });
      return;
    }
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.moved += Math.hypot(dx, dy);
    if (drag.moved < 5 && !dragged) return;
    if (!dragged) {
      dragged = true;
      stage.setPointerCapture(e.pointerId);
      stage.classList.add('dragging');
    }
    const degrees = 1 / ((size / 2 - 5) * view.zoom * RADIANS);
    const dt = Math.max(1, e.timeStamp - drag.at);
    view.lng = wrap(view.lng - dx * degrees);
    view.lat = clamp(view.lat + dy * degrees, -MAX_LAT, MAX_LAT);
    drag.speedLng = 0.6 * drag.speedLng + 0.4 * (-dx * degrees / dt);
    drag.speedLat = 0.6 * drag.speedLat + 0.4 * (dy * degrees / dt);
    Object.assign(drag, { x: e.clientX, y: e.clientY, at: e.timeStamp });
    redraw();
  });

  function release(e) {
    if (!touches.delete(e.pointerId)) return;
    stage.classList.remove('dragging');
    if (touches.size || !drag) return;
    if (dragged && drag.x !== null && e.timeStamp - drag.at < 80) coast(drag.speedLng, drag.speedLat);
    drag = null;
  }
  stage.addEventListener('pointerup', release);
  stage.addEventListener('pointercancel', release);

  // A drag that ends over a face must not open that person's page.
  stage.addEventListener('click', (e) => {
    if (dragged) {
      e.preventDefault();
      e.stopPropagation();
      dragged = false;
      return;
    }
    if (picking && e.target === canvas) {
      const spot = pointAt(local(e));
      if (spot) save(spot);
    }
  }, true);

  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    zoomAt(local(e), Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0018)));
  }, { passive: false });

  stage.addEventListener('dblclick', (e) => {
    if (picking || e.target !== canvas) return;
    const spot = pointAt(local(e));
    if (spot) flyTo({ ...spot, zoom: view.zoom * 3 }, 450);
  });

  canvas.tabIndex = 0;
  canvas.addEventListener('keydown', (e) => {
    const turn = 25 / view.zoom;
    const moves = { ArrowLeft: [-turn, 0], ArrowRight: [turn, 0], ArrowUp: [0, turn], ArrowDown: [0, -turn] };
    if (moves[e.key]) flyTo({ lng: view.lng + moves[e.key][0], lat: view.lat + moves[e.key][1] }, 200);
    else if (e.key === '+' || e.key === '=') flyTo({ zoom: view.zoom * 2 }, 200);
    else if (e.key === '-') flyTo({ zoom: view.zoom / 2 }, 200);
    else return;
    e.preventDefault();
  });

  function setPicking(on) {
    picking = on;
    stage.classList.toggle('picking', on);
    pinKey = '-';
    renderControls();
    redraw();
  }

  const centre = (rect) => rect && { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  const settle = (node) => node?.animate([{ transform: 'scale(0.4)' }, { transform: 'scale(1.25)' }, { transform: 'scale(1)' }], { duration: 380, easing: 'ease-out' });

  // The reader's picture floats between two points on the screen: down from the cloud under a
  // parachute, or up to it under a balloon. It sways a little and lands with a bounce.
  function travel(from, to, down) {
    if (reduceMotion || !from || !to) return Promise.resolve();
    const base = root.getBoundingClientRect();
    const gear = el('span', { className: 'traveler-gear' }, down ? '🪂' : '🎈');
    const traveler = el('div', { className: 'globe-traveler', ariaHidden: 'true' }, gear, avatar(me, { small: true }));
    root.append(traveler);
    const [x0, y0, x1, y1] = [from.x - base.left, from.y - base.top, to.x - base.left, to.y - base.top];
    const steps = 40;
    const frames = Array.from({ length: steps + 1 }, (_, i) => {
      const t = i / steps;
      // Down: a small hop off the cloud, then a slow fall. Up: a slow start, then the balloon takes it.
      const eased = down ? t * t * (3 - 2 * t) : t ** 1.6;
      const hop = down ? -26 * Math.sin(Math.min(1, t * 3) * Math.PI) * (1 - t) : 0;
      const sway = Math.sin(t * Math.PI * 4) * 16 * Math.sin(t * Math.PI);
      const x = x0 + (x1 - x0) * eased + sway;
      const y = y0 + (y1 - y0) * eased + hop;
      return { transform: `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) rotate(${(sway * 0.9).toFixed(1)}deg)` };
    });
    const flight = traveler.animate(frames, { duration: down ? 1700 : 1500, easing: 'linear', fill: 'forwards' });
    gear.animate(down
      ? [{ opacity: 0, transform: 'scale(0.3)' }, { opacity: 1, transform: 'scale(1)', offset: 0.18 }, { opacity: 1, transform: 'scale(1)', offset: 0.85 }, { opacity: 0, transform: 'translateY(-14px) scale(0.6)' }]
      : [{ opacity: 0, transform: 'scale(0.3)' }, { opacity: 1, transform: 'scale(1.1)', offset: 0.25 }, { opacity: 1, transform: 'scale(1)', offset: 0.9 }, { opacity: 0 }],
    { duration: down ? 1700 : 1500, fill: 'forwards' });
    return flight.finished.then(() => {
      if (down) {
        // Landing: a ripple spreads on the ground where the picture touched it.
        const ripple = el('span', { className: 'globe-ripple' });
        ripple.style.transform = `translate(${x1}px, ${y1}px)`;
        root.append(ripple);
        ripple.animate([{ opacity: 0.9, scale: 0.2 }, { opacity: 0, scale: 2.4 }], { duration: 700, easing: 'ease-out' }).finished.then(() => ripple.remove());
      }
      traveler.remove();
    }).catch(() => traveler.remove());
  }

  // The cloud opens before someone arrives on it and folds away once nobody is left.
  function resizeCloud(open) {
    if (reduceMotion || cloud.hidden) return Promise.resolve();
    const height = cloud.offsetHeight;
    const frames = [{ height: '0px', opacity: 0, marginBottom: '0px' }, { height: `${height}px`, opacity: 1, marginBottom: '4px' }];
    return cloud.animate(open ? frames : frames.reverse(), { duration: 450, easing: 'ease-in-out', fill: open ? 'none' : 'forwards' }).finished.catch(() => {});
  }

  async function save(spot) {
    const wasOnCloud = !me.place;
    const from = wasOnCloud ? centre(cloud.querySelector('.cloud-rider.mine')?.getBoundingClientRect()) : null;
    try {
      const { place } = await api('/me/place', { method: 'PUT', body: roundPlace(spot) });
      me.place = place;
      lastPlace(me.id, place);
      choosing = false;
      travelling = !!from && !reduceMotion;
      setPicking(false);
      flyTo({ ...place, zoom: Math.max(view.zoom, 3) }, travelling ? 1100 : 700);
      if (!travelling) {
        renderCloud();
        return;
      }
      // Until the picture lands, its seat on the cloud stays where it was, empty.
      cloud.querySelector('.cloud-rider.mine')?.classList.add('waiting');
      await travel(from, centre(stage.getBoundingClientRect()), true);
      travelling = false;
      pinKey = '-';
      draw();
      settle(pins.querySelector('.globe-pin.mine'));
      const empty = !friends.some((friend) => !friend.place);
      if (empty) await resizeCloud(false);
      renderCloud();
      // The folded height was held until the cloud was hidden; let go of it for the next time.
      for (const held of cloud.getAnimations()) if (!(held instanceof CSSAnimation)) held.cancel();
    } catch (err) {
      travelling = false;
      tell(err.message);
    }
  }

  async function hide() {
    const pinned = pins.querySelector('.globe-pin.mine');
    const stageBefore = stage.getBoundingClientRect();
    const fromPin = centre(pinned?.getBoundingClientRect());
    try {
      await api('/me/place', { method: 'DELETE' });
      me.place = null;
      choosing = false;
      travelling = !!fromPin && !reduceMotion;
      const wasHidden = cloud.hidden;
      pinKey = '-';
      renderControls();
      renderCloud();
      redraw();
      if (!travelling) return;
      if (wasHidden) await resizeCloud(true);
      // The cloud may have pushed the earth down; the picture starts where its pin stood.
      const shift = stage.getBoundingClientRect();
      const from = { x: fromPin.x + shift.left - stageBefore.left, y: fromPin.y + shift.top - stageBefore.top };
      const seat = cloud.querySelector('.cloud-rider.mine');
      await travel(from, centre(seat?.getBoundingClientRect()), false);
      travelling = false;
      seat?.classList.remove('waiting');
      settle(seat);
    } catch (err) {
      travelling = false;
      tell(err.message);
      renderControls();
    }
  }

  function fromDevice(button) {
    if (!navigator.geolocation) {
      tell('הדפדפן הזה לא יודע למסור מיקום. אפשר לבחור מקום על הכדור.');
      return;
    }
    button.disabled = true;
    navigator.geolocation.getCurrentPosition(
      (position) => save({ lat: position.coords.latitude, lng: position.coords.longitude }),
      () => {
        button.disabled = false;
        tell('לא קיבלנו מיקום מהמכשיר. אפשר לבחור מקום על הכדור במקום זה.');
      },
      { maximumAge: 600000, timeout: 15000 });
  }

  let choosing = false;

  function renderControls() {
    const shared = !!me.place;
    const toggle = el('input', {
      type: 'checkbox',
      role: 'switch',
      checked: shared || choosing,
      onchange: () => {
        if (toggle.checked) {
          choosing = true;
          renderControls();
        } else if (shared) {
          hide();
        } else {
          choosing = false;
          setPicking(false);
        }
      },
    });
    const parts = [el('label', { className: 'switch' }, toggle, el('span', { className: 'switch-track' }), el('span', {}, 'המקום שלי גלוי לחברים'))];

    if (picking) {
      parts.push(el('p', { className: 'globe-hint' }, 'סובבו וקרבו את הכדור, ואז לחצו על המקום שבו תרצו להופיע.'),
        el('div', { className: 'row' }, el('button', { className: 'plain', type: 'button', onclick: () => { choosing = shared ? false : choosing; setPicking(false); } }, 'ביטול')));
    } else if (choosing) {
      const remembered = lastPlace(me.id);
      const device = el('button', { className: 'small', type: 'button', onclick: () => fromDevice(device) }, 'המקום של המכשיר');
      parts.push(el('p', { className: 'globe-hint' }, 'איפה להציב אותך?'),
        el('div', { className: 'row' },
          remembered && !shared ? el('button', { className: 'small', type: 'button', onclick: () => save(remembered) }, 'איפה שהייתי קודם') : null,
          device,
          el('button', { className: 'small ghost', type: 'button', onclick: () => setPicking(true) }, 'לבחור על הכדור'),
          shared ? el('button', { className: 'plain', type: 'button', onclick: () => { choosing = false; renderControls(); } }, 'ביטול') : null),
        el('p', { className: 'note' }, 'נשמר רק בדיוק של קילומטר בערך, ורק החברים שלך רואים אותו. כיבוי מוחק את המקום מהשרת.'));
    } else if (shared) {
      parts.push(el('div', { className: 'row' },
        el('button', { className: 'plain', type: 'button', onclick: () => flyTo({ ...me.place, zoom: Math.max(view.zoom, 3) }) }, 'איפה אני?'),
        el('button', { className: 'plain', type: 'button', onclick: () => { choosing = true; renderControls(); } }, 'שינוי המקום')));
    } else {
      parts.push(el('p', { className: 'note' }, friends.length
        ? 'כרגע המקום שלך לא נשמר אצלנו, והחברים רואים אותך על הענן.'
        : 'כשיהיו לך חברים ברשת, הם יופיעו כאן.'));
    }
    if (demo && (choosing || picking)) parts.push(el('p', { className: 'note' }, 'בגרסת ההדגמה שום דבר לא נשמר, אז גם המקום לא.'));
    controls.replaceChildren(...parts);
  }

  new ResizeObserver(() => {
    const width = Math.round(stage.clientWidth);
    if (!width || width === size) return;
    size = width;
    const ratio = Math.min(2, devicePixelRatio || 1);
    canvas.width = Math.round(size * ratio);
    canvas.height = Math.round(size * ratio);
    pinKey = '-';
    draw();
    // The first time the globe is on screen it turns to where the friends are.
    if (!arrived) flyTo({ ...start, zoom: 1 }, 1600);
    arrived = true;
  }).observe(stage);

  // The update asks for this when a friend has moved.
  root.addEventListener('fly', (e) => flyTo({ ...e.detail, zoom: Math.max(view.zoom, 5) }, 1200));

  renderCloud();
  renderControls();
  loadLand('110m').then((shape) => {
    coarse = shape;
    redraw();
  }).catch(() => {});
  return root;
}
