// Per-piece silhouette drawers. Each takes (ctx, cx, cy, r, fg) where r is the inner radius
// and fg is the silhouette color. All drawings are normalized: paths use offsets in units of r.

const SPRITE_DRAWERS = {
  // ─── LIGHT SIDE ─────────────────────────────────────────

  knight(ctx, cx, cy, r, fg) {
    // Shield + sword
    ctx.fillStyle = fg;
    // Shield (kite)
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.7);
    ctx.lineTo(cx - r * 0.55, cy - r * 0.2);
    ctx.lineTo(cx - r * 0.4, cy + r * 0.55);
    ctx.lineTo(cx, cy + r * 0.75);
    ctx.lineTo(cx + r * 0.4, cy + r * 0.55);
    ctx.lineTo(cx + r * 0.55, cy - r * 0.2);
    ctx.closePath();
    ctx.fill();
    // Cross on shield (inverse)
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.fillRect(cx - r * 0.08, cy - r * 0.45, r * 0.16, r * 0.95);
    ctx.fillRect(cx - r * 0.4,  cy - r * 0.1,  r * 0.8,  r * 0.16);
  },

  archer(ctx, cx, cy, r, fg) {
    ctx.strokeStyle = fg;
    ctx.fillStyle = fg;
    ctx.lineWidth = r * 0.14;
    ctx.lineCap = 'round';
    // Hooded head
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.4, r * 0.28, 0, Math.PI * 2);
    ctx.fill();
    // Body
    ctx.fillRect(cx - r * 0.18, cy - r * 0.1, r * 0.36, r * 0.7);
    // Bow (arc)
    ctx.beginPath();
    ctx.arc(cx + r * 0.5, cy + r * 0.1, r * 0.55, -Math.PI / 2, Math.PI / 2);
    ctx.stroke();
    // Bowstring
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.5, cy - r * 0.45);
    ctx.lineTo(cx + r * 0.5, cy + r * 0.65);
    ctx.lineWidth = r * 0.06;
    ctx.stroke();
    // Arrow
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.3, cy + r * 0.1);
    ctx.lineTo(cx + r * 0.5, cy + r * 0.1);
    ctx.lineWidth = r * 0.08;
    ctx.stroke();
  },

  valkyrie(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Wings (V shape)
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.2);
    ctx.lineTo(cx - r * 0.95, cy - r * 0.6);
    ctx.lineTo(cx - r * 0.7,  cy + r * 0.1);
    ctx.lineTo(cx - r * 0.2,  cy);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.2);
    ctx.lineTo(cx + r * 0.95, cy - r * 0.6);
    ctx.lineTo(cx + r * 0.7,  cy + r * 0.1);
    ctx.lineTo(cx + r * 0.2,  cy);
    ctx.closePath();
    ctx.fill();
    // Head
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.45, r * 0.2, 0, Math.PI * 2);
    ctx.fill();
    // Body + spear
    ctx.fillRect(cx - r * 0.12, cy - r * 0.2, r * 0.24, r * 0.85);
    ctx.fillRect(cx + r * 0.35, cy - r * 0.7, r * 0.08, r * 1.4);
    // Spear tip
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.39, cy - r * 0.9);
    ctx.lineTo(cx + r * 0.25, cy - r * 0.65);
    ctx.lineTo(cx + r * 0.53, cy - r * 0.65);
    ctx.closePath();
    ctx.fill();
  },

  golem(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Boxy body
    ctx.fillRect(cx - r * 0.55, cy - r * 0.25, r * 1.1, r * 0.95);
    // Head
    ctx.fillRect(cx - r * 0.32, cy - r * 0.7, r * 0.64, r * 0.5);
    // Arms (rectangular)
    ctx.fillRect(cx - r * 0.85, cy - r * 0.15, r * 0.28, r * 0.7);
    ctx.fillRect(cx + r * 0.57, cy - r * 0.15, r * 0.28, r * 0.7);
    // Runes (subtractive)
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillRect(cx - r * 0.25, cy + r * 0.05, r * 0.1, r * 0.1);
    ctx.fillRect(cx + r * 0.15, cy + r * 0.05, r * 0.1, r * 0.1);
    ctx.fillRect(cx - r * 0.05, cy + r * 0.25, r * 0.1, r * 0.1);
    // Eyes
    ctx.fillStyle = 'rgba(255,200,80,0.9)';
    ctx.fillRect(cx - r * 0.18, cy - r * 0.55, r * 0.1, r * 0.1);
    ctx.fillRect(cx + r * 0.08, cy - r * 0.55, r * 0.1, r * 0.1);
  },

  unicorn(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Body (oval)
    ctx.beginPath();
    ctx.ellipse(cx - r * 0.05, cy + r * 0.25, r * 0.55, r * 0.32, 0, 0, Math.PI * 2);
    ctx.fill();
    // Head (tilted up)
    ctx.beginPath();
    ctx.ellipse(cx + r * 0.45, cy - r * 0.3, r * 0.28, r * 0.22, Math.PI * 0.15, 0, Math.PI * 2);
    ctx.fill();
    // Neck
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.1, cy);
    ctx.lineTo(cx + r * 0.25, cy + r * 0.25);
    ctx.lineTo(cx + r * 0.6, cy - r * 0.15);
    ctx.lineTo(cx + r * 0.45, cy - r * 0.45);
    ctx.closePath();
    ctx.fill();
    // Horn
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.55, cy - r * 0.45);
    ctx.lineTo(cx + r * 0.78, cy - r * 0.95);
    ctx.lineTo(cx + r * 0.65, cy - r * 0.4);
    ctx.closePath();
    ctx.fill();
    // Legs
    ctx.fillRect(cx - r * 0.4, cy + r * 0.45, r * 0.13, r * 0.4);
    ctx.fillRect(cx - r * 0.15, cy + r * 0.45, r * 0.13, r * 0.4);
    ctx.fillRect(cx + r * 0.1, cy + r * 0.45, r * 0.13, r * 0.4);
    ctx.fillRect(cx + r * 0.32, cy + r * 0.45, r * 0.13, r * 0.4);
    // Tail
    ctx.fillRect(cx - r * 0.65, cy, r * 0.12, r * 0.55);
  },

  djinni(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Lower swirl (smoke trail)
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.4, cy + r * 0.85);
    ctx.bezierCurveTo(cx - r * 0.6, cy + r * 0.4, cx - r * 0.2, cy + r * 0.3, cx, cy + r * 0.1);
    ctx.bezierCurveTo(cx + r * 0.2, cy + r * 0.3, cx + r * 0.6, cy + r * 0.4, cx + r * 0.4, cy + r * 0.85);
    ctx.bezierCurveTo(cx + r * 0.15, cy + r * 0.7, cx - r * 0.15, cy + r * 0.7, cx - r * 0.4, cy + r * 0.85);
    ctx.fill();
    // Torso (with crossed arms)
    ctx.beginPath();
    ctx.ellipse(cx, cy - r * 0.05, r * 0.42, r * 0.32, 0, 0, Math.PI * 2);
    ctx.fill();
    // Head
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.55, r * 0.27, 0, Math.PI * 2);
    ctx.fill();
    // Turban (subtractive jewel)
    ctx.fillStyle = 'rgba(255,220,80,0.85)';
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.75, r * 0.08, 0, Math.PI * 2);
    ctx.fill();
  },

  phoenix(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Body (circle)
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.3, 0, Math.PI * 2);
    ctx.fill();
    // Head + beak
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.4, r * 0.18, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.55);
    ctx.lineTo(cx + r * 0.2, cy - r * 0.45);
    ctx.lineTo(cx, cy - r * 0.35);
    ctx.closePath();
    ctx.fill();
    // Wings (flame-shaped)
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.3, cy);
    ctx.bezierCurveTo(cx - r * 0.95, cy - r * 0.4, cx - r * 0.75, cy + r * 0.3, cx - r * 0.15, cy + r * 0.2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.3, cy);
    ctx.bezierCurveTo(cx + r * 0.95, cy - r * 0.4, cx + r * 0.75, cy + r * 0.3, cx + r * 0.15, cy + r * 0.2);
    ctx.fill();
    // Tail flames
    ctx.beginPath();
    ctx.moveTo(cx, cy + r * 0.25);
    ctx.lineTo(cx - r * 0.2, cy + r * 0.85);
    ctx.lineTo(cx, cy + r * 0.6);
    ctx.lineTo(cx + r * 0.2, cy + r * 0.85);
    ctx.closePath();
    ctx.fill();
  },

  wizard(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Robe (triangle)
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.1);
    ctx.lineTo(cx - r * 0.6, cy + r * 0.85);
    ctx.lineTo(cx + r * 0.6, cy + r * 0.85);
    ctx.closePath();
    ctx.fill();
    // Hat (tall pointed cone)
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.95);
    ctx.lineTo(cx - r * 0.35, cy - r * 0.25);
    ctx.lineTo(cx + r * 0.35, cy - r * 0.25);
    ctx.closePath();
    ctx.fill();
    // Hat brim
    ctx.fillRect(cx - r * 0.45, cy - r * 0.28, r * 0.9, r * 0.08);
    // Beard
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.18, cy - r * 0.1);
    ctx.lineTo(cx, cy + r * 0.3);
    ctx.lineTo(cx + r * 0.18, cy - r * 0.1);
    ctx.closePath();
    ctx.fill();
    // Staff
    ctx.fillRect(cx + r * 0.45, cy - r * 0.6, r * 0.08, r * 1.4);
    // Star on staff top
    ctx.fillStyle = 'rgba(255,220,80,0.95)';
    drawStar(ctx, cx + r * 0.49, cy - r * 0.7, r * 0.18, 5);
  },

  // ─── DARK SIDE ─────────────────────────────────────────

  goblin(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Head (with pointy ears)
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.35, r * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.3, cy - r * 0.45);
    ctx.lineTo(cx - r * 0.55, cy - r * 0.7);
    ctx.lineTo(cx - r * 0.2, cy - r * 0.55);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.3, cy - r * 0.45);
    ctx.lineTo(cx + r * 0.55, cy - r * 0.7);
    ctx.lineTo(cx + r * 0.2, cy - r * 0.55);
    ctx.closePath();
    ctx.fill();
    // Body (hunched)
    ctx.beginPath();
    ctx.ellipse(cx, cy + r * 0.25, r * 0.32, r * 0.4, 0, 0, Math.PI * 2);
    ctx.fill();
    // Eyes
    ctx.fillStyle = 'rgba(255,200,40,0.95)';
    ctx.fillRect(cx - r * 0.18, cy - r * 0.4, r * 0.08, r * 0.08);
    ctx.fillRect(cx + r * 0.1,  cy - r * 0.4, r * 0.08, r * 0.08);
    // Curved jagged sword
    ctx.strokeStyle = fg;
    ctx.lineWidth = r * 0.1;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.45, cy + r * 0.6);
    ctx.quadraticCurveTo(cx + r * 0.75, cy, cx + r * 0.55, cy - r * 0.6);
    ctx.stroke();
  },

  manticore(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Body
    ctx.beginPath();
    ctx.ellipse(cx, cy + r * 0.2, r * 0.55, r * 0.3, 0, 0, Math.PI * 2);
    ctx.fill();
    // Head with mane
    ctx.beginPath();
    ctx.arc(cx - r * 0.4, cy - r * 0.2, r * 0.35, 0, Math.PI * 2);
    ctx.fill();
    // Mane spikes
    for (let i = 0; i < 8; i++) {
      const ang = -Math.PI + i * (Math.PI / 7);
      const x1 = cx - r * 0.4 + Math.cos(ang) * r * 0.35;
      const y1 = cy - r * 0.2 + Math.sin(ang) * r * 0.35;
      const x2 = cx - r * 0.4 + Math.cos(ang) * r * 0.55;
      const y2 = cy - r * 0.2 + Math.sin(ang) * r * 0.55;
      ctx.beginPath();
      ctx.lineWidth = r * 0.1;
      ctx.strokeStyle = fg;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    // Scorpion tail (curled up over back)
    ctx.strokeStyle = fg;
    ctx.lineWidth = r * 0.16;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.45, cy + r * 0.3);
    ctx.bezierCurveTo(cx + r * 0.85, cy, cx + r * 0.7, cy - r * 0.7, cx + r * 0.25, cy - r * 0.65);
    ctx.stroke();
    // Stinger
    ctx.fillStyle = fg;
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.2, cy - r * 0.7);
    ctx.lineTo(cx + r * 0.05, cy - r * 0.95);
    ctx.lineTo(cx + r * 0.3, cy - r * 0.55);
    ctx.closePath();
    ctx.fill();
    // Legs
    ctx.fillRect(cx - r * 0.35, cy + r * 0.4, r * 0.1, r * 0.4);
    ctx.fillRect(cx - r * 0.1, cy + r * 0.4, r * 0.1, r * 0.4);
    ctx.fillRect(cx + r * 0.15, cy + r * 0.4, r * 0.1, r * 0.4);
    ctx.fillRect(cx + r * 0.4, cy + r * 0.4, r * 0.1, r * 0.4);
  },

  harpy(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Wings spread
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx - r * 0.95, cy - r * 0.3);
    ctx.lineTo(cx - r * 0.85, cy + r * 0.1);
    ctx.lineTo(cx - r * 0.6, cy - r * 0.05);
    ctx.lineTo(cx - r * 0.5, cy + r * 0.25);
    ctx.lineTo(cx - r * 0.2, cy + r * 0.1);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * 0.95, cy - r * 0.3);
    ctx.lineTo(cx + r * 0.85, cy + r * 0.1);
    ctx.lineTo(cx + r * 0.6, cy - r * 0.05);
    ctx.lineTo(cx + r * 0.5, cy + r * 0.25);
    ctx.lineTo(cx + r * 0.2, cy + r * 0.1);
    ctx.closePath();
    ctx.fill();
    // Body
    ctx.beginPath();
    ctx.ellipse(cx, cy + r * 0.05, r * 0.2, r * 0.35, 0, 0, Math.PI * 2);
    ctx.fill();
    // Head
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.45, r * 0.2, 0, Math.PI * 2);
    ctx.fill();
    // Claws
    ctx.strokeStyle = fg;
    ctx.lineWidth = r * 0.08;
    ctx.lineCap = 'round';
    for (let i = -1; i <= 1; i++) {
      ctx.beginPath();
      ctx.moveTo(cx + i * r * 0.13, cy + r * 0.4);
      ctx.lineTo(cx + i * r * 0.18, cy + r * 0.75);
      ctx.stroke();
    }
  },

  troll(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Lumpy body
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.55, cy + r * 0.7);
    ctx.quadraticCurveTo(cx - r * 0.7, cy, cx - r * 0.5, cy - r * 0.3);
    ctx.quadraticCurveTo(cx, cy - r * 0.5, cx + r * 0.5, cy - r * 0.3);
    ctx.quadraticCurveTo(cx + r * 0.7, cy, cx + r * 0.55, cy + r * 0.7);
    ctx.closePath();
    ctx.fill();
    // Head (small)
    ctx.beginPath();
    ctx.arc(cx, cy - r * 0.55, r * 0.22, 0, Math.PI * 2);
    ctx.fill();
    // Tusk
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.05, cy - r * 0.45);
    ctx.lineTo(cx - r * 0.1, cy - r * 0.3);
    ctx.lineTo(cx + r * 0.02, cy - r * 0.4);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,250,220,0.9)';
    ctx.fill();
    // Club
    ctx.fillStyle = fg;
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.5, cy + r * 0.3);
    ctx.lineTo(cx + r * 0.9, cy - r * 0.4);
    ctx.lineTo(cx + r * 1.0, cy - r * 0.55);
    ctx.lineTo(cx + r * 0.6, cy + r * 0.2);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + r * 0.92, cy - r * 0.5, r * 0.18, 0, Math.PI * 2);
    ctx.fill();
    // Eyes
    ctx.fillStyle = 'rgba(255,80,40,0.95)';
    ctx.fillRect(cx - r * 0.12, cy - r * 0.6, r * 0.07, r * 0.07);
    ctx.fillRect(cx + r * 0.05, cy - r * 0.6, r * 0.07, r * 0.07);
  },

  basilisk(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    ctx.strokeStyle = fg;
    ctx.lineWidth = r * 0.22;
    ctx.lineCap = 'round';
    // S-curve body
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.7, cy + r * 0.7);
    ctx.bezierCurveTo(cx - r * 0.7, cy + r * 0.1, cx + r * 0.7, cy + r * 0.4, cx + r * 0.7, cy - r * 0.2);
    ctx.bezierCurveTo(cx + r * 0.7, cy - r * 0.5, cx + r * 0.2, cy - r * 0.6, cx + r * 0.1, cy - r * 0.45);
    ctx.stroke();
    // Head (slightly wider)
    ctx.beginPath();
    ctx.ellipse(cx + r * 0.15, cy - r * 0.45, r * 0.25, r * 0.18, Math.PI * 0.1, 0, Math.PI * 2);
    ctx.fill();
    // Eyes
    ctx.fillStyle = 'rgba(255,255,80,0.95)';
    ctx.beginPath();
    ctx.arc(cx + r * 0.05, cy - r * 0.5, r * 0.06, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + r * 0.22, cy - r * 0.45, r * 0.06, 0, Math.PI * 2);
    ctx.fill();
    // Scales (dots along body)
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    for (let i = 0; i < 6; i++) {
      const t = i / 6;
      const x = cx - r * 0.7 + t * r * 1.4;
      const y = cy + r * 0.55 - Math.sin(t * Math.PI) * r * 0.45;
      ctx.beginPath();
      ctx.arc(x, y, r * 0.05, 0, Math.PI * 2);
      ctx.fill();
    }
  },

  dragon(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Wings (large, spread)
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.1);
    ctx.lineTo(cx - r * 0.95, cy - r * 0.6);
    ctx.lineTo(cx - r * 0.75, cy - r * 0.2);
    ctx.lineTo(cx - r * 0.6,  cy - r * 0.5);
    ctx.lineTo(cx - r * 0.45, cy - r * 0.15);
    ctx.lineTo(cx - r * 0.3,  cy - r * 0.4);
    ctx.lineTo(cx - r * 0.15, cy);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.1);
    ctx.lineTo(cx + r * 0.95, cy - r * 0.6);
    ctx.lineTo(cx + r * 0.75, cy - r * 0.2);
    ctx.lineTo(cx + r * 0.6,  cy - r * 0.5);
    ctx.lineTo(cx + r * 0.45, cy - r * 0.15);
    ctx.lineTo(cx + r * 0.3,  cy - r * 0.4);
    ctx.lineTo(cx + r * 0.15, cy);
    ctx.closePath();
    ctx.fill();
    // Body
    ctx.beginPath();
    ctx.ellipse(cx, cy + r * 0.2, r * 0.3, r * 0.35, 0, 0, Math.PI * 2);
    ctx.fill();
    // Head (with horns)
    ctx.beginPath();
    ctx.ellipse(cx, cy - r * 0.35, r * 0.22, r * 0.2, 0, 0, Math.PI * 2);
    ctx.fill();
    // Horns
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.15, cy - r * 0.5);
    ctx.lineTo(cx - r * 0.3, cy - r * 0.8);
    ctx.lineTo(cx - r * 0.08, cy - r * 0.55);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(cx + r * 0.15, cy - r * 0.5);
    ctx.lineTo(cx + r * 0.3, cy - r * 0.8);
    ctx.lineTo(cx + r * 0.08, cy - r * 0.55);
    ctx.closePath();
    ctx.fill();
    // Tail
    ctx.strokeStyle = fg;
    ctx.lineWidth = r * 0.16;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, cy + r * 0.5);
    ctx.quadraticCurveTo(cx + r * 0.5, cy + r * 0.85, cx + r * 0.3, cy + r * 0.95);
    ctx.stroke();
    // Eyes (glowing)
    ctx.fillStyle = 'rgba(255,80,40,0.95)';
    ctx.beginPath();
    ctx.arc(cx - r * 0.08, cy - r * 0.38, r * 0.05, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + r * 0.08, cy - r * 0.38, r * 0.05, 0, Math.PI * 2);
    ctx.fill();
  },

  shapeshifter(ctx, cx, cy, r, fg, tMs = 0) {
    // Smoky undulating blob
    const phase = (tMs || 0) / 500;
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    const points = 12;
    for (let i = 0; i <= points; i++) {
      const ang = (i / points) * Math.PI * 2;
      const wob = 0.7 + Math.sin(phase + i * 1.3) * 0.12;
      const x = cx + Math.cos(ang) * r * wob;
      const y = cy + Math.sin(ang) * r * wob;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
    // Glowing eyes
    ctx.fillStyle = 'rgba(255,255,255,0.95)';
    ctx.beginPath();
    ctx.arc(cx - r * 0.2, cy - r * 0.15, r * 0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + r * 0.2, cy - r * 0.15, r * 0.1, 0, Math.PI * 2);
    ctx.fill();
  },

  sorceress(ctx, cx, cy, r, fg) {
    ctx.fillStyle = fg;
    // Tall robe
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.2);
    ctx.lineTo(cx - r * 0.55, cy + r * 0.9);
    ctx.lineTo(cx + r * 0.55, cy + r * 0.9);
    ctx.closePath();
    ctx.fill();
    // Hood
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.4, cy - r * 0.2);
    ctx.quadraticCurveTo(cx, cy - r * 0.9, cx + r * 0.4, cy - r * 0.2);
    ctx.closePath();
    ctx.fill();
    // Face (dark gap)
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.beginPath();
    ctx.ellipse(cx, cy - r * 0.3, r * 0.18, r * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();
    // Glowing magic orb in hand
    ctx.fillStyle = 'rgba(200,120,255,0.95)';
    ctx.beginPath();
    ctx.arc(cx + r * 0.5, cy + r * 0.3, r * 0.16, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.beginPath();
    ctx.arc(cx + r * 0.45, cy + r * 0.25, r * 0.06, 0, Math.PI * 2);
    ctx.fill();
    // Crown spikes on hood
    ctx.fillStyle = fg;
    for (let i = -1; i <= 1; i++) {
      ctx.beginPath();
      ctx.moveTo(cx + i * r * 0.15, cy - r * 0.7);
      ctx.lineTo(cx + i * r * 0.15 - r * 0.05, cy - r * 0.5);
      ctx.lineTo(cx + i * r * 0.15 + r * 0.05, cy - r * 0.5);
      ctx.closePath();
      ctx.fill();
    }
  },
};

function drawStar(ctx, cx, cy, r, points = 5) {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const ang = (i / (points * 2)) * Math.PI * 2 - Math.PI / 2;
    const rad = i % 2 === 0 ? r : r * 0.45;
    const x = cx + Math.cos(ang) * rad;
    const y = cy + Math.sin(ang) * rad;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
}

// Main entry — draws the full piece icon at (cx, cy) with cell size.
// piece animation params (bob, scale) are folded in by the caller.
function drawPieceSprite(ctx, piece, cx, cy, cellSize, tMs) {
  const def = PIECE_TYPES[piece.type];
  const r = cellSize * 0.42;

  // Glow halo for sovereigns
  if (def.isSovereign) {
    const pulse = 0.85 + Math.sin(tMs / 400 + piece.id) * 0.15;
    ctx.fillStyle = `rgba(255, 215, 0, ${0.18 * pulse})`;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.35, 0, Math.PI * 2);
    ctx.fill();
  }

  // Background disc
  ctx.fillStyle = def.color;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  // Outline
  ctx.strokeStyle = def.side === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.85)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Foreground silhouette
  const fg = def.side === 'light' ? '#15151c' : '#f4f4fc';
  const drawer = SPRITE_DRAWERS[piece.type];
  if (drawer) {
    ctx.save();
    drawer(ctx, cx, cy, r * 0.95, fg, tMs);
    ctx.restore();
  }

  // Sovereign crown indicator (ring)
  if (def.isSovereign) {
    ctx.strokeStyle = 'rgba(255,215,0,0.95)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.stroke();
  }
}
