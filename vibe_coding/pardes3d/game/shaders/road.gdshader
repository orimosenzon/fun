// כבישים: UV.x = רוחבי (0..1) + 2 * round(רוחב * 4) + 1000 * סוג, UV.y = מטרים לאורך.
// סוג: 0 אספלט, 1 עפר, 2 מדרכה, 3 מסילה, 4 מעבר חצייה.
shader_type spatial;
render_mode cull_disabled;

varying vec3 wpos;

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
	vec2 i = floor(p), f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

void vertex() {
	wpos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
	float kf = floor(UV.x / 1000.0 + 0.0001);
	int kind = int(kf);
	float rest = UV.x - kf * 1000.0;
	float w4 = floor(rest / 2.0 + 0.0001);
	float width = w4 / 4.0;
	float x = clamp(rest - w4 * 2.0, 0.0, 1.0);
	float along = 1.0 - UV.y; // trimesh הופך את V בייצוא
	float n = noise(wpos.xz * 2.0) * 0.6 + noise(wpos.xz * 9.0) * 0.4;
	vec3 col;
	float rough = 0.9;
	if (kind == 0) {
		col = vec3(0.075, 0.075, 0.08) * (0.85 + 0.3 * n);
		float edge_m = 0.35 / width;
		// קווי שוליים לבנים
		if (width >= 6.0 && (abs(x - edge_m) < 0.12 / width || abs(x - (1.0 - edge_m)) < 0.12 / width)) {
			col = vec3(0.85);
		}
		// קו מרוסק באמצע
		if (width >= 6.0 && abs(x - 0.5) < 0.07 / width && fract(along / 6.0) < 0.5) {
			col = vec3(0.88);
		}
		// שפה (אבן שפה) בצבע אפור בהיר בקצוות
		if (x < 0.25 / width || x > 1.0 - 0.25 / width) {
			col = vec3(0.45, 0.45, 0.43);
		}
		rough = 0.85;
	} else if (kind == 1) {
		col = mix(vec3(0.62, 0.52, 0.38), vec3(0.72, 0.62, 0.46), n);
		float rut = smoothstep(0.08, 0.0, abs(abs(x - 0.5) - 0.25));
		col *= 1.0 - 0.15 * rut;
	} else if (kind == 2) {
		// אבנים משתלבות
		vec2 t = vec2(x * width, along) * 3.0;
		float g = step(0.08, fract(t.x)) * step(0.08, fract(t.y + 0.5 * step(0.5, fract(t.x * 0.5))));
		col = mix(vec3(0.45, 0.43, 0.42), vec3(0.66, 0.64, 0.6), g) * (0.9 + 0.2 * n);
	} else if (kind == 4) {
		// מעבר חצייה: פסים לבנים של חצי מטר לרוחב הכביש
		col = fract(x * width / 1.0) < 0.5 ? vec3(0.86) : vec3(0.075, 0.075, 0.08) * (0.85 + 0.3 * n);
		rough = 0.8;
	} else {
		// מסילה: חצץ, אדנים ושתי פסים
		col = vec3(0.42, 0.38, 0.34) * (0.8 + 0.4 * n);
		if (fract(along / 0.6) < 0.35 && x > 0.15 && x < 0.85) col = vec3(0.35, 0.3, 0.26);
		if (abs(x - 0.32) < 0.02 || abs(x - 0.68) < 0.02) { col = vec3(0.55, 0.55, 0.57); rough = 0.3; METALLIC = 0.8; }
	}
	ALBEDO = col;
	ROUGHNESS = rough;
}
