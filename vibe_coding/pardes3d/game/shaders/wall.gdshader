// חומות וגדרות בין מגרשים. UV.x = מטרים לאורך + 10000 * סוג, 1 - UV.y = גובה במטרים.
// סוג: 0 חומת טיח (צבע הקודקוד), 1 חומת בלוקים, 2 גדר חיה.
shader_type spatial;

varying vec3 wpos;
varying float up;

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
	up = (MODEL_MATRIX * vec4(NORMAL, 0.0)).y;
}

void fragment() {
	float kf = floor(UV.x / 10000.0 + 0.0001);
	int kind = int(kf);
	float u = UV.x - kf * 10000.0;
	float v = 1.0 - UV.y;
	float top = step(0.5, up);
	if (kind == 2) {
		float n = noise(wpos.xz * 2.5 + wpos.y * 1.7) * 0.6 + noise(wpos.xz * 9.0 + wpos.y * 7.0) * 0.4;
		ALBEDO = mix(vec3(0.05, 0.12, 0.035), vec3(0.13, 0.24, 0.07), n);
		ROUGHNESS = 1.0;
	} else if (kind == 1) {
		vec2 b = vec2(u / 0.4 + 0.5 * step(0.5, fract(v / 0.4)), v / 0.2);
		float mortar = 1.0 - step(0.04, fract(b.x)) * step(0.08, fract(b.y));
		vec3 c = vec3(0.56, 0.55, 0.52) * (0.9 + 0.2 * hash(floor(b)));
		ALBEDO = mix(c, vec3(0.45), mortar) * (1.0 - 0.12 * top);
		ROUGHNESS = 0.95;
	} else {
		vec3 c = COLOR.rgb * (0.92 + 0.08 * noise(vec2(u, v) * 2.0));
		c *= mix(1.0, 0.82, top); // נדבך עליון כהה יותר
		ALBEDO = c * (1.0 - 0.15 * smoothstep(0.4, 0.0, v));
		ROUGHNESS = 0.9;
	}
}
