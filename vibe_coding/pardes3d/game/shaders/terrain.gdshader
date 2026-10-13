// קרקע: טקסטורת שימושי הקרקע שנבנתה מ-OSM, עם רעש פרטים בקרבה.
// ה-UV מחושב ממיקום העולם, כך שאין תלות במוסכמת ה-UV של הייצוא.
shader_type spatial;

uniform sampler2D ground : source_color, filter_linear_mipmap_anisotropic, repeat_disable;
uniform vec4 bounds; // west, south, east, north (מטרים)

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
	float north = -wpos.z;
	vec2 uv = vec2((wpos.x - bounds.x) / (bounds.z - bounds.x), (bounds.w - north) / (bounds.w - bounds.y));
	vec3 base = texture(ground, uv).rgb;
	float n = noise(wpos.xz * 0.35) * 0.5 + noise(wpos.xz * 1.7) * 0.3 + noise(wpos.xz * 7.0) * 0.2;
	float dist = length(wpos - CAMERA_POSITION_WORLD);
	float detail = mix(0.75 + 0.5 * n, 1.0, smoothstep(150.0, 800.0, dist));
	ALBEDO = base * detail;
	ROUGHNESS = 0.97;
}
