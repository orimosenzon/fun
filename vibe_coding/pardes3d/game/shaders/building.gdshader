// בניינים: טיח בצבע הקודקוד, חלונות ותריסים פרוצדורליים, גג בטון עם שוליים.
// UV.x = מטרים לאורך ההיקף, 1 - UV.y = מטרים מעל קומת הקרקע (trimesh הופך את V בייצוא). בגג UV.x = -1000.
shader_type spatial;

const float FLOOR_H = 3.2;
const float BAY = 3.6;

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
	vec3 wall = COLOR.rgb;
	if (UV.x < -500.0) {
		// גג שטוח: בטון אפרפר עם כתמים
		float n = noise(wpos.xz * 0.7) * 0.5 + noise(wpos.xz * 3.1) * 0.25;
		ALBEDO = mix(vec3(0.62, 0.61, 0.58), vec3(0.72, 0.70, 0.66), n);
		ROUGHNESS = 0.95;
	} else {
		float u = UV.x, v = 1.0 - UV.y;
		float fl = floor(v / FLOOR_H);
		float fv = v - fl * FLOOR_H;
		float bay = floor(u / BAY);
		float fu = u - bay * BAY;
		float seed = hash(vec2(bay, fl) + floor(wpos.xz / 40.0));
		// טיח: רעש עדין ולכלוך שנאסף למטה
		float grime = smoothstep(1.2, 0.0, v) * 0.18;
		vec3 col = wall * (0.93 + 0.07 * noise(vec2(u, v) * 1.7)) * (1.0 - grime);
		bool is_window = v > 0.0 && fv > 0.95 && fv < 2.25 && fu > 1.0 && fu < 2.6 && seed > 0.12;
		ROUGHNESS = 0.9;
		if (is_window) {
			if (seed > 0.72) {
				// תריס גלילה מוגף, בהיר עם פסים
				float slat = step(0.5, fract(fv * 12.0));
				col = mix(vec3(0.80, 0.80, 0.78), vec3(0.72, 0.72, 0.70), slat);
			} else {
				col = mix(vec3(0.10, 0.13, 0.16), vec3(0.22, 0.27, 0.32), fv - 0.95);
				ROUGHNESS = 0.15;
				METALLIC = 0.3;
			}
		} else if (v > 0.0 && fv > 0.85 && fv < 0.95 && fu > 0.9 && fu < 2.7 && seed > 0.12) {
			col *= 0.82; // אדן חלון
		}
		// פס בין קומות
		col *= 1.0 - 0.05 * step(fv, 0.12) * step(0.5, fl);
		ALBEDO = col;
	}
}
