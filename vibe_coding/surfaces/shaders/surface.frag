#version 330 core

in vec3 v_worldPos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3 u_camPos;
uniform vec3 u_lightDir;
uniform vec3 u_baseColor;
uniform bool u_showWireframe;
uniform float u_uMin;
uniform float u_uMax;
uniform float u_vMin;
uniform float u_vMax;
uniform bool u_doubleSided;

out vec4 fragColor;

void main() {
    vec3 N = normalize(v_normal);
    if (u_doubleSided && !gl_FrontFacing) N = -N;

    vec3 L = normalize(-u_lightDir);
    vec3 V = normalize(u_camPos - v_worldPos);
    vec3 H = normalize(L + V);

    // Color modulated by uv (parameter domain) — gives a sense of the chart
    float un = (v_uv.x - u_uMin) / max(u_uMax - u_uMin, 1e-6);
    float vn = (v_uv.y - u_vMin) / max(u_vMax - u_vMin, 1e-6);
    vec3 paramTint = mix(u_baseColor, vec3(un, 0.55, vn), 0.35);

    float ambient = 0.22;
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 48.0);

    vec3 color = paramTint * (ambient + 0.85 * diff) + vec3(1.0) * spec * 0.35;

    // Soft grid lines along the parameter domain
    float gu = abs(fract(un * 12.0) - 0.5);
    float gv = abs(fract(vn * 12.0) - 0.5);
    float grid = smoothstep(0.48, 0.5, max(gu, gv));
    color = mix(color, color * 0.55, grid * 0.4);

    fragColor = vec4(color, 1.0);
}
