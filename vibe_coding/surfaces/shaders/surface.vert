#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normalMatrix;

out vec3 v_worldPos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_worldPos = world.xyz;
    v_normal = normalize(u_normalMatrix * a_normal);
    v_uv = a_uv;
    gl_Position = u_proj * u_view * world;
}
