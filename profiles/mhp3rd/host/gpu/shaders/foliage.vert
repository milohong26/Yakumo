#version 450

// The scene's leaves, grass and cloth drawn again into the foliage channel
// of the water mask (post_process.hpp), tested against the depth they
// wrote: the same transform, sway and depth as ge.vert gives them, and
// their texture coordinates for their cut-outs.
layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_texcoord;
layout(location = 0) out vec2 frag_texcoord;

layout(push_constant) uniform Push {
    mat4 transform;
    vec4 viewport;  // x, y: the wind's time and sway; w: flags (8: swayed)
    vec4 texture_params;
    vec4 uv_transform;
    vec4 view_z;
} push;

// As ge.vert's.
vec3 swayed(vec3 position) {
    float height = clamp(position.y / 90.0, 0.0, 1.6);
    float t = push.viewport.x;
    float phase = dot(position.xz, vec2(0.021, 0.013));
    float gust = sin(t * 1.3 + phase) * 0.7 + sin(t * 3.7 + phase * 2.9) * 0.3;
    float flutter = sin(t * 9.0 + dot(position, vec3(0.31, 0.17, 0.23))) * 0.25;
    return position + vec3(1.0, 0.0, 0.55) * ((gust + flutter) * height * height * push.viewport.y);
}

void main() {
    frag_texcoord = in_texcoord * push.uv_transform.xy + push.uv_transform.zw;
    int enables = int(push.viewport.w + 0.5);
    vec3 position = (enables & 8) != 0 ? swayed(in_position.xyz) : in_position.xyz;
    vec4 clip = push.transform * vec4(position, 1.0);
    clip.z = (clip.z + clip.w) * 0.5;
    gl_Position = clip;
}
