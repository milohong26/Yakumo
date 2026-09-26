#version 450

// The scene's water surfaces drawn again into the water mask
// (post_process.hpp), tested against the depth they wrote: the same
// transform and depth as ge.vert gives them, and their vertex alpha, which
// fades the water out at its shores.
layout(location = 0) in vec4 in_position;
layout(location = 2) in vec4 in_color;
layout(location = 0) out float frag_alpha;

layout(push_constant) uniform Push {
    mat4 transform;
    vec4 viewport;
    vec4 texture_params;
    vec4 uv_transform;
    vec4 view_z;
} push;

void main() {
    frag_alpha = in_color.a;
    vec4 clip = push.transform * vec4(in_position.xyz, 1.0);
    clip.z = (clip.z + clip.w) * 0.5;
    gl_Position = clip;
}
