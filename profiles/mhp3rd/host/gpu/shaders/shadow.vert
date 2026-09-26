#version 450

// The sun's shadow map (post_process.hpp): a draw of the scene again, from
// the sun, in the GE's own vertex layout and push constants, with the
// transform taking the draw's model space to the shadow map's clip space.
layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_texcoord;
layout(location = 0) out vec2 frag_texcoord;

layout(push_constant) uniform Push {
    mat4 transform;
    vec4 viewport;
    vec4 texture_params;  // x: texture enabled, z: alpha cut-off, w: 1 cut out by alpha
    vec4 uv_transform;
    vec4 view_z;
} push;

void main() {
    frag_texcoord = in_texcoord * push.uv_transform.xy + push.uv_transform.zw;
    gl_Position = push.transform * vec4(in_position.xyz, 1.0);
}
