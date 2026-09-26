#version 450

layout(location = 0) in vec2 frag_texcoord;
layout(location = 0) out vec4 out_mask;
layout(set = 0, binding = 0) uniform sampler2D guest_texture;

layout(push_constant) uniform Push {
    mat4 transform;
    vec4 viewport;
    vec4 texture_params;  // z: the game's alpha reference, 0 to 255
    vec4 uv_transform;
    vec4 view_z;
} push;

void main() {
    if (texture(guest_texture, frag_texcoord).a * 255.0 < max(push.texture_params.z, 1.0)) discard;
    out_mask = vec4(0.0, 1.0, 0.0, 0.0);
}
