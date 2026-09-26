#version 450

// Leaves, fences and hair are cut out of their quads by alpha: their shadow
// is too.
layout(location = 0) in vec2 frag_texcoord;
layout(set = 0, binding = 0) uniform sampler2D guest_texture;

layout(push_constant) uniform Push {
    mat4 transform;
    vec4 viewport;
    vec4 texture_params;
    vec4 uv_transform;
    vec4 view_z;
} push;

void main() {
    if (push.texture_params.w > 0.5 && push.texture_params.x > 0.5 &&
        texture(guest_texture, frag_texcoord).a < push.texture_params.z)
        discard;
}
