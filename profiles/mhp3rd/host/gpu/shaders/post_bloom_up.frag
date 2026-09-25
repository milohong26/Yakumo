#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// Bloom, up: a 3x3 tent filter of the smaller level (binding 0), added to
// this level's own down-sampled light (binding 1).
//   p.a.x, p.a.y: 1 / width, 1 / height of the smaller level; p.a.z its weight
layout(set = 0, binding = 0) uniform sampler2D smaller;
layout(set = 0, binding = 1) uniform sampler2D level;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 t = p.a.xy;
    vec3 s = texture(smaller, uv + vec2(-t.x, -t.y)).rgb + texture(smaller, uv + vec2(t.x, -t.y)).rgb +
             texture(smaller, uv + vec2(-t.x, t.y)).rgb + texture(smaller, uv + vec2(t.x, t.y)).rgb +
             (texture(smaller, uv + vec2(-t.x, 0.0)).rgb + texture(smaller, uv + vec2(t.x, 0.0)).rgb +
              texture(smaller, uv + vec2(0.0, -t.y)).rgb + texture(smaller, uv + vec2(0.0, t.y)).rgb) * 2.0 +
             texture(smaller, uv).rgb * 4.0;
    out_color = vec4(texture(level, uv).rgb + s / 16.0 * p.a.z, 1.0);
}
