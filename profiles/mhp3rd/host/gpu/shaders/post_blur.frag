#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// A depth-aware blur of the ambient and light visibility at half resolution,
// in one pass over a 5x5 neighbourhood, to smooth the noise of GTAO's few
// directions without bleeding across edges.
layout(set = 0, binding = 0) uniform sampler2D visibility;
layout(set = 0, binding = 1) uniform sampler2D distances;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_visibility;

void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 limit = textureSize(visibility, 0) - 1;
    float centre = texelFetch(distances, texel, 0).r;
    float tolerance = 1.0 / max(centre * 0.03, 1e-3);
    vec2 sum = vec2(0.0);
    float total = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 at = clamp(texel + ivec2(x, y), ivec2(0), limit);
            float d = texelFetch(distances, at, 0).r;
            float w = exp(-float(x * x + y * y) / 6.0 - abs(d - centre) * tolerance);
            sum += texelFetch(visibility, at, 0).rg * w;
            total += w;
        }
    }
    out_visibility = sum / max(total, 1e-5);
}
