#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// How much of the scene the sun reaches, averaged over a grid of the
// half-resolution visibility: one pixel for the composite, which eases the
// shade where a view is nearly all shadow (a cave, under a cliff), as eyes
// adapt to it.
layout(set = 0, binding = 0) uniform sampler2D visibility;  // y: sunlight
layout(set = 0, binding = 1) uniform sampler2D distances;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_average;  // x: lit fraction of the scene's surfaces

void main() {
    float lit = 0.0;
    float count = 0.0;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 12; ++x) {
            vec2 at = (vec2(x, y) + 0.5) / vec2(12.0, 8.0);
            if (textureLod(distances, at, 0.0).r >= kSky * 0.5) continue;
            lit += textureLod(visibility, at, 0.0).y;
            count += 1.0;
        }
    }
    out_average = vec2(count > 0.0 ? lit / count : 1.0, count / 96.0);
}
