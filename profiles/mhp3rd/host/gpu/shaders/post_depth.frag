#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// The scene's depth at half resolution, as view-space distance: the same
// pixel of every 2x2 block. Taking the nearest instead would fatten
// foreground silhouettes by a texel and leave dark halos behind them.
layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
layout(location = 0) in vec2 uv;
layout(location = 0) out float out_distance;

void main() {
    ivec2 at = min(ivec2(gl_FragCoord.xy) * 2, textureSize(depth_buffer, 0) - 1);
    out_distance = view_distance(texelFetch(depth_buffer, at, 0).r);
}
