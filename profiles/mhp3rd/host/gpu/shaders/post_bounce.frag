#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
#include "post_sun.glsl"

// Sunlight bounced off what it lights, at a quarter of the resolution: the
// sunlit colour of the surfaces around each point, gathered over a wide,
// turning disc and weighted by how near they are in depth, so light does
// not leak from the far hills onto the foreground. The composite lets it
// fall on each surface by its own colour, most in the shade.
layout(set = 0, binding = 0) uniform sampler2D scene;
layout(set = 0, binding = 1) uniform sampler2D visibility;  // half resolution, y: sunlight
layout(set = 0, binding = 2) uniform sampler2D distances;   // half resolution
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_bounce;

const vec2 kDisc[12] = vec2[](vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457),
                              vec2(-0.203, 0.621), vec2(0.962, -0.195), vec2(0.473, -0.480),
                              vec2(0.519, 0.767), vec2(0.185, -0.893), vec2(0.507, 0.064),
                              vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

void main() {
    float centre = textureLod(distances, uv, 0.0).r;
    if (centre >= kSky * 0.5) {
        out_bounce = vec4(0.0);
        return;
    }
    float angle = ign(gl_FragCoord.xy) * 6.2831853;
    mat2 turn = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    // About a hunter's height around the point, on screen.
    float radius = clamp(p.a.x / centre, 0.02, 0.18);
    vec2 aspect = vec2(p.viewport.w / p.viewport.z, 1.0);
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < 12; ++i) {
        vec2 at = uv + turn * kDisc[i] * radius * aspect;
        if (any(lessThan(at, vec2(0.0))) || any(greaterThan(at, vec2(1.0)))) continue;
        float d = textureLod(distances, at, 0.0).r;
        if (d >= kSky * 0.5) continue;
        float w = exp(-abs(d - centre) / (0.25 * centre));
        vec3 lit = to_linear(textureLod(scene, at, 0.0).rgb) * textureLod(visibility, at, 0.0).y;
        sum += lit * w;
        total += w;
    }
    out_bounce = vec4(total > 0.0 ? sum / 12.0 : vec3(0.0), 1.0);
}
