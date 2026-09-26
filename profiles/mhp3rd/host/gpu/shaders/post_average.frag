#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
#include "post_sun.glsl"

// One pixel for the composite, from grids over the scene:
//   rgb: the sky's colour, averaged where the scene shows sky (or the far
//        dome beyond the shadow map), which tints the shade as the light of
//        the sky fills shadows;
//   a:   how much of the scene's surfaces the sun reaches, which eases the
//        shade where a view is nearly all shadow (a cave, under a cliff), as
//        eyes adapt to it.
layout(set = 0, binding = 0) uniform sampler2D visibility;  // y: sunlight
layout(set = 0, binding = 1) uniform sampler2D distances;
layout(set = 0, binding = 2) uniform sampler2D scene;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_average;

void main() {
    float lit = 0.0;
    float surfaces = 0.0;
    vec3 sky = vec3(0.0);
    float skies = 0.0;
    float far = max(sun.params.w * 2.5, 1.0);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 12; ++x) {
            vec2 at = (vec2(x, y) + 0.5) / vec2(12.0, 8.0);
            float dist = textureLod(distances, at, 0.0).r;
            if (dist >= far) {
                sky += to_linear(textureLod(scene, at, 0.0).rgb);
                skies += 1.0;
            }
            if (dist >= kSky * 0.5) continue;
            lit += textureLod(visibility, at, 0.0).y;
            surfaces += 1.0;
        }
    }
    out_average = vec4(skies > 0.0 ? sky / skies : vec3(0.0), surfaces > 0.0 ? lit / surfaces : 1.0);
}
