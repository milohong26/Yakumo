#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
#include "post_sun.glsl"

// Reflections on the scene's water, at half resolution: from each water
// pixel the view ray is mirrored about the rippled surface and marched
// through the half-resolution distances until it passes behind the scene;
// the scene's colour there is the reflection. A ray that leaves the screen
// or finds nothing takes the sky's colour (the composite does that, from
// alpha: how sure the hit is).
layout(set = 0, binding = 0) uniform sampler2D distances;  // half resolution
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D water_mask;  // full resolution
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_reflection;

void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    vec2 pixel = vec2(texel) * 2.0 + 1.0;
    float mask = textureLod(water_mask, pixel / p.viewport.zw, 0.0).r;
    float dist = texelFetch(distances, texel, 0).r;
    if (mask <= 0.0 || dist >= kSky * 0.5) {
        out_reflection = vec4(0.0);
        return;
    }
    vec3 origin = view_position(pixel, dist);
    vec3 world = (sun.view_to_world * vec4(origin, 1.0)).xyz;
    vec3 n = water_normal(world);
    vec3 v = normalize(origin);
    vec3 r = reflect(v, n);
    // Rays towards the camera find what is behind it: nothing on screen.
    if (r.z > 0.2) {
        out_reflection = vec4(0.0);
        return;
    }
    vec2 size = vec2(textureSize(distances, 0));
    // The same dither every frame (the ripples move the reflection anyway).
    float noise = ign(gl_FragCoord.xy);
    // Steps growing with distance, so near reflections are exact and far
    // ones still reach the hills.
    float step_length = max(dist * 0.025, 5.0);
    vec3 at = origin + r * step_length * noise;
    vec3 previous = origin;
    float hit = 0.0;
    vec2 hit_uv = vec2(0.0);
    for (int i = 0; i < 20; ++i) {
        previous = at;
        at += r * step_length;
        step_length *= 1.22;
        vec2 screen = target_pixel(at) / p.viewport.zw;
        if (any(lessThan(screen, vec2(0.0))) || any(greaterThan(screen, vec2(1.0))) || at.z > -1.0) break;
        float scene_dist = textureLod(distances, screen, 0.0).r;
        float depth = -at.z;
        if (depth > scene_dist && depth - scene_dist < max(step_length * 2.0, scene_dist * 0.05)) {
            // Between the last two points: four halvings.
            vec3 a = previous, b = at;
            for (int j = 0; j < 4; ++j) {
                vec3 mid = (a + b) * 0.5;
                vec2 s = target_pixel(mid) / p.viewport.zw;
                if (-mid.z > textureLod(distances, s, 0.0).r) b = mid;
                else a = mid;
            }
            hit_uv = target_pixel(b) / p.viewport.zw;
            // Fade at the screen's edges and for rays that went far.
            vec2 edge = min(hit_uv, 1.0 - hit_uv);
            hit = smoothstep(0.0, 0.08, min(edge.x, edge.y)) * (1.0 - float(i) / 20.0 * 0.5);
            break;
        }
    }
    vec3 color = hit > 0.0 ? to_linear(textureLod(scene, hit_uv, 0.0).rgb) : vec3(0.0);
    out_reflection = vec4(color, hit);
}
