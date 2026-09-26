#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
#include "post_sun.glsl"

// Light shafts, at a quarter of the target's resolution, in two parts.
//   x: sunlight scattered towards the camera by the air between it and the
//      scene, where the sun reaches that air: each pixel marches its view
//      ray through the sun's shadow map in a few dithered steps; the lit
//      fraction, weighted by how much air scatters light towards the eye at
//      that angle to the sun (Henyey-Greenstein).
//   y: beams from the sky around the sun through what stands against it
//      (leaves, branches, ridges beyond the shadow map): the bright sky is
//      blurred along the lines towards the sun's place on the screen, with
//      a decay, so every gap in a canopy throws a beam (after Mitchell,
//      "Volumetric Light Scattering as a Post-Process", GPU Gems 3).
//
//   p.a: x steps, y noise offset
layout(set = 0, binding = 0) uniform sampler2D distances;  // half resolution
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_rays;

// The screen-space beams towards the sun's place on the screen.
float beams(vec3 direction, float noise) {
    vec3 l = sun.direction.xyz;
    if (l.z > -0.05) return 0.0;  // the sun is behind the camera
    vec2 sun_uv = target_pixel(l * 10000.0) / p.viewport.zw;
    // Far off the screen, the beams fade out.
    vec2 outside = max(abs(sun_uv - 0.5) - 0.5, vec2(0.0));
    float onscreen = 1.0 - smoothstep(0.0, 0.6, length(outside));
    if (onscreen <= 0.0) return 0.0;
    const int kSteps = 28;
    vec2 delta = (sun_uv - uv) * (0.85 / float(kSteps));
    vec2 at = uv + delta * noise;
    float weight = 1.0;
    float sum = 0.0;
    float norm = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        at += delta;
        norm += weight;
        if (all(greaterThanEqual(at, vec2(0.0))) && all(lessThanEqual(at, vec2(1.0))) &&
            textureLod(distances, at, 0.0).r >= kSky * 0.5) {
            vec3 sky = to_linear(textureLod(scene, at, 0.0).rgb);
            // The bright sky near the sun makes the beams, not the blue
            // around it.
            sum += smoothstep(0.35, 0.95, luminance(sky)) * weight;
        }
        weight *= 0.95;
    }
    // Most looking towards the sun, as the air scatters light forward.
    float towards = max(dot(direction, l), 0.0);
    towards *= towards;
    return sum / norm * onscreen * (0.08 + 0.92 * towards * towards);
}

float phase(float cos_angle, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(1.0 + g2 - 2.0 * g * cos_angle, 1e-4), 1.5));
}

void main() {
    if (sun.rays.x <= 0.0 && sun.bounce.y <= 0.0) {
        out_rays = vec2(0.0);
        return;
    }
    // This pixel's centre in the target, and the scene's distance there.
    vec2 pixel = uv * p.viewport.zw;
    float dist = textureLod(distances, uv, 0.0).r;
    float reach = min(dist, sun.rays.y);
    vec3 direction = normalize(view_position(pixel, 1.0));
    int steps = int(p.a.x);
    float noise = ign(gl_FragCoord.xy + p.a.y);
    float screen_beams = beams(direction, noise);
    if (sun.direction.w < 0.5) {
        out_rays = vec2(0.0, screen_beams);
        return;
    }
    float lit = 0.0;
    for (int i = 0; i < steps; ++i) {
        float t = (float(i) + noise) / float(steps);
        // Denser near the camera, where shafts are seen against the scene.
        vec3 at = direction * (reach * t * t);
        vec4 s = sun.view_to_shadow * vec4(at, 1.0);
        lit += (s.z >= 1.0) ? 1.0 : texture(shadow_map, vec3(s.xy, s.z));
    }
    lit /= float(max(steps, 1));
    float cos_angle = dot(direction, sun.direction.xyz);
    // Forward scattering: shafts show looking towards the sun, and the air
    // elsewhere stays clear.
    float scatter = phase(cos_angle, sun.rays.z) * 4.0 * 3.14159265 * 0.35;
    // Air over a longer ray scatters more, up to the ray's full reach.
    float air = reach / max(sun.rays.y, 1.0);
    out_rays = vec2(lit * scatter * air, screen_beams);
}
