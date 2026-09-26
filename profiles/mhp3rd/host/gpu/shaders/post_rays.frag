#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
#include "post_sun.glsl"

// Light shafts: sunlight scattered towards the camera by the air between it
// and the scene, where the sun reaches that air. At a quarter of the
// target's resolution, each pixel marches its view ray through the sun's
// shadow map in a few dithered steps; the lit fraction, weighted by how
// much air scatters light towards the eye at that angle to the sun
// (Henyey-Greenstein), is what the composite adds.
//
//   p.a: x steps, y noise offset
layout(set = 0, binding = 0) uniform sampler2D distances;  // half resolution
layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_rays;

float phase(float cos_angle, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(1.0 + g2 - 2.0 * g * cos_angle, 1e-4), 1.5));
}

void main() {
    if (sun.direction.w < 0.5 || sun.rays.x <= 0.0) {
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
    out_rays = vec2(lit * scatter * air, 0.0);
}
