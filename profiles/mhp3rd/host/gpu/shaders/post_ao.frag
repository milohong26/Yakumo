#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// Ground-truth ambient occlusion (Jimenez et al. 2016, as Intel's XeGTAO
// formulates it) and screen-space contact shadows towards the key light, at
// half resolution from the half-resolution view distances.
//
//   p.a: x radius (view units), y falloff (fraction of the radius), z the
//        target pixels a view unit covers at distance 1, w frame noise offset
//   p.b: x shadow length (view units), y shadow thickness (view units),
//        z steps, w shadow fade distance (view units)
layout(set = 0, binding = 0) uniform sampler2D distances;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_visibility;  // x ambient, y light

const float kPi = 3.14159265;
const int kSlices = 2;
const int kSteps = 5;

float distance_at(ivec2 texel) {
    return texelFetch(distances, clamp(texel, ivec2(0), textureSize(distances, 0) - 1), 0).r;
}


// A half-resolution texel's view position: its pixel centre in the target.
vec3 position_at(ivec2 texel, float dist) { return view_position(vec2(texel) * 2.0 + 1.0, dist); }

// The surface normal from the neighbours, taking on each axis the side
// closest in depth, so edges do not bend it (Turanszki 2021).
vec3 normal_at(ivec2 texel, vec3 centre, float dist) {
    float l = distance_at(texel - ivec2(1, 0));
    float r = distance_at(texel + ivec2(1, 0));
    float u = distance_at(texel - ivec2(0, 1));
    float d = distance_at(texel + ivec2(0, 1));
    vec3 dx = abs(l - dist) < abs(r - dist) ? centre - position_at(texel - ivec2(1, 0), l)
                                             : position_at(texel + ivec2(1, 0), r) - centre;
    vec3 dy = abs(u - dist) < abs(d - dist) ? centre - position_at(texel - ivec2(0, 1), u)
                                             : position_at(texel + ivec2(0, 1), d) - centre;
    vec3 n = normalize(cross(dy, dx));
    // Towards the viewer.
    return dot(n, -centre) < 0.0 ? -n : n;
}

float fast_acos(float x) {
    float r = -0.156583 * abs(x) + kPi * 0.5;
    r *= sqrt(1.0 - abs(x));
    return x >= 0.0 ? r : kPi - r;
}

void main() {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    float dist = distance_at(texel);
    if (dist >= kSky * 0.5) {
        out_visibility = vec2(1.0);
        return;
    }
    vec3 centre = position_at(texel, dist);
    vec3 normal = normal_at(texel, centre, dist);
    vec3 view = normalize(-centre);

    float noise = ign(gl_FragCoord.xy + p.a.w);
    float noise2 = fract(noise * 1.61803 + 0.5);
    // The radius on screen, in half-resolution texels, kept to a range the
    // texture cache takes well.
    // XeGTAO's defaults: the radius scaled by 1.457, the last 61.5% of it
    // fading out, samples spread by the square of their index.
    float radius = p.a.x * 1.457;
    float screen_radius = clamp(radius * p.a.z / dist * 0.5, 1.0, 48.0);
    float falloff_range = p.a.y * radius;
    float falloff_from = radius - falloff_range;
    float falloff_mul = -1.0 / falloff_range;
    float falloff_add = falloff_from / falloff_range + 1.0;

    float visibility = 0.0;
    for (int slice = 0; slice < kSlices; ++slice) {
        float phi = (float(slice) + noise) * kPi / float(kSlices);
        vec2 omega = vec2(cos(phi), -sin(phi));  // on screen, y down
        vec3 direction = vec3(cos(phi), sin(phi), 0.0);
        vec3 ortho = direction - dot(direction, view) * view;
        vec3 axis = normalize(cross(ortho, view));
        vec3 projected = normal - axis * dot(normal, axis);
        float projected_length = length(projected);
        float sign_n = sign(dot(ortho, projected));
        float cos_n = clamp(dot(projected, view) / max(projected_length, 1e-4), 0.0, 1.0);
        float n = sign_n * fast_acos(cos_n);
        float low0 = cos(n + kPi * 0.5);
        float low1 = cos(n - kPi * 0.5);
        float horizon0 = low0;
        float horizon1 = low1;
        for (int step = 0; step < kSteps; ++step) {
            float t = (float(step) + noise2) / float(kSteps);
            t *= t;  // denser near the centre
            float reach = max(t * screen_radius, 1.0 + float(step));
            vec2 offset = omega * reach;
            for (int side = 0; side < 2; ++side) {
                ivec2 at = texel + ivec2(round(side == 0 ? offset : -offset));
                float d = distance_at(at);
                if (d >= kSky * 0.5) continue;
                vec3 delta = position_at(at, d) - centre;
                float length2 = dot(delta, delta);
                float inv = inversesqrt(max(length2, 1e-6));
                float shc = dot(delta, view) * inv;
                float weight = clamp(sqrt(length2) * falloff_mul + falloff_add, 0.0, 1.0);
                if (side == 0) horizon0 = max(horizon0, mix(low0, shc, weight));
                else horizon1 = max(horizon1, mix(low1, shc, weight));
            }
        }
        float h0 = -fast_acos(horizon1);
        float h1 = fast_acos(horizon0);
        h0 = n + clamp(h0 - n, -kPi * 0.5, kPi * 0.5);
        h1 = n + clamp(h1 - n, -kPi * 0.5, kPi * 0.5);
        float iarc0 = (cos_n + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cos_n + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) * 0.25;
        visibility += projected_length * (iarc0 + iarc1);
    }
    visibility = pow(clamp(visibility / float(kSlices), 0.0, 1.0), 2.2);

    // Contact shadow: march towards the light and look for something in front
    // of the ray within a thickness, fading with distance from the camera.
    float light = 1.0;
    vec3 to_light = p.light.xyz;
    int steps = int(p.b.z);
    if (p.light.w > 0.0 && steps > 0 && dist < p.b.w) {
        float facing = dot(normal, to_light);
        if (facing > 0.0) {
            float step_length = p.b.x / float(steps);
            vec3 ray = centre + normal * (0.02 * dist) + to_light * step_length * noise;
            float occlusion = 0.0;
            for (int i = 0; i < steps; ++i) {
                ray += to_light * step_length;
                vec2 at = target_pixel(ray) * 0.5;
                if (any(lessThan(at, vec2(0.0))) || any(greaterThanEqual(at, vec2(textureSize(distances, 0))))) break;
                float scene = distance_at(ivec2(at));
                float depth_delta = -ray.z - scene;
                if (depth_delta > 0.02 * dist && depth_delta < p.b.y) {
                    occlusion = 1.0 - float(i) / float(steps) * 0.5;
                    break;
                }
            }
            float fade = clamp((p.b.w - dist) / (p.b.w * 0.25), 0.0, 1.0);
            light = 1.0 - occlusion * fade;
        }
    }
    out_visibility = vec2(visibility, light);
}
