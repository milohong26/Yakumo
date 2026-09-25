#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// Bloom, down (Jimenez 2014, "Next generation post processing in Call of
// Duty: Advanced Warfare"): a 13-tap filter from the level above. The first
// level (p.a.x = 1) reads the scene itself: to linear light, a soft threshold
// (p.a.y threshold, p.a.z knee) and a Karis average against fireflies.
//   p.a.w: 1 / source width, p.b.x: 1 / source height
layout(set = 0, binding = 0) uniform sampler2D source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// The first level expands the scene as the composite does (the invertible
// shoulder, knee p.b.y), capped at p.b.z so that a white wall does not bloom
// like the sun.
vec3 fetch(vec2 at) {
    vec3 c = textureLod(source, at, 0.0).rgb;
    if (p.a.x < 0.5) return c;
    c = to_linear(c);
    float k = p.b.y;
    float peak = max(c.r, max(c.g, c.b));
    if (peak <= k) return c;
    float y = min(peak, 0.999);
    float expanded = min(k + (1.0 - k) * (y - k) / max(1.0 - y, 1.0 / 256.0), p.b.z);
    return c * (expanded / peak);
}

float karis(vec3 c) { return 1.0 / (1.0 + luminance(c)); }

void main() {
    vec2 texel = vec2(p.a.w, p.b.x);
    vec3 a = fetch(uv + texel * vec2(-2.0, -2.0));
    vec3 b = fetch(uv + texel * vec2(0.0, -2.0));
    vec3 c = fetch(uv + texel * vec2(2.0, -2.0));
    vec3 d = fetch(uv + texel * vec2(-2.0, 0.0));
    vec3 e = fetch(uv);
    vec3 f = fetch(uv + texel * vec2(2.0, 0.0));
    vec3 g = fetch(uv + texel * vec2(-2.0, 2.0));
    vec3 h = fetch(uv + texel * vec2(0.0, 2.0));
    vec3 i = fetch(uv + texel * vec2(2.0, 2.0));
    vec3 j = fetch(uv + texel * vec2(-1.0, -1.0));
    vec3 k = fetch(uv + texel * vec2(1.0, -1.0));
    vec3 l = fetch(uv + texel * vec2(-1.0, 1.0));
    vec3 m = fetch(uv + texel * vec2(1.0, 1.0));
    vec3 result;
    if (p.a.x > 0.5) {
        // Five groups, each weighted by its brightness, as Karis proposed.
        vec3 g0 = (j + k + l + m) * 0.25;
        vec3 g1 = (a + b + d + e) * 0.25;
        vec3 g2 = (b + c + e + f) * 0.25;
        vec3 g3 = (d + e + g + h) * 0.25;
        vec3 g4 = (e + f + h + i) * 0.25;
        float w0 = karis(g0) * 0.5, w1 = karis(g1) * 0.125, w2 = karis(g2) * 0.125, w3 = karis(g3) * 0.125,
              w4 = karis(g4) * 0.125;
        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
        // Soft threshold on the brightest channel.
        float bright = max(result.r, max(result.g, result.b));
        float knee = p.a.z;
        float soft = clamp(bright - p.a.y + knee, 0.0, 2.0 * knee);
        soft = soft * soft / (4.0 * knee + 1e-4);
        result *= max(soft, bright - p.a.y) / max(bright, 1e-4);
    } else {
        result = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    }
    out_color = vec4(max(result, vec3(0.0)), 1.0);
}
