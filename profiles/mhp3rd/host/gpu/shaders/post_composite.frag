#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"

// The finished 3D scene with everything applied, written back into the
// game's framebuffer before its interface is drawn over it.
//
// The game's picture is 8-bit and made to be shown as it is, so nothing here
// tone-maps it: the brightest part of its range is expanded by an invertible
// shoulder, light is added (bloom), and the same shoulder maps it back, so a
// pixel nothing touched comes out as it went in and only added light rolls
// off.
//
//   p.a: x ambient occlusion strength, y contact shadow strength, z bloom
//        strength, w sharpening (0 to 1)
//   p.b: x exposure, y contrast, z saturation, w vibrance
//   p.c: x vignette, y warm/cool split toning, z shoulder knee (linear),
//        w frame noise offset
//   p.light: x the GE fog's end, y its scale, z 1 when the scene is fogged,
//            w what MHP3RD_EFFECTS_DEBUG shows instead (1 occlusion,
//            2 contact shadows, 3 distance, 4 bloom)
layout(set = 0, binding = 0) uniform sampler2D scene;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
layout(set = 0, binding = 2) uniform sampler2D distances;  // half resolution
layout(set = 0, binding = 3) uniform sampler2D visibility; // half resolution: ambient, light
layout(set = 0, binding = 4) uniform sampler2D bloom;      // half resolution, expanded linear light
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// Contrast-adaptive sharpening (after AMD FidelityFX CAS): a cross of taps,
// weighted less where the neighbourhood is already contrasty.
vec4 sharpened(vec2 at, float amount) {
    vec4 centre = texture(scene, at);
    if (amount <= 0.0) return centre;
    vec2 t = 1.0 / vec2(textureSize(scene, 0));
    vec3 n = texture(scene, at + vec2(0.0, -t.y)).rgb;
    vec3 s = texture(scene, at + vec2(0.0, t.y)).rgb;
    vec3 w = texture(scene, at + vec2(-t.x, 0.0)).rgb;
    vec3 e = texture(scene, at + vec2(t.x, 0.0)).rgb;
    vec3 lo = min(centre.rgb, min(min(n, s), min(w, e)));
    vec3 hi = max(centre.rgb, max(max(n, s), max(w, e)));
    vec3 amp = sqrt(clamp(min(lo, 1.0 - hi) / max(hi, vec3(1e-4)), 0.0, 1.0));
    vec3 weight = -amp / mix(8.0, 5.0, amount);
    vec3 rgb = (centre.rgb + (n + s + w + e) * weight) / (1.0 + 4.0 * weight);
    return vec4(clamp(rgb, 0.0, 1.0), centre.a);
}

// Half-resolution visibility at this pixel: the four neighbours, weighted by
// how close their depth is to this pixel's (a joint bilateral upsample).
vec2 upsampled_visibility(float dist) {
    vec2 at = gl_FragCoord.xy * 0.5 - 0.5;
    ivec2 base = ivec2(floor(at));
    vec2 f = at - vec2(base);
    ivec2 limit = textureSize(distances, 0) - 1;
    vec2 sum = vec2(0.0);
    float total = 0.0;
    vec2 closest = vec2(1.0);
    float closest_delta = 1e30;
    for (int i = 0; i < 4; ++i) {
        ivec2 o = ivec2(i & 1, i >> 1);
        ivec2 texel = clamp(base + o, ivec2(0), limit);
        float bilinear = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        float delta = abs(texelFetch(distances, texel, 0).r - dist);
        vec2 v = texelFetch(visibility, texel, 0).rg;
        float w = bilinear * exp(-delta / max(0.05 * dist, 1e-3));
        sum += v * w;
        total += w;
        if (delta < closest_delta) {
            closest_delta = delta;
            closest = v;
        }
    }
    return total > 1e-4 ? sum / total : closest;
}

// GTAO's fit of occlusion with interreflections: bright surfaces lose less.
vec3 multi_bounce(float v, vec3 albedo) {
    vec3 a = 2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c = 2.7552 * albedo + 0.6903;
    return max(vec3(v), ((v * a + b) * v + c) * v);
}

// The invertible shoulder, on the brightest channel so hues stay: identity
// to the knee, above it y = k + (1-k)(x-k)/((x-k)+(1-k)).
float shoulder(float x, float k) { return x <= k ? x : k + (1.0 - k) * (x - k) / ((x - k) + (1.0 - k)); }
float expand(float y, float k) { return y <= k ? y : k + (1.0 - k) * (y - k) / max(1.0 - y, 1.0 / 256.0); }
vec3 scale_peak(vec3 c, float to) {
    float peak = max(c.r, max(c.g, c.b));
    return peak > 1e-5 ? c * (to / peak) : c;
}

// A grade that leaves the art's colours where they are: contrast about
// middle grey on luminance (so hues stay), saturation and vibrance (more for
// muted colours), and a light cool-shadow / warm-light split.
vec3 grade(vec3 color) {
    float luma = max(luminance(color), 1e-5);
    float graded = 0.18 * pow(luma / 0.18, p.b.y);
    color *= graded / luma;
    luma = graded;
    float hi = max(color.r, max(color.g, color.b));
    float lo = min(color.r, min(color.g, color.b));
    float saturation = hi > 1e-5 ? (hi - lo) / hi : 0.0;
    color = mix(vec3(luma), color, p.b.z * (1.0 + p.b.w * (1.0 - saturation)));
    float tone = smoothstep(0.02, 0.6, sqrt(luma));
    vec3 tint = mix(vec3(0.975, 0.99, 1.035), vec3(1.03, 1.0, 0.965), tone);
    return color * mix(vec3(1.0), tint, p.c.y);
}

void main() {
    vec4 base = sharpened(uv, p.a.w);
    vec3 color = to_linear(base.rgb);
    float dist = view_distance(texelFetch(depth_buffer, ivec2(gl_FragCoord.xy), 0).r);
    bool sky = dist >= kSky * 0.5;
    int debug = int(p.light.w + 0.5);
    if (debug != 0) {
        vec2 vis = sky ? vec2(1.0) : upsampled_visibility(dist);
        vec3 shown = debug == 1 ? vec3(vis.x)
                   : debug == 2 ? vec3(vis.y)
                   : debug == 3 ? vec3(sky ? 0.0 : fract(log2(dist) * 0.5))
                                : to_srgb(texture(bloom, uv).rgb * 0.25);
        out_color = vec4(shown, base.a);
        return;
    }

    if (!sky) {
        // The game's own fog covers what is far away; occlusion fades with it.
        float clear = p.light.z > 0.5 ? clamp((p.light.x - dist) * p.light.y, 0.0, 1.0) : 1.0;
        vec2 vis = upsampled_visibility(dist);
        vec3 occluded = multi_bounce(vis.x, clamp(color * 1.3, 0.0, 1.0));
        color *= mix(vec3(1.0), occluded, p.a.x * clear);
        color *= mix(1.0, 1.0 - p.a.y, (1.0 - vis.y) * clear);
    }

    // Expand, add light, map back.
    float k = p.c.z;
    float peak = max(color.r, max(color.g, color.b));
    color = scale_peak(color, expand(min(peak, 0.999), k)) * p.b.x;
    color += texture(bloom, uv).rgb * p.a.z;
    color = grade(max(color, vec3(0.0)));
    peak = max(color.r, max(color.g, color.b));
    color = scale_peak(max(color, vec3(0.0)), shoulder(peak, k));

    // A vignette to the window's shape.
    vec2 centred = (uv - 0.5) * vec2(p.viewport.z / p.viewport.w, 1.0);
    color *= 1.0 - p.c.x * smoothstep(0.4, 1.15, length(centred));

    vec3 srgb = to_srgb(color);
    // Triangular dither of one 8-bit step, to hide banding.
    float n0 = ign(gl_FragCoord.xy + p.c.w);
    float n1 = ign(gl_FragCoord.yx * 1.37 + 17.0 + p.c.w);
    srgb += (n0 + n1 - 1.0) / 255.0;
    out_color = vec4(clamp(srgb, 0.0, 1.0), base.a);
}
