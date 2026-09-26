#version 450
#extension GL_GOOGLE_include_directive : require
#include "post_common.glsl"
#include "post_sun.glsl"

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
//        w edge anti-aliasing: how much of a pixel's sub-pixel contrast it
//        blends away (0: off)
//   p.light: x the GE fog's end, y its scale, z 1 when the scene is fogged,
//            w what MHP3RD_EFFECTS_DEBUG shows instead (1 occlusion,
//            2 sunlight and shadows, 3 distance, 4 bloom)
layout(set = 0, binding = 0) uniform sampler2D scene;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
layout(set = 0, binding = 2) uniform sampler2D distances;  // half resolution
layout(set = 0, binding = 3) uniform sampler2D visibility; // half resolution: ambient, light
layout(set = 0, binding = 4) uniform sampler2D bloom;      // half resolution, expanded linear light
layout(set = 0, binding = 7) uniform sampler2D rays;       // quarter resolution: light shafts
layout(set = 0, binding = 8) uniform sampler2D average;    // 1x1: rgb the sky's colour, a how much the sun reaches
layout(set = 0, binding = 10) uniform sampler2D water_mask; // full resolution: how much of the pixel is water
layout(set = 0, binding = 11) uniform sampler2D reflection; // half resolution: rgb the reflection, a how sure
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

// Contrast-adaptive sharpening (after AMD FidelityFX CAS): a cross of taps,
// weighted less where the neighbourhood is already contrasty.
vec3 sharpened(vec3 centre, vec3 n, vec3 s, vec3 w, vec3 e, float amount) {
    if (amount <= 0.0) return centre;
    vec3 lo = min(centre, min(min(n, s), min(w, e)));
    vec3 hi = max(centre, max(max(n, s), max(w, e)));
    vec3 amp = sqrt(clamp(min(lo, 1.0 - hi) / max(hi, vec3(1e-4)), 0.0, 1.0));
    vec3 weight = -amp / mix(8.0, 5.0, amount);
    return clamp((centre + (n + s + w + e) * weight) / (1.0 + 4.0 * weight), 0.0, 1.0);
}

// Edge anti-aliasing on the 8-bit scene, before the interface is drawn: the
// scheme of FXAA's quality preset. Where the luma contrast around the pixel
// is high enough, the edge's direction is taken from the second differences
// across it, the edge is followed both ways to its ends, and the pixel is
// sampled shifted across the edge by how far it sits from the nearer end, as
// the edge's slope would have covered it. Thin features the ends do not find
// are blended by their contrast with the 3x3 average instead.
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float luma_at(vec2 at) { return luma(textureLod(scene, at, 0.0).rgb); }

bool antialiased(vec2 at, float subpixel, float m, float n, float s, float w, float e, out vec3 color);

// The pixel's own colour and its four neighbours, once for both the edge
// test and the sharpening; the corners and the walk along an edge only where
// there is one.
vec3 filtered(vec2 at, float subpixel, float sharpen) {
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 limit = textureSize(scene, 0) - 1;
    vec3 c_m = texelFetch(scene, texel, 0).rgb;
    vec3 c_n = texelFetch(scene, clamp(texel + ivec2(0, -1), ivec2(0), limit), 0).rgb;
    vec3 c_s = texelFetch(scene, clamp(texel + ivec2(0, 1), ivec2(0), limit), 0).rgb;
    vec3 c_w = texelFetch(scene, clamp(texel + ivec2(-1, 0), ivec2(0), limit), 0).rgb;
    vec3 c_e = texelFetch(scene, clamp(texel + ivec2(1, 0), ivec2(0), limit), 0).rgb;
    vec3 color;
    if (subpixel > 0.0 && antialiased(at, subpixel, luma(c_m), luma(c_n), luma(c_s), luma(c_w), luma(c_e), color))
        return color;
    return sharpened(c_m, c_n, c_s, c_w, c_e, sharpen);
}

bool antialiased(vec2 at, float subpixel, float m, float n, float s, float w, float e, out vec3 color) {
    vec2 t = 1.0 / vec2(textureSize(scene, 0));
    float lo = min(m, min(min(n, s), min(w, e)));
    float hi = max(m, max(max(n, s), max(w, e)));
    float range = hi - lo;
    if (range < max(0.0312, hi * 0.125)) return false;
    float nw = luma(textureLodOffset(scene, at, 0.0, ivec2(-1, -1)).rgb);
    float ne = luma(textureLodOffset(scene, at, 0.0, ivec2(1, -1)).rgb);
    float sw = luma(textureLodOffset(scene, at, 0.0, ivec2(-1, 1)).rgb);
    float se = luma(textureLodOffset(scene, at, 0.0, ivec2(1, 1)).rgb);

    // A horizontal edge changes from north to south.
    float across_h = abs(nw - 2.0 * w + sw) + 2.0 * abs(n - 2.0 * m + s) + abs(ne - 2.0 * e + se);
    float across_v = abs(nw - 2.0 * n + ne) + 2.0 * abs(w - 2.0 * m + e) + abs(sw - 2.0 * s + se);
    bool horizontal = across_v >= across_h;
    float before = horizontal ? n : w;
    float after = horizontal ? s : e;
    float step_across = horizontal ? t.y : t.x;
    float gradient_before = abs(before - m);
    float gradient_after = abs(after - m);
    float side;
    if (gradient_before >= gradient_after) {
        step_across = -step_across;
        side = 0.5 * (before + m);
    } else {
        side = 0.5 * (after + m);
    }
    float threshold = 0.25 * max(gradient_before, gradient_after);

    // Along the edge, half a pixel over, until the luma leaves the edge's.
    vec2 on_edge = at + (horizontal ? vec2(0.0, step_across * 0.5) : vec2(step_across * 0.5, 0.0));
    vec2 along = horizontal ? vec2(t.x, 0.0) : vec2(0.0, t.y);
    const float kSteps[10] = float[](1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0, 8.0);
    vec2 end_a = on_edge - along;
    vec2 end_b = on_edge + along;
    float delta_a = luma_at(end_a) - side;
    float delta_b = luma_at(end_b) - side;
    bool done_a = abs(delta_a) >= threshold;
    bool done_b = abs(delta_b) >= threshold;
    for (int i = 1; i < 10 && !(done_a && done_b); ++i) {
        if (!done_a) {
            end_a -= along * kSteps[i];
            delta_a = luma_at(end_a) - side;
            done_a = abs(delta_a) >= threshold;
        }
        if (!done_b) {
            end_b += along * kSteps[i];
            delta_b = luma_at(end_b) - side;
            done_b = abs(delta_b) >= threshold;
        }
    }
    float distance_a = horizontal ? at.x - end_a.x : at.y - end_a.y;
    float distance_b = horizontal ? end_b.x - at.x : end_b.y - at.y;
    bool nearer_a = distance_a < distance_b;
    float shift = 0.5 - min(distance_a, distance_b) / (distance_a + distance_b);
    // Only when the nearer end turns the way this pixel does.
    if (((nearer_a ? delta_a : delta_b) < 0.0) == (m < side)) shift = 0.0;

    float average = (2.0 * (n + s + w + e) + nw + ne + sw + se) / 12.0;
    float blend = clamp(abs(average - m) / range, 0.0, 1.0);
    blend = smoothstep(0.0, 1.0, blend);
    shift = max(shift, blend * blend * subpixel);
    color = textureLod(scene, at + (horizontal ? vec2(0.0, shift * step_across) : vec2(shift * step_across, 0.0)),
                       0.0).rgb;
    return true;
}

// Half-resolution visibility at this pixel: the four neighbours, weighted by
// how close their depth is to this pixel's (a joint bilateral upsample).
vec2 upsampled_visibility(float dist) {
    vec2 at = gl_FragCoord.xy * 0.5 - 0.5;
    vec2 base = floor(at);
    vec2 f = at - base;
    // The four texels around the point, in textureGather's order: (0,1),
    // (1,1), (1,0), (0,0).
    vec2 corner = (base + 1.0) / vec2(textureSize(distances, 0));
    vec4 d = textureGather(distances, corner, 0);
    vec4 ambient = textureGather(visibility, corner, 0);
    vec4 light = textureGather(visibility, corner, 1);
    vec4 bilinear = vec4((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
    vec4 delta = abs(d - dist);
    vec4 w = bilinear * exp(-delta / max(0.05 * dist, 1e-3));
    float total = dot(w, vec4(1.0));
    if (total > 1e-4) return vec2(dot(ambient, w), dot(light, w)) / total;
    // No neighbour at this depth: the closest one.
    float nearest = min(min(delta.x, delta.y), min(delta.z, delta.w));
    int i = nearest == delta.x ? 0 : nearest == delta.y ? 1 : nearest == delta.z ? 2 : 3;
    return vec2(ambient[i], light[i]);
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
    // An edge is smoothed and left unsharpened; the rest is sharpened.
    vec4 base = vec4(filtered(uv, p.c.w, p.a.w), texelFetch(scene, ivec2(gl_FragCoord.xy), 0).a);
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

    // The game's own fog covers what is far away; occlusion fades with it.
    float clear = 1.0;
    vec2 vis = vec2(1.0);
    if (!sky) {
        clear = p.light.z > 0.5 ? clamp((p.light.x - dist) * p.light.y, 0.0, 1.0) : 1.0;
        vis = upsampled_visibility(dist);
        vec3 occluded = multi_bounce(vis.x, clamp(color * 1.3, 0.0, 1.0));
        color *= mix(vec3(1.0), occluded, p.a.x * clear);
        if (sun.params.z <= 0.5) color *= mix(1.0, 1.0 - p.a.y, (1.0 - vis.y) * clear);
    }

    // Expand, add light, map back.
    float k = p.c.z;
    float peak = max(color.r, max(color.g, color.b));
    color = scale_peak(color, expand(min(peak, 0.999), k)) * p.b.x;
    if (sun.params.z > 0.5 && sun.color.w > 0.0 && sun.rays.x > 0.0) {
        // The air towards the sun glows: over the sky and what is far, as
        // the haze the sun lights (forward scattering), wide and faint with
        // a brighter core. Near things are left to the shafts.
        vec3 ray = normalize(view_position(gl_FragCoord.xy, 1.0));
        float towards = max(dot(ray, sun.direction.xyz), 0.0);
        float far = sky ? 1.0 : smoothstep(sun.params.w * 1.2, sun.params.w * 3.0, dist);
        float glow = pow(towards, 6.0) * 0.22 + pow(towards, 48.0) * 0.45;
        color += sun.color.rgb * (glow * far * sun.color.w * sun.rays.x * 5.0);
    }
    if (!sky && sun.water.w > 0.5) {
        float water = texelFetch(water_mask, ivec2(gl_FragCoord.xy), 0).r;
        if (water > 0.0) {
            // The rippled surface mirrors the scene (or the sky where the
            // reflection finds nothing) by Schlick's Fresnel term, more at
            // grazing angles, and the sun glints on it where it reaches.
            vec3 position = view_position(gl_FragCoord.xy, dist);
            vec3 world = (sun.view_to_world * vec4(position, 1.0)).xyz;
            vec3 n = water_normal(world);
            vec3 v = normalize(-position);
            float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
            vec4 found = texture(reflection, uv);
            vec3 sky_light = texelFetch(average, ivec2(0), 0).rgb;
            vec3 mirrored = mix(sky_light * 1.1, found.rgb, found.a);
            float amount = clamp(fresnel * 1.6 + 0.12, 0.0, 1.0) * water * sun.water.x * clear;
            color = mix(color, mirrored, amount);
            vec3 h = normalize(sun.direction.xyz + v);
            float glint = pow(max(dot(n, h), 0.0), 600.0) * 60.0 + pow(max(dot(n, h), 0.0), 60.0) * 0.6;
            color += sun.color.rgb * (glint * fresnel * 4.0 * vis.y * water * sun.water.x * clear * sun.color.w);
        }
    }
    if (!sky && sun.params.z > 0.5) {
        // Sunlight over the game's own lighting, as light added before the
        // shoulder: lit surfaces take on more light of the sun's colour,
        // shadowed ones fall towards the cooler light of the sky. Beyond the
        // shadow map it fades out, and the sky's dome and the far hills keep
        // the game's own light.
        float near = 1.0 - smoothstep(sun.params.w * 1.1, sun.params.w * 2.2, dist);
        vec3 lit = vec3(1.0) + sun.color.rgb * sun.color.w;
        // Where the sun reaches little of the view, the shade eases; its
        // tint leans towards the sky's own colour, which fills shadows.
        vec4 around = texelFetch(average, ivec2(0), 0);
        float adapted = smoothstep(0.04, 0.35, around.a);
        float sky_luma = luminance(around.rgb);
        vec3 tint = sky_luma > 0.02 ? mix(sun.shade.rgb, around.rgb / sky_luma, 0.5) : sun.shade.rgb;
        vec3 shaded = tint * mix(1.0 - (1.0 - sun.shade.w) * 0.35, sun.shade.w, adapted);
        color *= mix(vec3(1.0), mix(shaded, lit, vis.y), clear * near);
    }
    color += texture(bloom, uv).rgb * p.a.z;
    if (sun.rays.x > 0.0 && sun.direction.w > 0.5) {
        // Four bilinear taps around the pixel smooth the march's dither.
        vec2 t = 0.75 / vec2(textureSize(rays, 0));
        float shafts = 0.25 * (texture(rays, uv + vec2(-t.x, -t.y)).r + texture(rays, uv + vec2(t.x, -t.y)).r +
                               texture(rays, uv + vec2(-t.x, t.y)).r + texture(rays, uv + vec2(t.x, t.y)).r);
        color += sun.color.rgb * (shafts * sun.rays.x);
    }
    color = grade(max(color, vec3(0.0)));
    peak = max(color.r, max(color.g, color.b));
    color = scale_peak(max(color, vec3(0.0)), shoulder(peak, k));

    // A vignette to the window's shape.
    vec2 centred = (uv - 0.5) * vec2(p.viewport.z / p.viewport.w, 1.0);
    color *= 1.0 - p.c.x * smoothstep(0.4, 1.15, length(centred));

    vec3 srgb = to_srgb(color);
    // Triangular dither of one 8-bit step, to hide banding.
    float n0 = ign(gl_FragCoord.xy);
    float n1 = ign(gl_FragCoord.yx * 1.37 + 17.0);
    srgb += (n0 + n1 - 1.0) / 255.0;
    out_color = vec4(clamp(srgb, 0.0, 1.0), base.a);
}
