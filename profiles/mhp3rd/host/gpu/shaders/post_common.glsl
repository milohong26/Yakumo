// Shared by the post-processing passes (post_process.cpp): the scene camera,
// as the game's draws into the displayed framebuffer set it, and helpers to
// turn the depth buffer back into view-space positions.

layout(push_constant) uniform Params {
    vec4 proj;      // P[0], P[5], P[8], P[9] of the scene's projection (column major)
    vec4 depth;     // P[10], P[14], P[11], 0
    vec4 range;     // the viewport's minDepth, maxDepth; x, y of the viewport in target pixels
    vec4 viewport;  // width, height of the viewport in target pixels (height < 0: flipped); target width, height
    vec4 light;     // view-space direction towards the key light; w: its strength (0: none)
    vec4 a;         // per pass
    vec4 b;
    vec4 c;
} p;

const float kSky = 1.0e6;

// View-space distance (positive) of a depth-buffer value; kSky for the
// background and anything at or behind the far plane.
float view_distance(float d) {
    float span = p.range.y - p.range.x;
    if (abs(span) < 1.0e-8) return kSky;
    float z01 = (d - p.range.x) / span;
    if (z01 >= 0.99999 || z01 <= -0.5) return kSky;
    float ndc = z01 * 2.0 - 1.0;
    // clip.z = P10 z + P14, clip.w = P11 z (P11 = -1 for a GL projection).
    float z = p.depth.y / (p.depth.z * ndc - p.depth.x);
    return z < 0.0 ? -z : kSky;
}

// The view-space position of target pixel `pixel` (a pixel centre) at
// distance `dist`.
vec3 view_position(vec2 pixel, float dist) {
    vec2 ndc = (pixel - p.range.zw) / p.viewport.xy * 2.0 - 1.0;
    float z = -dist;
    float x = z * (p.depth.z * ndc.x - p.proj.z) / p.proj.x;
    float y = z * (p.depth.z * ndc.y - p.proj.w) / p.proj.y;
    return vec3(x, y, z);
}

// Target pixel of a view-space position.
vec2 target_pixel(vec3 v) {
    float w = p.depth.z * v.z;
    vec2 ndc = vec2((p.proj.x * v.x + p.proj.z * v.z) / w, (p.proj.y * v.y + p.proj.w * v.z) / w);
    return p.range.zw + (ndc * 0.5 + 0.5) * p.viewport.xy;
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// sRGB curves by polynomial fits (Ian Taylor), within about 0.1% and far
// cheaper than pow on every channel of every pixel.
vec3 to_linear(vec3 c) { return c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878); }
vec3 to_srgb(vec3 c) {
    c = max(c, vec3(0.0));
    vec3 s1 = sqrt(c);
    vec3 s2 = sqrt(s1);
    vec3 s3 = sqrt(s2);
    return 0.585122381 * s1 + 0.783140355 * s2 - 0.368262736 * s3;
}

// Interleaved gradient noise (Jimenez 2014).
float ign(vec2 pixel) { return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715)))); }
