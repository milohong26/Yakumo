// The sun (post_process.hpp): its direction, colour and shadow map, for the
// passes that light the scene with it.
layout(set = 0, binding = 5) uniform sampler2DShadow shadow_map;
layout(set = 0, binding = 9) uniform sampler2D shadow_depths;  // the same map, its depths
layout(set = 0, binding = 6) uniform Sun {
    mat4 view_to_shadow;  // view space to the shadow map: xy its coordinates, z its depth
    vec4 direction;       // view space, towards the sun; w: 1 when the shadow map has this scene
    vec4 color;           // rgb: sunlight's tint; w: how much it adds to a lit surface
    vec4 shade;           // rgb: the tint in shadow; w: how bright a shadowed surface stays
    vec4 params;          // x: a shadow map texel, y: world units per texel, z: 1 sun on, w: shadow range
    vec4 rays;            // x: light shaft strength, y: their reach, z: forward scattering, w: penumbra per unit
    mat4 view_to_world;   // for the water's ripples, which move in the world
    vec4 water;           // x: reflection strength, y: ripples, z: time in seconds, w: 1 when water was drawn
    vec4 clouds;          // x: how much their shadows darken, y: cover (0 to 1), z: size in world units,
                          // w: how much bright light lights the surfaces around it
} sun;

float cloud_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float cloud_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(cloud_hash(i), cloud_hash(i + vec2(1.0, 0.0)), f.x),
               mix(cloud_hash(i + vec2(0.0, 1.0)), cloud_hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// How much of the sun the clouds let through at a view-space position: a
// drifting layer of soft noise high above the world, looked up along the
// sun's direction, so its shadows lie over hills and valleys as a cloud's.
float cloud_light(vec3 position) {
    if (sun.clouds.x <= 0.0) return 1.0;
    vec3 world = (sun.view_to_world * vec4(position, 1.0)).xyz;
    vec3 to_sun = normalize(mat3(sun.view_to_world) * sun.direction.xyz);
    vec2 at = world.xz + to_sun.xz / max(to_sun.y, 0.2) * (2500.0 - world.y);
    at += vec2(38.0, 14.0) * sun.water.z;
    vec2 p = at / sun.clouds.z;
    float n = cloud_noise(p) * 0.6 + cloud_noise(p * 2.3 + 17.0) * 0.3 + cloud_noise(p * 5.1 + 3.0) * 0.1;
    float cover = sun.clouds.y;
    float density = smoothstep(1.0 - cover, 1.0 - cover + 0.3, n);
    return 1.0 - sun.clouds.x * density;
}

// The water's surface normal in view space at a world position: the world's
// up, bent by a few moving waves of different lengths and directions.
vec3 water_normal(vec3 world) {
    float t = sun.water.z;
    vec2 p = world.xz;
    vec2 slope = vec2(0.0);
    // (direction, wavelength in world units, speed, height) for each wave.
    const vec4 waves[4] = vec4[](vec4(0.83, 0.56, 90.0, 0.9), vec4(-0.45, 0.89, 51.0, 1.3),
                                 vec4(0.12, -0.99, 29.0, 1.9), vec4(-0.97, -0.24, 17.0, 2.6));
    const float heights[4] = float[](1.0, 0.7, 0.45, 0.3);
    for (int i = 0; i < 4; ++i) {
        vec2 d = waves[i].xy;
        float k = 6.2831853 / waves[i].z;
        float phase = dot(d, p) * k + t * waves[i].w;
        slope += d * (cos(phase) * heights[i]);
    }
    vec3 n = normalize(vec3(-slope.x * 0.06 * sun.water.y, 1.0, -slope.y * 0.06 * sun.water.y));
    // The view is a rotation and a move: its inverse rotation is the transpose.
    return normalize(transpose(mat3(sun.view_to_world)) * n);
}

