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
} sun;
