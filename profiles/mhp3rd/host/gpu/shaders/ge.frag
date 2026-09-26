#version 450

layout(location = 0) in vec2 frag_texcoord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in vec3 frag_specular;
layout(location = 3) in float frag_fog;
layout(location = 4) flat in vec4 frag_uv_rect;
layout(location = 5) in vec3 frag_normal;
layout(location = 6) in vec3 frag_to_eye;
layout(location = 7) in vec3 frag_ambient;
layout(location = 8) in vec3 frag_diffuse;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D guest_texture;

// False for pipelines of draws without an alpha test (texture_params.w 0):
// the discard below is then compiled out, so tiled GPUs keep their early
// depth test and hidden surface removal for them.
layout(constant_id = 0) const bool kAlphaTest = true;

layout(push_constant) uniform Push {
    mat4 transform;
    vec4 viewport;
    vec4 texture_params; // x: texture enabled, y: texture function, z: alpha ref, w: alpha func
    vec4 uv_transform;
    vec4 view_z;
} push;

// Both blocks are described in ge.vert.
layout(set = 1, binding = 0) uniform Environment {
    vec4 ambient;
    vec4 fog;
    vec4 fog_color;
    vec4 light_position[4];
    vec4 light_direction[4];
    vec4 light_attenuation[4];
    vec4 light_spot[4];
    vec4 light_ambient[4];
    vec4 light_diffuse[4];
    vec4 light_specular[4];
} lighting;
layout(set = 1, binding = 1) uniform Object {
    mat4 world;
    vec4 flags;
    vec4 emissive;
    vec4 material_ambient;
    vec4 material_diffuse;
    vec4 material_specular;
    vec4 eye;
    vec4 enhance;
} object;

// Per-pixel lighting of a lit draw, for the remaster look. The GE's own
// terms, evaluated at the pixel instead of the vertex: its directional lights'
// diffuse light (wrapped a little around the terminator, as skin and cloth
// scatter it) and the ambient light, brighter from above than below as a sky
// and the ground would give it. On top of them and after texturing, what the
// GE has no way to show: a soft highlight of each light, normalised
// Blinn-Phong with a dielectric Fresnel term, and a rim of sky light at
// grazing angles that sets models off from what is behind them.
struct Lit {
    vec3 diffuse;
    vec3 specular;
    vec3 rim;
};
Lit light_pixel() {
    vec3 n = frag_normal;
    float length_squared = dot(n, n);
    n = length_squared > 1e-12 ? n * inversesqrt(length_squared) : vec3(0.0, 0.0, 1.0);
    vec3 v = normalize(frag_to_eye + vec3(0.0, 1e-6, 0.0));
    float power = max(object.eye.w, 1.0);
    float wrap = object.enhance.z;
    float ground = object.enhance.w;
    float up = n.y * 0.5 + 0.5;
    Lit lit;
    lit.diffuse = frag_color.rgb + frag_ambient * mix(ground, 2.0 - ground, up);
    lit.specular = vec3(0.0);
    vec3 sky = vec3(0.0);
    vec3 back = vec3(0.0);
    float lights = 0.0;
    float grazing_edge = 1.0 - max(dot(n, v), 0.0);
    float edge3 = grazing_edge * grazing_edge * grazing_edge;
    for (int i = 0; i < 4; ++i) {
        if (lighting.light_position[i].w < 0.5 || int(lighting.light_direction[i].w + 0.5) != 0) continue;
        vec3 l = lighting.light_position[i].xyz;
        l = dot(l, l) > 0.0 ? normalize(l) : vec3(0.0, 0.0, 1.0);
        vec3 color = lighting.light_diffuse[i].rgb;
        float n_dot_l = dot(n, l);
        float diffuse = int(lighting.light_attenuation[i].w + 0.5) == 2
                            ? pow(max(n_dot_l, 0.0), object.emissive.w)
                            : max((n_dot_l + wrap) / (1.0 + wrap), 0.0);
        lit.diffuse += color * frag_diffuse * diffuse;
        if (n_dot_l > 0.0) {
            vec3 h = normalize(l + v);
            float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(v, h), 0.0), 5.0);
            lit.specular += color * (fresnel * (power + 8.0) / 8.0 * pow(max(dot(n, h), 0.0), power) * n_dot_l);
        }
        sky += color;
        // A light behind the model, as the camera sees it, outlines its
        // silhouette (a back light's rim).
        float behind = max(dot(-v, l), 0.0);
        back += color * (edge3 * behind * behind);
        lights += 1.0;
    }
    // The GE clamps light at 1, so wherever two of its lights meet a model
    // goes flat; past the knee, light rolls off towards 1 instead (an
    // exponential shoulder, continuous in slope), which keeps the shape.
    float knee = object.flags.z;
    lit.diffuse = max(lit.diffuse, vec3(0.0));
    if (knee > 0.0 && knee < 1.0) {
        vec3 over = max(lit.diffuse - knee, vec3(0.0));
        lit.diffuse = mix(lit.diffuse, 1.0 - (1.0 - knee) * exp(-over / (1.0 - knee)),
                          step(vec3(knee), lit.diffuse));
    }
    lit.diffuse = min(lit.diffuse, vec3(1.0));
    lit.specular *= object.enhance.x;
    float grazing = 1.0 - max(dot(n, v), 0.0);
    grazing *= grazing;
    lit.rim = sky / max(lights, 1.0) * (grazing * grazing * object.enhance.y * (0.35 + 0.65 * up)) +
              back * (object.enhance.y * 1.4);
    return lit;
}

void main() {
    vec4 color = frag_color;
    int enables = int(push.viewport.w + 0.5);
    bool per_pixel = (enables & 4) != 0;
    Lit lit = Lit(vec3(0.0), vec3(0.0), vec3(0.0));
    if (per_pixel) {
        lit = light_pixel();
        color.rgb = lit.diffuse;
    }
    vec4 texel = vec4(1.0);
    if (push.texture_params.x > 0.5) {
        texel = texture(guest_texture, clamp(frag_texcoord, frag_uv_rect.xy, frag_uv_rect.zw));
        int function = int(push.texture_params.y + 0.5);
        if (function == 0) {          // modulate
            color *= texel;
        } else if (function == 1) {   // decal
            color = vec4(mix(color.rgb, texel.rgb, texel.a), color.a);
        } else if (function == 2) {   // blend
            color = vec4(mix(color.rgb, texel.rgb, texel.rgb), color.a * texel.a);
        } else {                      // replace and everything else
            color = texel;
        }
    }

    // A separate specular term is added after texturing, then fog blends
    // towards its colour; neither touches alpha.
    color.rgb = min(color.rgb + frag_specular, vec3(1.0));
    // The highlight is the light's own colour; the rim takes on some of the
    // surface's.
    if (per_pixel) color.rgb = min(color.rgb + lit.specular + lit.rim * mix(vec3(1.0), texel.rgb, 0.5), vec3(1.0));
    if ((enables & 1) != 0) color.rgb = mix(lighting.fog_color.rgb, color.rgb, clamp(frag_fog, 0.0, 1.0));

    // PSP alpha test, evaluated per fragment.
    if (!kAlphaTest) {
        out_color = color;
        return;
    }
    int alpha_function = int(push.texture_params.w + 0.5);
    float reference = push.texture_params.z / 255.0;
    float alpha = color.a;
    bool passed = true;
    if (alpha_function == 1) passed = false;                     // never
    else if (alpha_function == 2) passed = abs(alpha - reference) < 0.002;
    else if (alpha_function == 3) passed = abs(alpha - reference) >= 0.002;
    else if (alpha_function == 4) passed = alpha < reference;
    else if (alpha_function == 5) passed = alpha <= reference;
    else if (alpha_function == 6) passed = alpha > reference;
    else if (alpha_function == 7) passed = alpha >= reference;
    if (!passed) discard;

    out_color = color;
}
