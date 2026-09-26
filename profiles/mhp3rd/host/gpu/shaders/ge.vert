#version 450

// PSP geometry. Transformed vertices arrive in object space and are multiplied
// by the combined world-view-projection matrix; "through" vertices are already
// in screen pixels and are mapped to clip space with the viewport size.
#ifndef GE_RAW_VERTICES
layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_texcoord;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec3 in_normal;
#endif

#ifdef GE_RAW_VERTICES
// GPU vertex decode (kGeRawVertexShader): a transformed draw's vertices are
// the guest's own bytes, and this decodes and skins them exactly as
// decode_vertices() in ge_state.cpp does on the CPU, into the four values the
// other build reads from vertex attributes.
vec4 in_position;
vec2 in_texcoord;
vec4 in_color;
vec3 in_normal;

// The draw's vertex format and bones; the layout matches RawBlock on the host.
layout(set = 1, binding = 2) uniform Raw {
    uvec4 format;  // x: stride, y: vertex type, z: weight | texcoord << 8 | colour << 16 | normal << 24 offsets,
                   // w: position offset (0xFF: a field the type lacks)
    uvec4 extra;   // x: the colour a vertex without one gets (0xAABBGGRR), y: check slot
    vec4 bones[32]; // four rows (x, y, z, translation) per bone, as the GE uploads them
} raw;

// The vertex buffer, as words: the draw's bytes start at gl_InstanceIndex
// (the draw is recorded with that first instance).
layout(std430, set = 1, binding = 3) readonly buffer RawBytes {
    uint raw_words[];
};

#ifdef GE_CHECK_DECODE
// MHP3RD_CHECK_GPU_DECODE (kGeCheckVertexShader): the decoded values of each
// vertex, for the host to compare with its own decode.
layout(std430, set = 1, binding = 4) writeonly buffer CheckOut {
    vec4 check_out[];
};
#endif

uint raw_byte(uint at) { return (raw_words[at >> 2u] >> ((at & 3u) * 8u)) & 0xFFu; }
uint raw_half(uint at) { return (raw_words[at >> 2u] >> ((at & 2u) * 8u)) & 0xFFFFu; }
uint raw_word(uint at) { return raw_words[at >> 2u]; }

// A component of format `type` (1: 8 bits, 2: 16 bits, 3: float).
float raw_signed(uint at, uint type) {
    if (type == 1u) return float(int(raw_byte(at) << 24u) >> 24);
    if (type == 2u) return float(int(raw_half(at) << 16u) >> 16);
    return uintBitsToFloat(raw_word(at));
}
float raw_unsigned(uint at, uint type) {
    if (type == 1u) return float(raw_byte(at));
    if (type == 2u) return float(raw_half(at));
    return uintBitsToFloat(raw_word(at));
}
float type_scale(uint type) { return type == 1u ? 1.0 / 128.0 : (type == 2u ? 1.0 / 32768.0 : 1.0); }
uint type_size(uint type) { return type == 3u ? 4u : type; }

// As a vertex attribute of format R8G8B8A8_UNORM reads it.
vec4 unpack_rgba8(uint value) { return unpackUnorm4x8(value); }

void decode_raw_vertex() {
    const uint stride = raw.format.x;
    const uint type = raw.format.y;
    const uint base = uint(gl_InstanceIndex) + uint(gl_VertexIndex) * stride;
    const uint texcoord_type = type & 3u;
    const uint color_type = (type >> 2u) & 7u;
    const uint normal_type = (type >> 5u) & 3u;
    const uint position_type = (type >> 7u) & 3u;
    const uint weight_type = (type >> 9u) & 3u;
    const uint weight_count = ((type >> 14u) & 7u) + 1u;

    in_texcoord = vec2(0.0);
    if (texcoord_type != 0u) {
        const uint at = base + ((raw.format.z >> 8u) & 0xFFu);
        const float scale = type_scale(texcoord_type);
        in_texcoord = vec2(raw_unsigned(at, texcoord_type) * scale,
                           raw_unsigned(at + type_size(texcoord_type), texcoord_type) * scale);
    }
    in_color = unpack_rgba8(raw.extra.x);
    if (color_type != 0u) {
        const uint at = base + ((raw.format.z >> 16u) & 0xFFu);
        const uint value = color_type == 7u ? raw_word(at) : raw_half(at);
        // The integer expansion of expand_color() in ge_state.cpp.
        uint r, g, b, a;
        if (color_type == 4u) {
            r = (value & 0x1Fu) * 255u / 31u;
            g = ((value >> 5u) & 0x3Fu) * 255u / 63u;
            b = ((value >> 11u) & 0x1Fu) * 255u / 31u;
            a = 255u;
        } else if (color_type == 5u) {
            r = (value & 0x1Fu) * 255u / 31u;
            g = ((value >> 5u) & 0x1Fu) * 255u / 31u;
            b = ((value >> 10u) & 0x1Fu) * 255u / 31u;
            a = ((value >> 15u) & 1u) * 255u;
        } else if (color_type == 6u) {
            r = (value & 0xFu) * 17u;
            g = ((value >> 4u) & 0xFu) * 17u;
            b = ((value >> 8u) & 0xFu) * 17u;
            a = ((value >> 12u) & 0xFu) * 17u;
        } else {
            r = value & 0xFFu;
            g = (value >> 8u) & 0xFFu;
            b = (value >> 16u) & 0xFFu;
            a = value >> 24u;
        }
        in_color = unpack_rgba8(r | g << 8u | b << 16u | a << 24u);
    }
    in_normal = vec3(0.0);
    if (normal_type != 0u) {
        const uint at = base + (raw.format.z >> 24u);
        const uint size = type_size(normal_type);
        const float scale = type_scale(normal_type);
        in_normal = vec3(raw_signed(at, normal_type), raw_signed(at + size, normal_type),
                         raw_signed(at + 2u * size, normal_type)) * scale;
    }
    {
        const uint at = base + raw.format.w;
        const uint size = type_size(position_type);
        const float scale = type_scale(position_type);
        in_position = vec4(vec3(raw_signed(at, position_type), raw_signed(at + size, position_type),
                                raw_signed(at + 2u * size, position_type)) * scale, 1.0);
    }
    if (weight_type != 0u) {
        // Skinned: blended into the bones' space by the weights, which is
        // what the world matrix then sees.
        const uint at = base + (raw.format.z & 0xFFu);
        const uint size = type_size(weight_type);
        const float scale = type_scale(weight_type);
        vec3 position = vec3(0.0);
        vec3 normal = vec3(0.0);
        for (uint bone = 0u; bone < min(weight_count, 8u); ++bone) {
            const float weight = raw_unsigned(at + bone * size, weight_type) * scale;
            if (weight == 0.0) continue;
            const vec3 x = raw.bones[bone * 4u].xyz;
            const vec3 y = raw.bones[bone * 4u + 1u].xyz;
            const vec3 z = raw.bones[bone * 4u + 2u].xyz;
            const vec3 t = raw.bones[bone * 4u + 3u].xyz;
            position += weight * (in_position.x * x + in_position.y * y + in_position.z * z + t);
            normal += weight * (in_normal.x * x + in_normal.y * y + in_normal.z * z);
        }
        in_position.xyz = position;
        in_normal = normal;
    }
#ifdef GE_CHECK_DECODE
    const uint slot = raw.extra.y + uint(gl_VertexIndex) * 3u;
    check_out[slot] = vec4(in_position.xyz, in_texcoord.x);
    check_out[slot + 1u] = vec4(in_normal, in_texcoord.y);
    check_out[slot + 2u] = in_color;
#endif
}
#endif

layout(location = 0) out vec2 frag_texcoord;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out vec3 frag_specular;
layout(location = 3) out float frag_fog;
// Through-mode tiles: the texture coordinates the tile may sample, as min.xy,
// max.xy. The vertex carries them in texels, in in_normal.xy and
// (in_normal.z, in_position.w), which through-mode vertices do not use
// otherwise; see clamp_through_quads() in vulkan_renderer.cpp.
layout(location = 4) flat out vec4 frag_uv_rect;
// Per-pixel lighting (push.viewport.w bit 4): what the fragment shader needs
// to light the pixel itself. frag_color then holds only the light that does
// not depend on the normal (emissive, point and spot lights).
layout(location = 5) out vec3 frag_normal;   // world space
layout(location = 6) out vec3 frag_to_eye;   // world space, towards the camera
layout(location = 7) out vec3 frag_ambient;  // ambient light times the ambient material
layout(location = 8) out vec3 frag_diffuse;  // the diffuse material

layout(push_constant) uniform Push {
    mat4 transform;      // WVP, or identity for through vertices
    vec4 viewport;       // xy: target size in PSP pixels (3D with wind: time, sway), z: through,
                         // w: 1 fog + 2 lighting + 4 per pixel + 8 wind
    vec4 texture_params; // x: texture enabled, y: texture function, z: alpha ref, w: alpha func
    vec4 uv_transform;   // xy: scale, zw: offset
    vec4 view_z;         // row of view * world that gives view-space z
} push;

// Leaves, grass and cloth cut out of their quads sway in the wind: more the
// higher they stand above their model's origin (where plants are rooted),
// in gusts that roll across the model.
vec3 swayed(vec3 position) {
    float height = clamp(position.y / 90.0, 0.0, 1.6);
    float t = push.viewport.x;
    float phase = dot(position.xz, vec2(0.021, 0.013));
    float gust = sin(t * 1.3 + phase) * 0.7 + sin(t * 3.7 + phase * 2.9) * 0.3;
    float flutter = sin(t * 9.0 + dot(position, vec3(0.31, 0.17, 0.23))) * 0.25;
    return position + vec3(1.0, 0.0, 0.55) * ((gust + flutter) * height * height * push.viewport.y);
}

// The lighting environment, shared by every draw until the game changes it;
// the layout matches EnvironmentBlock on the host. Colours are 0..1; small
// integers are stored as floats.
layout(set = 1, binding = 0) uniform Environment {
    vec4 ambient;           // global ambient light, rgba
    vec4 fog;               // x: end, y: scale
    vec4 fog_color;
    vec4 light_position[4];    // xyz; w: enabled
    vec4 light_direction[4];   // xyz; w: type (0 directional, 1 point, 2 spot)
    vec4 light_attenuation[4]; // xyz: constant, linear, quadratic; w: kind
    vec4 light_spot[4];        // x: exponent, y: cutoff
    vec4 light_ambient[4];
    vec4 light_diffuse[4];
    vec4 light_specular[4];
} lighting;

// A lit draw's world matrix and material; the layout matches ObjectBlock.
layout(set = 1, binding = 1) uniform Object {
    mat4 world;
    vec4 flags;             // y: vertex has a colour, z: per-pixel light's knee, w: material update mask
    vec4 emissive;          // rgb; w: specular power
    vec4 material_ambient;  // rgba
    vec4 material_diffuse;  // rgb; w: 1 keeps specular apart
    vec4 material_specular; // rgb; w: reverse normals
    vec4 eye;               // xyz: the camera in world space; w: highlight power
    vec4 enhance;           // x: highlight, y: rim, z: wrap, w: ground ambient
} object;

// The GE's per-vertex lighting, evaluated in world space: emissive, plus the
// global ambient light times the material ambient, plus for each enabled light
// its ambient, diffuse and specular terms, scaled by distance attenuation and
// the spot cone. The material update mask makes the vertex colour stand in for
// the ambient (bit 0), diffuse (bit 1) and specular (bit 2) material colours;
// a vertex without a colour keeps the material ones.
//
// `per_pixel`: the directional lights' diffuse terms and every ambient term
// are left to the fragment shader, which gets the normal and materials.
void light_vertex(bool per_pixel, out vec4 color, out vec3 separate_specular) {
    int mask = int(object.flags.w + 0.5);
    bool has_color = object.flags.y > 0.5;
    vec4 ambient_material = (has_color && (mask & 1) != 0) ? in_color : object.material_ambient;
    vec3 diffuse_material = (has_color && (mask & 2) != 0) ? in_color.rgb : object.material_diffuse.rgb;
    vec3 specular_material = (has_color && (mask & 4) != 0) ? in_color.rgb : object.material_specular.rgb;
    float power = object.emissive.w;

    vec3 world_position = (object.world * vec4(in_position.xyz, 1.0)).xyz;
    // Skinned normals come out of the bone matrices far from unit length, so
    // the GE's normalisation after the transform matters.
    vec3 normal = mat3(object.world) * in_normal;
    float length_squared = dot(normal, normal);
    normal = length_squared > 0.0 ? normal * inversesqrt(length_squared) : vec3(0.0, 0.0, 1.0);
    if (object.material_specular.w > 0.5) normal = -normal;

    vec3 ambient = lighting.ambient.rgb * ambient_material.rgb;
    vec3 sum = object.emissive.rgb;
    vec3 specular = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        if (lighting.light_position[i].w < 0.5) continue;
        int type = int(lighting.light_direction[i].w + 0.5);
        int kind = int(lighting.light_attenuation[i].w + 0.5);
        vec3 to_light = lighting.light_position[i].xyz;
        float scale = 1.0;
        if (type != 0) {
            to_light -= world_position;
            float distance = length(to_light);
            vec3 k = lighting.light_attenuation[i].xyz;
            scale = clamp(1.0 / max(k.x + k.y * distance + k.z * distance * distance, 1e-20), 0.0, 1.0);
        }
        to_light = dot(to_light, to_light) > 0.0 ? normalize(to_light) : vec3(0.0, 0.0, 1.0);
        if (type == 2) {
            vec3 axis = lighting.light_direction[i].xyz;
            axis = dot(axis, axis) > 0.0 ? normalize(axis) : vec3(0.0, 0.0, 1.0);
            float angle = dot(axis, -to_light);
            scale *= angle >= lighting.light_spot[i].y ? pow(max(angle, 0.0), lighting.light_spot[i].x) : 0.0;
        }
        float n_dot_l = dot(normal, to_light);
        float diffuse = max(n_dot_l, 0.0);
        if (kind == 2) diffuse = pow(diffuse, power);
        ambient += lighting.light_ambient[i].rgb * ambient_material.rgb * scale;
        if (!per_pixel || type != 0) sum += lighting.light_diffuse[i].rgb * diffuse_material * diffuse * scale;
        if (kind == 1 && n_dot_l >= 0.0) {
            // The viewer is taken to look down z, as the GE does.
            vec3 half_vector = normalize(to_light + vec3(0.0, 0.0, 1.0));
            specular += lighting.light_specular[i].rgb * specular_material *
                        pow(max(dot(normal, half_vector), 0.0), power) * scale;
        }
    }
    float alpha = lighting.ambient.a * ambient_material.a;
    if (per_pixel) {
        frag_normal = normal;
        frag_to_eye = object.eye.xyz - world_position;
        frag_ambient = ambient;
        frag_diffuse = diffuse_material;
    } else {
        sum += ambient;
    }
    if (object.material_diffuse.w > 0.5) {
        separate_specular = clamp(specular, 0.0, 1.0);
    } else {
        sum += specular;
        separate_specular = vec3(0.0);
    }
    color = clamp(vec4(sum, alpha), 0.0, 1.0);
}


void main() {
#ifdef GE_RAW_VERTICES
    decode_raw_vertex();
#endif
    frag_texcoord = in_texcoord * push.uv_transform.xy + push.uv_transform.zw;
    frag_color = in_color;
    frag_specular = vec3(0.0);
    frag_fog = 1.0;
    frag_uv_rect = vec4(-1e30, -1e30, 1e30, 1e30);
    frag_normal = vec3(0.0);
    frag_to_eye = vec3(0.0);
    frag_ambient = vec3(0.0);
    frag_diffuse = vec3(0.0);
    if (push.viewport.z > 0.5) {
        vec4 rect = vec4(in_normal.xy, in_normal.z, in_position.w);
        frag_uv_rect = vec4(rect.xy * push.uv_transform.xy + push.uv_transform.zw,
                            rect.zw * push.uv_transform.xy + push.uv_transform.zw);
        // Screen-space vertices: pixels to clip space.
        vec2 ndc = vec2(in_position.x / push.viewport.x, in_position.y / push.viewport.y) * 2.0 - 1.0;
        gl_Position = vec4(ndc, clamp(in_position.z / 65535.0, 0.0, 1.0), 1.0);
    } else {
        int enables = int(push.viewport.w + 0.5);
        if ((enables & 2) != 0) light_vertex((enables & 4) != 0, frag_color, frag_specular);
        // Fog runs linearly from 1 (clear) to 0 (fogged) with view-space z,
        // which is negative in front of the camera: (z + end) * scale.
        if ((enables & 1) != 0)
            frag_fog = (dot(push.view_z, vec4(in_position.xyz, 1.0)) + lighting.fog.x) * lighting.fog.y;
        vec3 position = (enables & 8) != 0 ? swayed(in_position.xyz) : in_position.xyz;
        vec4 clip = push.transform * vec4(position, 1.0);
        // PSP clip space follows OpenGL with z in [-w, w]; Vulkan clips against
        // [0, w], so without this remap the near half of every frustum is lost.
        // The PSP viewport's z scale and offset are folded into the Vulkan
        // viewport's min/max depth, which expects this [0, 1] device z.
        clip.z = (clip.z + clip.w) * 0.5;
        gl_Position = clip;
    }
}
