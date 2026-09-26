#version 450

layout(location = 0) in float frag_alpha;
layout(location = 0) out vec4 out_mask;

void main() {
    // Even nearly clear water reflects fully: its alpha is how much of the
    // bottom it hides, not how much it mirrors. Where it fades to nothing
    // at its shores, so does the reflection.
    out_mask = vec4(smoothstep(0.0, 0.3, frag_alpha));
}
