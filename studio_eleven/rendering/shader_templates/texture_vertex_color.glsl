// Texture x Vertex Color
// The texture 0 of the material times the vertex color, no lighting.
// The vertex program of the game is kept: write a VERTEX_SHADER section in place of the pragma to change it.

#pragma vertex_program "#FIX_MDS"

#ifdef FRAGMENT_SHADER
uniform sampler2D unf_frg_txt_2d_0;
varying vec4 var_clr;
varying vec4 var_tx0;

void main() {
    vec4 tex0 = texture2D(unf_frg_txt_2d_0, var_tx0.xy);

    // The vertex program halves the color, the combiner doubles it back
    gl_FragColor = vec4(tex0.rgb * var_clr.rgb * 2.0, tex0.a * var_clr.a * 2.0);
}
#endif
