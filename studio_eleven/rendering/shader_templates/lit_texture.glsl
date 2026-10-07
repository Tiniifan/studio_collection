// Lit Texture
// The texture 0 of the material lit by the lighting of the material (.mtr) and the vertex color, as #FIX_TON_12.
// The vertex program of the game is kept: write a VERTEX_SHADER section in place of the pragma to change it.

#pragma vertex_program "#FIX_MDS"

#ifdef FRAGMENT_SHADER
uniform sampler2D unf_frg_txt_2d_0;
varying vec4 var_clr;
varying vec4 var_tx0;
// Lighting of the material (.mtr), computed by the hardware: diffuse and specular
varying vec4 clr_1st;
varying vec4 clr_2nd;

void main() {
    vec4 tex0 = texture2D(unf_frg_txt_2d_0, var_tx0.xy);
    vec3 lit = (clr_1st.rgb + clr_2nd.rgb) * tex0.rgb;

    gl_FragColor = vec4(lit * var_clr.rgb * 2.0, (clr_1st.a + clr_2nd.a) * tex0.a * var_clr.a * 2.0);
}
#endif
