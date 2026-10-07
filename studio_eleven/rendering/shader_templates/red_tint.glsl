// Red Tint
// The lit texture turned halfway to red: change the color or the factor of mix to tint it otherwise.
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
    vec3 lit = (clr_1st.rgb + clr_2nd.rgb) * tex0.rgb * var_clr.rgb * 2.0;

    gl_FragColor = vec4(mix(lit, vec3(1.0, 0.0, 0.0), 0.5), (clr_1st.a + clr_2nd.a) * tex0.a * var_clr.a * 2.0);
}
#endif
