// Custom Vertex: Rigid Model
// The vertex program of the models without bones (#FIX_MDL) as the game runs it, with the lighting.

#ifdef VERTEX_SHADER
attribute vec4 atr_pos;
attribute vec4 atr_clr;
attribute vec3 atr_nrm;
attribute vec4 atr_tx0;
attribute vec4 atr_tx1;
attribute vec4 atr_tx2;

uniform vec4 unf_vtx_cmr_prj[4]; // projection
uniform vec4 unf_vtx_glb_cmr[3]; // world to camera
uniform vec4 unf_vtx_lcl_glb[3]; // model to world
uniform vec4 unf_vtx_clr; // color of the material, animated by the material animations
uniform vec4 unf_vtx_txt[6]; // texture projections (UV animations), two rows per texture

varying vec4 var_clr;
varying vec4 var_nrm;
varying vec4 var_tx0;
varying vec4 var_tx1;
varying vec4 var_tx2;
varying vec4 var_prm;

void main() {
    // Model to world to camera
    vec4 world = vec4(dot(unf_vtx_lcl_glb[0], atr_pos), dot(unf_vtx_lcl_glb[1], atr_pos), dot(unf_vtx_lcl_glb[2], atr_pos), 1.0);
    vec4 eye = vec4(dot(unf_vtx_glb_cmr[0], world), dot(unf_vtx_glb_cmr[1], world), dot(unf_vtx_glb_cmr[2], world), 1.0);
    vec3 world_normal = vec3(dot(unf_vtx_lcl_glb[0].xyz, atr_nrm), dot(unf_vtx_lcl_glb[1].xyz, atr_nrm), dot(unf_vtx_lcl_glb[2].xyz, atr_nrm));
    vec3 eye_normal = normalize(vec3(dot(unf_vtx_glb_cmr[0].xyz, world_normal), dot(unf_vtx_glb_cmr[1].xyz, world_normal), dot(unf_vtx_glb_cmr[2].xyz, world_normal)));

    gl_Position = vec4(dot(unf_vtx_cmr_prj[0], eye), dot(unf_vtx_cmr_prj[1], eye), dot(unf_vtx_cmr_prj[2], eye), dot(unf_vtx_cmr_prj[3], eye));

    // The combiners double the color
    var_clr = unf_vtx_clr * atr_clr * 0.5;

    // Texture coordinates through the texture projections
    var_tx0 = vec4(dot(unf_vtx_txt[0], atr_tx0), dot(unf_vtx_txt[1], atr_tx0), atr_tx0.z, 0.0);
    var_tx1 = vec4(dot(unf_vtx_txt[2], atr_tx1), dot(unf_vtx_txt[3], atr_tx1), 0.0, 0.0);
    var_tx2 = vec4(dot(unf_vtx_txt[4], atr_tx2), dot(unf_vtx_txt[5], atr_tx2), 0.0, 0.0);

    // Lighting: the normal and the position in the camera space
    var_nrm = vec4(eye_normal, 0.0);
    var_prm = vec4(eye.xyz, 0.0);
}
#endif

#ifdef FRAGMENT_SHADER
uniform sampler2D unf_frg_txt_2d_0;
varying vec4 var_clr;
varying vec4 var_tx0;
// Lighting of the material (.mtr), computed by the hardware: diffuse and specular
varying vec4 clr_1st;
varying vec4 clr_2nd;

void main() {
    vec4 tex0 = texture2D(unf_frg_txt_2d_0, var_tx0.xy);
    gl_FragColor = vec4((clr_1st.rgb + clr_2nd.rgb) * tex0.rgb * var_clr.rgb * 2.0, (clr_1st.a + clr_2nd.a) * tex0.a * var_clr.a * 2.0);
}
#endif
