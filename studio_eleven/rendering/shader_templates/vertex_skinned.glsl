// Custom Vertex: Skinned Character
// The vertex program of the characters (#FIX_SIL_VTX_MDS) as the game runs it: 4 bones per vertex,
// the lighting and the outline. Change what you need, the texture coordinates for example.

#ifdef VERTEX_SHADER
attribute vec4 atr_pos;
attribute vec4 atr_clr;
attribute vec3 atr_nrm;
attribute vec4 atr_tx0;
attribute vec4 atr_tx1;
attribute vec4 atr_tx2;
attribute vec4 atr_pr0; // weights of the 4 bones
attribute vec4 atr_pr1; // the 4 bones, their number in the node table of the mesh
attribute vec4 atr_pr2; // outline of the vertex: x width, y second color

uniform vec4 unf_vtx_cmr_prj[4]; // projection
uniform vec4 unf_vtx_glb_cmr[3]; // world to camera
uniform vec4 unf_vtx_lcl_glb[3]; // model to world
uniform vec4 unf_vtx_clr; // color of the material, animated by the material animations
uniform vec4 unf_vtx_txt[6]; // texture projections (UV animations), two rows per texture
uniform vec4 unf_vtx_bone[72]; // 24 bones, three rows each
uniform vec4 unf_vtx_silhouette_0; // outline: on, depth min, depth max, per vertex
uniform vec4 unf_vtx_silhouette_1; // outline: width, first color, second color, depth clamp

varying vec4 var_clr;
varying vec4 var_nrm;
varying vec4 var_tx0;
varying vec4 var_tx1;
varying vec4 var_tx2;
varying vec4 var_prm;

void main() {
    // Skinning
    vec4 position = vec4(0.0);
    vec3 normal = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        int bone = int(atr_pr1[i] * 3.0);
        float weight = atr_pr0[i];
        position.x += dot(unf_vtx_bone[bone], atr_pos) * weight;
        position.y += dot(unf_vtx_bone[bone + 1], atr_pos) * weight;
        position.z += dot(unf_vtx_bone[bone + 2], atr_pos) * weight;
        normal.x += dot(unf_vtx_bone[bone].xyz, atr_nrm) * weight;
        normal.y += dot(unf_vtx_bone[bone + 1].xyz, atr_nrm) * weight;
        normal.z += dot(unf_vtx_bone[bone + 2].xyz, atr_nrm) * weight;
    }
    position.w = 1.0;
    normal = normalize(normal);

    // Model to world to camera
    vec4 world = vec4(dot(unf_vtx_lcl_glb[0], position), dot(unf_vtx_lcl_glb[1], position), dot(unf_vtx_lcl_glb[2], position), 1.0);
    vec4 eye = vec4(dot(unf_vtx_glb_cmr[0], world), dot(unf_vtx_glb_cmr[1], world), dot(unf_vtx_glb_cmr[2], world), 1.0);
    vec3 world_normal = vec3(dot(unf_vtx_lcl_glb[0].xyz, normal), dot(unf_vtx_lcl_glb[1].xyz, normal), dot(unf_vtx_lcl_glb[2].xyz, normal));
    vec3 eye_normal = normalize(vec3(dot(unf_vtx_glb_cmr[0].xyz, world_normal), dot(unf_vtx_glb_cmr[1].xyz, world_normal), dot(unf_vtx_glb_cmr[2].xyz, world_normal)));

    vec4 clip = vec4(dot(unf_vtx_cmr_prj[0], eye), dot(unf_vtx_cmr_prj[1], eye), dot(unf_vtx_cmr_prj[2], eye), dot(unf_vtx_cmr_prj[3], eye));

    // Outline: the pass of the outline moves the vertex along its normal, by a width that keeps its size on screen
    vec4 moved = vec4(eye.xyz + eye_normal, 1.0);
    vec4 extruded = vec4(dot(unf_vtx_cmr_prj[0], moved), dot(unf_vtx_cmr_prj[1], moved), dot(unf_vtx_cmr_prj[2], moved), dot(unf_vtx_cmr_prj[3], moved));
    float depth = mix(clip.w, clip.w / max(unf_vtx_silhouette_0.y, min(unf_vtx_silhouette_0.z, clip.w)), unf_vtx_silhouette_1.w);
    float flag_x = 1.0 + (atr_pr2.x - 1.0) * unf_vtx_silhouette_0.w;
    vec2 offset = unf_vtx_silhouette_0.x * (eye_normal.yx * depth * (unf_vtx_silhouette_1.x * flag_x));
    gl_Position = clip + (extruded - clip) * length(offset);

    // The color of the outline darkens the vertex color, the combiners double the color
    // (the V1 game, Inazuma Eleven Go, leaves out the alpha of the vertex: unf_vtx_clr.a alone)
    float flag_y = unf_vtx_silhouette_0.w * atr_pr2.y;
    float shade = mix(unf_vtx_silhouette_1.y * floor(flag_x + 0.95), unf_vtx_silhouette_1.z, flag_y);
    var_clr = vec4(unf_vtx_clr.rgb * (atr_clr.rgb * (1.0 - unf_vtx_silhouette_0.x * shade)), unf_vtx_clr.a * atr_clr.a) * 0.5;

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
