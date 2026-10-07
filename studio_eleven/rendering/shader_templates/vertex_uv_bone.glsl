// Custom Vertex: Texture Moved by a Bone
// A skinned model whose texture 0 slides with the location of a bone: animate the bone, the texture scrolls.
// The bone is the one of the node table of the mesh at UV_BONE (0 is the first one), its location x / y
// move U / V. Several meshes can share the bone, so they share one scroll (a UV animation can't be shared).

#ifdef VERTEX_SHADER
attribute vec4 atr_pos;
attribute vec4 atr_clr;
attribute vec3 atr_nrm;
attribute vec4 atr_tx0;
attribute vec4 atr_pr0; // weights of the 4 bones
attribute vec4 atr_pr1; // the 4 bones, their number in the node table of the mesh

uniform vec4 unf_vtx_cmr_prj[4];
uniform vec4 unf_vtx_glb_cmr[3];
uniform vec4 unf_vtx_lcl_glb[3];
uniform vec4 unf_vtx_clr;
uniform vec4 unf_vtx_bone[72]; // 24 bones, three rows each, the location in the w of the rows

varying vec4 var_clr;
varying vec4 var_nrm;
varying vec4 var_tx0;
varying vec4 var_prm;

const int UV_BONE = 1;

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

    vec4 world = vec4(dot(unf_vtx_lcl_glb[0], position), dot(unf_vtx_lcl_glb[1], position), dot(unf_vtx_lcl_glb[2], position), 1.0);
    vec4 eye = vec4(dot(unf_vtx_glb_cmr[0], world), dot(unf_vtx_glb_cmr[1], world), dot(unf_vtx_glb_cmr[2], world), 1.0);
    vec3 world_normal = vec3(dot(unf_vtx_lcl_glb[0].xyz, normal), dot(unf_vtx_lcl_glb[1].xyz, normal), dot(unf_vtx_lcl_glb[2].xyz, normal));
    vec3 eye_normal = normalize(vec3(dot(unf_vtx_glb_cmr[0].xyz, world_normal), dot(unf_vtx_glb_cmr[1].xyz, world_normal), dot(unf_vtx_glb_cmr[2].xyz, world_normal)));

    gl_Position = vec4(dot(unf_vtx_cmr_prj[0], eye), dot(unf_vtx_cmr_prj[1], eye), dot(unf_vtx_cmr_prj[2], eye), dot(unf_vtx_cmr_prj[3], eye));
    var_clr = vec4(unf_vtx_clr.rgb * atr_clr.rgb, unf_vtx_clr.a * atr_clr.a) * 0.5;

    // The location of the bone moves the texture coordinates
    vec2 scroll = vec2(unf_vtx_bone[UV_BONE * 3].w, unf_vtx_bone[UV_BONE * 3 + 1].w);
    var_tx0 = vec4(atr_tx0.xy + scroll, 0.0, 0.0);

    var_nrm = vec4(eye_normal, 0.0);
    var_prm = vec4(eye.xyz, 0.0);
}
#endif

#ifdef FRAGMENT_SHADER
uniform sampler2D unf_frg_txt_2d_0;
varying vec4 var_clr;
varying vec4 var_tx0;

void main() {
    vec4 tex0 = texture2D(unf_frg_txt_2d_0, var_tx0.xy);
    gl_FragColor = vec4(tex0.rgb * var_clr.rgb * 2.0, tex0.a * var_clr.a * 2.0);
}
#endif
