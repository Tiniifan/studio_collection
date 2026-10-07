import bpy
from bpy.props import StringProperty, EnumProperty, BoolProperty, PointerProperty

from ...rendering import project as rendering_project, custom_shaders, render_defaults

##########################################
# CONST
##########################################

# What the code can read and write, shown in the Level 5 tab of the Text Editor
HELP_LINES = (
    "#pragma vertex_program \"#NAME\"",
    "#pragma combiner \"#NAME\"",
    "    take the one of the game",
    "VERTEX_SHADER:",
    "    atr_pos atr_nrm atr_clr",
    "    atr_tx0-2 atr_pr0-2",
    "    unf_vtx_lcl_glb[3]",
    "    unf_vtx_glb_cmr[3]",
    "    unf_vtx_cmr_prj[4]",
    "    unf_vtx_txt[6] unf_vtx_clr",
    "    unf_vtx_bone[72]",
    "    gl_Position var_clr",
    "    var_tx0-2 var_nrm var_prm",
    "FRAGMENT_SHADER:",
    "    texture2D(unf_frg_txt_2d_N,",
    "        var_txN.xy)",
    "    var_clr clr_1st clr_2nd",
    "    + - * mix dot3 add_signed",
    "    * 2.0 or * 4.0, 6 stages",
)

##########################################
# Shader Editor Function
##########################################

def get_editor_shader(context):
    """The shader the Text Editor shows."""
    space = context.space_data

    if space is not None and space.type == 'TEXT_EDITOR' and custom_shaders.is_shader(space.text):
        return space.text

    return None

def get_active_mesh(context):
    obj = context.active_object

    if obj is not None and obj.type == 'MESH':
        return obj

    return None

def get_unique_name(name):
    unique = name
    count = 1

    while custom_shaders.find_shader(unique) is not None or bpy.data.texts.get(unique + custom_shaders.SHADER_EXTENSION) is not None:
        count += 1
        unique = f"{name}_{count}"

    return unique

def show_text(context, text):
    """The text in the Text Editor of the context, or in a new window."""
    space = context.space_data

    if space is None or space.type != 'TEXT_EDITOR':
        bpy.ops.wm.window_new()
        window = context.window_manager.windows[-1]
        area = window.screen.areas[0]
        area.type = 'TEXT_EDITOR'
        space = area.spaces.active

    space.text = text
    space.top = 0
    space.show_region_ui = True

def source_items(self, context):
    """What a new shader starts from: the templates, then the render defaults of the scene engine."""
    items = []

    for shader_template in custom_shaders.get_templates():
        items.append((f"TEMPLATE:{shader_template['file_name']}", shader_template["label"], "Template of Studio Eleven", 'TEXT', len(items)))

    for item in rendering_project.render_default_items(context.scene):
        items.append((f"RENDER:{item[0]}", f"Copy of {item[1]}", item[2], 'NONE', len(items)))

    return remember_items(items)

remembered_items = {}

def remember_items(items):
    # Blender keeps a reference to the list of an enum
    key = tuple(item[0] for item in items)
    if key not in remembered_items:
        remembered_items[key] = items
    return remembered_items[key]

def get_source_code(context, source):
    if source.startswith("TEMPLATE:"):
        return custom_shaders.read_template(source[len("TEMPLATE:"):])

    engine_id = rendering_project.get_scene_engine_id(context.scene)
    render_default = render_defaults.find_render_default(engine_id, source[len("RENDER:"):])

    if render_default is None:
        return None

    return custom_shaders.make_code_from_render_default(engine_id, render_default)

def assign_shader(context, text):
    meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']

    for obj in meshes:
        rendering_project.set_mesh_render_default(obj.data, custom_shaders.get_shader_name(text))

    custom_shaders.clear_render_cache()

    return len(meshes)

##########################################
# Register class
##########################################

class ShaderEditor_New(bpy.types.Operator):
    bl_idname = "studio_eleven.new_shader"
    bl_label = "New Shader"
    bl_description = "Make a custom shader of the blend: GLSL code from a template or from a render default of the game"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(
        name="Name",
        description="Name of the render program, a mesh stores its crc32. The name of a render default of fix.xr hides it for the archive",
        default="#CUSTOM"
    )
    source: EnumProperty(
        name="From",
        description="Code the shader starts with",
        items=source_items
    )
    assign: BoolProperty(
        name="Assign to Selected Meshes",
        description="Give the new shader to the selected meshes",
        default=True
    )

    def invoke(self, context, event):
        self.name = get_unique_name("#CUSTOM")

        # A copy of the render default of the active mesh by default
        mesh = get_active_mesh(context)
        if mesh is not None:
            wanted = f"RENDER:{rendering_project.get_mesh_render_default(mesh.data).name}"
            for item in source_items(self, context):
                if item[0] == wanted:
                    self.source = wanted

        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        code = get_source_code(context, self.source)

        if code is None:
            self.report({'ERROR'}, f"{self.source} not found")
            return {'CANCELLED'}

        text = custom_shaders.new_shader(get_unique_name(self.name), code)

        if self.assign:
            assign_shader(context, text)

        show_text(context, text)

        self.report({'INFO'}, f"Shader {custom_shaders.get_shader_name(text)} made")

        return {'FINISHED'}

class ShaderEditor_Template(bpy.types.Operator):
    bl_idname = "studio_eleven.shader_template"
    bl_label = "Level 5 Shader Template"
    bl_description = "Make a custom shader of the blend from this template"
    bl_options = {'REGISTER', 'UNDO'}

    file_name: StringProperty()

    def execute(self, context):
        text = custom_shaders.new_shader(get_unique_name("#CUSTOM"), custom_shaders.read_template(self.file_name))
        show_text(context, text)

        return {'FINISHED'}

class ShaderEditor_MakeShader(bpy.types.Operator):
    bl_idname = "studio_eleven.make_shader"
    bl_label = "Use as a Level 5 Shader"
    bl_description = "The text becomes a custom shader of the blend, compiled as it is typed"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space is not None and space.type == 'TEXT_EDITOR' and space.text is not None

    def execute(self, context):
        text = context.space_data.text

        # The extension .osl gives the coloring of a C-like language
        if not text.name.endswith(custom_shaders.SHADER_EXTENSION):
            text.name = get_unique_name(custom_shaders.get_shader_name(text)) + custom_shaders.SHADER_EXTENSION

        text.level5_shader.is_shader = True
        text.use_fake_user = True
        custom_shaders.compile_shader(text)

        return {'FINISHED'}

class ShaderEditor_Compile(bpy.types.Operator):
    bl_idname = "studio_eleven.compile_shader"
    bl_label = "Compile"
    bl_description = "Compile the shader now (it is done by itself half a second after the code stops changing)"

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def execute(self, context):
        error = custom_shaders.compile_shader(get_editor_shader(context))

        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}

        self.report({'INFO'}, "The shader compiles")

        return {'FINISHED'}

class ShaderEditor_Assign(bpy.types.Operator):
    bl_idname = "studio_eleven.assign_shader"
    bl_label = "Assign to Selected Meshes"
    bl_description = "Give the shader to the selected meshes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def execute(self, context):
        text = get_editor_shader(context)
        count = assign_shader(context, text)

        if count == 0:
            self.report({'ERROR'}, "No mesh selected")
            return {'CANCELLED'}

        self.report({'INFO'}, f"{custom_shaders.get_shader_name(text)} given to {count} meshes")

        return {'FINISHED'}

class ShaderEditor_EditMeshShader(bpy.types.Operator):
    bl_idname = "studio_eleven.edit_mesh_shader"
    bl_label = "Edit Shader"
    bl_description = "Open the code of the shader of the mesh in a Text Editor"

    def execute(self, context):
        mesh = get_active_mesh(context)
        text = None
        if mesh is not None:
            text = custom_shaders.get_mesh_shader(mesh.data)

        if text is None:
            self.report({'ERROR'}, "The mesh has no custom shader")
            return {'CANCELLED'}

        show_text(context, text)

        return {'FINISHED'}

class TEXT_MT_level5_templates(bpy.types.Menu):
    bl_idname = "TEXT_MT_level5_templates"
    bl_label = "Level 5 Shaders"

    def draw(self, context):
        layout = self.layout

        for shader_template in custom_shaders.get_templates():
            operator = layout.operator("studio_eleven.shader_template", text=shader_template["label"])
            operator.file_name = shader_template["file_name"]

def draw_template_menu(self, context):
    self.layout.menu(TEXT_MT_level5_templates.bl_idname)

def draw_status(layout, text):
    properties = text.level5_shader

    if properties.error:
        box = layout.box()
        box.label(text="Error", icon='ERROR')
        # The message is cut in lines the panel can show
        words = properties.error.split(" ")
        line = ""
        for word in words:
            if len(line) + len(word) > 42:
                box.label(text=line)
                line = ""
            line += word + " "
        if line:
            box.label(text=line)
        box.label(text="The last code that compiled is still used", icon='INFO')

    column = layout.column(align=True)

    if properties.vertex_program:
        column.label(text=f"Vertex: {properties.vertex_program} of the game", icon='MESH_DATA')
    elif properties.instruction_count:
        column.label(text=f"Vertex: {properties.instruction_count} / 512 instructions", icon='MESH_DATA')

    if properties.combiner:
        column.label(text=f"Fragment: {properties.combiner} of the game", icon='SHADING_TEXTURE')
    elif properties.stage_count:
        column.label(text=f"Fragment: {properties.stage_count} / 6 stages", icon='SHADING_TEXTURE')

class TEXT_PT_level5_shader(bpy.types.Panel):
    bl_label = "Level 5 Shader"
    bl_idname = "TEXT_PT_level5_shader"
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Level 5"

    def draw(self, context):
        layout = self.layout
        text = context.space_data.text

        layout.operator("studio_eleven.new_shader", icon='ADD')
        layout.menu(TEXT_MT_level5_templates.bl_idname, icon='TEXT')

        if text is None:
            return

        if not custom_shaders.is_shader(text):
            layout.operator("studio_eleven.make_shader", icon='SHADING_TEXTURE')
            return

        layout.label(text=f"Render program: {custom_shaders.get_shader_name(text)}")
        draw_status(layout, text)

        column = layout.column(align=True)
        column.operator("studio_eleven.assign_shader", icon='CHECKMARK')
        column.operator("studio_eleven.compile_shader", icon='FILE_REFRESH')

class TEXT_PT_level5_shader_help(bpy.types.Panel):
    bl_label = "Names"
    bl_idname = "TEXT_PT_level5_shader_help"
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Level 5"
    bl_parent_id = "TEXT_PT_level5_shader"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        column = self.layout.column(align=True)
        for line in HELP_LINES:
            column.label(text=line)

classes = (
    custom_shaders.Level5ShaderProperties,
    ShaderEditor_New,
    ShaderEditor_Template,
    ShaderEditor_MakeShader,
    ShaderEditor_Compile,
    ShaderEditor_Assign,
    ShaderEditor_EditMeshShader,
    TEXT_MT_level5_templates,
    TEXT_PT_level5_shader,
    TEXT_PT_level5_shader_help,
)

##########################################
# Register
##########################################

def register_shader_editor():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Text.level5_shader = PointerProperty(type=custom_shaders.Level5ShaderProperties)
    bpy.types.TEXT_MT_templates.append(draw_template_menu)

    # The code typed in the Text Editor is compiled half a second after it stops changing
    bpy.app.timers.register(custom_shaders.watch_shaders, first_interval=1.0, persistent=True)

def unregister_shader_editor():
    if bpy.app.timers.is_registered(custom_shaders.watch_shaders):
        bpy.app.timers.unregister(custom_shaders.watch_shaders)

    bpy.types.TEXT_MT_templates.remove(draw_template_menu)
    del bpy.types.Text.level5_shader

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
