import bpy
import nodeitems_utils
from bpy.props import StringProperty, EnumProperty, BoolProperty
from nodeitems_utils import NodeCategory, NodeItem

from ...rendering import project as rendering_project, custom_shaders, render_defaults
from ...rendering.studio_render import resources

##########################################
# CONST
##########################################

NODE_CATEGORY_ID = "LEVEL5_SHADER_NODES"

##########################################
# Shader Editor Function
##########################################

def get_editor_shader(context):
    """The shader of the Shader Editor, or of the text the Text Editor shows."""
    space = context.space_data

    if space is not None and space.type == 'NODE_EDITOR' and custom_shaders.is_shader(space.node_tree):
        return space.node_tree

    if space is not None and space.type == 'TEXT_EDITOR' and space.text is not None:
        for tree in custom_shaders.get_shaders():
            if tree.code == space.text:
                return tree

    return None

def get_active_mesh(context):
    obj = context.active_object

    if obj is not None and obj.type == 'MESH':
        return obj

    return None

def open_window(context, area_type):
    """A new window with one editor, the editor of the Studio Eleven panel stays as it is."""
    bpy.ops.wm.window_new()
    window = context.window_manager.windows[-1]
    area = window.screen.areas[0]
    area.type = area_type

    return area

def open_shader_editor(context, tree):
    area = open_window(context, 'NODE_EDITOR')
    area.ui_type = custom_shaders.TREE_TYPE

    if tree is not None:
        area.spaces.active.node_tree = tree

def source_items(self, context):
    """Render defaults a new shader starts from: the custom shaders, then the ones of the scene engine."""
    items = []

    for tree in custom_shaders.get_shaders():
        items.append((tree.name, tree.name, "Custom shader of the blend", 'NODE_MATERIAL', len(items)))

    for item in rendering_project.render_default_items(context.scene):
        items.append((item[0], item[1], item[2], 'NONE', len(items)))

    return custom_shaders.remember_items(items)

def get_default_source(context):
    mesh = get_active_mesh(context)

    if mesh is not None:
        return rendering_project.get_mesh_render_default(mesh.data).name

    return render_defaults.get_default_render_default(rendering_project.get_scene_engine_id(context.scene)).name

##########################################
# Register class
##########################################

class ShaderEditor_Open(bpy.types.Operator):
    bl_idname = "studio_eleven.open_shader_editor"
    bl_label = "Shader Editor"
    bl_description = "Open the Shader Editor in a new window: the custom shaders of the blend as nodes, with their code"

    def execute(self, context):
        tree = None
        mesh = get_active_mesh(context)

        if mesh is not None:
            tree = custom_shaders.get_mesh_shader(mesh.data)

        if tree is None and len(custom_shaders.get_shaders()) > 0:
            tree = custom_shaders.get_shaders()[0]

        open_shader_editor(context, tree)

        return {'FINISHED'}

class ShaderEditor_New(bpy.types.Operator):
    bl_idname = "studio_eleven.new_shader"
    bl_label = "New Shader"
    bl_description = "Make a custom shader from a render default: its vertex program and its combiner as nodes"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(
        name="Name",
        description="Name of the render default, a mesh stores its crc32. The name of a render default of fix.xr hides it for the archive",
        default="#CUSTOM"
    )
    source: EnumProperty(
        name="Copy Of",
        description="Render default the shader starts from",
        items=source_items
    )
    assign: BoolProperty(
        name="Assign to Selected Meshes",
        description="Give the new shader to the selected meshes",
        default=True
    )

    def invoke(self, context, event):
        default = get_default_source(context)

        for item in source_items(self, context):
            if item[0] == default:
                self.source = default

        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        engine_id = rendering_project.get_scene_engine_id(context.scene)
        shader = bpy.data.node_groups.get(self.source)

        if custom_shaders.is_shader(shader):
            source = custom_shaders.make_render_default(shader, engine_id)
        else:
            source = render_defaults.find_render_default(engine_id, self.source)

        if source is None:
            self.report({'ERROR'}, f"Render default {self.source} not found")
            return {'CANCELLED'}

        stages = resources.render_default_stages(engine_id, source)
        stage_dicts = None
        if stages:
            stage_dicts = [stage.to_dict() for stage in stages]

        # The shader keeps the combiner of the game as it is until its nodes are edited
        game_combiner = ""
        if source.data.get("stages") is None:
            game_combiner = source.data["combiner"]

        tree = custom_shaders.new_shader(self.name, source.data["vertex_program"], stage_dicts, game_combiner, engine_id)

        if self.assign:
            for obj in context.selected_objects:
                if obj.type == 'MESH':
                    rendering_project.set_mesh_render_default(obj.data, tree.name)

        space = context.space_data
        if space is not None and space.type == 'NODE_EDITOR':
            space.node_tree = tree

        custom_shaders.clear_render_cache()
        self.report({'INFO'}, f"Shader {tree.name} made from {source.name}")

        return {'FINISHED'}

class ShaderEditor_Fill(bpy.types.Operator):
    bl_idname = "studio_eleven.fill_shader"
    bl_label = "Fill with Default Nodes"
    bl_description = "Give the empty shader an Output node fed by a texture lit by the vertex color, and the vertex program of the engine default"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def execute(self, context):
        custom_shaders.fill_shader(get_editor_shader(context), rendering_project.get_scene_engine_id(context.scene))
        return {'FINISHED'}

class ShaderEditor_Assign(bpy.types.Operator):
    bl_idname = "studio_eleven.assign_shader"
    bl_label = "Assign to Selected Meshes"
    bl_description = "Give this shader to the selected meshes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def execute(self, context):
        tree = get_editor_shader(context)
        meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if len(meshes) == 0:
            self.report({'ERROR'}, "No mesh selected")
            return {'CANCELLED'}

        for obj in meshes:
            rendering_project.set_mesh_render_default(obj.data, tree.name)

        custom_shaders.clear_render_cache()
        self.report({'INFO'}, f"{tree.name} given to {len(meshes)} meshes")

        return {'FINISHED'}

class ShaderEditor_EditCode(bpy.types.Operator):
    bl_idname = "studio_eleven.edit_shader_code"
    bl_label = "Edit Code"
    bl_description = "Open the code of the shader in a Text Editor, Apply Code makes the nodes again from it"

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def execute(self, context):
        tree = get_editor_shader(context)
        custom_shaders.shader_changed(tree)

        text = custom_shaders.get_code_text(tree)

        # Writing the code leaves the cursor after its last line, the editor would open below the code
        text.current_line_index = 0
        text.current_character = 0
        text.select_end_line_index = 0
        text.select_end_character = 0

        area = open_window(context, 'TEXT_EDITOR')
        area.spaces.active.text = text
        area.spaces.active.top = 0

        return {'FINISHED'}

class ShaderEditor_ApplyCode(bpy.types.Operator):
    bl_idname = "studio_eleven.apply_shader_code"
    bl_label = "Apply Code"
    bl_description = "Compile the code of the shader and make its nodes again from it now (it is done by itself half a second after the code stops changing)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def execute(self, context):
        tree = get_editor_shader(context)
        error = custom_shaders.apply_code(tree)

        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}

        self.report({'INFO'}, f"{tree.name}: the nodes follow the code")

        return {'FINISHED'}

class ShaderEditor_EditMeshShader(bpy.types.Operator):
    bl_idname = "studio_eleven.edit_mesh_shader"
    bl_label = "Edit Shader"
    bl_description = "Open the shader of the mesh in the Shader Editor"

    def execute(self, context):
        mesh = get_active_mesh(context)
        tree = None
        if mesh is not None:
            tree = custom_shaders.get_mesh_shader(mesh.data)

        open_shader_editor(context, tree)

        return {'FINISHED'}

def draw_status(layout, tree):
    if custom_shaders.find_output_node(tree) is None:
        box = layout.box()
        box.label(text="Empty shader", icon='ERROR')
        box.label(text="Add nodes with Shift A, at least an Output")
        box.operator("studio_eleven.fill_shader", icon='NODETREE')
        return

    if tree.error:
        box = layout.box()
        box.alert = True
        for line in split_text(tree.error, 40):
            box.label(text=line)
        return

    if tree.game_combiner:
        layout.label(text=f"Combiner {tree.game_combiner} of the game", icon='INFO')
    else:
        stages = custom_shaders.get_shader_stages(tree)
        if stages is None:
            layout.label(text="Not compiled yet, link the Output node", icon='INFO')
        else:
            layout.label(text=f"{len(stages)} / 6 stages", icon='CHECKMARK')

    if not tree.vertex_program:
        layout.label(text="Choose a vertex program", icon='ERROR')

def split_text(text, width):
    lines = []
    line = ""

    for word in text.split(" "):
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        elif line:
            line += " " + word
        else:
            line = word

    if line:
        lines.append(line)

    return lines

class NODE_PT_level5_shader(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Level 5"
    bl_label = "Shader"

    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == custom_shaders.TREE_TYPE

    def draw(self, context):
        layout = self.layout
        tree = get_editor_shader(context)

        layout.operator("studio_eleven.new_shader", icon='ADD')

        if tree is None:
            return

        layout.prop(tree, "name", text="Name")
        layout.prop(tree, "vertex_program_choice", text="Vertex Program")
        draw_status(layout, tree)

        column = layout.column(align=True)
        column.operator("studio_eleven.assign_shader", icon='CHECKMARK')
        column.operator("studio_eleven.edit_shader_code", icon='TEXT')
        column.operator("studio_eleven.apply_shader_code", icon='FILE_REFRESH')

class TEXT_PT_level5_shader(bpy.types.Panel):
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Level 5"
    bl_label = "Shader"

    @classmethod
    def poll(cls, context):
        return get_editor_shader(context) is not None

    def draw(self, context):
        layout = self.layout
        tree = get_editor_shader(context)

        layout.label(text=tree.name, icon='NODE_MATERIAL')
        layout.operator("studio_eleven.apply_shader_code", icon='FILE_REFRESH')
        draw_status(layout, tree)

        column = layout.box().column(align=True)
        column.label(text="texture0..3, vertex_color")
        column.label(text="light_diffuse, light_specular")
        column.label(text="Channels: .rgb .a .r .g .b")
        column.label(text="+ - * mix dot3 add_signed")
        column.label(text="x * 2.0, x * 4.0: scale")
        column.label(text="Results clamped to 0..1")

class Level5ShaderNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == custom_shaders.TREE_TYPE

NODE_CATEGORIES = [
    Level5ShaderNodeCategory("LEVEL5_INPUT", "Input", items=[
        NodeItem("Level5NodeTexture"),
        NodeItem("Level5NodeVertexColor"),
        NodeItem("Level5NodeLighting"),
        NodeItem("Level5NodeColor"),
        NodeItem("Level5NodeValue"),
    ]),
    Level5ShaderNodeCategory("LEVEL5_OPERATION", "Operation", items=[
        NodeItem("Level5NodeMath"),
        NodeItem("Level5NodeSeparate"),
    ]),
    Level5ShaderNodeCategory("LEVEL5_OUTPUT", "Output", items=[
        NodeItem("Level5NodeOutput"),
    ]),
]

classes = custom_shaders.classes + (
    ShaderEditor_Open,
    ShaderEditor_New,
    ShaderEditor_Fill,
    ShaderEditor_Assign,
    ShaderEditor_EditCode,
    ShaderEditor_ApplyCode,
    ShaderEditor_EditMeshShader,
    NODE_PT_level5_shader,
    TEXT_PT_level5_shader,
)

##########################################
# Register
##########################################

def register_shader_editor():
    for cls in classes:
        bpy.utils.register_class(cls)

    nodeitems_utils.register_node_categories(NODE_CATEGORY_ID, NODE_CATEGORIES)

    # The code typed in the Text Editor follows the nodes and the other way
    bpy.app.timers.register(custom_shaders.watch_code, first_interval=1.0, persistent=True)

def unregister_shader_editor():
    if bpy.app.timers.is_registered(custom_shaders.watch_code):
        bpy.app.timers.unregister(custom_shaders.watch_code)

    nodeitems_utils.unregister_node_categories(NODE_CATEGORY_ID)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
