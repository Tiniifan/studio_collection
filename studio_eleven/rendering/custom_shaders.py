import bpy
import json
import zlib
import re

from bpy.props import StringProperty, EnumProperty, FloatVectorProperty, FloatProperty, PointerProperty

from ..formats import cmb, res
from . import render_defaults, combiner_code

##########################################
# CONST
##########################################

TREE_TYPE = "Level5ShaderTree"

# Name in the code -> (node, output index)
SOURCE_OUTPUTS = {
    "texture0": ("Level5NodeTexture", 0),
    "texture1": ("Level5NodeTexture", 0),
    "texture2": ("Level5NodeTexture", 0),
    "texture3": ("Level5NodeTexture", 0),
    "vertex_color": ("Level5NodeVertexColor", 0),
    "light_diffuse": ("Level5NodeLighting", 0),
    "light_specular": ("Level5NodeLighting", 2),
}

OPERATION_ITEMS = [
    ('MULTIPLY', "Multiply", "A × B"),
    ('ADD', "Add", "A + B"),
    ('SUBTRACT', "Subtract", "A - B"),
    ('MIX', "Mix", "Mix from A to B by Factor, mix(A, B, Factor)"),
    ('MULTIPLY_ADD', "Multiply Add", "A × B + C, one stage"),
    ('ADD_MULTIPLY', "Add Multiply", "(A + B) × C, one stage"),
    ('INVERT', "Invert", "1 - A"),
    ('SCALE', "Scale", "A × the scale of the node"),
    ('ADD_SIGNED', "Add Signed", "A + B - 0.5"),
    ('DOT3', "Dot3", "4 × dot(A - 0.5, B - 0.5), a normal map"),
]

# Names of the A, B and C inputs per operation, None hides the input
OPERATION_INPUTS = {
    'MULTIPLY': ("A", "B", None),
    'ADD': ("A", "B", None),
    'SUBTRACT': ("A", "B", None),
    'MIX': ("A", "B", "Factor"),
    'MULTIPLY_ADD': ("A", "B", "C"),
    'ADD_MULTIPLY': ("A", "B", "C"),
    'INVERT': ("A", None, None),
    'SCALE': ("A", None, None),
    'ADD_SIGNED': ("A", "B", None),
    'DOT3': ("A", "B", None),
}

DATA_TYPE_ITEMS = [
    ('COLOR', "Color", "The operation works on colors (RGB)"),
    ('VALUE', "Value", "The operation works on values (alpha, one channel)"),
]

SCALE_ITEMS = [
    ('1', "× 1", "The result as it is"),
    ('2', "× 2", "The result is doubled, the vertex color of the game is halved and needs it"),
    ('4', "× 4", "The result is multiplied by 4"),
]

UNIT_ITEMS = [
    ('0', "Texture 0", "First texture slot of the material"),
    ('1', "Texture 1", "Second texture slot of the material"),
    ('2', "Texture 2", "Third texture slot of the material"),
    ('3', "Texture 3", "Procedural texture unit of the 3DS"),
]

COLUMN_WIDTH = 240
ROW_HEIGHT = 190

# Nodes made by the code are not a change of the user
building = False

# A texture lit by the vertex color, what an empty shader is filled with
DEFAULT_CODE = "color = texture0 * vertex_color * 2.0;\nalpha = texture0.a * vertex_color.a * 2.0;\n"

##########################################
# Shader Function
##########################################

def is_shader(tree):
    return tree is not None and tree.bl_idname == TREE_TYPE

def get_shaders():
    return [tree for tree in bpy.data.node_groups if tree.bl_idname == TREE_TYPE]

def get_shader_stages(tree):
    """The stages compiled from the nodes, None when the shader uses a combiner of the game or doesn't compile."""
    if tree.game_combiner or not tree.stages:
        return None
    return json.loads(tree.stages)

def make_render_default(tree, engine_id):
    stages = get_shader_stages(tree)

    combiner = tree.name
    if tree.game_combiner:
        combiner = tree.game_combiner

    if stages is None:
        fragment_lighting = render_defaults.find_combiner_fragment_lighting(engine_id, combiner)
    else:
        fragment_lighting = False
        for stage in stages:
            if cmb.CombinerStage.from_dict(stage).uses_fragment_lighting:
                fragment_lighting = True

    data = {
        "name": tree.name,
        "vertex_program": tree.vertex_program,
        "combiner": combiner,
        "skinned": render_defaults.find_vertex_program_skinned(engine_id, tree.vertex_program),
        "fragment_lighting": fragment_lighting,
        "stages": stages,
        "key": json.dumps([tree.name, tree.vertex_program, tree.game_combiner, stages]),
    }

    return render_defaults.RenderDefault(engine_id, data)

def get_mesh_shader(mesh):
    """The custom shader of a mesh, the copies StudioRender draws find it by the name of the mesh."""
    properties = mesh.level5_properties

    if is_shader(properties.shader):
        return properties.shader

    original = bpy.data.meshes.get(mesh.name)
    if original is not None and is_shader(original.level5_properties.shader):
        return original.level5_properties.shader

    return None

def clear_render_cache():
    # StudioRender keeps the programs it built from the combiners
    from . import project

    if project.is_studio_render_enabled():
        from .studio_render import material
        material.clear_cache()

##########################################
# Nodes -> code
##########################################

def find_output_node(tree):
    for node in tree.nodes:
        if node.bl_idname == "Level5NodeOutput":
            return node
    return None

def variable_name(node, socket, used):
    base = node.label or node.name
    if node.bl_idname == "Level5NodeSeparate":
        base = f"{base}_{socket.name}"

    name = re.sub(r"[^a-z0-9_]", "_", base.lower()).strip("_")
    if not name or name[0].isdigit():
        name = "v_" + name
    if name in combiner_code.SOURCE_NAMES or name in ("color", "alpha", "vec3", "float", "mix", "dot3", "add_signed"):
        name = "v_" + name

    unique = name
    index = 1
    while unique in used:
        unique = f"{name}_{index}"
        index += 1

    return unique

class NodeReader:
    """Reads the nodes from the output: a node output linked to several inputs is a variable of the code."""

    def __init__(self, tree):
        self.tree = tree
        self.links = {}
        self.uses = {}
        self.variables = []
        self.names = {}
        self.visiting = set()

        # A link just made is only marked valid after the update of the tree, the loops are found while reading
        for link in tree.links:
            if not link.is_muted and link.from_socket.enabled and link.to_socket.enabled:
                self.links[link.to_socket.as_pointer()] = link.from_socket

    def count(self, socket):
        key = socket.as_pointer()
        self.uses[key] = self.uses.get(key, 0) + 1

        if self.uses[key] == 1:
            node = socket.node
            for input_socket in node.inputs:
                if input_socket.enabled and input_socket.as_pointer() in self.links:
                    self.count(self.links[input_socket.as_pointer()])

    def input(self, socket):
        """The expression an input reads: the linked output, or the value of the input as a constant."""
        if socket.as_pointer() in self.links:
            source = self.links[socket.as_pointer()]

            if socket.type == 'VALUE' and source.type == 'RGBA':
                raise combiner_code.CodeError(f"{socket.node.name}: a color is linked to the value {socket.name}, read one channel with a Separate node")

            return self.output(source)

        if socket.type == 'RGBA':
            values = [min(max(value, 0.0), 1.0) for value in socket.default_value[0:3]]

            # A gray is written as a number, the alpha of the stage constant can give it
            if abs(values[0] - values[1]) < 1e-6 and abs(values[0] - values[2]) < 1e-6:
                return {"type": "number", "value": values[0]}

            return {"type": "vector", "values": values}

        return {"type": "number", "value": min(max(socket.default_value, 0.0), 1.0)}

    def output(self, socket):
        key = socket.as_pointer()
        if key in self.names:
            return {"type": "name", "name": self.names[key], "line": 0}

        node = socket.node
        if node.as_pointer() in self.visiting:
            raise combiner_code.CodeError(f"{node.name}: the nodes make a loop")

        self.visiting.add(node.as_pointer())
        expression = self.node_output(node, socket)
        self.visiting.discard(node.as_pointer())

        # A computed value read by several nodes is written once
        if self.uses.get(key, 0) > 1 and node.bl_idname in ("Level5NodeMath", "Level5NodeSeparate", "Level5NodeColor", "Level5NodeValue"):
            name = variable_name(node, socket, set(self.names.values()))
            self.names[key] = name
            self.variables.append({"name": name, "type": None, "expression": expression})
            return {"type": "name", "name": name, "line": 0}

        return expression

    def node_output(self, node, socket):
        index = list(node.outputs).index(socket)

        if node.bl_idname == "Level5NodeTexture":
            name = {"type": "name", "name": f"texture{node.unit}", "line": 0}
            if index == 1:
                return {"type": "swizzle", "argument": name, "swizzle": "a"}
            return name

        if node.bl_idname in ("Level5NodeVertexColor", "Level5NodeLighting"):
            name = "vertex_color"
            if node.bl_idname == "Level5NodeLighting":
                name = "light_diffuse"
                if index >= 2:
                    name = "light_specular"
            expression = {"type": "name", "name": name, "line": 0}
            if index % 2 == 1:
                return {"type": "swizzle", "argument": expression, "swizzle": "a"}
            return expression

        if node.bl_idname == "Level5NodeColor":
            return {"type": "vector", "values": list(node.color)}

        if node.bl_idname == "Level5NodeValue":
            return {"type": "number", "value": node.value}

        if node.bl_idname == "Level5NodeSeparate":
            return {"type": "swizzle", "argument": self.input(node.inputs[0]), "swizzle": "rgb"[index]}

        if node.bl_idname == "Level5NodeMath":
            return self.math(node)

        raise combiner_code.CodeError(f"{node.name} can't be part of a Level 5 shader")

    def math(self, node):
        offset = 0
        if node.data_type == 'VALUE':
            offset = 3

        a = self.input(node.inputs[offset])
        operation = node.operation

        if operation == 'INVERT':
            expression = {"type": "binary", "operator": "-", "left": {"type": "number", "value": 1.0}, "right": a}
        elif operation == 'SCALE':
            expression = a
        else:
            b = self.input(node.inputs[offset + 1])
            if operation == 'MULTIPLY':
                expression = {"type": "binary", "operator": "*", "left": a, "right": b}
            elif operation == 'ADD':
                expression = {"type": "binary", "operator": "+", "left": a, "right": b}
            elif operation == 'SUBTRACT':
                expression = {"type": "binary", "operator": "-", "left": a, "right": b}
            elif operation == 'ADD_SIGNED':
                expression = {"type": "call", "name": "add_signed", "arguments": [a, b]}
            elif operation == 'DOT3':
                expression = {"type": "call", "name": "dot3", "arguments": [a, b]}
            else:
                c = self.input(node.inputs[offset + 2])
                if operation == 'MIX':
                    expression = {"type": "call", "name": "mix", "arguments": [a, b, c]}
                elif operation == 'MULTIPLY_ADD':
                    product = {"type": "binary", "operator": "*", "left": a, "right": b}
                    expression = {"type": "binary", "operator": "+", "left": product, "right": c}
                else:
                    total = {"type": "binary", "operator": "+", "left": a, "right": b}
                    expression = {"type": "binary", "operator": "*", "left": total, "right": c}

        if node.scale != '1':
            expression = {"type": "binary", "operator": "*", "left": expression, "right": {"type": "number", "value": float(node.scale)}}

        return expression

    def read(self):
        output = find_output_node(self.tree)
        if output is None:
            raise combiner_code.CodeError("The shader has no Output node")

        # The tree is updated while the Output node makes its sockets
        if len(output.inputs) < 2:
            raise combiner_code.CodeError("The Output node isn't made yet")

        for socket in output.inputs:
            if socket.as_pointer() in self.links:
                self.count(self.links[socket.as_pointer()])

        alpha = self.input(output.inputs[1])
        color = self.input(output.inputs[0])

        return {"variables": self.variables, "color": color, "alpha": alpha}

def read_nodes(tree):
    """The program (parsed code) the nodes of a shader make."""
    return NodeReader(tree).read()

##########################################
# Code -> nodes
##########################################

class NodeBuilder:
    def __init__(self, tree, program):
        self.tree = tree
        self.program = program
        self.sources = {}
        self.variables = {}
        self.separates = {}

    def link(self, socket, target):
        self.tree.links.new(socket, target)

    def source_node(self, name):
        key = name
        if name in ("light_diffuse", "light_specular"):
            key = "lighting"

        if key not in self.sources:
            node_type = SOURCE_OUTPUTS[name][0]
            node = self.tree.nodes.new(node_type)
            if node_type == "Level5NodeTexture":
                node.unit = name[-1]
            self.sources[key] = node

        return self.sources[key]

    def build(self, expression, target):
        """Gives the expression to an input: a constant is the value of the input, the rest a linked output."""
        if expression["type"] == "number" and target is not None:
            if target.type == 'RGBA':
                target.default_value = (expression["value"], expression["value"], expression["value"], 1.0)
            else:
                target.default_value = expression["value"]
            return

        if expression["type"] == "vector" and target is not None and target.type == 'RGBA':
            target.default_value = (expression["values"][0], expression["values"][1], expression["values"][2], 1.0)
            return

        self.link(self.socket(expression), target)

    def socket(self, expression):
        """The output socket that gives an expression."""
        kind = expression["type"]

        if kind == "number":
            node = self.tree.nodes.new("Level5NodeValue")
            node.value = expression["value"]
            return node.outputs[0]

        if kind == "vector":
            node = self.tree.nodes.new("Level5NodeColor")
            node.color = expression["values"]
            return node.outputs[0]

        if kind == "name" and expression["name"] in SOURCE_OUTPUTS:
            node = self.source_node(expression["name"])
            return node.outputs[SOURCE_OUTPUTS[expression["name"]][1]]

        if kind == "name":
            name = expression["name"]
            if name not in self.variables:
                variable = combiner_code.get_variable(self.program, name)
                socket = self.socket(variable["expression"])
                if socket.node.bl_idname not in ("Level5NodeTexture", "Level5NodeVertexColor", "Level5NodeLighting"):
                    socket.node.label = name
                self.variables[name] = socket
            return self.variables[name]

        if kind == "swizzle":
            argument = expression["argument"]
            swizzle = expression["swizzle"]

            if swizzle == "rgb":
                return self.socket(argument)

            if swizzle == "a" and argument["type"] == "name" and argument["name"] in SOURCE_OUTPUTS:
                node = self.source_node(argument["name"])
                return node.outputs[SOURCE_OUTPUTS[argument["name"]][1] + 1]

            if swizzle == "a":
                raise combiner_code.CodeError("Only an input of the combiner has an alpha (.a), a computed color doesn't")

            key = id(argument)
            if argument["type"] == "name":
                key = argument["name"]
            if key not in self.separates:
                node = self.tree.nodes.new("Level5NodeSeparate")
                self.build(argument, node.inputs[0])
                self.separates[key] = node
            return self.separates[key].outputs["rgb".index(swizzle)]

        return self.math(expression)

    def math(self, expression):
        kind = expression["type"]

        # x * 2.0 and x * 4.0 are the scale of the node that computes x
        if kind == "binary" and expression["operator"] == "*" and combiner_code.is_scale(expression):
            value = expression["right"]
            other = expression["left"]
            if not combiner_code.is_number(value, 2.0) and not combiner_code.is_number(value, 4.0):
                value = expression["left"]
                other = expression["right"]

            socket = self.socket(other)
            node = socket.node
            if node.bl_idname == "Level5NodeMath" and node.scale == '1' and socket not in self.variables.values():
                node.scale = str(int(value["value"]))
                return socket

            node = self.new_math('SCALE', other)
            node.scale = str(int(value["value"]))
            self.build(other, self.math_input(node, 0))
            return self.math_output(node)

        if kind == "binary" and expression["operator"] == "-" and combiner_code.is_number(expression["left"], 1.0):
            node = self.new_math('INVERT', expression)
            self.build(expression["right"], self.math_input(node, 0))
            return self.math_output(node)

        # a * b + c and (a + b) * c are one node, like the stage of the game that computes them
        if kind == "binary" and expression["operator"] in ("+", "*"):
            fused = self.fused_arguments(expression)
            if fused is not None:
                operation = 'MULTIPLY_ADD'
                if expression["operator"] == "*":
                    operation = 'ADD_MULTIPLY'
                node = self.new_math(operation, expression)
                for index, argument in enumerate(fused):
                    self.build(argument, self.math_input(node, index))
                return self.math_output(node)

        if kind == "binary":
            operation = {"*": 'MULTIPLY', "+": 'ADD', "-": 'SUBTRACT'}[expression["operator"]]
            arguments = [expression["left"], expression["right"]]
        else:
            operation = {"mix": 'MIX', "add_signed": 'ADD_SIGNED', "dot3": 'DOT3'}[expression["name"]]
            arguments = expression["arguments"]

        node = self.new_math(operation, expression)
        for index, argument in enumerate(arguments):
            self.build(argument, self.math_input(node, index))

        return self.math_output(node)

    def fused_arguments(self, expression):
        """The A, B and C of a * b + c or (a + b) * c, None for another expression."""
        inner_operator = "*"
        if expression["operator"] == "*":
            inner_operator = "+"

        for inner, other in ((expression["left"], expression["right"]), (expression["right"], expression["left"])):
            if inner["type"] == "binary" and inner["operator"] == inner_operator and not combiner_code.is_scale(inner):
                return [inner["left"], inner["right"], other]

        return None

    def new_math(self, operation, expression):
        node = self.tree.nodes.new("Level5NodeMath")
        node.operation = operation
        if combiner_code.type_of(self.program, expression) == "value" and operation != 'DOT3':
            node.data_type = 'VALUE'
        else:
            node.data_type = 'COLOR'
        return node

    def math_input(self, node, index):
        if node.data_type == 'VALUE':
            return node.inputs[index + 3]
        return node.inputs[index]

    def math_output(self, node):
        if node.data_type == 'VALUE':
            return node.outputs[1]
        return node.outputs[0]

    def build_all(self):
        output = self.tree.nodes.new("Level5NodeOutput")
        self.build(self.program["color"], output.inputs[0])
        self.build(self.program["alpha"], output.inputs[1])
        arrange_nodes(self.tree, output)

def arrange_nodes(tree, output):
    """Columns from the output to the inputs, a node sits left of every node it feeds."""
    depth = {output.as_pointer(): 0}
    changed = True

    while changed:
        changed = False
        for link in tree.links:
            target = link.to_node.as_pointer()
            source = link.from_node.as_pointer()
            if target in depth and depth.get(source, -1) < depth[target] + 1:
                depth[source] = depth[target] + 1
                changed = True

    rows = {}
    for node in tree.nodes:
        column = depth.get(node.as_pointer(), 0)
        row = rows.get(column, 0)
        rows[column] = row + 1
        node.location = (-column * COLUMN_WIDTH, -row * ROW_HEIGHT)

def build_nodes(tree, program):
    global building

    building = True
    try:
        tree.nodes.clear()
        NodeBuilder(tree, program).build_all()
    finally:
        building = False

##########################################
# Synchronisation
##########################################

def get_code_text(tree):
    if tree.code is None:
        tree.code = bpy.data.texts.new(f"{tree.name}.glsl")
        tree.code.from_string(tree.node_code)
        tree.seen_code = tree.node_code
    return tree.code

def compile_shader(tree, program):
    try:
        stages = combiner_code.compile_program(program)
    except combiner_code.CodeError as error:
        tree.error = str(error)
        tree.stages = ""
        return False

    tree.error = ""
    tree.stages = json.dumps(stages)

    return True

def write_code(tree, text):
    get_code_text(tree).from_string(text)
    tree.node_code = text
    tree.seen_code = text

def same_stages(text, program):
    """Whether the code typed computes what the nodes compute, then it is kept as it was typed."""
    try:
        return combiner_code.compile_code(text) == combiner_code.compile_program(program)
    except combiner_code.CodeError:
        return False

def shader_changed(tree):
    """The nodes changed: the code is written again and compiled."""
    if building or not is_shader(tree):
        return

    try:
        program = read_nodes(tree)
    except combiner_code.CodeError as error:
        tree.error = str(error)
        clear_render_cache()
        return

    text = combiner_code.write_program(program)

    # A redraw of the nodes changes nothing, the combiner of the game stays used as it is
    if text == tree.node_code and tree.code is not None and tree.code.as_string().strip():
        return

    tree.node_code = text
    tree.game_combiner = ""

    code = get_code_text(tree)
    if not same_stages(code.as_string(), program):
        code.from_string(text)
        tree.seen_code = text

    compile_shader(tree, program)
    clear_render_cache()
    redraw_editors()

def apply_code(tree, text=None):
    """The code was edited: the nodes are made again from it, the code stays as it was typed. Returns the error, empty when it compiles."""
    if text is None:
        text = get_code_text(tree).as_string()

    tree.seen_code = text

    try:
        program = combiner_code.parse_code(text)
        combiner_code.compile_program(program)
    except combiner_code.CodeError as error:
        tree.error = str(error)
        redraw_editors()
        return str(error)

    build_nodes(tree, program)
    tree.node_code = combiner_code.write_program(read_nodes(tree))
    tree.game_combiner = ""
    compile_shader(tree, program)
    clear_render_cache()
    redraw_editors()

    return ""

def redraw_editors():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ('NODE_EDITOR', 'TEXT_EDITOR', 'VIEW_3D'):
                area.tag_redraw()

# Code typed and not applied yet, it is applied once it stops changing
_pending = {}

def watch_code():
    """Timer: the code of a shader typed in the Text Editor makes its nodes again when it compiles."""
    for tree in get_shaders():
        if tree.code is None:
            continue

        text = tree.code.as_string()
        if text == tree.seen_code:
            _pending.pop(tree.name, None)
            continue

        if _pending.get(tree.name) != text:
            _pending[tree.name] = text
            continue

        del _pending[tree.name]
        apply_code(tree, text)

    return 0.4

def set_shader_stages(tree, stages, game_combiner=""):
    """Makes the nodes and the code of a combiner (dicts of formats/material/cmb.py)."""
    text = combiner_code.decompile_code(stages)
    program = combiner_code.parse_code(text)

    build_nodes(tree, program)
    tree.game_combiner = game_combiner
    write_code(tree, combiner_code.write_program(read_nodes(tree)))

    # The combiner of the game is kept as it was read, the nodes only show it
    if game_combiner:
        tree.stages = json.dumps(stages)
        tree.error = ""
    else:
        compile_shader(tree, read_nodes(tree))

def fill_shader(tree, engine_id):
    """Gives an empty shader the nodes of a texture lit by the vertex color, and the vertex program of the engine default."""
    if not tree.vertex_program:
        tree.vertex_program = render_defaults.get_default_render_default(engine_id).data["vertex_program"]

    build_nodes(tree, combiner_code.parse_code(DEFAULT_CODE))
    tree.game_combiner = ""
    write_code(tree, combiner_code.write_program(read_nodes(tree)))
    compile_shader(tree, read_nodes(tree))
    clear_render_cache()

def new_shader(name, vertex_program, stages, game_combiner="", engine_id=None):
    tree = bpy.data.node_groups.new(name, TREE_TYPE)
    tree.use_fake_user = True
    tree.vertex_program = vertex_program
    tree.code = bpy.data.texts.new(f"{tree.name}.glsl")

    if stages:
        set_shader_stages(tree, stages, game_combiner)
    else:
        fill_shader(tree, engine_id)

    return tree

##########################################
# Import / Export
##########################################

def same_shader(tree, vertex_program, game_combiner, stages):
    if tree.vertex_program != vertex_program or tree.game_combiner != game_combiner:
        return False

    if game_combiner:
        return True

    return get_shader_stages(tree) == stages

def import_render_library(library, files, engine_id):
    """The render programs of an archive become shaders of the blend, returns {render program hash: shader}."""
    from .studio_render import resources

    shaders = {}
    combiners = {}

    for index, item in enumerate(library["combiners"]):
        data = files["cmb"].get(index)
        if data is not None:
            version, stages = cmb.read_cmb(data)
            combiners[item["name"]] = [stage.to_dict() for stage in stages]

    vertex_programs = {}
    for index, item in enumerate(library["vertex_programs"]):
        vertex_programs[item["name"]] = item

    for item in library["render_programs"]:
        game_combiner = ""
        stages = combiners.get(item["combiner"])

        # A combiner the archive doesn't hold is the one of the game
        if stages is None:
            game_combiner = item["combiner"]
            engine_stages = resources.load_combiner(engine_id, game_combiner)
            if engine_stages is not None:
                stages = [stage.to_dict() for stage in engine_stages]

        shader = None
        for tree in get_shaders():
            if tree.name == item["name"] and same_shader(tree, item["vertex_program"], game_combiner, stages):
                shader = tree

        if shader is None:
            shader = new_shader(item["name"], item["vertex_program"], stages, game_combiner)

        # A vertex program of the archive goes with the shader, with its binary
        embedded = vertex_programs.get(item["vertex_program"])
        if embedded is not None:
            binary_index = int.from_bytes(embedded["data"][6:8], 'little')
            shader.embedded_vertex_program = embedded["data"].hex()
            if binary_index < len(library["shader_binaries"]):
                shader.embedded_shader_entry = library["shader_binaries"][binary_index]["data"].hex()
                shader.embedded_shader_binary = files["shdc"].get(binary_index, bytes()).hex()

        shaders[zlib.crc32(item["name"].encode("shift-jis"))] = shader

    return shaders

def get_library_files(shaders, file_version):
    """The render library an archive needs for its shaders: the dict of formats/archive/res.py and the numbered files."""
    library = res.new_render_library()
    files = {}
    binaries = []

    for tree in shaders:
        if not tree.vertex_program:
            raise ValueError(f"Shader {tree.name} has no vertex program")

        combiner = tree.name
        if tree.game_combiner:
            combiner = tree.game_combiner

        library["render_programs"].append({"name": tree.name, "vertex_program": tree.vertex_program, "combiner": combiner})

        if not tree.game_combiner:
            stages = get_shader_stages(tree)
            if stages is None:
                raise ValueError(f"Shader {tree.name}: {tree.error or 'the nodes are not compiled'}")

            files[f"{len(library['combiners']):03d}.cmb"] = cmb.write_cmb([cmb.CombinerStage.from_dict(stage) for stage in stages], file_version)
            library["combiners"].append({"name": tree.name, "data": bytes()})

        if tree.embedded_vertex_program:
            binary = bytes.fromhex(tree.embedded_shader_binary)
            if binary not in binaries:
                binaries.append(binary)
                files[f"SHD{len(library['shader_binaries']):03d}.shdc"] = binary
                library["shader_binaries"].append({"name": "", "data": bytes.fromhex(tree.embedded_shader_entry)})

            # The vertex program points at its binary by its number in the archive
            entry = bytearray(bytes.fromhex(tree.embedded_vertex_program))
            entry[6:8] = binaries.index(binary).to_bytes(2, 'little')
            library["vertex_programs"].append({"name": tree.vertex_program, "data": bytes(entry)})

    return library, files

##########################################
# Register class
##########################################

def update_shader(self, context):
    shader_changed(self.id_data)

def update_vertex_program(self, context):
    clear_render_cache()

def get_vertex_program_items(self, context):
    from . import project
    from .studio_render import resources

    names = sorted(resources.load_vertex_programs(project.get_scene_engine_id(context.scene)))
    if self.vertex_program and self.vertex_program not in names:
        names.insert(0, self.vertex_program)

    items = [(name, name, "") for name in names]
    if not self.vertex_program:
        items.insert(0, ("NONE", "Choose a vertex program", "The shader has no vertex program yet"))

    return remember_items(items)

_items = {}

def remember_items(items):
    # Blender keeps a reference to the list of an enum
    key = tuple(item[0] for item in items)
    if key not in _items:
        _items[key] = items
    return _items[key]

def get_vertex_program_choice(self):
    names = [item[0] for item in get_vertex_program_items(self, bpy.context)]
    if self.vertex_program in names:
        return names.index(self.vertex_program)
    return 0

def set_vertex_program_choice(self, value):
    name = get_vertex_program_items(self, bpy.context)[value][0]
    if name != "NONE":
        self.vertex_program = name
    clear_render_cache()

class Level5ShaderTree(bpy.types.NodeTree):
    """The combiner and the vertex program of a custom render default, written in the render library of the archives"""
    bl_idname = TREE_TYPE
    bl_label = "Level 5 Shader"
    bl_icon = 'NODE_MATERIAL'

    vertex_program: StringProperty(
        name="Vertex Program",
        description="Vertex program of the shader, the game looks it up in the archive then in its fix.xr",
        update=update_vertex_program
    )
    vertex_program_choice: EnumProperty(
        name="Vertex Program",
        description="Vertex program of the shader: skinned (MDS), rigid (MDL), with an outline (SIL)...",
        items=get_vertex_program_items,
        get=get_vertex_program_choice,
        set=set_vertex_program_choice
    )
    game_combiner: StringProperty(
        name="Combiner of the Game",
        description="The shader uses this combiner of fix.xr as it is until its nodes are edited"
    )
    code: PointerProperty(type=bpy.types.Text)
    # The code of the nodes at the last synchronisation, and the last code typed that was read
    node_code: StringProperty()
    seen_code: StringProperty()
    stages: StringProperty()
    error: StringProperty()
    embedded_vertex_program: StringProperty()
    embedded_shader_entry: StringProperty()
    embedded_shader_binary: StringProperty()

    def update(self):
        shader_changed(self)

class Level5ShaderNode:
    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == TREE_TYPE

    def socket_value_update(self, context):
        shader_changed(self.id_data)

class Level5NodeOutput(Level5ShaderNode, bpy.types.Node):
    """The color and the alpha the combiner gives"""
    bl_idname = "Level5NodeOutput"
    bl_label = "Output"
    bl_icon = 'OUTPUT'

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Color")
        self.inputs.new('NodeSocketFloat', "Alpha").default_value = 1.0

class Level5NodeTexture(Level5ShaderNode, bpy.types.Node):
    """A texture slot of the material (Material > Level 5 > Textures)"""
    bl_idname = "Level5NodeTexture"
    bl_label = "Texture"
    bl_icon = 'TEXTURE'

    unit: EnumProperty(name="Texture", items=UNIT_ITEMS, default='0', update=update_shader)

    def init(self, context):
        self.outputs.new('NodeSocketColor', "Color")
        self.outputs.new('NodeSocketFloat', "Alpha")

    def draw_buttons(self, context, layout):
        layout.prop(self, "unit", text="")

class Level5NodeVertexColor(Level5ShaderNode, bpy.types.Node):
    """The color of the vertices (Tint) times the color of the material, halved by the vertex program"""
    bl_idname = "Level5NodeVertexColor"
    bl_label = "Vertex Color"
    bl_icon = 'GROUP_VCOL'

    def init(self, context):
        self.outputs.new('NodeSocketColor', "Color")
        self.outputs.new('NodeSocketFloat', "Alpha")

class Level5NodeLighting(Level5ShaderNode, bpy.types.Node):
    """The lighting of the material (.mtr): diffuse and specular"""
    bl_idname = "Level5NodeLighting"
    bl_label = "Lighting"
    bl_icon = 'LIGHT'

    def init(self, context):
        self.outputs.new('NodeSocketColor', "Diffuse")
        self.outputs.new('NodeSocketFloat', "Diffuse Alpha")
        self.outputs.new('NodeSocketColor', "Specular")
        self.outputs.new('NodeSocketFloat', "Specular Alpha")

class Level5NodeColor(Level5ShaderNode, bpy.types.Node):
    """A constant color"""
    bl_idname = "Level5NodeColor"
    bl_label = "Color"
    bl_icon = 'COLOR'

    color: FloatVectorProperty(name="Color", subtype='COLOR_GAMMA', size=3, default=(1.0, 1.0, 1.0), min=0.0, max=1.0, update=update_shader)

    def init(self, context):
        self.outputs.new('NodeSocketColor', "Color")

    def draw_buttons(self, context, layout):
        layout.prop(self, "color", text="")

class Level5NodeValue(Level5ShaderNode, bpy.types.Node):
    """A constant value"""
    bl_idname = "Level5NodeValue"
    bl_label = "Value"

    value: FloatProperty(name="Value", default=1.0, min=0.0, max=1.0, update=update_shader)

    def init(self, context):
        self.outputs.new('NodeSocketFloat', "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value", text="")

class Level5NodeSeparate(Level5ShaderNode, bpy.types.Node):
    """The red, green and blue of a color as values"""
    bl_idname = "Level5NodeSeparate"
    bl_label = "Separate Color"

    def init(self, context):
        self.inputs.new('NodeSocketColor', "Color")
        self.outputs.new('NodeSocketFloat', "R")
        self.outputs.new('NodeSocketFloat', "G")
        self.outputs.new('NodeSocketFloat', "B")

def update_math(self, context):
    self.update_sockets()
    shader_changed(self.id_data)

class Level5NodeMath(Level5ShaderNode, bpy.types.Node):
    """An operation of the combiner, every result is clamped to 0..1 like the game does"""
    bl_idname = "Level5NodeMath"
    bl_label = "Math"

    operation: EnumProperty(name="Operation", items=OPERATION_ITEMS, default='MULTIPLY', update=update_math)
    data_type: EnumProperty(name="Data Type", items=DATA_TYPE_ITEMS, default='COLOR', update=update_math)
    scale: EnumProperty(name="Scale", items=SCALE_ITEMS, default='1', update=update_shader)

    def init(self, context):
        for name in ("A", "B", "C"):
            self.inputs.new('NodeSocketColor', name)
        for name in ("A", "B", "C"):
            self.inputs.new('NodeSocketFloat', name).default_value = 1.0
        self.outputs.new('NodeSocketColor', "Result")
        self.outputs.new('NodeSocketFloat', "Result")
        self.update_sockets()

    def update_sockets(self):
        names = OPERATION_INPUTS[self.operation]
        is_value = self.data_type == 'VALUE'

        for index, name in enumerate(names):
            color_socket = self.inputs[index]
            value_socket = self.inputs[index + 3]
            color_socket.enabled = name is not None and not is_value
            value_socket.enabled = name is not None and is_value
            if name is not None:
                color_socket.name = name
                value_socket.name = name

        self.outputs[0].enabled = not is_value
        self.outputs[1].enabled = is_value

    def draw_label(self):
        for identifier, name, description in OPERATION_ITEMS:
            if identifier == self.operation:
                return name
        return "Math"

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation", text="")
        row = layout.row(align=True)
        row.prop(self, "data_type", text="")
        row.prop(self, "scale", text="")

classes = (
    Level5ShaderTree,
    Level5NodeOutput,
    Level5NodeTexture,
    Level5NodeVertexColor,
    Level5NodeLighting,
    Level5NodeColor,
    Level5NodeValue,
    Level5NodeSeparate,
    Level5NodeMath,
)
