import os
import bpy
import json
import zlib

from bpy.props import StringProperty, BoolProperty, IntProperty

from ..formats import cmb, res
from ..compression import compress
from . import render_defaults, combiner_code, shader_code, vertex_code
from .shader_code import CodeError

##########################################
# CONST
##########################################

# The Text Editor of Blender colors a text by the extension of its name, OSL is the C-like language it knows
SHADER_EXTENSION = ".osl"
SHADER_EXTENSIONS = (".osl", ".glsl")

TEMPLATE_EXTENSION = ".glsl"

TEMPLATE_DIRECTORY = os.path.join(os.path.dirname(__file__), "shader_templates")

# The binaries of an archive take numbers after those of fix.xr (the draw only uploads a code whose number changes),
# spread over a range so that two models drawn together rarely take the same one
FIRST_FREE_BINARY = 5
BINARY_SPREAD = 32

# Entry of a binary that holds one vertex program and no geometry program
CUSTOM_BINARY_ENTRY = bytes([1, 0, 0, 0])

# Entry of a vertex program after its hash and name: no GLSL, program 0 of its binary, no geometry program
CUSTOM_VERTEX_PROGRAM_ENTRY = bytes([0, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0])

# A texture lit by the vertex color, what a new shader starts with
DEFAULT_TEMPLATE = "texture_vertex_color.glsl"

# Code typed and not compiled yet, it is compiled once it stops changing
pending = {}

# Vertex stages written from the vertex binaries of imported archives, by (entry, binary)
embedded_vertex_stages = {}

##########################################
# Shader Function
##########################################

def is_shader(text):
    return text is not None and isinstance(text, bpy.types.Text) and text.level5_shader.is_shader

def get_shaders():
    return [text for text in bpy.data.texts if text.level5_shader.is_shader]

def get_shader_name(text):
    """The name of the render program of a shader: the name of its text without .osl."""
    name = text.name
    for extension in SHADER_EXTENSIONS:
        if name.endswith(extension):
            return name[:-len(extension)]
    return name

def find_shader(name):
    for text in get_shaders():
        if get_shader_name(text) == name:
            return text
    return None

def get_mesh_shader(mesh):
    """The custom shader of a mesh, the copies StudioRender draws find it by the name of the mesh."""
    properties = mesh.level5_properties

    if is_shader(properties.shader):
        return properties.shader

    original = bpy.data.meshes.get(mesh.name)
    if original is not None and is_shader(original.level5_properties.shader):
        return original.level5_properties.shader

    return None

def get_shader_stages(text):
    """The stages compiled from the FRAGMENT_SHADER section, the embedded ones of a #pragma combiner, or None."""
    properties = text.level5_shader

    if properties.stages:
        return json.loads(properties.stages)

    if properties.combiner and properties.combiner == properties.embedded_combiner_name and properties.embedded_combiner:
        return json.loads(properties.embedded_combiner)

    return None

def get_vertex_program_name(text):
    properties = text.level5_shader

    if properties.vertex_program:
        return properties.vertex_program

    return f"{get_shader_name(text)}_VTX"

def get_combiner_name(text):
    properties = text.level5_shader

    if properties.combiner:
        return properties.combiner

    return get_shader_name(text)

def make_render_default(text, engine_id):
    properties = text.level5_shader
    stages = get_shader_stages(text)
    combiner = get_combiner_name(text)
    vertex_program = get_vertex_program_name(text)

    if stages is None:
        fragment_lighting = render_defaults.find_combiner_fragment_lighting(engine_id, combiner)
    else:
        fragment_lighting = False
        for stage in stages:
            if cmb.CombinerStage.from_dict(stage).uses_fragment_lighting:
                fragment_lighting = True

    if properties.vertex_glsl:
        skinned = "studio_unf_vtx_bone" in properties.vertex_glsl
    else:
        skinned = render_defaults.find_vertex_program_skinned(engine_id, vertex_program)

    # The vertex program of an imported archive is only a binary, StudioRender draws it written as GLSL
    vertex_glsl = properties.vertex_glsl
    if not vertex_glsl and vertex_program == properties.embedded_vertex_program_name and properties.embedded_shader_binary:
        vertex_glsl = get_embedded_vertex_glsl(text)

    data = {
        "name": get_shader_name(text),
        "vertex_program": vertex_program,
        "combiner": combiner,
        "skinned": skinned,
        "fragment_lighting": fragment_lighting,
        "stages": stages,
        "vertex_glsl": vertex_glsl,
        "key": json.dumps([get_shader_name(text), vertex_program, combiner, stages, vertex_glsl]),
    }

    return render_defaults.RenderDefault(engine_id, data)

def clear_render_cache():
    # StudioRender keeps the programs it built from the combiners
    from . import project

    if project.is_studio_render_enabled():
        from .studio_render import material
        material.clear_cache()

def redraw_editors():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ('TEXT_EDITOR', 'VIEW_3D', 'PROPERTIES'):
                area.tag_redraw()

##########################################
# Compile
##########################################

def get_embedded_vertex_glsl(text):
    """The vertex binary of an imported archive written as a StudioRender vertex stage, empty when it can't be written."""
    from .studio_render import vertex_binary

    properties = text.level5_shader
    key = (properties.embedded_vertex_program, properties.embedded_shader_binary)

    if key not in embedded_vertex_stages:
        embedded_vertex_stages[key] = ""

        try:
            entry = bytes.fromhex(properties.embedded_vertex_program)
            data = bytes.fromhex(properties.embedded_shader_binary)
            embedded_vertex_stages[key] = vertex_binary.make_vertex_glsl(data, vertex_binary.get_program_index(entry))
        except Exception as error:
            print(f"[Studio Eleven] {get_shader_name(text)}: the vertex program {properties.embedded_vertex_program_name} is drawn with the vertex stage of StudioRender ({error})")

    return embedded_vertex_stages[key]

def compile_shader(text):
    """Compiles the code of a shader into its properties, returns the error, empty when it compiles."""
    properties = text.level5_shader
    code = text.as_string()
    properties.compiled_code = code

    # A shader of an older blend takes the extension the Text Editor colors
    if not text.name.endswith(SHADER_EXTENSION):
        name = get_shader_name(text) + SHADER_EXTENSION
        if bpy.data.texts.get(name) is None:
            text.name = name

    try:
        parsed = shader_code.parse_shader(code)

        stages = ""
        combiner = parsed["pragmas"].get("combiner", "")
        if parsed["fragment"] is not None:
            stages = json.dumps(combiner_code.compile_fragment(parsed["fragment"]))
        elif not combiner:
            raise CodeError("The shader needs a FRAGMENT_SHADER section or a #pragma combiner \"#NAME\"")

        vertex_glsl = ""
        instruction_count = 0
        vertex_program = parsed["pragmas"].get("vertex_program", "")
        if parsed["vertex"] is not None:
            compiled = vertex_code.compile_vertex(parsed["vertex"], 2)
            instruction_count = compiled["instruction_count"]
            vertex_glsl = vertex_code.make_blender_vertex(parsed["vertex"])
        elif not vertex_program:
            raise CodeError("The shader needs a VERTEX_SHADER section or a #pragma vertex_program \"#NAME\"")

    except CodeError as error:
        properties.error = str(error)
        redraw_editors()
        return str(error)

    properties.error = ""
    properties.stages = stages
    properties.combiner = combiner
    properties.vertex_program = vertex_program
    properties.vertex_glsl = vertex_glsl
    properties.instruction_count = instruction_count
    properties.stage_count = 0
    if stages:
        properties.stage_count = len(json.loads(stages))

    clear_render_cache()
    redraw_editors()

    return ""

def watch_shaders():
    """Timer: the code of a shader is compiled once it stops changing."""
    for text in get_shaders():
        code = text.as_string()
        if code == text.level5_shader.compiled_code:
            pending.pop(text.name, None)
            continue

        if pending.get(text.name) != code:
            pending[text.name] = code
            continue

        del pending[text.name]
        compile_shader(text)

    return 0.4

def new_shader(name, code):
    text = bpy.data.texts.new(name + SHADER_EXTENSION)
    text.from_string(code)
    text.use_fake_user = True
    text.level5_shader.is_shader = True

    # The cursor at the start, the editor opens on the first line
    text.current_line_index = 0
    text.current_character = 0
    text.select_end_line_index = 0
    text.select_end_character = 0

    compile_shader(text)

    return text

##########################################
# Templates
##########################################

def get_templates():
    """The shipped templates: [{file_name, label}], the label is the first line of the file after '// '."""
    templates = []

    if not os.path.isdir(TEMPLATE_DIRECTORY):
        return templates

    for file_name in sorted(os.listdir(TEMPLATE_DIRECTORY)):
        if not file_name.endswith(TEMPLATE_EXTENSION):
            continue

        with open(os.path.join(TEMPLATE_DIRECTORY, file_name), "r", encoding="utf-8") as file:
            first_line = file.readline().strip()

        label = file_name[:-len(TEMPLATE_EXTENSION)]
        if first_line.startswith("//"):
            label = first_line[2:].strip()

        templates.append({"file_name": file_name, "label": label})

    return templates

def read_template(file_name):
    with open(os.path.join(TEMPLATE_DIRECTORY, file_name), "r", encoding="utf-8") as file:
        return file.read()

def make_code_from_render_default(engine_id, render_default):
    """The code of a render default of a game: its vertex program by name, its combiner as code."""
    from .studio_render import resources

    lines = [f"// Copy of the render default {render_default.name}"]
    lines.append(f"#pragma vertex_program \"{render_default.data['vertex_program']}\"")
    lines.append("")

    stages = resources.render_default_stages(engine_id, render_default)
    fragment = None

    if stages:
        stage_dicts = [stage.to_dict() for stage in stages]
        if not combiner_code.uses_palette(stage_dicts):
            try:
                fragment = combiner_code.decompile_code(stage_dicts)
            except CodeError:
                fragment = None

    if fragment is None:
        # The colors of the masks of the characters come from the model, only the combiner of the game knows them
        lines.append(f"#pragma combiner \"{render_default.data['combiner']}\"")
    else:
        lines.append(fragment.rstrip("\n"))

    return "\n".join(lines) + "\n"

##########################################
# Import / Export
##########################################

def import_render_library(library, files, engine_id):
    """The render programs of an archive become shaders of the blend, returns {render program hash: shader}."""
    shaders = {}
    combiners = {}

    for index, item in enumerate(library["combiners"]):
        data = files["cmb"].get(index)
        if data is not None:
            read = cmb.read_cmb(data)
            combiners[item["name"]] = [stage.to_dict() for stage in read[1]]

    vertex_programs = {}
    for item in library["vertex_programs"]:
        vertex_programs[item["name"]] = item

    for item in library["render_programs"]:
        lines = []
        embedded_combiner = None

        embedded = vertex_programs.get(item["vertex_program"])
        if embedded is not None:
            lines.append(f"// The vertex program {item['vertex_program']} is a binary of the archive, kept as it is")
        lines.append(f"#pragma vertex_program \"{item['vertex_program']}\"")
        lines.append("")

        stages = combiners.get(item["combiner"])
        if stages is None:
            # A combiner the archive doesn't hold is the one of the game
            lines.append(f"#pragma combiner \"{item['combiner']}\"")
        elif combiner_code.uses_palette(stages):
            lines.append(f"// The combiner {item['combiner']} of the archive uses the mask colors of the model, kept as it is")
            lines.append(f"#pragma combiner \"{item['combiner']}\"")
            embedded_combiner = stages
        else:
            try:
                lines.append(combiner_code.decompile_code(stages).rstrip("\n"))
            except CodeError:
                lines.append(f"#pragma combiner \"{item['combiner']}\"")
                embedded_combiner = stages

        code = "\n".join(lines) + "\n"

        shader = find_shader(item["name"])
        if shader is None or shader.as_string() != code:
            shader = new_shader(item["name"], code)

        if embedded_combiner is not None:
            shader.level5_shader.embedded_combiner_name = item["combiner"]
            shader.level5_shader.embedded_combiner = json.dumps(embedded_combiner)

        # A vertex program of the archive goes with the shader, with its binary
        if embedded is not None:
            binary_index = int.from_bytes(embedded["data"][6:8], 'little')
            shader.level5_shader.embedded_vertex_program_name = item["vertex_program"]
            shader.level5_shader.embedded_vertex_program = embedded["data"].hex()
            if binary_index < len(library["shader_binaries"]):
                shader.level5_shader.embedded_shader_entry = library["shader_binaries"][binary_index]["data"].hex()
                shader.level5_shader.embedded_shader_binary = files["shdc"].get(binary_index, bytes()).hex()

        compile_shader(shader)
        shaders[zlib.crc32(item["name"].encode("shift-jis"))] = shader

    return shaders

def get_first_free_binary(engine_id):
    """The first binary number fix.xr of the game doesn't use, a default engine takes the highest of the games."""
    from .studio_render import resources

    highest = -1
    for entry in resources.load_vertex_programs(engine_id).values():
        if entry.get("shader_binary") is not None:
            highest = max(highest, entry["shader_binary"])

    if highest < 0:
        return FIRST_FREE_BINARY

    return highest + 1

def add_binary(binaries, name, data, entry, first):
    """Gives a binary its number in the archive: after those of fix.xr, at a place taken from its name."""
    for index, binary in binaries.items():
        if binary["data"] == data:
            return index

    index = first + zlib.crc32(name.encode("shift-jis")) % BINARY_SPREAD
    while index in binaries:
        index += 1

    binaries[index] = {"data": data, "entry": entry}

    return index

def get_library_files(shaders, file_version, engine_id):
    """The render library an archive needs for its shaders: the dict of formats/archive/res.py and the numbered files."""
    library = res.new_render_library()
    files = {}
    binaries = {}
    vertex_programs = []
    first = get_first_free_binary(engine_id)

    for text in shaders:
        name = get_shader_name(text)
        properties = text.level5_shader

        try:
            parsed = shader_code.parse_shader(text.as_string())
        except CodeError as error:
            raise ValueError(f"Shader {name}: {error}")

        vertex_program = parsed["pragmas"].get("vertex_program")
        combiner = parsed["pragmas"].get("combiner")

        if parsed["vertex"] is not None:
            vertex_program = f"{name}_VTX"
            try:
                compiled = vertex_code.compile_vertex(parsed["vertex"], file_version)
            except CodeError as error:
                raise ValueError(f"Shader {name}: {error}")

            index = add_binary(binaries, vertex_program, compress(compiled["data"]), CUSTOM_BINARY_ENTRY, first)
            entry = bytearray(CUSTOM_VERTEX_PROGRAM_ENTRY)
            entry[6:8] = index.to_bytes(2, 'little')
            vertex_programs.append({"name": vertex_program, "data": bytes(entry)})

        elif vertex_program is not None and vertex_program == properties.embedded_vertex_program_name and properties.embedded_shader_binary:
            data = bytes.fromhex(properties.embedded_shader_binary)
            index = add_binary(binaries, vertex_program, data, bytes.fromhex(properties.embedded_shader_entry), first)
            entry = bytearray(bytes.fromhex(properties.embedded_vertex_program))
            entry[6:8] = index.to_bytes(2, 'little')
            vertex_programs.append({"name": vertex_program, "data": bytes(entry)})

        elif vertex_program is None:
            raise ValueError(f"Shader {name} has no VERTEX_SHADER section and no #pragma vertex_program")

        if parsed["fragment"] is not None:
            combiner = name
            try:
                stages = combiner_code.compile_fragment(parsed["fragment"])
            except CodeError as error:
                raise ValueError(f"Shader {name}: {error}")

            files[f"{len(library['combiners']):03d}.cmb"] = cmb.write_cmb([cmb.CombinerStage.from_dict(stage) for stage in stages], file_version)
            library["combiners"].append({"name": combiner, "data": bytes()})

        elif combiner is not None and combiner == properties.embedded_combiner_name and properties.embedded_combiner:
            stages = json.loads(properties.embedded_combiner)
            files[f"{len(library['combiners']):03d}.cmb"] = cmb.write_cmb([cmb.CombinerStage.from_dict(stage) for stage in stages], file_version)
            library["combiners"].append({"name": combiner, "data": bytes()})

        elif combiner is None:
            raise ValueError(f"Shader {name} has no FRAGMENT_SHADER section and no #pragma combiner")

        library["render_programs"].append({"name": name, "vertex_program": vertex_program, "combiner": combiner})

    for vertex_program in vertex_programs:
        if vertex_program not in library["vertex_programs"]:
            library["vertex_programs"].append(vertex_program)

    # The loader reads the binaries 0 to the highest number, the numbers without a file are skipped
    if binaries:
        for index in range(max(binaries) + 1):
            entry = CUSTOM_BINARY_ENTRY
            if index in binaries:
                entry = binaries[index]["entry"]
                files[f"SHD{index:03d}.shdc"] = binaries[index]["data"]
            library["shader_binaries"].append({"name": "", "data": entry})

    return library, files

##########################################
# Register class
##########################################

class Level5ShaderProperties(bpy.types.PropertyGroup):
    is_shader: BoolProperty(
        name="Level 5 Shader",
        description="The text is a custom shader of the blend, the meshes can use it and the export writes it in their archive",
        default=False
    )
    # The last code compiled and what it gave
    compiled_code: StringProperty()
    error: StringProperty()
    stages: StringProperty()
    combiner: StringProperty()
    vertex_program: StringProperty()
    vertex_glsl: StringProperty()
    stage_count: IntProperty()
    instruction_count: IntProperty()
    # A vertex program or a combiner of an imported archive, written back when the code names it
    embedded_vertex_program_name: StringProperty()
    embedded_vertex_program: StringProperty()
    embedded_shader_entry: StringProperty()
    embedded_shader_binary: StringProperty()
    embedded_combiner_name: StringProperty()
    embedded_combiner: StringProperty()

classes = (
    Level5ShaderProperties,
)
