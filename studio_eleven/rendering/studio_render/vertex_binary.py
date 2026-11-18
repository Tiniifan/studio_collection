"""A PICA200 vertex program (a DVLE of a SHDnnn.shdc) written as a vertex stage of StudioRender.

The instructions become GLSL statements on the registers of the program: v0-v15 filled from the attributes StudioRender has,
c0-c95 from its uniforms by their name in the uniform table, then the constants of the DVLE. The outputs go to the frg_* of
the fragment stage. The mesh is already deformed by its armature, so every bone is the identity, as in make_blender_vertex.
"""
from ...compression import compressor
from ...formats.material import shdc
from . import shaders

##########################################
# CONST
##########################################

INPUT_COUNT = 16
TEMPORARY_COUNT = 16
OUTPUT_COUNT = 16
FLOAT_UNIFORM_COUNT = 96
INT_UNIFORM_COUNT = 4
BOOL_UNIFORM_COUNT = 16

# Registers of the uniform table: inputs, then the float, int and bool uniforms
FLOAT_UNIFORM_START = 0x10
INT_UNIFORM_START = 0x70
BOOL_UNIFORM_START = 0x78

# Calls are inlined, the PICA200 has a call stack of 4
MAX_CALL_DEPTH = 8

# The texture coordinates are read and written as the game does (shaders.NATIVE_UV_GLSL): some programs keep data in them
ATTRIBUTES = {
    "atr_pos": "vec4(atr_pos, 1.0)",
    "atr_clr": "atr_clr",
    "atr_nrm": "vec4(atr_nrm, 1.0)",
    "atr_tng": "vec4(1.0, 0.0, 0.0, 1.0)",
    "atr_tx0": "studio_file_uv(atr_tx0)",
    "atr_tx1": "studio_file_uv(atr_tx1)",
    "atr_tx2": "studio_file_uv(atr_tx2)",
    "atr_pr0": "vec4(1.0, 0.0, 0.0, 0.0)",
    "atr_pr1": "vec4(0.0)",
    "atr_pr2": "vec4(0.0)",
}

# The game gives the rows of its matrices
MATRIX_UNIFORMS = {
    "unf_vtx_cmr_prj": "unf_vtx_cmr_prj",
    "unf_vtx_glb_cmr": "unf_vtx_glb_cmr",
    "unf_vtx_lcl_glb": "unf_vtx_lcl_glb",
}

# The outline is drawn by its own pass, here it is off
VECTOR_UNIFORMS = {
    "unf_vtx_clr": "unf_vtx_clr",
    "unf_vtx_eye": "vec4(inverse(unf_vtx_glb_cmr)[3].xyz, 1.0)",
    "unf_vtx_silhouette_0": "vec4(0.0, 0.01, 100000.0, 0.0)",
    "unf_vtx_silhouette_1": "vec4(0.0)",
    "unf_vtx_str": "vec4(0.0)",
}

COMPARE_OPERATORS = {
    0: "==",
    1: "!=",
    2: "<",
    3: "<=",
    4: ">",
    5: ">=",
}

# Opcodes of the format 1 instructions, by the number of sources they read
UNARY_OPCODES = (0x05, 0x06, 0x0B, 0x0E, 0x0F, 0x12, 0x13)
BINARY_OPCODES = (0x00, 0x01, 0x02, 0x03, 0x04, 0x08, 0x09, 0x0A, 0x0C, 0x0D)
INVERTED_OPCODES = (0x18, 0x19, 0x1A, 0x1B)

# The 3DS gives 0 for 0 x infinity, where GLSL gives NaN
HELPERS_GLSL = """
vec4 studio_v[16];
vec4 studio_r[16];
vec4 studio_o[16];
vec4 studio_c[96];
ivec2 studio_a0;
int studio_al;
bvec2 studio_cmp;

vec4 studio_mul(vec4 a, vec4 b) {
    vec4 product = a * b;
    return mix(product, mix(mix(vec4(0.0), product, isnan(b)), product, isnan(a)), isnan(product));
}

vec4 studio_uniform(int index) {
    if (index < 0 || index >= 96) {
        return vec4(1.0);
    }
    return studio_c[index];
}

vec3 studio_rotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

const vec4 studio_bone_rows[3] = vec4[3](vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0));
"""

##########################################
# Vertex Binary Function
##########################################

def get_shader(data):
    """The DVLB of a SHDnnn.shdc as shdc.open_shdc reads it, the file of an archive is compressed."""
    if data[:4] != b"DVLB":
        data = compressor.decompress(data)

    return shdc.open_shdc(data)

def get_program_index(entry):
    """The DVLE of a vertex program entry of a render library (the bytes after its hash and name)."""
    return entry[3]

def get_binary_index(entry):
    return int.from_bytes(entry[6:8], 'little')

def get_uniform_name(name):
    # The attributes are named with their components, atr_pos.xyzw
    return name.split(".")[0]

def get_swizzle(bits):
    return shdc.get_swizzle_text(bits)

def get_mask(bits):
    mask = ""

    for index, component in enumerate("xyzw"):
        if bits & (8 >> index):
            mask += component

    return mask

def float_text(value):
    text = repr(float(value))

    if text in ("inf", "-inf", "nan"):
        return "0.0"

    return text

def read_source(register, offset):
    if register < FLOAT_UNIFORM_START:
        return f"studio_v[{register}]"

    if register < 0x20:
        return f"studio_r[{register - 0x10}]"

    index = register - 0x20

    if offset == 1:
        return f"studio_uniform({index} + studio_a0.x)"
    elif offset == 2:
        return f"studio_uniform({index} + studio_a0.y)"
    elif offset == 3:
        return f"studio_uniform({index} + studio_al)"

    return f"studio_c[{index}]"

def read_operand(register, offset, descriptor, source):
    shift = 4 + 9 * source
    negate = (descriptor >> shift) & 1
    swizzle = get_swizzle((descriptor >> (shift + 1)) & 0xFF)

    text = f"{read_source(register, offset)}.{swizzle}"
    if negate:
        text = f"(-{text})"

    return text

def write_destination(register):
    if register < 0x10:
        return f"studio_o[{register}]"

    return f"studio_r[{register - 0x10}]"

def write_assignment(register, descriptor, value):
    mask = get_mask(descriptor & 0xF)

    if mask == "":
        return []

    return [f"{write_destination(register)}.{mask} = ({value}).{mask};"]

def get_operation(opcode, sources):
    first = sources[0]
    second = ""
    if len(sources) > 1:
        second = sources[1]

    if opcode == 0x00:
        return f"{first} + {second}"
    elif opcode == 0x01:
        return f"vec4(dot(studio_mul({first}, {second}).xyz, vec3(1.0)))"
    elif opcode == 0x02:
        return f"vec4(dot(studio_mul({first}, {second}), vec4(1.0)))"
    elif opcode in (0x03, 0x18):
        return f"vec4(dot(studio_mul(vec4({first}.xyz, 1.0), {second}), vec4(1.0)))"
    elif opcode in (0x04, 0x19):
        return f"vec4(1.0, {first}.y * {second}.y, {first}.z, {second}.w)"
    elif opcode == 0x05:
        return f"vec4(exp2({first}.x))"
    elif opcode == 0x06:
        return f"vec4(log2({first}.x))"
    elif opcode == 0x08:
        return f"studio_mul({first}, {second})"
    elif opcode in (0x09, 0x1A):
        return f"vec4(greaterThanEqual({first}, {second}))"
    elif opcode in (0x0A, 0x1B):
        return f"vec4(lessThan({first}, {second}))"
    elif opcode == 0x0B:
        return f"floor({first})"
    elif opcode == 0x0C:
        return f"max({first}, {second})"
    elif opcode == 0x0D:
        return f"min({first}, {second})"
    elif opcode == 0x0E:
        return f"vec4(1.0 / {first}.x)"
    elif opcode == 0x0F:
        return f"vec4(inversesqrt({first}.x))"
    elif opcode == 0x13:
        return first

    raise Exception(f"Unsupported vertex instruction: {opcode:#04x}")

def get_condition(word):
    refx = (word >> 25) & 1
    refy = (word >> 24) & 1
    operation = (word >> 22) & 3

    x = "studio_cmp.x"
    if not refx:
        x = "!studio_cmp.x"

    y = "studio_cmp.y"
    if not refy:
        y = "!studio_cmp.y"

    if operation == 0:
        return f"({x} || {y})"
    elif operation == 1:
        return f"({x} && {y})"
    elif operation == 2:
        return f"({x})"

    return f"({y})"

##########################################
# Vertex Binary Write Function
##########################################

def write_instruction(shader, word):
    opcode = word >> 26
    descriptors = shader["operand_descriptors"]

    if opcode >= 0x30:
        destination = (word >> 24) & 0x1F
        offset = (word >> 22) & 3
        descriptor = descriptors[word & 0x1F]
        first = read_operand((word >> 17) & 0x1F, 0, descriptor, 0)

        # mad reads a uniform through its second source, madi through its third
        if opcode >= 0x38:
            second = read_operand((word >> 10) & 0x7F, offset, descriptor, 1)
            third = read_operand((word >> 5) & 0x1F, 0, descriptor, 2)
        else:
            second = read_operand((word >> 12) & 0x1F, 0, descriptor, 1)
            third = read_operand((word >> 5) & 0x7F, offset, descriptor, 2)

        return write_assignment(destination, descriptor, f"studio_mul({first}, {second}) + {third}")

    if opcode in (0x2E, 0x2F):
        compare_x = (word >> 24) & 7
        compare_y = (word >> 21) & 7
        offset = (word >> 19) & 3
        descriptor = descriptors[word & 0x7F]
        first = read_operand((word >> 12) & 0x7F, offset, descriptor, 0)
        second = read_operand((word >> 7) & 0x1F, 0, descriptor, 1)

        lines = []
        for lane, compare in (("x", compare_x), ("y", compare_y)):
            if compare in COMPARE_OPERATORS:
                lines.append(f"studio_cmp.{lane} = {first}.{lane} {COMPARE_OPERATORS[compare]} {second}.{lane};")
            else:
                lines.append(f"studio_cmp.{lane} = true;")

        return lines

    destination = (word >> 21) & 0x1F
    offset = (word >> 19) & 3
    descriptor = descriptors[word & 0x7F]

    if opcode in INVERTED_OPCODES:
        sources = [read_operand((word >> 14) & 0x1F, 0, descriptor, 0), read_operand((word >> 7) & 0x7F, offset, descriptor, 1)]
    elif opcode in BINARY_OPCODES:
        sources = [read_operand((word >> 12) & 0x7F, offset, descriptor, 0), read_operand((word >> 7) & 0x1F, 0, descriptor, 1)]
    elif opcode in UNARY_OPCODES:
        sources = [read_operand((word >> 12) & 0x7F, offset, descriptor, 0)]
    else:
        raise Exception(f"Unsupported vertex instruction: {opcode:#04x}")

    if opcode == 0x12:
        lines = []
        mask = get_mask(descriptor & 0xF)
        for lane in "xy":
            if lane in mask:
                lines.append(f"studio_a0.{lane} = int({sources[0]}.{lane});")

        return lines

    return write_assignment(destination, descriptor, get_operation(opcode, sources))

def write_block(shader, start, end, depth, lines, indent):
    """Writes the instructions from start to end (excluded), the flow control as GLSL blocks."""
    code = shader["code"]
    position = start

    while position < end and position < len(code):
        word = code[position]
        opcode = word >> 26
        target = (word >> 10) & 0xFFF
        count = word & 0xFF
        uniform = (word >> 22) & 0xF

        if opcode == 0x22:
            lines.append(indent + "return;")
            position += 1
        elif opcode == 0x21:
            position += 1
        elif opcode == 0x20:
            lines.append(indent + "break;")
            position += 1
        elif opcode == 0x23:
            lines.append(indent + f"if {get_condition(word)} {{ break; }}")
            position += 1
        elif opcode in (0x24, 0x25, 0x26):
            if depth >= MAX_CALL_DEPTH:
                raise Exception("Vertex program calls nested too deep")

            if opcode == 0x24:
                lines.append(indent + "{")
            elif opcode == 0x25:
                lines.append(indent + f"if {get_condition(word)} {{")
            else:
                lines.append(indent + f"if (studio_b[{uniform}]) {{")

            write_block(shader, target, target + count, depth + 1, lines, indent + "    ")
            lines.append(indent + "}")
            position += 1
        elif opcode in (0x27, 0x28):
            if opcode == 0x27:
                lines.append(indent + f"if (studio_b[{uniform}]) {{")
            else:
                lines.append(indent + f"if {get_condition(word)} {{")

            write_block(shader, position + 1, target, depth, lines, indent + "    ")

            if count > 0:
                lines.append(indent + "} else {")
                write_block(shader, target, target + count, depth, lines, indent + "    ")

            lines.append(indent + "}")
            position = target + count
        elif opcode == 0x29:
            lines.append(indent + f"studio_al = studio_i[{uniform & 3}].y;")
            lines.append(indent + f"for (int studio_loop{depth}_{position} = 0; studio_loop{depth}_{position} <= studio_i[{uniform & 3}].x; studio_loop{depth}_{position}++) {{")
            write_block(shader, position + 1, target + 1, depth, lines, indent + "    ")
            lines.append(indent + f"    studio_al += studio_i[{uniform & 3}].z;")
            lines.append(indent + "}")
            position = target + 1
        elif opcode in (0x2A, 0x2B, 0x2C, 0x2D, 0x07):
            raise Exception(f"Unsupported vertex instruction: {opcode:#04x}")
        else:
            for line in write_instruction(shader, word):
                lines.append(indent + line)
            position += 1

def write_constants(program):
    lines = []
    booleans = ["false"] * BOOL_UNIFORM_COUNT
    integers = ["ivec4(0)"] * INT_UNIFORM_COUNT

    for constant in program["constants"]:
        values = constant["values"]
        register = constant["register"]

        if constant["type"] == 2:
            if register < FLOAT_UNIFORM_COUNT:
                components = [float_text(value) for value in values]
                lines.append(f"    studio_c[{register}] = vec4({', '.join(components)});")
        elif constant["type"] == 1:
            if register < INT_UNIFORM_COUNT:
                # An int uniform holds 8 bit values: the count, the start and the step of a loop
                components = []
                for value in values[:4]:
                    value = int(value) & 0xFF
                    if value >= 0x80:
                        value -= 0x100
                    components.append(str(value))
                integers[register] = f"ivec4({', '.join(components)})"
        elif register < BOOL_UNIFORM_COUNT:
            if values[0]:
                booleans[register] = "true"
            else:
                booleans[register] = "false"

    declarations = [
        f"const bool studio_b[16] = bool[16]({', '.join(booleans)});",
        f"const ivec4 studio_i[4] = ivec4[4]({', '.join(integers)});",
    ]

    return declarations, lines

def write_uniforms(program):
    lines = []

    for index in range(INPUT_COUNT):
        lines.append(f"    studio_v[{index}] = vec4(0.0, 0.0, 0.0, 1.0);")

    for index in range(FLOAT_UNIFORM_COUNT):
        lines.append(f"    studio_c[{index}] = vec4(0.0);")

    for uniform in program["uniforms"]:
        name = get_uniform_name(uniform["name"])
        start = uniform["start"]
        size = uniform["end"] - start + 1

        if start < FLOAT_UNIFORM_START:
            if name in ATTRIBUTES:
                lines.append(f"    studio_v[{start}] = {ATTRIBUTES[name]};")
            continue

        if start >= INT_UNIFORM_START:
            continue

        first = start - FLOAT_UNIFORM_START

        if name in MATRIX_UNIFORMS:
            for row in range(min(size, 4)):
                lines.append(f"    studio_c[{first + row}] = transpose({MATRIX_UNIFORMS[name]})[{row}];")
        elif name == "unf_vtx_txt":
            for row in range(min(size, 6)):
                if row % 2 == 0:
                    lines.append(f"    studio_c[{first + row}] = studio_native_row_0(unf_vtx_txt_{row + 1});")
                else:
                    lines.append(f"    studio_c[{first + row}] = studio_native_row_1(unf_vtx_txt_{row - 1});")
        elif name == "unf_vtx_bone":
            for row in range(size):
                lines.append(f"    studio_c[{first + row}] = studio_bone_rows[{row % 3}];")
        elif name in VECTOR_UNIFORMS:
            lines.append(f"    studio_c[{first}] = {VECTOR_UNIFORMS[name]};")

    return lines

def write_outputs(program):
    registers = {}
    masks = {}

    for output in program["outputs"]:
        registers[output["type"]] = f"studio_o[{output['register']}]"
        masks[output["type"]] = output["mask"]

    lines = []

    if "POSITION" in registers:
        lines.append(f"    gl_Position = {registers['POSITION']};")
    else:
        lines.append("    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);")

    if "COLOR" in registers:
        lines.append(f"    frg_clr = {registers['COLOR']};")
    else:
        lines.append("    frg_clr = vec4(0.5);")

    for unit, name in enumerate(("TEXCOORD0", "TEXCOORD1", "TEXCOORD2")):
        if name in registers and len(masks[name]) >= 2:
            lines.append(f"    frg_tx{unit} = studio_upright_uv({registers[name]}.{masks[name][:2]});")
        elif name in registers:
            lines.append(f"    frg_tx{unit} = studio_upright_uv(vec2({registers[name]}.{masks[name]}, 0.0));")
        else:
            lines.append(f"    frg_tx{unit} = vec2(0.0);")

    # The view output is the opposite of the eye space position
    if "VIEW" in registers:
        lines.append(f"    frg_prm = -{registers['VIEW']}.xyz;")
    else:
        lines.append("    frg_prm = vec3(0.0);")

    if "NORMAL_QUATERNION" in registers:
        lines.append(f"    vec4 quaternion = {registers['NORMAL_QUATERNION']};")
        lines.append("    if (dot(quaternion, quaternion) > 0.0) {")
        lines.append("        frg_nrm = normalize(studio_rotate(normalize(quaternion), vec3(0.0, 0.0, 1.0)));")
        lines.append("    } else {")
        lines.append("        frg_nrm = vec3(0.0, 0.0, 1.0);")
        lines.append("    }")
    else:
        lines.append("    frg_nrm = vec3(0.0, 0.0, 1.0);")

    return lines

def make_vertex_glsl(data, program_index=0):
    """The vertex stage of StudioRender of one program of a SHDnnn.shdc, raises on an instruction it can't write."""
    shader = get_shader(data)

    if program_index >= len(shader["programs"]):
        raise Exception(f"The vertex binary has no program {program_index}")

    program = shader["programs"][program_index]
    if program["type"] != 0:
        raise Exception("The program is not a vertex program")

    body = []
    write_block(shader, program["main"], program["endmain"], 0, body, "    ")

    declarations, constants = write_constants(program)

    lines = [shaders.VERTEX_COMMON, shaders.NATIVE_UV_GLSL, HELPERS_GLSL]
    lines += declarations
    lines.append("")
    lines.append("void studio_program() {")
    lines += body
    lines.append("}")
    lines.append("")
    lines.append("void main() {")
    lines += write_uniforms(program)
    lines += constants
    lines.append("    studio_a0 = ivec2(0);")
    lines.append("    studio_al = 0;")
    lines.append("    studio_cmp = bvec2(false);")
    lines.append("")

    for index in range(TEMPORARY_COUNT):
        lines.append(f"    studio_r[{index}] = vec4(0.0);")

    for index in range(OUTPUT_COUNT):
        lines.append(f"    studio_o[{index}] = vec4(0.0);")

    lines.append("")
    lines.append("    studio_program();")
    lines.append("")
    lines += write_outputs(program)
    lines.append("}")

    return "\n".join(lines) + "\n"
