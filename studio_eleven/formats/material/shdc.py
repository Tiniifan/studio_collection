import io
import struct

##########################################
# CONST
##########################################

OPCODES = {
    "add": 0x00,
    "dp3": 0x01,
    "dp4": 0x02,
    "dph": 0x03,
    "ex2": 0x05,
    "lg2": 0x06,
    "mul": 0x08,
    "sge": 0x09,
    "slt": 0x0A,
    "flr": 0x0B,
    "max": 0x0C,
    "min": 0x0D,
    "rcp": 0x0E,
    "rsq": 0x0F,
    "mova": 0x12,
    "mov": 0x13,
    "sgei": 0x1A,
    "slti": 0x1B,
    "nop": 0x21,
    "end": 0x22,
}

OPCODE_NAMES = {v: k for k, v in OPCODES.items()}

# The second source of these takes any register, the first one only an input or a temporary
INVERTED_OPCODES = ("sgei", "slti")

OUTPUT_TYPES = {
    0: "POSITION",
    1: "NORMAL_QUATERNION",
    2: "COLOR",
    3: "TEXCOORD0",
    4: "TEXCOORD0_W",
    5: "TEXCOORD1",
    6: "TEXCOORD2",
    8: "VIEW",
    9: "DUMMY",
}

OUTPUT_TYPE_IDS = {v: k for k, v in OUTPUT_TYPES.items()}

# Registers in an instruction: inputs v0-v15, temporaries r0-r15 and the float uniforms c0-c95 after them
INPUT_REGISTER = 0x00
TEMPORARY_REGISTER = 0x10
UNIFORM_REGISTER = 0x20

# Registers in the uniform table of a program: the float uniforms start at 16
UNIFORM_TABLE_FLOAT = 0x10

# Version of the DVLB of the V1 and V2 games
SHADER_VERSIONS = {
    1: 0x0A02,
    2: 0x0C02,
}

##########################################
# SHDC Function
##########################################

def float24_to_float(value):
    if value & 0x7FFFFF == 0:
        return 0.0

    sign = (value >> 23) & 1
    exponent = (value >> 16) & 0x7F
    mantissa = value & 0xFFFF

    bits = (sign << 31) | ((exponent - 63 + 127) << 23) | (mantissa << 7)

    return struct.unpack("<f", struct.pack("<I", bits))[0]

def float_to_float24(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = bits >> 31
    exponent = ((bits >> 23) & 0xFF) - 127 + 63
    mantissa = (bits >> 7) & 0xFFFF

    if value == 0.0 or exponent <= 0:
        return sign << 23

    if exponent >= 0x7F:
        exponent = 0x7F
        mantissa = 0

    return (sign << 23) | (exponent << 16) | mantissa

def get_swizzle_bits(swizzle):
    bits = 0

    for index, component in enumerate(swizzle):
        bits |= "xyzw".index(component) << (6 - 2 * index)

    return bits

def get_swizzle_text(bits):
    text = ""

    for index in range(4):
        text += "xyzw"[(bits >> (6 - 2 * index)) & 3]

    return text

def get_mask_bits(mask):
    bits = 0

    for index, component in enumerate("xyzw"):
        if component in mask:
            bits |= 8 >> index

    return bits

def get_output_mask_bits(mask):
    # The components of an output are bit 0 = x to bit 3 = w, the reverse of an operand descriptor
    bits = 0

    for index, component in enumerate("xyzw"):
        if component in mask:
            bits |= 1 << index

    return bits

def get_output_mask_text(bits):
    text = ""

    for index, component in enumerate("xyzw"):
        if bits & (1 << index):
            text += component

    return text

def make_operand_descriptor(mask, sources):
    """sources is a list of (negate, swizzle) for the sources of the instruction, at most three."""
    descriptor = get_mask_bits(mask)

    # The second int flags the fields the instruction uses: the destination then each source
    used = 1
    for index in range(len(sources)):
        used |= 2 << index
    descriptor |= used << 32

    for index, source in enumerate(sources):
        negate = source[0]
        swizzle = source[1]
        shift = 4 + index * 9

        if negate:
            descriptor |= 1 << shift

        descriptor |= get_swizzle_bits(swizzle) << (shift + 1)

    return descriptor

def read_operand_descriptor(descriptor):
    mask = ""
    for index, component in enumerate("xyzw"):
        if descriptor & (8 >> index):
            mask += component

    sources = []
    for index in range(3):
        shift = 4 + index * 9
        negate = (descriptor >> shift) & 1 == 1
        sources.append((negate, get_swizzle_text((descriptor >> (shift + 1)) & 0xFF)))

    return {
        "mask": mask,
        "sources": sources,
    }

def encode_instruction(instruction):
    """instruction is {op, dst, src (register numbers), desc, relative}, mad / madi take three sources."""
    op = instruction["op"]
    sources = instruction.get("src", [])
    relative = instruction.get("relative", 0)
    descriptor = instruction.get("desc", 0)
    dst = instruction.get("dst", 0)

    if op == "mad":
        word = 0b111 << 29
        word |= (dst & 0x1F) << 24
        word |= relative << 22
        word |= (sources[0] & 0x1F) << 17
        word |= (sources[1] & 0x7F) << 10
        word |= (sources[2] & 0x1F) << 5
        word |= descriptor & 0x1F
        return word

    if op == "madi":
        word = 0b110 << 29
        word |= (dst & 0x1F) << 24
        word |= relative << 22
        word |= (sources[0] & 0x1F) << 17
        word |= (sources[1] & 0x1F) << 12
        word |= (sources[2] & 0x7F) << 5
        word |= descriptor & 0x1F
        return word

    word = OPCODES[op] << 26

    if op in ("end", "nop"):
        return word

    word |= (dst & 0x1F) << 21
    word |= relative << 19

    if op in INVERTED_OPCODES:
        word |= (sources[0] & 0x1F) << 14
        word |= (sources[1] & 0x7F) << 7
    else:
        word |= (sources[0] & 0x7F) << 12
        if len(sources) > 1:
            word |= (sources[1] & 0x1F) << 7

    word |= descriptor & 0x7F

    return word

def read_symbol(data, offset):
    end = data.index(b"\0", offset)
    return data[offset:end].decode("ascii")

##########################################
# SHDC Write Function
##########################################

def write_program(program):
    """One DVLE: the header then the constants, labels, outputs, uniforms and the symbols."""
    symbols = bytes()
    symbol_offsets = {}

    for name in [label["name"] for label in program["labels"]] + [uniform["name"] for uniform in program["uniforms"]]:
        if name not in symbol_offsets:
            symbol_offsets[name] = len(symbols)
            symbols += name.encode("ascii") + b"\0"

    constants = bytes()
    for constant in program["constants"]:
        constants += int(constant["type"]).to_bytes(2, 'little')
        constants += int(constant["register"]).to_bytes(2, 'little')

        if constant["type"] == 2:
            for value in constant["values"]:
                constants += int(float_to_float24(value)).to_bytes(4, 'little')
        else:
            for value in constant["values"]:
                constants += int(value).to_bytes(4, 'little')

    labels = bytes()
    for label in program["labels"]:
        labels += int(label["id"]).to_bytes(1, 'little')
        labels += bytes(label.get("unk", bytes(3)))
        labels += int(label["offset"]).to_bytes(4, 'little')
        labels += int(label.get("size", 0xFFFFFFFF)).to_bytes(4, 'little')
        labels += int(symbol_offsets[label["name"]]).to_bytes(4, 'little')

    outputs = bytes()
    for output in program["outputs"]:
        outputs += int(OUTPUT_TYPE_IDS[output["type"]]).to_bytes(2, 'little')
        outputs += int(output["register"]).to_bytes(2, 'little')
        outputs += int(get_output_mask_bits(output["mask"])).to_bytes(2, 'little')
        outputs += bytes(2)

    uniforms = bytes()
    for uniform in program["uniforms"]:
        uniforms += int(symbol_offsets[uniform["name"]]).to_bytes(4, 'little')
        uniforms += int(uniform["start"]).to_bytes(2, 'little')
        uniforms += int(uniform["end"]).to_bytes(2, 'little')

    constant_offset = 0x40
    label_offset = constant_offset + len(constants)
    output_offset = label_offset + len(labels)
    uniform_offset = output_offset + len(outputs)
    symbol_offset = uniform_offset + len(uniforms)

    out = bytes()

    out += b"DVLE"
    out += int(program["version"]).to_bytes(2, 'little')
    out += int(program["type"]).to_bytes(1, 'little')
    out += int(program.get("merge_outputs", 0)).to_bytes(1, 'little')
    out += int(program["main"]).to_bytes(4, 'little')
    out += int(program["endmain"]).to_bytes(4, 'little')
    out += int(program["input_mask"]).to_bytes(2, 'little')
    out += int(program["output_mask"]).to_bytes(2, 'little')
    out += bytes(program.get("geometry", bytes(4)))
    out += int(constant_offset).to_bytes(4, 'little')
    out += int(len(program["constants"])).to_bytes(4, 'little')
    out += int(label_offset).to_bytes(4, 'little')
    out += int(len(program["labels"])).to_bytes(4, 'little')
    out += int(output_offset).to_bytes(4, 'little')
    out += int(len(program["outputs"])).to_bytes(4, 'little')
    out += int(uniform_offset).to_bytes(4, 'little')
    out += int(len(program["uniforms"])).to_bytes(4, 'little')
    out += int(symbol_offset).to_bytes(4, 'little')
    out += int(len(symbols)).to_bytes(4, 'little')
    out += constants + labels + outputs + uniforms + symbols

    while len(out) % 4:
        out += b"\0"

    return out

def write_shdc(shader):
    """The DVLB of a SHDnnn.shdc: the shared code (DVLP) and one DVLE per program, not compressed."""
    file_names = bytes()
    for name in shader.get("file_names", []):
        file_names += name.encode("ascii") + b"\0"

    code = bytes()
    for word in shader["code"]:
        code += int(word).to_bytes(4, 'little')

    descriptors = bytes()
    for descriptor in shader["operand_descriptors"]:
        descriptors += int(descriptor).to_bytes(8, 'little')

    code_offset = 0x28
    descriptor_offset = code_offset + len(code)
    file_name_offset = descriptor_offset + len(descriptors)

    dvlp = bytes()
    dvlp += b"DVLP"
    dvlp += int(shader["version"]).to_bytes(4, 'little')
    dvlp += int(code_offset).to_bytes(4, 'little')
    dvlp += int(len(shader["code"])).to_bytes(4, 'little')
    dvlp += int(descriptor_offset).to_bytes(4, 'little')
    dvlp += int(len(shader["operand_descriptors"])).to_bytes(4, 'little')
    dvlp += int(file_name_offset).to_bytes(4, 'little')
    dvlp += int(0).to_bytes(4, 'little')
    dvlp += int(file_name_offset).to_bytes(4, 'little')
    dvlp += int(len(file_names)).to_bytes(4, 'little')
    dvlp += code + descriptors + file_names

    while len(dvlp) % 4:
        dvlp += b"\0"

    programs = [write_program(program) for program in shader["programs"]]

    out = bytes()
    out += b"DVLB"
    out += int(len(programs)).to_bytes(4, 'little')

    offset = 8 + 4 * len(programs) + len(dvlp)
    for program in programs:
        out += int(offset).to_bytes(4, 'little')
        offset += len(program)

    out += dvlp

    for program in programs:
        out += program

    return out

##########################################
# SHDC Open Function
##########################################

def open_program(data, offset):
    reader = io.BytesIO(data)
    reader.seek(offset)

    magic = reader.read(4)
    if magic != b"DVLE":
        raise Exception(f"Unknown shader program magic: {magic}")

    version = struct.unpack("<H", reader.read(2))[0]
    shader_type = reader.read(1)[0]
    merge_outputs = reader.read(1)[0]
    main = struct.unpack("<I", reader.read(4))[0]
    endmain = struct.unpack("<I", reader.read(4))[0]
    input_mask = struct.unpack("<H", reader.read(2))[0]
    output_mask = struct.unpack("<H", reader.read(2))[0]
    geometry = reader.read(4)
    constant_offset = struct.unpack("<I", reader.read(4))[0]
    constant_count = struct.unpack("<I", reader.read(4))[0]
    label_offset = struct.unpack("<I", reader.read(4))[0]
    label_count = struct.unpack("<I", reader.read(4))[0]
    output_offset = struct.unpack("<I", reader.read(4))[0]
    output_count = struct.unpack("<I", reader.read(4))[0]
    uniform_offset = struct.unpack("<I", reader.read(4))[0]
    uniform_count = struct.unpack("<I", reader.read(4))[0]
    symbol_offset = struct.unpack("<I", reader.read(4))[0]
    symbols = offset + symbol_offset

    constants = []
    reader.seek(offset + constant_offset)
    for i in range(constant_count):
        constant_type = struct.unpack("<H", reader.read(2))[0]
        register = struct.unpack("<H", reader.read(2))[0]
        raw = struct.unpack("<IIII", reader.read(16))

        if constant_type == 2:
            values = [float24_to_float(value) for value in raw]
        else:
            values = list(raw)

        constants.append({"type": constant_type, "register": register, "values": values})

    labels = []
    reader.seek(offset + label_offset)
    for i in range(label_count):
        label_id = reader.read(1)[0]
        unk = reader.read(3)
        label_start = struct.unpack("<I", reader.read(4))[0]
        size = struct.unpack("<I", reader.read(4))[0]
        name_offset = struct.unpack("<I", reader.read(4))[0]
        labels.append({"id": label_id, "unk": unk, "offset": label_start, "size": size, "name": read_symbol(data, symbols + name_offset)})

    outputs = []
    reader.seek(offset + output_offset)
    for i in range(output_count):
        output_type = struct.unpack("<H", reader.read(2))[0]
        register = struct.unpack("<H", reader.read(2))[0]
        mask = struct.unpack("<H", reader.read(2))[0]
        reader.read(2)
        outputs.append({"type": OUTPUT_TYPES.get(output_type, output_type), "register": register, "mask": get_output_mask_text(mask)})

    uniforms = []
    reader.seek(offset + uniform_offset)
    for i in range(uniform_count):
        name_offset = struct.unpack("<I", reader.read(4))[0]
        start = struct.unpack("<H", reader.read(2))[0]
        end = struct.unpack("<H", reader.read(2))[0]
        uniforms.append({"name": read_symbol(data, symbols + name_offset), "start": start, "end": end})

    reader.close()

    return {
        "version": version,
        "type": shader_type,
        "merge_outputs": merge_outputs,
        "main": main,
        "endmain": endmain,
        "input_mask": input_mask,
        "output_mask": output_mask,
        "geometry": geometry,
        "constants": constants,
        "labels": labels,
        "outputs": outputs,
        "uniforms": uniforms,
    }

def open_shdc(data):
    """data is the DVLB of a SHDnnn.shdc, already decompressed."""
    reader = io.BytesIO(data)

    magic = reader.read(4)
    if magic != b"DVLB":
        raise Exception(f"Unknown shader binary magic: {magic}")

    count = struct.unpack("<I", reader.read(4))[0]
    program_offsets = struct.unpack(f"<{count}I", reader.read(4 * count))

    dvlp = reader.tell()
    magic = reader.read(4)
    if magic != b"DVLP":
        raise Exception(f"Unknown shader code magic: {magic}")

    version = struct.unpack("<I", reader.read(4))[0]
    code_offset = struct.unpack("<I", reader.read(4))[0]
    code_size = struct.unpack("<I", reader.read(4))[0]
    descriptor_offset = struct.unpack("<I", reader.read(4))[0]
    descriptor_count = struct.unpack("<I", reader.read(4))[0]
    unk_offset = struct.unpack("<I", reader.read(4))[0]
    unk_count = struct.unpack("<I", reader.read(4))[0]
    file_name_offset = struct.unpack("<I", reader.read(4))[0]
    file_name_size = struct.unpack("<I", reader.read(4))[0]

    reader.seek(dvlp + code_offset)
    code = list(struct.unpack(f"<{code_size}I", reader.read(4 * code_size)))

    reader.seek(dvlp + descriptor_offset)
    operand_descriptors = []
    for i in range(descriptor_count):
        operand_descriptors.append(struct.unpack("<Q", reader.read(8))[0])

    reader.seek(dvlp + file_name_offset)
    file_names = [name.decode("ascii") for name in reader.read(file_name_size).split(b"\0") if name]

    reader.close()

    return {
        "version": version,
        "code": code,
        "operand_descriptors": operand_descriptors,
        "file_names": file_names,
        "programs": [open_program(data, offset) for offset in program_offsets],
    }
