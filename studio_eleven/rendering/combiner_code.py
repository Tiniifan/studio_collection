"""The FRAGMENT_SHADER section of a custom shader: GLSL that compiles to the stages of the texture combiner and back.

    void main() {
        vec4 tex0 = texture2D(unf_frg_txt_2d_0, var_tx0.xy);
        vec3 lit = (clr_1st.rgb + clr_2nd.rgb) * tex0.rgb;
        gl_FragColor = vec4(lit * var_clr.rgb * 2.0, tex0.a * var_clr.a * 2.0);
    }

The code is read lane by lane (rgb and alpha) into a program of the source names below, every operation clamps its result
to 0..1 like the hardware, so the code computes what the game draws.
"""
from .shader_code import CodeError

##########################################
# CONST
##########################################

MAX_STAGES = 6

# Name in the code -> source of the combiner
SOURCE_NAMES = {
    "texture0": "TEXTURE0",
    "texture1": "TEXTURE1",
    "texture2": "TEXTURE2",
    "texture3": "TEXTURE3",
    "vertex_color": "PRIMARY_COLOR",
    "light_diffuse": "FRAGMENT_PRIMARY_COLOR",
    "light_specular": "FRAGMENT_SECONDARY_COLOR",
}

SOURCE_CODE_NAMES = {v: k for k, v in SOURCE_NAMES.items()}

SWIZZLES = ("rgb", "a", "r", "g", "b")

SCALES = (2.0, 4.0)

FUNCTION_ARITY = {
    "REPLACE": 1,
    "MODULATE": 2,
    "ADD": 2,
    "SUBTRACT": 2,
    "ADD_SIGNED": 2,
    "DOT3_RGB": 2,
    "INTERPOLATE": 3,
    "ADD_MULT": 3,
    "MULT_ADD": 3,
}

RGB_OPERANDS = {
    ("rgb", False): "SRC_COLOR",
    ("rgb", True): "ONE_MINUS_SRC_COLOR",
    ("a", False): "SRC_ALPHA",
    ("a", True): "ONE_MINUS_SRC_ALPHA",
    ("r", False): "SRC_R",
    ("r", True): "ONE_MINUS_SRC_R",
    ("g", False): "SRC_G",
    ("g", True): "ONE_MINUS_SRC_G",
    ("b", False): "SRC_B",
    ("b", True): "ONE_MINUS_SRC_B",
}

OPERAND_SELECTS = {v: k for k, v in RGB_OPERANDS.items()}

# Names of the FRAGMENT_SHADER section -> name of the source in the program
FRAGMENT_INPUTS = {
    "var_clr": "vertex_color",
    "clr_1st": "light_diffuse",
    "clr_2nd": "light_specular",
}

FRAGMENT_SAMPLERS = {
    "unf_frg_txt_2d_0": 0,
    "unf_frg_txt_2d_1": 1,
    "unf_frg_txt_2d_2": 2,
}

TEXTURE_COORDINATES = {
    "var_tx0": 0,
    "var_tx1": 1,
    "var_tx2": 2,
}

# The hardware reads texture N at the texture coordinates N
TEXTURE_UNIT_COORDINATES = {v: k for k, v in TEXTURE_COORDINATES.items()}

# Source of a stage -> name in the FRAGMENT_SHADER section
SOURCE_GLSL = {
    "TEXTURE0": "tex0",
    "TEXTURE1": "tex1",
    "TEXTURE2": "tex2",
    "PRIMARY_COLOR": "var_clr",
    "FRAGMENT_PRIMARY_COLOR": "clr_1st",
    "FRAGMENT_SECONDARY_COLOR": "clr_2nd",
}

##########################################
# Fragment section (code -> program)
##########################################

class FragmentReader:
    """Runs the FRAGMENT_SHADER section on the lanes of the combiner: a vec4 is an rgb expression and an alpha expression."""

    def __init__(self, tree):
        self.tree = tree
        self.scopes = [{}]
        self.variables = []
        self.names = set()
        self.output = {"rgb": None, "a": None}
        self.returned = None
        self.call_depth = 0

    def error(self, message, line):
        raise CodeError(message, line)

    def read(self):
        for declaration in self.tree["declarations"]:
            self.declare_global(declaration)

        if "main" not in self.tree["functions"]:
            raise CodeError("The FRAGMENT_SHADER section has no main function")

        self.call_function(self.tree["functions"]["main"], [], self.tree["functions"]["main"]["line"])

        if self.output["rgb"] is None:
            raise CodeError("The fragment shader must write gl_FragColor")

        alpha = self.output["a"]
        if alpha is None:
            alpha = {"type": "number", "value": 1.0}

        return {"variables": self.variables, "color": self.output["rgb"], "alpha": alpha}

    def declare_global(self, declaration):
        name = declaration["name"]
        line = declaration["line"]

        if name in FRAGMENT_SAMPLERS or name in FRAGMENT_INPUTS or name in TEXTURE_COORDINATES:
            return

        if "uniform" in declaration["qualifiers"] or "varying" in declaration["qualifiers"] or "in" in declaration["qualifiers"]:
            self.error(f"The combiner can't read {name}: textures {', '.join(FRAGMENT_SAMPLERS)}, colors var_clr, clr_1st, clr_2nd", line)

        self.declare_local(declaration)

    def lookup(self, name, line):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]

        if name in FRAGMENT_INPUTS:
            source = {"type": "name", "name": FRAGMENT_INPUTS[name], "line": line}
            return {"type": "vec4", "rgb": source, "a": {"type": "swizzle", "argument": source, "swizzle": "a"}}

        if name in FRAGMENT_SAMPLERS:
            return {"type": "sampler2D", "unit": FRAGMENT_SAMPLERS[name]}

        if name in TEXTURE_COORDINATES:
            return {"type": "coordinates", "unit": TEXTURE_COORDINATES[name]}

        if name == "gl_FragColor":
            return {"type": "output"}

        self.error(f"Unknown name {name}", line)

    def declare_local(self, declaration):
        line = declaration["line"]
        value_type = declaration["type"]

        if value_type not in ("vec4", "vec3", "float"):
            self.error(f"The combiner only works on vec4, vec3 and float values, not {value_type}", line)
        if declaration["size"] is not None:
            self.error("The combiner has no arrays", line)

        if declaration["value"] is None:
            value = self.make_number_value(value_type, 0.0)
        else:
            value = self.convert(self.evaluate(declaration["value"]), value_type, line)

        self.scopes[-1][declaration["name"]] = self.name_value(declaration["name"], value)

    def make_number_value(self, value_type, number):
        constant = {"type": "number", "value": number}
        if value_type == "float":
            return {"type": "float", "value": constant}
        if value_type == "vec3":
            return {"type": "vec3", "rgb": constant}
        return {"type": "vec4", "rgb": constant, "a": constant}

    def name_value(self, name, value):
        """A computed lane becomes a variable of the program, so the stages compute it once."""
        result = dict(value)

        for lane in ("rgb", "a", "value"):
            expression = value.get(lane)
            if expression is None or is_simple(expression):
                continue

            variable = name
            if lane == "a" and value["type"] == "vec4":
                variable = f"{name}_a"

            unique = variable
            count = 1
            while unique in self.names or unique in SOURCE_NAMES:
                count += 1
                unique = f"{variable}_{count}"

            self.names.add(unique)
            declared = "float"
            if lane == "rgb":
                declared = "vec3"
            self.variables.append({"name": unique, "type": declared, "expression": expression})
            result[lane] = {"type": "name", "name": unique, "line": 0}

        return result

    def convert(self, value, value_type, line):
        if value["type"] == value_type:
            return value

        if value["type"] == "float" and value_type in ("vec3", "vec4"):
            rgb = value["value"]
            # A number given to a color is a gray constant, a color of the combiner
            if rgb["type"] == "number":
                rgb = {"type": "vector", "values": [rgb["value"]] * 3}
            result = {"type": value_type, "rgb": rgb}
            if value_type == "vec4":
                result["a"] = value["value"]
            return result

        self.error(f"A {value['type']} can't be given to a {value_type}", line)

    def get_lane(self, value, lane, line):
        if value["type"] == "float":
            return value["value"]
        if value["type"] not in ("vec3", "vec4"):
            self.error(f"A {value['type']} can't be used in a computation", line)
        if lane == "a" and value["type"] == "vec3":
            self.error("A vec3 has no alpha", line)
        return value[lane]

    ##########################################
    # Statements
    ##########################################

    def execute_block(self, statements):
        for statement in statements:
            self.execute(statement)
            if self.returned is not None:
                return

    def execute(self, statement):
        kind = statement["kind"]
        line = statement["line"]

        if kind == "declare":
            self.declare_local(statement)
        elif kind == "expression":
            self.evaluate(statement["value"])
        elif kind == "block":
            self.scopes.append({})
            self.execute_block(statement["body"])
            self.scopes.pop()
        elif kind == "return":
            value = None
            if statement["value"] is not None:
                value = self.evaluate(statement["value"])
            self.returned = {"value": value}
        elif kind in ("if", "for"):
            self.error(f"The combiner has no '{kind}': every pixel runs the same stages", line)
        else:
            self.error(f"Unknown statement {kind}", line)

    def call_function(self, function, arguments, line):
        if self.call_depth > 16:
            self.error(f"{function['name']} calls itself, a shader can't", line)

        if len(arguments) != len(function["parameters"]):
            self.error(f"{function['name']} takes {len(function['parameters'])} arguments", line)

        scope = {}
        for index, parameter in enumerate(function["parameters"]):
            scope[parameter["name"]] = self.convert(arguments[index], parameter["type"], line)

        saved_scopes = self.scopes
        saved_returned = self.returned
        self.scopes = [self.scopes[0], scope]
        self.returned = None
        self.call_depth += 1

        self.execute_block(function["body"])

        result = self.returned
        self.call_depth -= 1
        self.scopes = saved_scopes
        self.returned = saved_returned

        if function["type"] == "void":
            return {"type": "void"}

        if result is None or result["value"] is None:
            self.error(f"{function['name']} doesn't return a value", function["line"])

        return self.convert(result["value"], function["type"], line)

    ##########################################
    # Expressions
    ##########################################

    def evaluate(self, expression):
        kind = expression["kind"]
        line = expression["line"]

        if kind == "number":
            return {"type": "float", "value": {"type": "number", "value": expression["value"]}}

        if kind == "name":
            return self.lookup(expression["name"], line)

        if kind == "binary":
            left = self.evaluate(expression["left"])
            right = self.evaluate(expression["right"])
            return self.combine_values(expression["operator"], left, right, line)

        if kind == "unary":
            if expression["operator"] != "-":
                self.error(f"The combiner has no '{expression['operator']}'", line)
            zero = {"type": "float", "value": {"type": "number", "value": 0.0}}
            return self.combine_values("-", zero, self.evaluate(expression["operand"]), line)

        if kind == "swizzle":
            return self.evaluate_swizzle(self.evaluate(expression["operand"]), expression["components"], line)

        if kind == "assign":
            return self.evaluate_assign(expression)

        if kind == "call":
            return self.evaluate_call(expression)

        self.error(f"The combiner can't compute a {kind}", line)

    def combine_values(self, operator, left, right, line):
        if operator == "/":
            if right["type"] != "float" or right["value"]["type"] != "number" or right["value"]["value"] == 0.0:
                self.error("The combiner only divides by a number", line)
            right = {"type": "float", "value": {"type": "number", "value": 1.0 / right["value"]["value"]}}
            operator = "*"

        if operator not in ("+", "-", "*"):
            self.error(f"The combiner has no '{operator}'", line)

        value_type = left["type"]
        if right["type"] != "float":
            value_type = right["type"]
        if left["type"] != "float" and right["type"] != "float" and left["type"] != right["type"]:
            self.error(f"{left['type']} {operator} {right['type']}: the sizes don't match", line)

        if value_type == "float":
            return {"type": "float", "value": fold_binary(operator, left["value"], right["value"])}

        result = {"type": value_type}
        lanes = ["rgb"]
        if value_type == "vec4":
            lanes.append("a")
        for lane in lanes:
            result[lane] = fold_binary(operator, self.get_lane(left, lane, line), self.get_lane(right, lane, line))

        return result

    def evaluate_swizzle(self, value, components, line):
        if value["type"] not in ("vec3", "vec4", "float"):
            self.error(f"A {value['type']} has no components", line)

        letters = components
        for letter_set, mapped in (("xyzw", "rgba"), ("stpq", "rgba")):
            if all(letter in letter_set for letter in letters):
                letters = "".join(mapped[letter_set.index(letter)] for letter in letters)

        if value["type"] == "float":
            if letters != "r":
                self.error(f"A float has no '.{components}'", line)
            return value

        if letters == "rgba" and value["type"] == "vec4":
            return value
        if letters == "rgb":
            return {"type": "vec3", "rgb": value["rgb"]}
        if letters == "a":
            return {"type": "float", "value": self.get_lane(value, "a", line)}
        if letters in ("r", "g", "b"):
            return {"type": "float", "value": {"type": "swizzle", "argument": value["rgb"], "swizzle": letters}}
        if len(letters) == 3 and len(set(letters)) == 1 and letters[0] in "rgb":
            return {"type": "vec3", "rgb": {"type": "swizzle", "argument": value["rgb"], "swizzle": letters[0]}}

        self.error(f"The combiner reads .rgb, .a, .r, .g or .b, not '.{components}'", line)

    def evaluate_assign(self, expression):
        line = expression["line"]
        target = expression["target"]
        value = self.evaluate(expression["value"])

        if expression["operator"] != "=":
            current = self.evaluate(target)
            value = self.combine_values(expression["operator"][0], current, value, line)

        if target["kind"] == "name" and target["name"] == "gl_FragColor":
            value = self.convert(value, "vec4", line)
            self.output["rgb"] = value["rgb"]
            self.output["a"] = value["a"]
            return value

        if target["kind"] == "swizzle" and target["operand"]["kind"] == "name" and target["operand"]["name"] == "gl_FragColor":
            letters = target["components"].replace("x", "r").replace("y", "g").replace("z", "b").replace("w", "a")
            if letters == "rgb":
                self.output["rgb"] = self.convert(value, "vec3", line)["rgb"]
            elif letters == "a":
                if value["type"] != "float":
                    self.error("gl_FragColor.a takes a float", line)
                self.output["a"] = value["value"]
            elif letters == "rgba":
                self.assign_output(value, line)
            else:
                self.error("Write gl_FragColor, gl_FragColor.rgb or gl_FragColor.a", line)
            return value

        if target["kind"] != "name":
            self.error("The combiner can only write a whole variable or gl_FragColor", line)

        name = target["name"]
        current = self.lookup(name, line)
        if current["type"] in ("sampler2D", "coordinates") or name in FRAGMENT_INPUTS:
            self.error(f"{name} is an input, it can't be written", line)

        value = self.convert(value, current["type"], line)
        named = self.name_value(name, value)

        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = named
                break

        return named

    def assign_output(self, value, line):
        value = self.convert(value, "vec4", line)
        self.output["rgb"] = value["rgb"]
        self.output["a"] = value["a"]

    def evaluate_call(self, expression):
        line = expression["line"]
        name = expression["name"]

        # dot3 / add_signed may be defined in the code to keep it valid GLSL, the combiner has them
        if name in self.tree["functions"] and name not in ("dot3", "add_signed"):
            arguments = [self.evaluate(argument) for argument in expression["arguments"]]
            return self.call_function(self.tree["functions"][name], arguments, line)

        if name in ("texture2D", "texture"):
            return self.sample(expression)

        arguments = [self.evaluate(argument) for argument in expression["arguments"]]

        if name in ("vec3", "vec4", "float"):
            return self.construct(name, arguments, line)

        if name == "mix":
            if len(arguments) != 3:
                self.error("mix takes 3 arguments", line)
            return self.lanes_call("mix", arguments, line)

        if name == "add_signed":
            if len(arguments) != 2:
                self.error("add_signed takes 2 arguments", line)
            return self.lanes_call("add_signed", arguments, line)

        if name == "dot3":
            if len(arguments) != 2:
                self.error("dot3 takes 2 arguments", line)
            rgb = {"type": "call", "name": "dot3", "arguments": [self.get_lane(arguments[0], "rgb", line), self.get_lane(arguments[1], "rgb", line)]}
            return {"type": "vec3", "rgb": rgb}

        if name == "clamp":
            if len(arguments) != 3 or not is_number_value(arguments[1], 0.0) or not is_number_value(arguments[2], 1.0):
                self.error("The combiner only clamps to 0.0 .. 1.0, which it always does", line)
            return arguments[0]

        self.error(f"The combiner has no {name}: it knows texture2D, mix, dot3, add_signed and clamp(x, 0.0, 1.0)", line)

    def lanes_call(self, name, arguments, line):
        value_type = "float"
        for argument in arguments[:2]:
            if argument["type"] != "float":
                value_type = argument["type"]

        if value_type == "float":
            return {"type": "float", "value": {"type": "call", "name": name, "arguments": [argument["value"] for argument in arguments]}}

        result = {"type": value_type}
        lanes = ["rgb"]
        if value_type == "vec4":
            lanes.append("a")
        for lane in lanes:
            result[lane] = {"type": "call", "name": name, "arguments": [self.get_lane(argument, lane, line) for argument in arguments]}

        return result

    def sample(self, expression):
        line = expression["line"]
        arguments = expression["arguments"]

        if len(arguments) != 2 or arguments[0]["kind"] != "name":
            self.error("texture2D(unf_frg_txt_2d_N, var_txN.xy) expected", line)

        sampler = self.lookup(arguments[0]["name"], line)
        if sampler["type"] != "sampler2D":
            self.error(f"{arguments[0]['name']} is not a texture, use {', '.join(FRAGMENT_SAMPLERS)}", line)

        coordinates = arguments[1]
        while coordinates["kind"] == "swizzle":
            coordinates = coordinates["operand"]

        unit = sampler["unit"]
        expected = TEXTURE_UNIT_COORDINATES[unit]
        if coordinates["kind"] != "name" or coordinates["name"] != expected:
            self.error(f"Texture {unit} is read at {expected} by the hardware: texture2D({arguments[0]['name']}, {expected}.xy)", line)

        source = {"type": "name", "name": f"texture{unit}", "line": line}

        return {"type": "vec4", "rgb": source, "a": {"type": "swizzle", "argument": source, "swizzle": "a"}}

    def construct(self, name, arguments, line):
        if name == "float":
            if len(arguments) != 1 or arguments[0]["type"] != "float":
                self.error("float() takes one float", line)
            return arguments[0]

        if len(arguments) == 1:
            return self.convert(arguments[0], name, line)

        numbers = []
        for argument in arguments:
            if argument["type"] == "float" and argument["value"]["type"] == "number":
                numbers.append(argument["value"]["value"])
            else:
                numbers = None
                break

        if numbers is not None and name == "vec3" and len(numbers) == 3:
            return {"type": "vec3", "rgb": {"type": "vector", "values": numbers}}

        if numbers is not None and name == "vec4" and len(numbers) == 4:
            return {"type": "vec4", "rgb": {"type": "vector", "values": numbers[:3]}, "a": {"type": "number", "value": numbers[3]}}

        if name == "vec4" and len(arguments) == 2 and arguments[0]["type"] == "vec3" and arguments[1]["type"] == "float":
            return {"type": "vec4", "rgb": arguments[0]["rgb"], "a": arguments[1]["value"]}

        self.error(f"{name}: the combiner builds a color from one value, three numbers, or vec4(vec3, float)", line)

def is_simple(expression):
    if expression["type"] in ("number", "vector", "name"):
        return True
    return expression["type"] == "swizzle" and expression["argument"]["type"] == "name"

def is_number_value(value, number):
    return value["type"] == "float" and value["value"]["type"] == "number" and abs(value["value"]["value"] - number) < 1e-6

def fold_binary(operator, left, right):
    if left["type"] == "number" and right["type"] == "number":
        if operator == "+":
            return {"type": "number", "value": left["value"] + right["value"]}
        if operator == "-":
            return {"type": "number", "value": left["value"] - right["value"]}
        return {"type": "number", "value": left["value"] * right["value"]}

    return {"type": "binary", "operator": operator, "left": left, "right": right}

def read_fragment(tree):
    """The program (variables, color, alpha) of a parsed FRAGMENT_SHADER section."""
    program = FragmentReader(tree).read()
    check_names(program)
    return program

def compile_fragment(tree):
    return compile_program(read_fragment(tree))

def walk_names(expression, found):
    if expression["type"] == "name":
        found.append(expression)
    elif expression["type"] == "binary":
        walk_names(expression["left"], found)
        walk_names(expression["right"], found)
    elif expression["type"] == "swizzle":
        walk_names(expression["argument"], found)
    elif expression["type"] == "call":
        for argument in expression["arguments"]:
            walk_names(argument, found)

def check_names(program):
    known = set(SOURCE_NAMES)

    for variable in program["variables"]:
        found = []
        walk_names(variable["expression"], found)
        for name in found:
            if name["name"] not in known:
                raise CodeError(f"Line {name['line']}: {name['name']} is not defined before")
        known.add(variable["name"])

    for output in ("color", "alpha"):
        found = []
        walk_names(program[output], found)
        for name in found:
            if name["name"] not in known:
                raise CodeError(f"Line {name['line']}: {name['name']} is not defined")

##########################################
# Types
##########################################

def get_variable(program, name):
    for variable in program["variables"]:
        if variable["name"] == name:
            return variable
    return None

def type_of(program, expression):
    """'color' (vec3) or 'value' (float)."""
    kind = expression["type"]

    if kind == "number":
        return "value"
    if kind == "vector":
        return "color"
    if kind == "name":
        if expression["name"] in SOURCE_NAMES:
            return "color"
        return type_of(program, get_variable(program, expression["name"])["expression"])
    if kind == "swizzle":
        if expression["swizzle"] == "rgb":
            return "color"
        return "value"
    if kind == "call":
        if expression["name"] == "dot3":
            return "color"
        for argument in expression["arguments"][:2]:
            if type_of(program, argument) == "color":
                return "color"
        return "value"

    if type_of(program, expression["left"]) == "color" or type_of(program, expression["right"]) == "color":
        return "color"

    return "value"

##########################################
# Lowering (code -> operations)
##########################################

def is_number(expression, value=None):
    if expression["type"] != "number":
        return False
    if value is None:
        return True
    return abs(expression["value"] - value) < 1e-6

def is_scale(expression):
    if expression["type"] != "binary" or expression["operator"] != "*":
        return False
    for side in ("left", "right"):
        for value in SCALES:
            if is_number(expression[side], value):
                return True
    return False

class Lowering:
    """Turns the code into operations, one per stage lane, that share the values the code names."""

    def __init__(self, program):
        self.program = program
        self.nodes = []
        self.memo = {}

    def new_node(self, lane, function, arguments, scale=1):
        node = {"id": len(self.nodes), "lane": lane, "function": function, "arguments": arguments, "scale": scale}
        self.nodes.append(node)
        return node

    def lower_argument(self, expression, lane, select=None):
        """An argument of an operation: a source, a constant or the result of another operation, read with a channel."""
        kind = expression["type"]

        if kind == "number":
            if expression["value"] < 0.0 or expression["value"] > 1.0:
                raise CodeError(f"The constant {expression['value']} is out of 0..1, only * 2.0 and * 4.0 can scale a result")
            return {"kind": "constant", "value": expression["value"], "select": "value", "invert": False}

        if kind == "vector":
            if lane == "alpha" and select not in ("r", "g", "b"):
                raise CodeError("A vec3 constant is a color, the alpha takes a number or one channel of it")
            for value in expression["values"]:
                if value < 0.0 or value > 1.0:
                    raise CodeError(f"The color {expression['values']} is out of 0..1")
            if select in ("r", "g", "b"):
                return {"kind": "constant", "value": expression["values"]["rgb".index(select)], "select": "value", "invert": False}
            return {"kind": "constant", "value": tuple(expression["values"]), "select": "rgb", "invert": False}

        if kind == "binary" and expression["operator"] == "-" and is_number(expression["left"], 1.0):
            argument = self.lower_argument(expression["right"], lane, select)
            argument = dict(argument)
            argument["invert"] = not argument["invert"]
            return argument

        if kind == "swizzle":
            inner = expression["swizzle"]
            if select is not None and select != "rgb" and inner != "rgb":
                raise CodeError(f"A channel of a channel (.{inner}.{select}) can't be read")
            if inner == "rgb":
                return self.lower_argument(expression["argument"], lane, select)
            return self.lower_argument(expression["argument"], lane, inner)

        if kind == "name" and expression["name"] in SOURCE_NAMES:
            if select is None:
                if lane == "alpha":
                    code_name = SOURCE_GLSL.get(SOURCE_NAMES[expression["name"]], expression["name"])
                    raise CodeError(f"{code_name}.rgb is a color, the alpha reads one channel of it: {code_name}.a")
                select = "rgb"
            return {"kind": "source", "source": SOURCE_NAMES[expression["name"]], "select": select, "invert": False}

        if kind == "name":
            return self.lower_variable(expression["name"], lane, select)

        node_lane = operation_lane(lane, select)
        if lane == "rgb" and select is None and type_of(self.program, expression) == "value":
            node_lane = "alpha"

        node = self.lower_operation(expression, node_lane, None)
        return self.read_node(node, lane, select)

    def read_node(self, node, lane, select):
        if node["lane"] == "alpha":
            return {"kind": "node", "node": node["id"], "select": "a", "invert": False}
        if select in ("r", "g", "b"):
            return {"kind": "node", "node": node["id"], "select": select, "invert": False}
        if lane == "alpha":
            raise CodeError("A color can't be the alpha, read one channel of it (.r .g or .b)")
        return {"kind": "node", "node": node["id"], "select": "rgb", "invert": False}

    def lower_variable(self, name, lane, select):
        expression = get_variable(self.program, name)["expression"]

        # A leaf is read where it is used, a computed value once per lane and shared
        if expression["type"] in ("number", "vector", "swizzle") or (expression["type"] == "name" and expression["name"] in SOURCE_NAMES):
            return self.lower_argument(expression, lane, select)
        if expression["type"] == "binary" and expression["operator"] == "-" and is_number(expression["left"], 1.0):
            return self.lower_argument(expression, lane, select)
        if expression["type"] == "name":
            return self.lower_variable(expression["name"], lane, select)

        # The value already computed in the other lane is read there (the alpha read by the color and the other way)
        if lane == "rgb" and select in (None, "a") and (name, "alpha") in self.memo and type_of(self.program, expression) == "value":
            return self.read_node(self.memo[(name, "alpha")], lane, "a")
        if select in ("r", "g", "b") and (name, "rgb") in self.memo:
            return self.read_node(self.memo[(name, "rgb")], lane, select)

        node_lane = operation_lane(lane, select)
        if lane == "rgb" and select is None and type_of(self.program, expression) == "value":
            node_lane = "alpha"

        key = (name, node_lane)
        if key not in self.memo:
            self.memo[key] = self.lower_operation(expression, node_lane, None)

        return self.read_node(self.memo[key], lane, select)

    def constants_fit(self, arguments, lane):
        return find_constant([(lane, arguments)]) is not None

    def lower_operation(self, expression, lane, select):
        """The operation that computes an expression in a lane, the fused functions of the hardware first."""
        kind = expression["type"]

        if kind == "binary" and expression["operator"] == "*" and is_scale(expression):
            value = expression["right"]
            other = expression["left"]
            if not is_number(value, 2.0) and not is_number(value, 4.0):
                value = expression["left"]
                other = expression["right"]

            node = None
            if other["type"] in ("binary", "call") and not (other["type"] == "binary" and other["operator"] == "-" and is_number(other["left"], 1.0)):
                node = self.lower_operation(other, lane, select)

            if node is None or node["scale"] != 1 or node in self.memo.values():
                node = self.new_node(lane, "REPLACE", [self.lower_argument(other, lane, select)])

            node["scale"] = int(value["value"])
            return node

        if kind == "binary" and expression["operator"] in ("+", "*"):
            return self.lower_fused(expression, lane)

        if kind == "binary" and expression["operator"] == "-":
            return self.new_node(lane, "SUBTRACT", [self.lower_argument(expression["left"], lane), self.lower_argument(expression["right"], lane)])

        if kind == "call" and expression["name"] == "mix":
            first, second, factor = expression["arguments"]
            return self.new_node(lane, "INTERPOLATE", [self.lower_argument(second, lane), self.lower_argument(first, lane), self.lower_argument(factor, lane)])

        if kind == "call" and expression["name"] == "add_signed":
            return self.new_node(lane, "ADD_SIGNED", [self.lower_argument(argument, lane) for argument in expression["arguments"]])

        if kind == "call" and expression["name"] == "dot3":
            if lane == "alpha":
                raise CodeError("dot3 gives a color, the alpha can read one channel of it")
            return self.new_node(lane, "DOT3_RGB", [self.lower_argument(argument, lane) for argument in expression["arguments"]])

        if kind == "name" and expression["name"] not in SOURCE_NAMES:
            variable = get_variable(self.program, expression["name"])["expression"]
            if variable["type"] in ("binary", "call"):
                argument = self.lower_variable(expression["name"], lane, select)
                if argument["kind"] == "node" and self.nodes[argument["node"]]["lane"] == lane and argument["select"] in ("rgb", "a"):
                    return self.nodes[argument["node"]]
                return self.new_node(lane, "REPLACE", [argument])

        return self.new_node(lane, "REPLACE", [self.lower_argument(expression, lane, select)])

    def lower_fused(self, expression, lane):
        """a * b + c and (a + b) * c take one stage (MULT_ADD, ADD_MULT) when their constants fit in it."""
        if expression["operator"] == "+":
            inner_operator = "*"
            fused = "MULT_ADD"
            single = "ADD"
            inner_single = "MODULATE"
        else:
            inner_operator = "+"
            fused = "ADD_MULT"
            single = "MODULATE"
            inner_single = "ADD"

        for inner, other in ((expression["left"], expression["right"]), (expression["right"], expression["left"])):
            if inner["type"] == "binary" and inner["operator"] == inner_operator and not is_scale(inner):
                first = self.lower_argument(inner["left"], lane)
                second = self.lower_argument(inner["right"], lane)
                third = self.lower_argument(other, lane)

                if self.constants_fit([first, second, third], lane):
                    return self.new_node(lane, fused, [first, second, third])

                inner_node = self.new_node(lane, inner_single, [first, second])
                inner_argument = self.read_node(inner_node, lane, None)
                if inner is expression["left"]:
                    return self.new_node(lane, single, [inner_argument, third])
                return self.new_node(lane, single, [third, inner_argument])

        return self.new_node(lane, single, [self.lower_argument(expression["left"], lane), self.lower_argument(expression["right"], lane)])

    def lower_output(self, expression, lane):
        argument = self.lower_argument(expression, lane)

        # The last stage gives the outputs: a value read as it is, an operation of the lane
        if argument["kind"] == "node" and not argument["invert"]:
            node = self.nodes[argument["node"]]
            if node["lane"] == lane and argument["select"] in ("rgb", "a"):
                return node

        return self.new_node(lane, "REPLACE", [argument])

def operation_lane(lane, select):
    """The lane that computes an expression read with a channel: .a in the alpha, .r .g .b in the color."""
    if select in ("r", "g", "b"):
        return "rgb"
    if select == "a":
        return "alpha"
    return lane

def lower_program(program):
    lowering = Lowering(program)

    # The alpha first: the color can read the values it computes
    alpha = lowering.lower_output(program["alpha"], "alpha")
    color = lowering.lower_output(program["color"], "rgb")

    return lowering.nodes, color["id"], alpha["id"]

##########################################
# Constants
##########################################

def constant_options(lane, argument):
    """The ways a register can give a constant argument: (component, value, select)."""
    if argument["select"] == "rgb":
        options = [("rgb", argument["value"], "rgb")]

        # A gray is also the alpha of the constant read on every channel
        red, green, blue = argument["value"]
        if lane == "rgb" and abs(red - green) < 1e-6 and abs(red - blue) < 1e-6:
            options.append(("a", red, "a"))

        return options

    value = argument["value"]
    options = [("a", value, "a")]

    if lane == "rgb":
        options.append(("rgb", (value, value, value), "rgb"))
    else:
        for channel in ("r", "g", "b"):
            options.append((channel, value, channel))

    return options

def merge_constant(constant, component, value):
    if component == "rgb":
        for i in range(3):
            if constant[i] is not None and abs(constant[i] - value[i]) > 1e-6:
                return None
        merged = list(constant)
        merged[0:3] = list(value)
        return tuple(merged)

    index = "rgba".index(component)
    if constant[index] is not None and abs(constant[index] - value) > 1e-6:
        return None

    merged = list(constant)
    merged[index] = value
    return tuple(merged)

def find_constant(lane_arguments, buffer=("", ""), buffer_color=(None, None, None, None)):
    """(constant, initial buffer color, {(lane, index): (source, select)}) that give the constants of a stage, or None."""
    wanted = []
    for lane, arguments in lane_arguments:
        for index, argument in enumerate(arguments):
            if argument["kind"] == "constant":
                wanted.append((lane, index, argument))

    return search_constant(wanted, 0, (None, None, None, None), buffer_color, buffer, {})

def search_constant(wanted, position, constant, buffer_color, buffer, placements):
    if position == len(wanted):
        return constant, buffer_color, dict(placements)

    lane, index, argument = wanted[position]

    for component, value, select in constant_options(lane, argument):
        merged = merge_constant(constant, component, value)
        if merged is not None:
            placements[(lane, index)] = ("CONSTANT", select)
            found = search_constant(wanted, position + 1, merged, buffer_color, buffer, placements)
            if found is not None:
                return found
            del placements[(lane, index)]

        buffer_index = 0
        if component == "a":
            buffer_index = 1

        if buffer[buffer_index] == "INIT":
            merged = merge_constant(buffer_color, component, value)
            if merged is not None:
                placements[(lane, index)] = ("PREVIOUS_BUFFER", select)
                found = search_constant(wanted, position + 1, constant, merged, buffer, placements)
                if found is not None:
                    return found
                del placements[(lane, index)]

    return None

##########################################
# Scheduler (operations -> stages)
##########################################

class Scheduler:
    """Places each operation in a stage: a value computed before is read in the previous stage or kept in the buffer."""

    def __init__(self, nodes, color, alpha):
        self.nodes = nodes
        self.color = color
        self.alpha = alpha
        self.failed = set()

        self.consumers = {}
        for node in nodes:
            self.consumers[node["id"]] = []
        for node in nodes:
            for argument in node["arguments"]:
                if argument["kind"] == "node":
                    self.consumers[argument["node"]].append(node["id"])

    def needed(self, node_id, placed):
        if node_id is None or node_id == "INIT":
            return False
        if node_id in (self.color, self.alpha):
            return True
        for consumer in self.consumers[node_id]:
            if consumer not in placed:
                return True
        return False

    def ready(self, node, previous, buffer):
        """The sources of the arguments of an operation (PREVIOUS or PREVIOUS_BUFFER), None when one isn't reachable."""
        sources = []

        for argument in node["arguments"]:
            if argument["kind"] != "node":
                sources.append(None)
                continue

            index = 0
            if self.nodes[argument["node"]]["lane"] == "alpha":
                index = 1

            if previous[index] == argument["node"]:
                sources.append("PREVIOUS")
            elif buffer[index] == argument["node"]:
                sources.append("PREVIOUS_BUFFER")
            else:
                return None

        return sources

    def choices(self, lane, placed, previous, buffer):
        found = [None]
        for node in self.nodes:
            if node["lane"] == lane and node["id"] not in placed:
                sources = self.ready(node, previous, buffer)
                if sources is not None:
                    found.append((node, sources))
        return found

    def remaining(self, placed):
        remaining = 0
        for lane in ("rgb", "alpha"):
            lane_count = 0
            for node in self.nodes:
                if node["lane"] == lane and node["id"] not in placed:
                    lane_count += 1
            remaining = max(remaining, lane_count)
        return remaining

    def search(self, stage, count, placed, previous, buffer, buffer_color, stages):
        if stage > 0 and previous[0] == self.color and previous[1] == self.alpha and len(placed) == len(self.nodes):
            return list(stages), buffer_color

        if self.remaining(placed) > count - stage:
            return None

        key = (stage, frozenset(placed), previous, buffer, buffer_color)
        if key in self.failed:
            return None

        # The buffer of a lane takes the previous stage (stages 1 to n-2), only worth it for a value read later
        captures = [(False, False)]
        if stage > 0:
            for capture in ((True, False), (False, True), (True, True)):
                useful = True
                for index in range(2):
                    if capture[index] and (previous[index] == buffer[index] or not self.needed(previous[index], placed)):
                        useful = False
                if useful:
                    captures.append(capture)

        for rgb_choice in self.choices("rgb", placed, previous, buffer):
            for alpha_choice in self.choices("alpha", placed, previous, buffer):
                lane_arguments = []
                if rgb_choice is not None:
                    lane_arguments.append(("rgb", rgb_choice[0]["arguments"]))
                if alpha_choice is not None:
                    lane_arguments.append(("alpha", alpha_choice[0]["arguments"]))

                found_constant = find_constant(lane_arguments, buffer, buffer_color)
                if found_constant is None:
                    continue

                constant, next_buffer_color, placements = found_constant

                next_previous = list(previous)
                next_placed = set(placed)
                for index, choice in enumerate((rgb_choice, alpha_choice)):
                    if choice is not None:
                        next_previous[index] = choice[0]["id"]
                        next_placed.add(choice[0]["id"])
                    elif not self.needed(previous[index], placed):
                        next_previous[index] = None

                for capture in captures:
                    next_buffer = list(buffer)
                    for index in range(2):
                        if capture[index]:
                            next_buffer[index] = previous[index]

                    stages.append({"rgb": rgb_choice, "alpha": alpha_choice, "capture": capture, "constant": constant, "placements": placements})
                    found = self.search(stage + 1, count, next_placed, tuple(next_previous), tuple(next_buffer), next_buffer_color, stages)
                    if found is not None:
                        return found
                    stages.pop()

        self.failed.add(key)
        return None

    def schedule(self):
        for count in range(1, MAX_STAGES + 1):
            self.failed.clear()
            found = self.search(0, count, set(), (None, None), ("INIT", "INIT"), (None, None, None, None), [])
            if found is not None:
                return found

        raise CodeError(f"The code needs more than {MAX_STAGES} stages, the combiner of the 3DS can't do it")

def color_bytes(values):
    result = []
    for value in values:
        if value is None:
            value = 1.0
        result.append(round(value * 255))
    return result

def emit_stages(scheduled, buffer_color):
    stages = []

    for scheduled_stage in scheduled:
        stage = {}

        for lane in ("rgb", "alpha"):
            choice = scheduled_stage[lane]
            if lane == "rgb":
                operand_default = "SRC_COLOR"
            else:
                operand_default = "SRC_ALPHA"

            if choice is None:
                stage[lane] = {"source": ["PREVIOUS"] * 3, "operand": [operand_default] * 3, "combine": "REPLACE", "scale": 1, "buffer_input": "PREVIOUS"}
                continue

            node, sources = choice
            source_names = []
            operands = []

            for index, argument in enumerate(node["arguments"]):
                select = argument["select"]
                if argument["kind"] == "source":
                    source_names.append(argument["source"])
                elif argument["kind"] == "constant":
                    source, select = scheduled_stage["placements"][(lane, index)]
                    source_names.append(source)
                else:
                    source_names.append(sources[index])

                if lane == "alpha" and select == "rgb":
                    raise CodeError("A color can't be read by the alpha")

                operands.append(RGB_OPERANDS[(select, argument["invert"])])

            while len(source_names) < 3:
                source_names.append("PREVIOUS")
                operands.append(operand_default)

            stage[lane] = {"source": source_names, "operand": operands, "combine": node["function"], "scale": node["scale"], "buffer_input": "PREVIOUS"}

        stage["constant_color"] = color_bytes(scheduled_stage["constant"])
        stage["buffer_color"] = [0, 0, 0, 0]
        stages.append(stage)

    initial = []
    for value in buffer_color:
        if value is None:
            value = 0.0
        initial.append(round(value * 255))
    stages[0]["buffer_color"] = initial

    # A stage that doesn't capture keeps the buffer when a later stage reads what it holds
    for stage_index, scheduled_stage in enumerate(scheduled):
        for lane_index, lane in enumerate(("rgb", "alpha")):
            if 0 < stage_index < len(stages) - 1 and not scheduled_stage["capture"][lane_index] and reads_buffer_later(scheduled, stage_index, lane_index):
                stages[stage_index][lane]["buffer_input"] = "PREVIOUS_BUFFER"

    return stages

def reads_buffer(scheduled_stage, lane_index):
    for lane in ("rgb", "alpha"):
        choice = scheduled_stage[lane]
        if choice is None:
            continue

        node, sources = choice
        for index, argument in enumerate(node["arguments"]):
            owner = 0
            if argument["kind"] == "constant":
                source, select = scheduled_stage["placements"][(lane, index)]
                if source != "PREVIOUS_BUFFER":
                    continue
            elif sources[index] != "PREVIOUS_BUFFER":
                continue
            else:
                select = argument["select"]

            if select == "a":
                owner = 1
            if owner == lane_index:
                return True

    return False

def reads_buffer_later(scheduled, stage_index, lane_index):
    """Whether a stage after this one reads the buffer of a lane before it is captured again."""
    for later in range(stage_index + 1, len(scheduled)):
        if reads_buffer(scheduled[later], lane_index):
            return True
        if scheduled[later]["capture"][lane_index]:
            return False
    return False

def compile_program(program):
    """The stages (dicts of formats/material/cmb.py) of a parsed program."""
    nodes, color, alpha = lower_program(program)
    scheduled, buffer_color = Scheduler(nodes, color, alpha).schedule()
    return emit_stages(scheduled, buffer_color)

##########################################
# Decompiler (stages -> code)
##########################################

def read_operand(lane_value, select, invert):
    """What a stage reads from a lane value with an operand: a lane value that is itself an argument keeps its own channel."""
    argument = dict(lane_value)

    if argument["kind"] == "constant":
        value = argument["value"]
        if argument["select"] == "rgb" and select in ("r", "g", "b"):
            argument["value"] = value["rgb".index(select)]
            argument["select"] = "value"
        argument["invert"] = argument["invert"] != invert
        return argument

    if argument["select"] == "rgb":
        argument["select"] = select
    argument["invert"] = argument["invert"] != invert

    return argument

def decompile_stages(stages):
    """The operations of a stage list: {"nodes", "color", "alpha"} with the outputs as arguments."""
    nodes = []
    zero = {"kind": "constant", "value": 0.0, "select": "value", "invert": False}
    previous = [zero, zero]

    initial = [value / 255 for value in stages[0]["buffer_color"]]
    buffer = [
        {"kind": "constant", "value": tuple(initial[0:3]), "select": "rgb", "invert": False},
        {"kind": "constant", "value": initial[3], "select": "value", "invert": False},
    ]

    for stage_index, stage in enumerate(stages):
        constant = [value / 255 for value in stage["constant_color"]]
        values = []

        for lane in ("rgb", "alpha"):
            channel = stage[lane]
            arguments = []

            if channel["combine"] not in FUNCTION_ARITY:
                raise CodeError(f"The combine function {channel['combine']} can't be written as code")

            for source, operand in zip(channel["source"], channel["operand"]):
                select, invert = OPERAND_SELECTS[operand]

                if source == "CONSTANT":
                    if select == "rgb":
                        argument = {"kind": "constant", "value": tuple(constant[0:3]), "select": "rgb", "invert": invert}
                    else:
                        argument = {"kind": "constant", "value": constant["rgba".index(select)], "select": "value", "invert": invert}
                elif source in ("PREVIOUS", "PREVIOUS_BUFFER"):
                    held = previous
                    if source == "PREVIOUS_BUFFER":
                        held = buffer
                    if select == "a":
                        argument = read_operand(held[1], select, invert)
                    else:
                        argument = read_operand(held[0], select, invert)
                else:
                    argument = {"kind": "source", "source": source, "select": select, "invert": invert}

                arguments.append(argument)

            arguments = arguments[:FUNCTION_ARITY[channel["combine"]]]

            if channel["combine"] == "REPLACE" and channel["scale"] == 1:
                values.append(arguments[0])
            else:
                node = {"id": len(nodes), "lane": lane, "function": channel["combine"], "arguments": arguments, "scale": channel["scale"]}
                nodes.append(node)
                if lane == "rgb":
                    values.append({"kind": "node", "node": node["id"], "select": "rgb", "invert": False})
                else:
                    values.append({"kind": "node", "node": node["id"], "select": "a", "invert": False})

        # The update of a stage shows from the next one
        if 0 < stage_index < len(stages) - 1:
            for index, lane in enumerate(("rgb", "alpha")):
                if stage[lane]["buffer_input"] == "PREVIOUS":
                    buffer[index] = previous[index]

        previous = values

    return {"nodes": nodes, "color": previous[0], "alpha": previous[1]}

##########################################
# Writer (operations -> code)
##########################################

def format_number(value):
    text = f"{value:.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text

# Priority of an expression: 1 sum, 2 product, 3 atom (name, call, parentheses)
SUM = 1
PRODUCT = 2
ATOM = 3

def wrap(written, level):
    text, written_level = written
    if written_level < level:
        return f"({text})"
    return text

def format_vector(values):
    if abs(values[0] - values[1]) < 1e-6 and abs(values[0] - values[2]) < 1e-6:
        return f"vec3({format_number(values[0])})"
    return "vec3(" + ", ".join(format_number(value) for value in values) + ")"

class Writer:
    """The stages as the FRAGMENT_SHADER section of a custom shader."""

    def __init__(self, decompiled):
        self.nodes = decompiled["nodes"]
        self.decompiled = decompiled
        self.uses = {}
        self.names = {}
        self.lines = []
        self.sources = set()
        self.helpers = set()

        for node in self.nodes:
            self.uses[node["id"]] = 0
        for node in self.nodes:
            for argument in node["arguments"]:
                self.count_use(argument)
        self.count_use(decompiled["color"])
        self.count_use(decompiled["alpha"])

    def count_use(self, argument):
        if argument is not None and argument["kind"] == "node":
            self.uses[argument["node"]] += 1

    def is_color(self, argument):
        """Whether an argument is a vec3 in the code, the other ones are floats."""
        if argument["kind"] == "node":
            return self.nodes[argument["node"]]["lane"] == "rgb" and argument["select"] == "rgb"
        return argument["select"] == "rgb"

    def argument(self, argument):
        """The code of an argument and its priority, inverted with 1.0 - x."""
        if argument["kind"] == "constant":
            if argument["select"] == "rgb":
                written = (format_vector(argument["value"]), ATOM)
            else:
                written = (format_number(argument["value"]), ATOM)
        elif argument["kind"] == "source":
            if argument["source"] not in SOURCE_GLSL:
                raise CodeError(f"The source {argument['source']} can't be written as code")
            self.sources.add(argument["source"])
            name = SOURCE_GLSL[argument["source"]]
            written = (f"{name}.{argument['select']}", ATOM)
        else:
            written = self.node(self.nodes[argument["node"]])
            if argument["select"] in ("r", "g", "b"):
                written = (f"{wrap(written, ATOM)}.{argument['select']}", ATOM)

        if argument["invert"]:
            return (f"(1.0 - {wrap(written, PRODUCT)})", ATOM)

        return written

    def node(self, node):
        if node["id"] in self.names:
            return (self.names[node["id"]], ATOM)

        written = self.expression(node)

        # A value read by several operations is named once
        if self.uses[node["id"]] > 1:
            if node["lane"] == "rgb":
                name = f"color{len(self.names)}"
                declared = "vec3"
            else:
                name = f"value{len(self.names)}"
                declared = "float"
            self.lines.append(f"    {declared} {name} = {written[0]};")
            self.names[node["id"]] = name
            return (name, ATOM)

        return written

    def call_argument(self, node, argument):
        """A float given to a function of the rgb lane is written vec3(x), so the code stays valid GLSL."""
        written = self.argument(argument)
        if node["lane"] == "rgb" and not self.is_color(argument):
            return f"vec3({written[0]})"
        return written[0]

    def expression(self, node):
        a = [self.argument(argument) for argument in node["arguments"]]
        function = node["function"]

        if function == "REPLACE":
            written = a[0]
        elif function == "MODULATE":
            written = (f"{wrap(a[0], PRODUCT)} * {wrap(a[1], ATOM)}", PRODUCT)
        elif function == "ADD":
            written = (f"{wrap(a[0], SUM)} + {wrap(a[1], PRODUCT)}", SUM)
        elif function == "SUBTRACT":
            written = (f"{wrap(a[0], SUM)} - {wrap(a[1], PRODUCT)}", SUM)
        elif function == "ADD_SIGNED":
            self.helpers.add("add_signed")
            written = (f"add_signed({self.call_argument(node, node['arguments'][0])}, {self.call_argument(node, node['arguments'][1])})", ATOM)
        elif function == "DOT3_RGB":
            self.helpers.add("dot3")
            written = (f"dot3({self.call_argument(node, node['arguments'][0])}, {self.call_argument(node, node['arguments'][1])})", ATOM)
        elif function == "INTERPOLATE":
            first = self.call_argument(node, node["arguments"][1])
            second = self.call_argument(node, node["arguments"][0])
            written = (f"mix({first}, {second}, {a[2][0]})", ATOM)
        elif function == "ADD_MULT":
            written = (f"({wrap(a[0], SUM)} + {wrap(a[1], PRODUCT)}) * {wrap(a[2], ATOM)}", PRODUCT)
        else:
            written = (f"{wrap(a[0], PRODUCT)} * {wrap(a[1], ATOM)} + {wrap(a[2], PRODUCT)}", SUM)

        if node["scale"] != 1:
            written = (f"{wrap(written, PRODUCT)} * {node['scale']}.0", PRODUCT)

        return written

    def write(self):
        alpha = self.argument(self.decompiled["alpha"])[0]
        color = self.argument(self.decompiled["color"])[0]
        if not self.is_color(self.decompiled["color"]):
            color = f"vec3({color})"

        lines = ["#ifdef FRAGMENT_SHADER"]

        for source in ("TEXTURE0", "TEXTURE1", "TEXTURE2"):
            if source in self.sources:
                lines.append(f"uniform sampler2D unf_frg_txt_2d_{source[-1]};")

        varyings = []
        if "PRIMARY_COLOR" in self.sources:
            varyings.append("var_clr")
        for source in ("TEXTURE0", "TEXTURE1", "TEXTURE2"):
            if source in self.sources:
                varyings.append(f"var_tx{source[-1]}")
        for name in varyings:
            lines.append(f"varying vec4 {name};")

        if "FRAGMENT_PRIMARY_COLOR" in self.sources or "FRAGMENT_SECONDARY_COLOR" in self.sources:
            lines.append("// Lighting of the material (.mtr), computed by the hardware: diffuse and specular")
            lines.append("varying vec4 clr_1st;")
            lines.append("varying vec4 clr_2nd;")

        if "dot3" in self.helpers:
            lines.append("")
            lines.append("// Stages of the combiner, every result is clamped to 0..1")
            lines.append("vec3 dot3(vec3 a, vec3 b) { return vec3(4.0 * dot(a - 0.5, b - 0.5)); }")
        if "add_signed" in self.helpers:
            if "dot3" not in self.helpers:
                lines.append("")
                lines.append("// Stages of the combiner, every result is clamped to 0..1")
            lines.append("vec3 add_signed(vec3 a, vec3 b) { return a + b - 0.5; }")
            lines.append("float add_signed(float a, float b) { return a + b - 0.5; }")

        lines.append("")
        lines.append("void main() {")
        for source in ("TEXTURE0", "TEXTURE1", "TEXTURE2"):
            if source in self.sources:
                unit = source[-1]
                lines.append(f"    vec4 tex{unit} = texture2D(unf_frg_txt_2d_{unit}, var_tx{unit}.xy);")
        lines += self.lines
        lines.append(f"    gl_FragColor = vec4({color}, {alpha});")
        lines.append("}")
        lines.append("#endif")

        return "\n".join(lines) + "\n"

def decompile_code(stages):
    """The FRAGMENT_SHADER section of a stage list (dicts of formats/material/cmb.py)."""
    return Writer(decompile_stages(stages)).write()

def uses_palette(stages):
    """The mask stage of the characters takes its constant from a color of the model, the code can't say that."""
    for stage in stages:
        rgb = stage["rgb"]
        if rgb["source"][:2] == ["CONSTANT", "TEXTURE0"] and rgb["operand"][1] in ("SRC_R", "SRC_G", "SRC_B"):
            return True
    return False
