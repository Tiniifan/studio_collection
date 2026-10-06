"""The combiner of a custom shader written as code: a small GLSL that compiles to the stages of the texture combiner and back.

    vec3 lit = (light_diffuse + light_specular) * texture0;
    color = lit * vertex_color * 2.0;
    alpha = texture0.a * vertex_color.a * 2.0;

Every operation clamps its result to 0..1 like the hardware, so the code computes what the game draws.
"""
import re

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

TOKEN_PATTERN = re.compile(r"\s*(?:(//[^\n]*|/\*.*?\*/)|(\d+\.\d*|\.\d+|\d+)|([A-Za-z_][A-Za-z_0-9]*)|(.))", re.S)

class CodeError(Exception):
    pass

##########################################
# Parser
##########################################

def tokenize(text):
    tokens = []
    line = 1
    position = 0

    while position < len(text):
        match = TOKEN_PATTERN.match(text, position)
        if match is None:
            break

        line += text.count("\n", position, match.start(0) + len(match.group(0)) - len(match.group(0).lstrip()))

        comment, number, name, symbol = match.groups()
        if number is not None:
            tokens.append(("number", float(number), line))
        elif name is not None:
            tokens.append(("name", name, line))
        elif symbol is not None and not symbol.isspace():
            tokens.append(("symbol", symbol, line))

        if comment is not None:
            line += comment.count("\n")

        position = match.end(0)

    tokens.append(("end", None, line))

    return tokens

class Parser:
    def __init__(self, text):
        self.tokens = tokenize(text)
        self.index = 0

    def peek(self):
        return self.tokens[self.index]

    def next(self):
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, value):
        token = self.next()
        if token[1] != value:
            raise CodeError(f"Line {token[2]}: '{value}' expected, found '{token[1]}'")
        return token

    def parse_program(self):
        program = {"variables": [], "color": None, "alpha": None}
        names = {}

        while self.peek()[0] != "end":
            token = self.next()
            if token[0] != "name":
                raise CodeError(f"Line {token[2]}: a statement starts with a name, found '{token[1]}'")

            declared = None
            if token[1] in ("vec3", "float"):
                declared = token[1]
                token = self.next()
                if token[0] != "name":
                    raise CodeError(f"Line {token[2]}: a variable name is expected after {declared}")

            name = token[1]
            line = token[2]
            self.expect("=")
            expression = self.parse_expression()
            self.expect(";")

            if name in ("color", "alpha"):
                program[name] = expression
            elif name in SOURCE_NAMES:
                raise CodeError(f"Line {line}: {name} is an input of the combiner, it can't be assigned")
            else:
                if name in names:
                    raise CodeError(f"Line {line}: {name} is already defined")
                names[name] = expression
                program["variables"].append({"name": name, "type": declared, "expression": expression})

        if program["color"] is None:
            raise CodeError("The code must assign color")

        if program["alpha"] is None:
            program["alpha"] = {"type": "number", "value": 1.0}

        check_names(program)

        return program

    def parse_expression(self):
        left = self.parse_term()

        while self.peek()[1] in ("+", "-"):
            operator = self.next()[1]
            left = {"type": "binary", "operator": operator, "left": left, "right": self.parse_term()}

        return left

    def parse_term(self):
        left = self.parse_postfix()

        while self.peek()[1] == "*":
            self.next()
            left = {"type": "binary", "operator": "*", "left": left, "right": self.parse_postfix()}

        return left

    def parse_postfix(self):
        expression = self.parse_primary()

        while self.peek()[1] == ".":
            self.next()
            token = self.next()
            if token[1] not in SWIZZLES:
                raise CodeError(f"Line {token[2]}: .{token[1]} is not a channel, use .rgb .a .r .g or .b")
            expression = {"type": "swizzle", "argument": expression, "swizzle": token[1]}

        return expression

    def parse_primary(self):
        token = self.next()

        if token[0] == "number":
            return {"type": "number", "value": token[1]}

        if token[1] == "(":
            expression = self.parse_expression()
            self.expect(")")
            return expression

        if token[0] != "name":
            raise CodeError(f"Line {token[2]}: unexpected '{token[1]}'")

        if self.peek()[1] != "(":
            return {"type": "name", "name": token[1], "line": token[2]}

        self.next()
        arguments = []
        if self.peek()[1] != ")":
            arguments.append(self.parse_expression())
            while self.peek()[1] == ",":
                self.next()
                arguments.append(self.parse_expression())
        self.expect(")")

        if token[1] == "vec3":
            values = []
            for argument in arguments:
                if argument["type"] != "number":
                    raise CodeError(f"Line {token[2]}: vec3 only takes numbers, the color of a constant")
                values.append(argument["value"])
            if len(values) == 1:
                values = values * 3
            if len(values) != 3:
                raise CodeError(f"Line {token[2]}: vec3 takes 1 or 3 numbers")
            return {"type": "vector", "values": values}

        arity = {"mix": 3, "dot3": 2, "add_signed": 2}
        if token[1] not in arity:
            raise CodeError(f"Line {token[2]}: unknown function {token[1]}, the combiner knows mix, dot3 and add_signed")
        if len(arguments) != arity[token[1]]:
            raise CodeError(f"Line {token[2]}: {token[1]} takes {arity[token[1]]} arguments")

        return {"type": "call", "name": token[1], "arguments": arguments}

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

def parse_code(text):
    return Parser(text).parse_program()

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
                    raise CodeError(f"{expression['name']} is a color, the alpha reads one channel of it: {expression['name']}.a")
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

def compile_code(text):
    return compile_program(parse_code(text))

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
    def __init__(self, decompiled):
        self.nodes = decompiled["nodes"]
        self.decompiled = decompiled
        self.uses = {}
        self.names = {}
        self.lines = []

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

    def argument(self, argument):
        """The code of an argument and its priority, inverted with 1.0 - x."""
        if argument["kind"] == "constant":
            if argument["select"] == "rgb":
                written = (format_vector(argument["value"]), ATOM)
            else:
                written = (format_number(argument["value"]), ATOM)
        elif argument["kind"] == "source":
            name = SOURCE_CODE_NAMES[argument["source"]]
            if argument["select"] == "rgb":
                written = (name, ATOM)
            else:
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
            self.lines.append(f"{declared} {name} = {written[0]};")
            self.names[node["id"]] = name
            return (name, ATOM)

        return written

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
            written = (f"add_signed({a[0][0]}, {a[1][0]})", ATOM)
        elif function == "DOT3_RGB":
            written = (f"dot3({a[0][0]}, {a[1][0]})", ATOM)
        elif function == "INTERPOLATE":
            written = (f"mix({a[1][0]}, {a[0][0]}, {a[2][0]})", ATOM)
        elif function == "ADD_MULT":
            written = (f"({wrap(a[0], SUM)} + {wrap(a[1], PRODUCT)}) * {wrap(a[2], ATOM)}", PRODUCT)
        else:
            written = (f"{wrap(a[0], PRODUCT)} * {wrap(a[1], ATOM)} + {wrap(a[2], PRODUCT)}", SUM)

        if node["scale"] != 1:
            written = (f"{wrap(written, PRODUCT)} * {node['scale']}.0", PRODUCT)

        return written

    def write(self):
        alpha = self.argument(self.decompiled["alpha"])
        color = self.argument(self.decompiled["color"])

        lines = list(self.lines)
        lines.append(f"color = {color[0]};")
        lines.append(f"alpha = {alpha[0]};")

        return "\n".join(lines) + "\n"

def decompile_code(stages):
    """The code of a stage list (dicts of formats/material/cmb.py)."""
    return Writer(decompile_stages(stages)).write()

##########################################
# Program writer (parsed code -> code)
##########################################

def write_expression(expression):
    """The code of a parsed expression and its priority."""
    kind = expression["type"]

    if kind == "number":
        return (format_number(expression["value"]), ATOM)
    if kind == "vector":
        return (format_vector(expression["values"]), ATOM)
    if kind == "name":
        return (expression["name"], ATOM)
    if kind == "swizzle":
        return (f"{wrap(write_expression(expression['argument']), ATOM)}.{expression['swizzle']}", ATOM)
    if kind == "call":
        arguments = ", ".join(write_expression(argument)[0] for argument in expression["arguments"])
        return (f"{expression['name']}({arguments})", ATOM)

    left = write_expression(expression["left"])
    right = write_expression(expression["right"])

    if expression["operator"] == "*":
        return (f"{wrap(left, PRODUCT)} * {wrap(right, ATOM)}", PRODUCT)

    return (f"{wrap(left, SUM)} {expression['operator']} {wrap(right, PRODUCT)}", SUM)

def write_program(program):
    lines = []

    for variable in program["variables"]:
        declared = variable["type"]
        if declared is None:
            declared = "vec3"
            if type_of(program, variable["expression"]) == "value":
                declared = "float"
        lines.append(f"{declared} {variable['name']} = {write_expression(variable['expression'])[0]};")

    lines.append(f"color = {write_expression(program['color'])[0]};")
    lines.append(f"alpha = {write_expression(program['alpha'])[0]};")

    return "\n".join(lines) + "\n"
