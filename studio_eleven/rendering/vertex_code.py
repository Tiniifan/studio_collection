"""The VERTEX_SHADER section of a custom shader compiled to a PICA200 vertex program (a DVLB with one DVLE).

The code is read into a graph of operations (functions inlined, loops unrolled, an if on a value becomes a select),
the graph is written as instructions, then the temporaries get their registers. The game binds the attributes and
the uniforms by their name in the DVLE, so the code uses the names of the game: atr_*, unf_vtx_*, var_*.
"""
import math
import struct

from ..formats.material import shdc
from . import shader_code
from .shader_code import CodeError

##########################################
# CONST
##########################################

# Input register of each attribute, the same in every program of the games
ATTRIBUTES = {
    "atr_pos": 0,
    "atr_clr": 1,
    "atr_nrm": 2,
    "atr_tng": 3,
    "atr_tx0": 4,
    "atr_tx1": 5,
    "atr_tx2": 6,
    "atr_pr0": 7,
    "atr_pr1": 8,
    "atr_pr2": 9,
}

ATTRIBUTE_TYPES = {
    "atr_nrm": "vec3",
    "atr_tng": "vec3",
}

# Uniforms the game fills, by their number in the uniform table (the float registers start at 16) and their array size
UNIFORMS = {
    "unf_vtx_cmr_prj": {"start": 16, "size": 4},
    "unf_vtx_glb_cmr": {"start": 20, "size": 3},
    "unf_vtx_lcl_glb": {"start": 23, "size": 3},
    "unf_vtx_eye": {"start": 26, "size": None},
    "unf_vtx_clr": {"start": 27, "size": None},
    "unf_vtx_txt": {"start": 28, "size": 6},
    "unf_vtx_bone": {"start": 34, "size": 72},
    "unf_vtx_silhouette_0": {"start": 106, "size": None},
    "unf_vtx_silhouette_1": {"start": 107, "size": None},
    "unf_vtx_str": {"start": 108, "size": None},
}

VARYINGS = ("var_pos", "var_clr", "var_nrm", "var_tng", "var_tx0", "var_tx1", "var_tx2", "var_prm", "var_gen")

# The outputs of the programs of the games: register, components, and what fills them
OUTPUTS = (
    ("POSITION", 0, "xyzw"),
    ("COLOR", 1, "xyzw"),
    ("TEXCOORD0", 2, "xy"),
    ("TEXCOORD0_W", 2, "z"),
    ("TEXCOORD1", 3, "xy"),
    ("TEXCOORD2", 4, "xy"),
    ("VIEW", 5, "xyz"),
    ("NORMAL_QUATERNION", 6, "xyzw"),
)

TYPE_WIDTHS = {
    "bool": 1,
    "int": 1,
    "float": 1,
    "vec2": 2,
    "vec3": 3,
    "vec4": 4,
    "ivec2": 2,
    "ivec3": 3,
    "ivec4": 4,
    "bvec2": 2,
    "bvec3": 3,
    "bvec4": 4,
}

FLOAT_TYPES = {
    1: "float",
    2: "vec2",
    3: "vec3",
    4: "vec4",
}

SWIZZLE_SETS = ("xyzw", "rgba", "stpq")

# Operations written by one instruction, component by component, that can write into any lanes
COMPONENT_OPS = ("add", "mul", "min", "max", "floor", "slt", "sge")

# Operations that give one value written in every lane
SCALAR_OPS = ("dot", "rcp", "rsq", "ex2", "lg2")

INSTRUCTIONS = {
    "add": "add",
    "mul": "mul",
    "min": "min",
    "max": "max",
    "floor": "flr",
    "slt": "slt",
    "sge": "sge",
    "rcp": "rcp",
    "rsq": "rsq",
    "ex2": "ex2",
    "lg2": "lg2",
}

TEMPORARY_COUNT = 16
FLOAT_UNIFORM_COUNT = 96
MAX_INSTRUCTIONS = 512
MAX_DESCRIPTORS = 128
MAX_MAD_DESCRIPTORS = 32
MAX_ITERATIONS = 256

##########################################
# Graph
##########################################

class Graph:
    """The operations of the code, two equal operations are one node, constants are folded."""

    def __init__(self):
        self.nodes = []
        self.keys = {}

    def add(self, op, args, width, extra=None):
        if extra is None:
            extra = {}

        key = (op, tuple(args), width, tuple(sorted(extra.items())))
        if key in self.keys:
            return self.keys[key]

        node = {"op": op, "args": list(args), "width": width}
        node.update(extra)
        self.nodes.append(node)
        self.keys[key] = len(self.nodes) - 1

        return len(self.nodes) - 1

    def node(self, index):
        return self.nodes[index]

    def width(self, index):
        return self.nodes[index]["width"]

    def const(self, values):
        values = tuple(float(value) for value in values)
        return self.add("const", [], len(values), {"values": values})

    def const_values(self, index):
        node = self.nodes[index]
        if node["op"] == "const":
            return node["values"]
        return None

    def swizzle(self, index, components):
        components = tuple(components)
        node = self.nodes[index]

        if components == tuple(range(node["width"])):
            return index

        if node["op"] == "const":
            return self.const([node["values"][component] for component in components])

        if node["op"] == "swizzle":
            return self.swizzle(node["args"][0], [node["components"][component] for component in components])

        if node["op"] == "combine":
            parts = [node["parts"][component] for component in components]
            return self.combine(parts)

        return self.add("swizzle", [index], len(components), {"components": components})

    def combine(self, parts):
        """parts is a list of (node, component), one per component of the result."""
        # The parts of a combine of combines are taken from the inner one
        flat = []
        for part in parts:
            inner = self.nodes[part[0]]
            if inner["op"] == "combine":
                flat.append(inner["parts"][part[1]])
            elif inner["op"] == "swizzle":
                flat.append((inner["args"][0], inner["components"][part[1]]))
            else:
                flat.append((part[0], part[1]))

        if len(flat) == 1:
            return self.swizzle(flat[0][0], [flat[0][1]])

        sources = set(part[0] for part in flat)
        if len(sources) == 1:
            return self.swizzle(flat[0][0], [part[1] for part in flat])

        values = []
        for part in flat:
            constant = self.const_values(part[0])
            if constant is None:
                break
            values.append(constant[part[1]])

        if len(values) == len(flat):
            return self.const(values)

        return self.add("combine", [part[0] for part in flat], len(flat), {"parts": tuple(flat)})

    def broadcast(self, index, width):
        if self.width(index) == width:
            return index
        if self.width(index) != 1:
            raise CodeError(f"A vector of {self.width(index)} components can't be used as one of {width}")
        return self.swizzle(index, [0] * width)

    def unary(self, op, index):
        node = self.nodes[index]
        constant = self.const_values(index)

        if op == "neg":
            if constant is not None:
                return self.const([-value for value in constant])
            if node["op"] == "neg":
                return node["args"][0]
            return self.add("neg", [index], node["width"])

        if constant is not None:
            return self.const([fold_unary(op, value) for value in constant])

        return self.add(op, [index], node["width"])

    def binary(self, op, left, right):
        width = max(self.width(left), self.width(right))
        left = self.broadcast(left, width)
        right = self.broadcast(right, width)

        left_values = self.const_values(left)
        right_values = self.const_values(right)

        if left_values is not None and right_values is not None:
            return self.const([fold_binary(op, left_values[i], right_values[i]) for i in range(width)])

        if op in ("add", "mul", "min", "max") and left_values is not None:
            swapped = left
            left = right
            right = swapped
            right_values = left_values

        if right_values is not None and op == "add" and all_equal(right_values, 0.0):
            return left

        if right_values is not None and op == "mul" and all_equal(right_values, 1.0):
            return left

        if right_values is not None and op == "mul" and all_equal(right_values, 0.0):
            return right

        if right_values is not None and op == "mul" and all_equal(right_values, -1.0):
            return self.unary("neg", left)

        # Same order for a commutative operation, so a * b and b * a are one node
        if op in ("add", "mul", "min", "max") and right < left and self.const_values(left) is None:
            swapped = left
            left = right
            right = swapped

        return self.add(op, [left, right], width)

    def dot(self, left, right):
        width = self.width(left)
        if width != self.width(right):
            raise CodeError("dot needs two vectors of the same size")

        left_values = self.const_values(left)
        right_values = self.const_values(right)
        if left_values is not None and right_values is not None:
            total = 0.0
            for i in range(width):
                total += left_values[i] * right_values[i]
            return self.const([total])

        if width == 1:
            return self.binary("mul", left, right)

        # dp3 / dp4 read three or four lanes, a vec2 is two products
        if width == 2:
            first = self.binary("mul", self.swizzle(left, [0]), self.swizzle(right, [0]))
            second = self.binary("mul", self.swizzle(left, [1]), self.swizzle(right, [1]))
            return self.binary("add", first, second)

        if right < left:
            swapped = left
            left = right
            right = swapped

        return self.add("dot", [left, right], 1, {"size": width})

def all_equal(values, value):
    for item in values:
        if item != value:
            return False
    return True

def fold_unary(op, value):
    if op == "rcp":
        if value == 0.0:
            return math.inf
        return 1.0 / value

    if op == "rsq":
        if value <= 0.0:
            return math.inf
        return 1.0 / math.sqrt(value)

    if op == "floor":
        return float(math.floor(value))

    if op == "int":
        return float(math.trunc(value))

    if op == "ex2":
        return 2.0 ** value

    if op == "lg2":
        if value <= 0.0:
            return -math.inf
        return math.log2(value)

    raise CodeError(f"Unknown operation {op}")

def fold_binary(op, left, right):
    if op == "add":
        return left + right
    if op == "mul":
        return left * right
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    if op == "slt":
        if left < right:
            return 1.0
        return 0.0
    if op == "sge":
        if left >= right:
            return 1.0
        return 0.0

    raise CodeError(f"Unknown operation {op}")

##########################################
# Reader
##########################################

def make_value(value_type, node):
    return {"type": value_type, "node": node}

def get_float_type(width, value_type="float"):
    if value_type in ("int", "ivec2", "ivec3", "ivec4") and width == 1:
        return "int"
    return FLOAT_TYPES[width]

def get_swizzle_components(text, width, line):
    for letters in SWIZZLE_SETS:
        if all(letter in letters for letter in text):
            components = [letters.index(letter) for letter in text]
            for component in components:
                if component >= width:
                    raise CodeError(f"'.{text}' reads past the {width} components of the value", line)
            if len(components) > 4:
                break
            return components

    raise CodeError(f"Unknown component '.{text}'", line)

class Reader:
    """Runs the code on the graph: every value is a node, the variables are scopes of values."""

    def __init__(self, tree, graph):
        self.tree = tree
        self.graph = graph
        self.scopes = [{}]
        self.declared = {}
        self.used_uniforms = set()
        self.call_depth = 0
        self.returned = None

    def error(self, message, line):
        raise CodeError(message, line)

    def read(self):
        for declaration in self.tree["declarations"]:
            self.declare_global(declaration)

        if "main" not in self.tree["functions"]:
            raise CodeError("The VERTEX_SHADER section has no main function")

        self.call_function(self.tree["functions"]["main"], [], self.tree["functions"]["main"]["line"])

        return self.make_outputs()

    def declare_global(self, declaration):
        name = declaration["name"]
        line = declaration["line"]
        qualifiers = declaration["qualifiers"]

        if "attribute" in qualifiers or ("in" in qualifiers and name in ATTRIBUTES):
            if name not in ATTRIBUTES:
                self.error(f"{name} is not an attribute of the game ({', '.join(ATTRIBUTES)})", line)
            self.scopes[0][name] = self.make_attribute(name, declaration["type"], line)
            return

        if "uniform" in qualifiers:
            if name not in UNIFORMS:
                self.error(f"The game doesn't fill the uniform {name} ({', '.join(UNIFORMS)})", line)
            if UNIFORMS[name]["size"] != declaration["size"]:
                self.error(f"{name} is an array of {UNIFORMS[name]['size']} vec4 in the game", line)
            self.scopes[0][name] = self.make_uniform(name)
            return

        if "varying" in qualifiers or "out" in qualifiers:
            if name not in VARYINGS:
                self.error(f"{name} is not a varying of the game ({', '.join(VARYINGS)})", line)
            self.scopes[0][name] = self.make_default(declaration["type"], declaration["size"], line)
            return

        self.declare_local(declaration)

    def make_attribute(self, name, value_type, line):
        if value_type not in ("vec2", "vec3", "vec4", "float"):
            self.error(f"The attribute {name} must be a float or a vector", line)

        node = self.graph.add("attribute", [], 4, {"register": ATTRIBUTES[name]})
        width = TYPE_WIDTHS[value_type]

        return make_value(value_type, self.graph.swizzle(node, range(width)))

    def make_uniform(self, name):
        self.used_uniforms.add(name)
        uniform = UNIFORMS[name]

        if uniform["size"] is None:
            return make_value("vec4", self.graph.add("uniform", [], 4, {"register": uniform["start"]}))

        return {"type": "vec4", "uniform_array": name}

    def make_default(self, value_type, size, line):
        if value_type not in TYPE_WIDTHS:
            self.error(f"The type {value_type} can't be used in a vertex shader", line)

        width = TYPE_WIDTHS[value_type]
        value = make_value(value_type, self.graph.const([0.0] * width))

        if size is not None:
            return {"type": value_type, "array": [value] * size}

        return value

    def lookup(self, name, line):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]

        # The names of the game can be used without a declaration
        if name in ATTRIBUTES:
            value_type = ATTRIBUTE_TYPES.get(name, "vec4")
            self.scopes[0][name] = self.make_attribute(name, value_type, line)
            return self.scopes[0][name]

        if name in UNIFORMS:
            self.scopes[0][name] = self.make_uniform(name)
            return self.scopes[0][name]

        if name in VARYINGS:
            self.scopes[0][name] = self.make_default("vec4", None, line)
            return self.scopes[0][name]

        if name == "gl_Position":
            self.scopes[0][name] = self.make_default("vec4", None, line)
            return self.scopes[0][name]

        self.error(f"Unknown name {name}", line)

    def store(self, name, value, line):
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return

        self.lookup(name, line)
        self.store(name, value, line)

    def declare_local(self, declaration):
        line = declaration["line"]
        value_type = declaration["type"]

        if value_type not in TYPE_WIDTHS:
            self.error(f"The type {value_type} can't be used in a vertex shader", line)

        if declaration["value"] is not None:
            value = self.convert(self.evaluate(declaration["value"]), value_type, line)
        else:
            value = self.make_default(value_type, declaration["size"], line)

        self.scopes[-1][declaration["name"]] = value

    def convert(self, value, value_type, line):
        """An int given to a float and a float given to an int are the same value, a size change is an error."""
        if "node" not in value:
            if value.get("type") != value_type:
                self.error(f"A {value.get('type')} array can't be given to a {value_type}", line)
            return value

        width = TYPE_WIDTHS[value_type]
        if self.graph.width(value["node"]) != width:
            if self.graph.width(value["node"]) == 1:
                return make_value(value_type, self.graph.broadcast(value["node"], width))
            self.error(f"A {value['type']} can't be given to a {value_type}", line)

        return make_value(value_type, value["node"])

    ##########################################
    # Statements
    ##########################################

    def execute_block(self, statements):
        """Runs statements, stops at a return (self.returned is set)."""
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
        elif kind == "if":
            self.execute_if(statement)
        elif kind == "for":
            self.execute_for(statement)
        elif kind == "return":
            value = None
            if statement["value"] is not None:
                value = self.evaluate(statement["value"])
            self.returned = {"value": value}
        else:
            self.error(f"Unknown statement {kind}", line)

    def get_constant(self, value):
        if "node" not in value:
            return None
        return self.graph.const_values(value["node"])

    def execute_if(self, statement):
        condition = self.evaluate(statement["condition"])
        constant = self.get_constant(condition)

        if constant is not None:
            branch = statement["else"]
            if constant[0] != 0.0:
                branch = statement["then"]
            self.scopes.append({})
            self.execute_block(branch)
            self.scopes.pop()
            return

        # A condition known only on the GPU: both branches run, every variable takes the value of its branch
        before = [dict(scope) for scope in self.scopes]

        self.scopes.append({})
        self.execute_block(statement["then"])
        self.scopes.pop()
        then_scopes = self.scopes
        then_returned = self.returned

        self.scopes = [dict(scope) for scope in before]
        self.returned = None
        self.scopes.append({})
        self.execute_block(statement["else"])
        self.scopes.pop()
        else_scopes = self.scopes
        else_returned = self.returned

        if (then_returned is None) != (else_returned is None):
            self.error("A return inside an if on a value needs a return in the else too", statement["line"])

        merged = []
        for index, scope in enumerate(then_scopes):
            result = dict(scope)
            for name, value in scope.items():
                other = else_scopes[index].get(name)
                if other is not None and other is not value:
                    result[name] = self.select(condition, value, other, statement["line"])
            merged.append(result)

        self.scopes = merged
        self.returned = None

        if then_returned is not None and then_returned["value"] is not None:
            self.returned = {"value": self.select(condition, then_returned["value"], else_returned["value"], statement["line"])}
        elif then_returned is not None:
            self.returned = {"value": None}

    def select(self, condition, first, second, line):
        if "array" in first:
            values = []
            for index in range(len(first["array"])):
                values.append(self.select(condition, first["array"][index], second["array"][index], line))
            return {"type": first["type"], "array": values}

        if "node" not in first:
            return first

        width = self.graph.width(first["node"])
        test = self.graph.broadcast(condition["node"], width)
        difference = self.graph.binary("add", first["node"], self.graph.unary("neg", second["node"]))
        node = self.graph.binary("add", second["node"], self.graph.binary("mul", difference, test))

        return make_value(first["type"], node)

    def execute_for(self, statement):
        line = statement["line"]
        self.scopes.append({})

        for initial in statement["initial"]:
            self.execute(initial)

        count = 0
        while True:
            if statement["condition"] is not None:
                condition = self.get_constant(self.evaluate(statement["condition"]))
                if condition is None:
                    self.error("A for loop must have a count known before the shader runs (for (int i = 0; i < 4; i++))", line)
                if condition[0] == 0.0:
                    break

            count += 1
            if count > MAX_ITERATIONS:
                self.error(f"A for loop can't run more than {MAX_ITERATIONS} times", line)

            self.scopes.append({})
            self.execute_block(statement["body"])
            self.scopes.pop()

            if self.returned is not None:
                self.error("A return inside a for loop is not supported", line)

            if statement["step"] is not None:
                self.evaluate(statement["step"])

        self.scopes.pop()

    ##########################################
    # Expressions
    ##########################################

    def evaluate(self, expression):
        kind = expression["kind"]
        line = expression["line"]

        if kind == "value":
            return expression["value"]

        if kind == "number":
            value_type = "float"
            if expression["int"]:
                value_type = "int"
            return make_value(value_type, self.graph.const([expression["value"]]))

        if kind == "name":
            return self.lookup(expression["name"], line)

        if kind == "binary":
            return self.evaluate_binary(expression)

        if kind == "unary":
            operand = self.read_node(self.evaluate(expression["operand"]), line)
            if expression["operator"] == "-":
                return make_value(operand["type"], self.graph.unary("neg", operand["node"]))
            one = self.graph.const([1.0] * self.graph.width(operand["node"]))
            return make_value("bool", self.graph.binary("add", one, self.graph.unary("neg", operand["node"])))

        if kind == "ternary":
            condition = self.read_node(self.evaluate(expression["condition"]), line)
            constant = self.get_constant(condition)
            if constant is not None:
                if constant[0] != 0.0:
                    return self.evaluate(expression["first"])
                return self.evaluate(expression["second"])
            first = self.read_node(self.evaluate(expression["first"]), line)
            second = self.read_node(self.evaluate(expression["second"]), line)
            return self.select(condition, first, second, line)

        if kind == "swizzle":
            operand = self.read_node(self.evaluate(expression["operand"]), line)
            components = get_swizzle_components(expression["components"], self.graph.width(operand["node"]), line)
            return make_value(get_float_type(len(components), operand["type"]), self.graph.swizzle(operand["node"], components))

        if kind == "index":
            return self.evaluate_index(expression)

        if kind == "assign":
            return self.evaluate_assign(expression)

        if kind == "increment":
            one = {"kind": "number", "value": 1.0, "int": True, "line": line}
            operator = "+="
            if expression["operator"] == "--":
                operator = "-="
            return self.evaluate_assign({"kind": "assign", "operator": operator, "target": expression["target"], "value": one, "line": line})

        if kind == "call":
            return self.evaluate_call(expression)

        self.error(f"Unknown expression {kind}", line)

    def read_node(self, value, line):
        if "node" not in value:
            self.error("An array can't be used as a value, take one of its elements", line)
        return value

    def evaluate_binary(self, expression):
        line = expression["line"]
        operator = expression["operator"]
        left = self.read_node(self.evaluate(expression["left"]), line)
        right = self.read_node(self.evaluate(expression["right"]), line)
        graph = self.graph

        width = max(graph.width(left["node"]), graph.width(right["node"]))
        value_type = left["type"]
        if graph.width(right["node"]) > graph.width(left["node"]):
            value_type = right["type"]

        if graph.width(left["node"]) != graph.width(right["node"]) and min(graph.width(left["node"]), graph.width(right["node"])) != 1:
            self.error(f"{left['type']} {operator} {right['type']}: the sizes don't match", line)

        a = left["node"]
        b = right["node"]

        if operator == "+":
            return make_value(value_type, graph.binary("add", a, b))
        if operator == "-":
            return make_value(value_type, graph.binary("add", a, graph.unary("neg", b)))
        if operator == "*":
            return make_value(value_type, graph.binary("mul", a, b))
        if operator == "/":
            return make_value(value_type, graph.binary("mul", a, self.reciprocal(graph.broadcast(b, width))))

        if operator == "<":
            return make_value("bool", graph.binary("slt", a, b))
        if operator == ">":
            return make_value("bool", graph.binary("slt", b, a))
        if operator == "<=":
            return make_value("bool", graph.binary("sge", b, a))
        if operator == ">=":
            return make_value("bool", graph.binary("sge", a, b))
        if operator == "==":
            return make_value("bool", graph.binary("mul", graph.binary("sge", a, b), graph.binary("sge", b, a)))
        if operator == "!=":
            equal = graph.binary("mul", graph.binary("sge", a, b), graph.binary("sge", b, a))
            return make_value("bool", graph.binary("add", graph.const([1.0] * width), graph.unary("neg", equal)))
        if operator == "&&":
            return make_value("bool", graph.binary("mul", a, b))
        if operator == "||":
            return make_value("bool", graph.binary("max", a, b))

        self.error(f"Unknown operator {operator}", line)

    def reciprocal(self, node):
        return self.per_component("rcp", node)

    def per_component(self, op, node):
        """rcp / rsq / ex2 / lg2 work on one value, a vector is done component by component."""
        width = self.graph.width(node)
        if width == 1:
            return self.graph.unary(op, node)

        parts = []
        for component in range(width):
            parts.append((self.graph.unary(op, self.graph.swizzle(node, [component])), 0))

        return self.graph.combine(parts)

    def evaluate_index(self, expression):
        line = expression["line"]
        operand = self.evaluate(expression["operand"])
        index = self.read_node(self.evaluate(expression["index"]), line)
        constant = self.get_constant(index)

        if "uniform_array" in operand:
            name = operand["uniform_array"]
            uniform = UNIFORMS[name]

            if constant is not None:
                position = int(constant[0])
                if position < 0 or position >= uniform["size"]:
                    self.error(f"{name}[{position}] is outside of its {uniform['size']} elements", line)
                return make_value("vec4", self.graph.add("uniform", [], 4, {"register": uniform["start"] + position}))

            split = self.split_index(index["node"])
            node = self.graph.add("uniform_dynamic", [split["node"]], 4, {"register": uniform["start"] + split["offset"], "array": name})
            return make_value("vec4", node)

        if constant is None:
            self.error("Only a uniform array can be read at an index computed by the shader", line)

        position = int(constant[0])

        if "array" in operand:
            if position < 0 or position >= len(operand["array"]):
                self.error(f"Index {position} is outside of the {len(operand['array'])} elements", line)
            return operand["array"][position]

        width = self.graph.width(operand["node"])
        if position < 0 or position >= width:
            self.error(f"Index {position} is outside of the {width} components", line)

        return make_value(get_float_type(1, operand["type"]), self.graph.swizzle(operand["node"], [position]))

    def split_index(self, node):
        """index = constant + value: the constant goes in the register, the value in the address register."""
        offset = 0

        while self.graph.node(node)["op"] == "add":
            args = self.graph.node(node)["args"]
            constant = self.graph.const_values(args[1])
            if constant is None:
                constant = self.graph.const_values(args[0])
                if constant is None:
                    break
                offset += int(constant[0])
                node = args[1]
            else:
                offset += int(constant[0])
                node = args[0]

        return {"offset": offset, "node": node}

    def evaluate_assign(self, expression):
        line = expression["line"]
        value = self.evaluate(expression["value"])
        operator = expression["operator"]

        if operator != "=":
            current = self.read_node(self.evaluate(expression["target"]), line)
            value = self.read_node(value, line)
            ops = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}
            value = self.evaluate_binary({
                "kind": "binary",
                "operator": ops[operator],
                "left": {"kind": "value", "value": current, "line": line},
                "right": {"kind": "value", "value": value, "line": line},
                "line": line,
            })

        self.assign(expression["target"], value, line)

        return value

    def assign(self, target, value, line):
        kind = target["kind"]

        if kind == "name":
            current = self.lookup(target["name"], line)
            if target["name"] in ATTRIBUTES or target["name"] in UNIFORMS:
                self.error(f"{target['name']} is filled by the game, it can't be written", line)
            if "node" in current:
                value = self.convert(value, current["type"], line)
            self.store(target["name"], value, line)
            return

        if kind == "swizzle":
            base = self.read_node(self.evaluate(target["operand"]), line)
            width = self.graph.width(base["node"])
            components = get_swizzle_components(target["components"], width, line)
            value = self.read_node(value, line)
            node = self.graph.broadcast(value["node"], len(components))

            parts = []
            for component in range(width):
                parts.append((base["node"], component))
            for position, component in enumerate(components):
                parts.append(None)
                parts[component] = (node, position)
            parts = parts[:width]

            self.assign(target["operand"], make_value(base["type"], self.graph.combine(parts)), line)
            return

        if kind == "index":
            container = self.evaluate(target["operand"])
            index = self.get_constant(self.read_node(self.evaluate(target["index"]), line))
            if index is None:
                self.error("An element can only be written at an index known before the shader runs", line)
            position = int(index[0])

            if "array" in container:
                if position < 0 or position >= len(container["array"]):
                    self.error(f"Index {position} is outside of the {len(container['array'])} elements", line)
                values = list(container["array"])
                values[position] = self.convert(value, container["type"], line)
                self.assign(target["operand"], {"type": container["type"], "array": values}, line)
                return

            swizzle = {"kind": "swizzle", "operand": target["operand"], "components": "xyzw"[position], "line": line}
            self.assign(swizzle, value, line)
            return

        if kind == "value":
            self.error("This can't be written", line)

        self.error("Only a variable, a component or an element can be written", line)

    def evaluate_call(self, expression):
        line = expression["line"]
        name = expression["name"]

        if name in self.tree["functions"]:
            arguments = [self.evaluate(argument) for argument in expression["arguments"]]
            return self.call_function(self.tree["functions"][name], arguments, line)

        arguments = []
        for argument in expression["arguments"]:
            arguments.append(self.read_node(self.evaluate(argument), line))

        if name in TYPE_WIDTHS:
            return self.construct(name, arguments, line)

        return self.call_builtin(name, arguments, line)

    def call_function(self, function, arguments, line):
        if self.call_depth > 16:
            self.error(f"{function['name']} calls itself, a shader can't", line)

        if len(arguments) != len(function["parameters"]):
            self.error(f"{function['name']} takes {len(function['parameters'])} arguments", line)

        scope = {}
        for index, parameter in enumerate(function["parameters"]):
            if "out" in parameter["qualifiers"] or "inout" in parameter["qualifiers"]:
                self.error("out parameters are not supported, return the value", line)
            scope[parameter["name"]] = self.convert(arguments[index], parameter["type"], line)

        saved_scopes = self.scopes
        saved_returned = self.returned
        self.scopes = [self.scopes[0], scope]
        self.returned = None
        self.call_depth += 1

        self.execute_block(function["body"])

        result = self.returned
        self.call_depth -= 1
        globals_scope = self.scopes[0]
        self.scopes = saved_scopes
        self.scopes[0] = globals_scope
        self.returned = saved_returned

        if function["type"] == "void":
            return {"type": "void"}

        if result is None or result["value"] is None:
            self.error(f"{function['name']} doesn't return a value", function["line"])

        return self.convert(result["value"], function["type"], line)

    def construct(self, value_type, arguments, line):
        width = TYPE_WIDTHS[value_type]

        if len(arguments) == 1 and self.graph.width(arguments[0]["node"]) == 1:
            node = arguments[0]["node"]
            if value_type == "int":
                node = self.graph.unary("int", node)
            return make_value(value_type, self.graph.broadcast(node, width))

        parts = []
        for argument in arguments:
            for component in range(self.graph.width(argument["node"])):
                parts.append((argument["node"], component))

        if len(parts) < width:
            self.error(f"{value_type} needs {width} components, {len(parts)} given", line)

        return make_value(value_type, self.graph.combine(parts[:width]))

    def call_builtin(self, name, arguments, line):
        graph = self.graph
        nodes = [argument["node"] for argument in arguments]
        value_type = "float"
        if len(arguments) > 0:
            value_type = arguments[0]["type"]

        counts = {
            "dot": 2, "normalize": 1, "length": 1, "distance": 2, "min": 2, "max": 2, "clamp": 3, "mix": 3,
            "floor": 1, "fract": 1, "ceil": 1, "abs": 1, "sign": 1, "step": 2, "sqrt": 1, "inversesqrt": 1,
            "pow": 2, "exp2": 1, "log2": 1, "exp": 1, "log": 1, "reflect": 2, "cross": 2,
        }

        if name in ("sin", "cos", "tan", "asin", "acos", "atan", "texture2D", "texture"):
            self.error(f"{name} doesn't exist on the vertex unit of the 3DS", line)

        if name not in counts:
            self.error(f"Unknown function {name}", line)

        if len(arguments) != counts[name]:
            self.error(f"{name} takes {counts[name]} arguments", line)

        if name == "dot":
            return make_value("float", graph.dot(nodes[0], nodes[1]))

        if name == "normalize":
            scale = graph.unary("rsq", graph.dot(nodes[0], nodes[0]))
            return make_value(value_type, graph.binary("mul", nodes[0], scale))

        if name == "length":
            return make_value("float", self.square_root(graph.dot(nodes[0], nodes[0])))

        if name == "distance":
            difference = graph.binary("add", nodes[0], graph.unary("neg", nodes[1]))
            return make_value("float", self.square_root(graph.dot(difference, difference)))

        if name in ("min", "max"):
            return make_value(value_type, graph.binary(name, nodes[0], nodes[1]))

        if name == "clamp":
            return make_value(value_type, graph.binary("min", graph.binary("max", nodes[0], nodes[1]), nodes[2]))

        if name == "mix":
            difference = graph.binary("add", nodes[1], graph.unary("neg", nodes[0]))
            return make_value(value_type, graph.binary("add", nodes[0], graph.binary("mul", difference, nodes[2])))

        if name == "floor":
            return make_value(value_type, graph.unary("floor", nodes[0]))

        if name == "fract":
            return make_value(value_type, graph.binary("add", nodes[0], graph.unary("neg", graph.unary("floor", nodes[0]))))

        if name == "ceil":
            return make_value(value_type, graph.unary("neg", graph.unary("floor", graph.unary("neg", nodes[0]))))

        if name == "abs":
            return make_value(value_type, graph.binary("max", nodes[0], graph.unary("neg", nodes[0])))

        if name == "sign":
            zero = graph.const([0.0] * graph.width(nodes[0]))
            positive = graph.binary("slt", zero, nodes[0])
            negative = graph.binary("slt", nodes[0], zero)
            return make_value(value_type, graph.binary("add", positive, graph.unary("neg", negative)))

        if name == "step":
            return make_value(arguments[1]["type"], graph.binary("sge", nodes[1], nodes[0]))

        if name == "sqrt":
            return make_value(value_type, self.square_root(nodes[0]))

        if name == "inversesqrt":
            return make_value(value_type, self.per_component("rsq", nodes[0]))

        if name == "exp2":
            return make_value(value_type, self.per_component("ex2", nodes[0]))

        if name == "log2":
            return make_value(value_type, self.per_component("lg2", nodes[0]))

        if name == "exp":
            return make_value(value_type, self.per_component("ex2", graph.binary("mul", nodes[0], graph.const([1.0 / math.log(2.0)]))))

        if name == "log":
            return make_value(value_type, graph.binary("mul", self.per_component("lg2", nodes[0]), graph.const([math.log(2.0)])))

        if name == "pow":
            logarithm = self.per_component("lg2", nodes[0])
            return make_value(value_type, self.per_component("ex2", graph.binary("mul", logarithm, nodes[1])))

        if name == "reflect":
            twice = graph.binary("mul", graph.dot(nodes[1], nodes[0]), graph.const([2.0]))
            return make_value(value_type, graph.binary("add", nodes[0], graph.unary("neg", graph.binary("mul", nodes[1], twice))))

        first = graph.binary("mul", graph.swizzle(nodes[0], [1, 2, 0]), graph.swizzle(nodes[1], [2, 0, 1]))
        second = graph.binary("mul", graph.swizzle(nodes[0], [2, 0, 1]), graph.swizzle(nodes[1], [1, 2, 0]))
        return make_value("vec3", graph.binary("add", first, graph.unary("neg", second)))

    def square_root(self, node):
        return self.per_component("rcp", self.per_component("rsq", node))

    ##########################################
    # Outputs
    ##########################################

    def get_written(self, name, default):
        value = self.scopes[0].get(name)
        if value is None or "node" not in value:
            return self.graph.const(default)

        node = value["node"]
        width = self.graph.width(node)
        if width < 4:
            parts = []
            for component in range(4):
                if component < width:
                    parts.append((node, component))
                else:
                    parts.append((self.graph.const([default[component]]), 0))
            node = self.graph.combine(parts)

        return node

    def make_outputs(self):
        """The node of every output: position, color, texture coordinates, view vector and normal quaternion."""
        graph = self.graph

        if "gl_Position" not in self.scopes[0]:
            raise CodeError("The vertex shader must write gl_Position")

        position = self.get_written("gl_Position", (0.0, 0.0, 0.0, 1.0))
        # The combiners double the color, an unwritten var_clr is white
        color = self.get_written("var_clr", (0.5, 0.5, 0.5, 0.5))
        texcoord0 = self.get_written("var_tx0", (0.0, 0.0, 0.0, 0.0))
        texcoord1 = self.get_written("var_tx1", (0.0, 0.0, 0.0, 0.0))
        texcoord2 = self.get_written("var_tx2", (0.0, 0.0, 0.0, 0.0))

        # The lighting reads the vector from the vertex to the eye, the code gives the position in the camera space
        eye = self.get_written("var_prm", (0.0, 0.0, 0.0, 0.0))
        view = graph.unary("neg", graph.swizzle(eye, [0, 1, 2]))

        normal = self.scopes[0].get("var_nrm")
        if normal is None or "node" not in normal:
            quaternion = graph.const((0.0, 0.0, 1.0, 0.0))
        else:
            quaternion = self.make_quaternion(graph.swizzle(self.get_written("var_nrm", (0.0, 0.0, 1.0, 0.0)), [0, 1, 2]))

        return {
            "POSITION": position,
            "COLOR": color,
            "TEXCOORD0": graph.swizzle(texcoord0, [0, 1]),
            "TEXCOORD0_W": graph.swizzle(texcoord0, [2]),
            "TEXCOORD1": graph.swizzle(texcoord1, [0, 1]),
            "TEXCOORD2": graph.swizzle(texcoord2, [0, 1]),
            "VIEW": view,
            "NORMAL_QUATERNION": quaternion,
        }

    def make_quaternion(self, normal):
        """The rotation of (0, 0, 1) to the normal, as the programs of the games compute it."""
        graph = self.graph
        normal = graph.binary("mul", normal, graph.unary("rsq", graph.dot(normal, normal)))
        z = graph.swizzle(normal, [2])

        # (1 + z) / 2, kept above 0 for a normal looking away from the camera
        half = graph.binary("max", graph.binary("mul", graph.binary("add", z, graph.const([1.0])), graph.const([0.5])), graph.const([0.00001]))
        inverse = graph.unary("rsq", half)
        xy = graph.binary("mul", graph.binary("mul", graph.swizzle(normal, [0, 1]), graph.const([0.5])), graph.broadcast(inverse, 2))

        return graph.combine([(xy, 0), (xy, 1), (graph.unary("rcp", inverse), 0), (graph.const([0.0]), 0)])

##########################################
# Instructions
##########################################

class ProgramWriter:
    """Writes the graph as instructions on virtual temporaries, then gives them the 16 registers."""

    def __init__(self, graph, outputs, used_uniforms):
        self.graph = graph
        self.outputs = outputs
        self.used_uniforms = used_uniforms
        self.instructions = []
        self.locations = {}
        self.temporary_count = 0
        self.address = {1: None, 2: None}
        self.address_order = [1, 2]
        self.references = {}
        self.constant_registers = []
        self.free_registers = self.get_free_registers()

    def get_free_registers(self):
        used = set()
        for name in self.used_uniforms:
            uniform = UNIFORMS[name]
            size = 1
            if uniform["size"] is not None:
                size = uniform["size"]
            for register in range(uniform["start"], uniform["start"] + size):
                used.add(register - shdc.UNIFORM_TABLE_FLOAT)

        free = []
        for register in range(FLOAT_UNIFORM_COUNT - 1, -1, -1):
            if register not in used:
                free.append(register)

        return free

    def find_readers(self):
        """The nodes and outputs that read each node, a swizzle gives its readers to its source."""
        order = []
        seen = set()
        stack = list(self.outputs.values())

        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            order.append(index)
            for argument in self.graph.node(index)["args"]:
                stack.append(argument)

        readers = {}
        for output_type, index in self.outputs.items():
            readers.setdefault(index, set()).add(("output", output_type))

        # A node always has a higher index than its arguments: its readers are all known before its sources are reached
        for index in sorted(order, reverse=True):
            node = self.graph.node(index)
            for argument in node["args"]:
                if node["op"] == "swizzle":
                    readers.setdefault(argument, set()).update(readers.get(index, set()))
                else:
                    readers.setdefault(argument, set()).add(index)

        self.references = readers

    def read_only_by(self, index, owners):
        return self.references.get(index, set()) <= owners

    def new_temporary(self):
        self.temporary_count += 1
        return self.temporary_count - 1

    ##########################################
    # Operands
    ##########################################

    def get_constant_operand(self, values):
        rounded = [shdc.float24_to_float(shdc.float_to_float24(value)) for value in values]

        for register in self.constant_registers:
            lanes = []
            for value in rounded:
                if value in register["values"]:
                    lanes.append(register["values"].index(value))
            if len(lanes) == len(rounded):
                return {"kind": "c", "index": register["register"], "lanes": lanes, "negate": False, "dynamic": None}

        for register in self.constant_registers:
            missing = []
            for value in rounded:
                if value not in register["values"] and value not in missing:
                    missing.append(value)
            if len(register["values"]) + len(missing) <= 4:
                for value in missing:
                    register["values"].append(value)
                return self.get_constant_operand(values)

        if len(self.free_registers) == 0:
            raise CodeError("No register left for the constants of the vertex shader")

        self.constant_registers.append({"register": self.free_registers.pop(0), "values": []})

        return self.get_constant_operand(values)

    def get_operand(self, index):
        """Where the components of a node are: the register and the lane of each component."""
        node = self.graph.node(index)
        op = node["op"]

        if op == "const":
            return self.get_constant_operand(node["values"])

        if op == "attribute":
            return {"kind": "v", "index": node["register"], "lanes": [0, 1, 2, 3], "negate": False, "dynamic": None}

        if op == "uniform":
            return {"kind": "c", "index": node["register"] - shdc.UNIFORM_TABLE_FLOAT, "lanes": [0, 1, 2, 3], "negate": False, "dynamic": None}

        if op == "uniform_dynamic":
            return {"kind": "c", "index": node["register"] - shdc.UNIFORM_TABLE_FLOAT, "lanes": [0, 1, 2, 3], "negate": False, "dynamic": node["args"][0]}

        if op == "swizzle":
            operand = self.get_operand(node["args"][0])
            lanes = [operand["lanes"][component] for component in node["components"]]
            return self.with_lanes(operand, lanes)

        if op == "neg":
            operand = dict(self.get_operand(node["args"][0]))
            operand["negate"] = not operand["negate"]
            return operand

        self.compute(index)
        location = self.locations[index]

        return {"kind": "r", "index": location, "lanes": list(range(node["width"])), "negate": False, "dynamic": None}

    def with_lanes(self, operand, lanes):
        result = dict(operand)
        result["lanes"] = lanes
        return result

    def is_wide(self, operand):
        # Only a 7 bit source field reaches the uniforms
        return operand["kind"] == "c"

    def to_temporary(self, operand):
        """A uniform copied to a temporary, when an instruction would read two of them."""
        temporary = self.new_temporary()
        lanes = list(operand["lanes"])
        while len(lanes) < 4:
            lanes.append(lanes[-1])

        self.emit("mov", {"kind": "r", "index": temporary}, "xyzw", [self.with_lanes(operand, lanes)], {0: 0, 1: 1, 2: 2, 3: 3})

        return {"kind": "r", "index": temporary, "lanes": [0, 1, 2, 3][:len(operand["lanes"])], "negate": False, "dynamic": None}

    ##########################################
    # Emit
    ##########################################

    def load_address(self, operand):
        """The address register that holds the index of a uniform read at a computed index."""
        index = operand["dynamic"]

        for lane in (1, 2):
            if self.address[lane] == index:
                self.address_order.remove(lane)
                self.address_order.append(lane)
                return lane

        lane = self.address_order.pop(0)
        self.address_order.append(lane)

        # mova truncates the value, int() is done by the move
        source = index
        if self.graph.node(index)["op"] == "int":
            source = self.graph.node(index)["args"][0]

        value = self.get_operand(source)
        lanes = [value["lanes"][0]] * 4
        self.instructions.append({
            "op": "mova",
            "dst": None,
            "mask": "xy"[lane - 1],
            "src": [self.with_lanes(value, lanes)],
        })
        self.address[lane] = index

        return lane

    def emit(self, op, target, mask, operands, mapping):
        """One instruction: target lane L takes component mapping[L] of each operand (a scalar op ignores the mapping)."""
        sources = []

        for operand in operands:
            lanes = []
            for lane in range(4):
                component = mapping.get(lane)
                if component is None:
                    component = mapping[min(mapping)]
                if op in ("dp3", "dp4"):
                    component = lane
                if component >= len(operand["lanes"]):
                    component = len(operand["lanes"]) - 1
                lanes.append(operand["lanes"][component])
            sources.append(self.with_lanes(operand, lanes))

        relative = 0
        for source in sources:
            if source["dynamic"] is not None:
                relative = self.load_address(source)

        self.instructions.append({
            "op": op,
            "dst": target,
            "mask": mask,
            "src": sources,
            "relative": relative,
        })

    def emit_node(self, index, target, mapping):
        """Writes a node into the lanes of a target, mapping is lane -> component of the node."""
        node = self.graph.node(index)
        op = node["op"]
        mask = ""
        for lane, component in enumerate("xyzw"):
            if lane in mapping:
                mask += component

        if op == "add":
            fused = self.find_product(index, node)
            if fused is not None:
                self.emit_mad(fused, target, mask, mapping)
                return

        if op in ("add", "mul", "min", "max"):
            first = self.get_operand(node["args"][0])
            second = self.get_operand(node["args"][1])
            if self.is_wide(second) and not self.is_wide(first):
                swapped = first
                first = second
                second = swapped
            if self.is_wide(second):
                second = self.to_temporary(second)
            self.emit(INSTRUCTIONS[op], target, mask, [first, second], mapping)
            return

        if op in ("slt", "sge"):
            first = self.get_operand(node["args"][0])
            second = self.get_operand(node["args"][1])
            if self.is_wide(first) and self.is_wide(second):
                second = self.to_temporary(second)
            if self.is_wide(second):
                self.emit(INSTRUCTIONS[op] + "i", target, mask, [first, second], mapping)
            else:
                self.emit(INSTRUCTIONS[op], target, mask, [first, second], mapping)
            return

        if op == "dot":
            first = self.get_operand(node["args"][0])
            second = self.get_operand(node["args"][1])
            if self.is_wide(second) and not self.is_wide(first):
                swapped = first
                first = second
                second = swapped
            if self.is_wide(second):
                second = self.to_temporary(second)
            instruction = "dp4"
            if node["size"] == 3:
                instruction = "dp3"
            self.emit(instruction, target, mask, [first, second], mapping)
            return

        if op in ("rcp", "rsq", "ex2", "lg2", "floor"):
            operand = self.get_operand(node["args"][0])
            if op != "floor":
                mapping = {lane: 0 for lane in mapping}
            self.emit(INSTRUCTIONS[op], target, mask, [operand], mapping)
            return

        if op == "int":
            # A truncation used as a value, the index of a uniform array takes it from mova
            operand = self.get_operand(node["args"][0])
            self.emit("flr", target, mask, [operand], mapping)
            return

        operand = self.get_operand(index)
        self.emit("mov", target, mask, [operand], mapping)

    def find_product(self, index, node):
        """a * b + c written as one mad when the product is read only there."""
        for position in (0, 1):
            argument = node["args"][position]
            other = node["args"][1 - position]
            negate = False
            product = self.graph.node(argument)

            if product["op"] == "neg":
                negate = True
                argument = product["args"][0]
                product = self.graph.node(argument)

            if product["op"] != "mul" or argument in self.locations:
                continue
            owner = index
            if negate:
                owner = node["args"][position]
                if not self.read_only_by(owner, {index}):
                    continue
            if not self.read_only_by(argument, {owner}):
                continue
            if product["width"] != node["width"]:
                continue

            return {"product": product, "addend": other, "negate": negate}

        return None

    def emit_mad(self, fused, target, mask, mapping):
        first = self.get_operand(fused["product"]["args"][0])
        second = self.get_operand(fused["product"]["args"][1])
        addend = self.get_operand(fused["addend"])

        if fused["negate"]:
            first = dict(first)
            first["negate"] = not first["negate"]

        if self.is_wide(first) and not self.is_wide(second):
            swapped = first
            first = second
            second = swapped

        if self.is_wide(first):
            first = self.to_temporary(first)

        if self.is_wide(second) and self.is_wide(addend):
            addend = self.to_temporary(addend)

        if self.is_wide(addend):
            self.emit("madi", target, mask, [first, second, addend], mapping)
        else:
            self.emit("mad", target, mask, [first, second, addend], mapping)

    def compute(self, index):
        """A node kept in a temporary for its readers."""
        if index in self.locations:
            return

        node = self.graph.node(index)

        if node["op"] == "combine":
            temporary = self.new_temporary()
            parts = [(lane, part[0], part[1]) for lane, part in enumerate(node["parts"])]
            self.write_parts({"kind": "r", "index": temporary}, parts, {index})
            self.locations[index] = temporary
            return

        temporary = self.new_temporary()
        mapping = {}
        for lane in range(node["width"]):
            mapping[lane] = lane
        if node["op"] in SCALAR_OPS:
            mapping = {0: 0}

        self.emit_node(index, {"kind": "r", "index": temporary}, mapping)
        self.locations[index] = temporary

    def flatten_parts(self, parts, owners):
        """Swizzles and combines read only here are opened, so their sources are written in place."""
        flat = []

        for part in parts:
            lane = part[0]
            index = part[1]
            component = part[2]
            node = self.graph.node(index)

            while True:
                if node["op"] == "swizzle":
                    component = node["components"][component]
                    index = node["args"][0]
                elif node["op"] == "combine" and index not in self.locations and self.read_only_by(index, owners):
                    owners.add(index)
                    inner = node["parts"][component]
                    index = inner[0]
                    component = inner[1]
                else:
                    break
                node = self.graph.node(index)

            flat.append((lane, index, component))

        return flat

    def write_parts(self, target, parts, owners):
        """parts is a list of (lane of the target, node, component of the node), owners are the readers that write here."""
        owners = set(owners)
        parts = self.flatten_parts(parts, owners)
        groups = {}
        order = []

        for part in parts:
            if part[1] not in groups:
                groups[part[1]] = {}
                order.append(part[1])
            groups[part[1]][part[0]] = part[2]

        for index in order:
            mapping = groups[index]
            node = self.graph.node(index)
            direct = index not in self.locations and self.read_only_by(index, owners)

            if direct and (node["op"] in COMPONENT_OPS or node["op"] in SCALAR_OPS or node["op"] == "int"):
                self.emit_node(index, target, mapping)
            else:
                operand = self.get_operand(index)
                mask = ""
                for lane, letter in enumerate("xyzw"):
                    if lane in mapping:
                        mask += letter
                self.emit("mov", target, mask, [operand], mapping)

    def write(self):
        self.find_readers()

        for output in OUTPUTS:
            output_type = output[0]
            register = output[1]
            mask = output[2]
            index = self.outputs[output_type]

            parts = []
            for position, letter in enumerate(mask):
                parts.append(("xyzw".index(letter), index, position))

            self.write_parts({"kind": "o", "index": register}, parts, {("output", output_type)})

        self.instructions.append({"op": "end"})
        self.instructions.append({"op": "nop"})

        return self.assign_registers()

    ##########################################
    # Registers
    ##########################################

    def assign_registers(self):
        """Linear scan over the instructions: a temporary lives from its first write to its last read."""
        first = {}
        last = {}

        for position, instruction in enumerate(self.instructions):
            for source in instruction.get("src", []):
                if source["kind"] == "r":
                    last[source["index"]] = position
            target = instruction.get("dst")
            if target is not None and target["kind"] == "r":
                if target["index"] not in first:
                    first[target["index"]] = position
                last[target["index"]] = max(last.get(target["index"], position), position)

        ends = {}
        for temporary in last:
            ends.setdefault(last[temporary], []).append(temporary)

        starts = {}
        for temporary in first:
            starts.setdefault(first[temporary], []).append(temporary)

        registers = {}
        free = list(range(TEMPORARY_COUNT))

        for position in range(len(self.instructions)):
            # A value read for the last time here gives its register to a value written here
            for temporary in ends.get(position, []):
                if temporary in registers and first.get(temporary, -1) < position:
                    free.insert(0, registers[temporary])
            for temporary in starts.get(position, []):
                if len(free) == 0:
                    raise CodeError("The vertex shader needs more than 16 temporary registers, keep fewer values alive at once")
                registers[temporary] = free.pop(0)
            for temporary in ends.get(position, []):
                if first.get(temporary, -1) == position and temporary in registers:
                    free.insert(0, registers[temporary])

        return registers

##########################################
# Binary
##########################################

def get_register_field(operand, registers):
    if operand["kind"] == "v":
        return shdc.INPUT_REGISTER + operand["index"]
    if operand["kind"] == "r":
        return shdc.TEMPORARY_REGISTER + registers[operand["index"]]
    return shdc.UNIFORM_REGISTER + operand["index"]

def get_target_field(target, registers):
    if target["kind"] == "o":
        return target["index"]
    return shdc.TEMPORARY_REGISTER + registers[target["index"]]

def get_lane_text(lanes):
    text = ""
    for lane in lanes:
        text += "xyzw"[lane]
    return text

def encode_program(writer, registers):
    """The instructions with their operand descriptors, those of mad first (a mad has 5 bits to name it)."""
    descriptors = []
    keys = {}

    for madness in (True, False):
        for instruction in writer.instructions:
            if instruction["op"] in ("end", "nop"):
                continue
            if (instruction["op"] in ("mad", "madi")) != madness:
                continue
            sources = [(source["negate"], get_lane_text(source["lanes"])) for source in instruction["src"]]
            descriptor = shdc.make_operand_descriptor(instruction["mask"], sources)
            if descriptor not in keys:
                keys[descriptor] = len(descriptors)
                descriptors.append(descriptor)
            instruction["desc"] = keys[descriptor]

    for instruction in writer.instructions:
        if instruction["op"] in ("mad", "madi") and instruction["desc"] >= MAX_MAD_DESCRIPTORS:
            raise CodeError("The vertex shader has too many different mad operations")

    if len(descriptors) > MAX_DESCRIPTORS:
        raise CodeError("The vertex shader has too many different operand layouts")

    code = []
    for instruction in writer.instructions:
        op = instruction["op"]
        if op in ("end", "nop"):
            code.append(shdc.encode_instruction({"op": op}))
            continue

        fields = [get_register_field(source, registers) for source in instruction["src"]]
        target = 0
        if instruction["dst"] is not None:
            target = get_target_field(instruction["dst"], registers)

        code.append(shdc.encode_instruction({
            "op": op,
            "dst": target,
            "src": fields,
            "desc": instruction["desc"],
            "relative": instruction.get("relative", 0),
        }))

    if len(code) > MAX_INSTRUCTIONS:
        raise CodeError(f"The vertex shader is {len(code)} instructions long, the 3DS runs {MAX_INSTRUCTIONS} at most")

    return {"code": code, "operand_descriptors": descriptors}

def make_uniform_table(used_uniforms):
    uniforms = []

    for name in ATTRIBUTES:
        uniforms.append({"name": f"{name}.xyzw", "start": ATTRIBUTES[name], "end": ATTRIBUTES[name]})

    for name in UNIFORMS:
        if name in used_uniforms:
            uniform = UNIFORMS[name]
            size = 1
            if uniform["size"] is not None:
                size = uniform["size"]
            uniforms.append({"name": name, "start": uniform["start"], "end": uniform["start"] + size - 1})

    return uniforms

##########################################
# Vertex Code Function
##########################################

def read_vertex(tree):
    graph = Graph()
    reader = Reader(tree, graph)
    outputs = reader.read()

    return {"graph": graph, "outputs": outputs, "used_uniforms": reader.used_uniforms}

def compile_vertex(tree, file_version):
    """The DVLB (not compressed) of the VERTEX_SHADER section of a parsed shader."""
    vertex = read_vertex(tree)
    writer = ProgramWriter(vertex["graph"], vertex["outputs"], vertex["used_uniforms"])
    registers = writer.write()
    encoded = encode_program(writer, registers)

    constants = []
    for register in writer.constant_registers:
        values = list(register["values"])
        while len(values) < 4:
            values.append(0.0)
        constants.append({"type": 2, "register": register["register"], "values": values})

    version = shdc.SHADER_VERSIONS[file_version]
    end = len(encoded["code"])
    outputs = [{"type": output[0], "register": output[1], "mask": output[2]} for output in OUTPUTS]

    program = {
        "version": version,
        "type": 0,
        "merge_outputs": 0,
        "main": 0,
        "endmain": end,
        "input_mask": 0x3FF,
        "output_mask": 0x7F,
        "geometry": bytes(4),
        "constants": constants,
        "labels": [
            {"id": 0, "unk": bytes(3), "offset": 0, "size": 0xFFFFFFFF, "name": "main"},
            {"id": 1, "unk": bytes([0, 1, 0]), "offset": end, "size": 0xFFFFFFFF, "name": "endmain"},
        ],
        "outputs": outputs,
        "uniforms": make_uniform_table(vertex["used_uniforms"]),
    }

    shader = {
        "version": version,
        "code": encoded["code"],
        "operand_descriptors": encoded["operand_descriptors"],
        "file_names": ["custom.obj"],
        "programs": [program],
    }

    return {
        "data": shdc.write_shdc(shader),
        "instruction_count": len(encoded["code"]),
        "temporary_count": len(set(registers.values())),
    }

def compile_vertex_code(text, file_version):
    tree = shader_code.parse_section(text)
    return compile_vertex(tree, file_version)

##########################################
# StudioRender
##########################################

# What StudioRender gives in place of the attributes and uniforms of the game, the mesh it draws is already deformed by its armature
BLENDER_VERTEX_HEADER = """
in vec3 atr_pos;
in vec3 atr_nrm;
in vec4 atr_clr;
in vec2 atr_tx0;
in vec2 atr_tx1;
in vec2 atr_tx2;

out vec4 frg_clr;
out vec2 frg_tx0;
out vec2 frg_tx1;
out vec2 frg_tx2;
out vec3 frg_nrm;
out vec3 frg_prm;

uniform mat4 unf_vtx_lcl_glb;
uniform mat4 unf_vtx_glb_cmr;
uniform mat4 unf_vtx_cmr_prj;
uniform vec4 unf_vtx_txt_0;
uniform vec4 unf_vtx_txt_1;
uniform vec4 unf_vtx_txt_2;
uniform vec4 unf_vtx_txt_3;
uniform vec4 unf_vtx_txt_4;
uniform vec4 unf_vtx_txt_5;
uniform vec4 unf_vtx_clr;

vec4 studio_atr_pos;
vec4 studio_atr_clr;
vec3 studio_atr_nrm;
vec3 studio_atr_tng;
vec4 studio_atr_tx0;
vec4 studio_atr_tx1;
vec4 studio_atr_tx2;
vec4 studio_atr_pr0;
vec4 studio_atr_pr1;
vec4 studio_atr_pr2;
vec4 studio_unf_vtx_cmr_prj[4];
vec4 studio_unf_vtx_glb_cmr[3];
vec4 studio_unf_vtx_lcl_glb[3];
vec4 studio_unf_vtx_eye;
vec4 studio_unf_vtx_txt[6];
vec4 studio_unf_vtx_bone[72];
vec4 studio_unf_vtx_silhouette_0;
vec4 studio_unf_vtx_silhouette_1;
vec4 studio_unf_vtx_str;
vec4 var_pos;
vec4 var_clr;
vec4 var_nrm;
vec4 var_tng;
vec4 var_tx0;
vec4 var_tx1;
vec4 var_tx2;
vec4 var_prm;
vec4 var_gen;
"""

BLENDER_VERTEX_MAIN = """
void main() {
    studio_atr_pos = vec4(atr_pos, 1.0);
    studio_atr_clr = atr_clr;
    studio_atr_nrm = atr_nrm;
    studio_atr_tng = vec3(1.0, 0.0, 0.0);
    // The texture coordinates as the game gives them (studio_render/shaders.py NATIVE_UV_GLSL)
    studio_atr_tx0 = studio_file_uv(atr_tx0);
    studio_atr_tx1 = studio_file_uv(atr_tx1);
    studio_atr_tx2 = studio_file_uv(atr_tx2);

    // Every bone is the identity and the vertex follows bone 0 with its whole weight
    studio_atr_pr0 = vec4(1.0, 0.0, 0.0, 0.0);
    studio_atr_pr1 = vec4(0.0);
    studio_atr_pr2 = vec4(0.0);
    for (int i = 0; i < 24; i++) {
        studio_unf_vtx_bone[3 * i] = vec4(1.0, 0.0, 0.0, 0.0);
        studio_unf_vtx_bone[3 * i + 1] = vec4(0.0, 1.0, 0.0, 0.0);
        studio_unf_vtx_bone[3 * i + 2] = vec4(0.0, 0.0, 1.0, 0.0);
    }

    // The game gives the rows of the matrices
    for (int i = 0; i < 4; i++) {
        studio_unf_vtx_cmr_prj[i] = vec4(unf_vtx_cmr_prj[0][i], unf_vtx_cmr_prj[1][i], unf_vtx_cmr_prj[2][i], unf_vtx_cmr_prj[3][i]);
    }
    for (int i = 0; i < 3; i++) {
        studio_unf_vtx_glb_cmr[i] = vec4(unf_vtx_glb_cmr[0][i], unf_vtx_glb_cmr[1][i], unf_vtx_glb_cmr[2][i], unf_vtx_glb_cmr[3][i]);
        studio_unf_vtx_lcl_glb[i] = vec4(unf_vtx_lcl_glb[0][i], unf_vtx_lcl_glb[1][i], unf_vtx_lcl_glb[2][i], unf_vtx_lcl_glb[3][i]);
    }
    studio_unf_vtx_txt[0] = studio_native_row_0(unf_vtx_txt_1);
    studio_unf_vtx_txt[1] = studio_native_row_1(unf_vtx_txt_0);
    studio_unf_vtx_txt[2] = studio_native_row_0(unf_vtx_txt_3);
    studio_unf_vtx_txt[3] = studio_native_row_1(unf_vtx_txt_2);
    studio_unf_vtx_txt[4] = studio_native_row_0(unf_vtx_txt_5);
    studio_unf_vtx_txt[5] = studio_native_row_1(unf_vtx_txt_4);
    studio_unf_vtx_eye = vec4(0.0);
    studio_unf_vtx_str = vec4(0.0);

    // The outline is drawn by its own pass, here it is off
    studio_unf_vtx_silhouette_0 = vec4(0.0, 0.01, 100000.0, 0.0);
    studio_unf_vtx_silhouette_1 = vec4(0.0);

    var_pos = vec4(0.0);
    var_clr = vec4(0.5);
    var_nrm = vec4(0.0, 0.0, 1.0, 0.0);
    var_tng = vec4(0.0);
    var_tx0 = vec4(0.0);
    var_tx1 = vec4(0.0);
    var_tx2 = vec4(0.0);
    var_prm = vec4(0.0);
    var_gen = vec4(0.0);

    studio_main();

    frg_clr = var_clr;
    frg_tx0 = studio_upright_uv(var_tx0.xy);
    frg_tx1 = studio_upright_uv(var_tx1.xy);
    frg_tx2 = studio_upright_uv(var_tx2.xy);
    frg_nrm = normalize(var_nrm.xyz);
    frg_prm = var_prm.xyz;
}
"""

def make_blender_vertex(tree):
    """The VERTEX_SHADER section as a vertex stage of StudioRender, with its interface."""
    # Imported here: the studio_render package imports the project, which imports this module
    from .studio_render import shaders

    renames = {"main": "studio_main"}

    for name in ATTRIBUTES:
        renames[name] = f"studio_{name}"

    for name in UNIFORMS:
        renames[name] = f"studio_{name}"

    # StudioRender sets unf_vtx_clr itself, with the same type as the game
    renames["unf_vtx_clr"] = "unf_vtx_clr"

    lines = [BLENDER_VERTEX_HEADER, shaders.NATIVE_UV_GLSL]

    for declaration in tree["declarations"]:
        qualifiers = declaration["qualifiers"]
        if "attribute" in qualifiers or "uniform" in qualifiers or "varying" in qualifiers or "in" in qualifiers or "out" in qualifiers:
            continue
        lines.append(shader_code.write_declaration(declaration, renames))

    for name, function in tree["functions"].items():
        lines += shader_code.write_function(function, renames)
        lines.append("")

    lines.append(BLENDER_VERTEX_MAIN)

    return "\n".join(lines)
