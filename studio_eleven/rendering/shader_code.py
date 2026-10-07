"""The code of a custom shader: a GLSL file with a vertex and a fragment section, parsed into a tree the compilers read.

    #pragma vertex_program "#FIX_SIL_VTX_MDS"

    #ifdef FRAGMENT_SHADER
    void main() {
        vec4 tex0 = texture2D(unf_frg_txt_2d_0, var_tx0.xy);
        gl_FragColor = vec4(tex0.rgb * var_clr.rgb * 2.0, tex0.a);
    }
    #endif
"""
import re
import bisect

##########################################
# CONST
##########################################

SECTIONS = ("VERTEX_SHADER", "FRAGMENT_SHADER")

PRAGMAS = ("vertex_program", "combiner")

TYPES = (
    "void", "bool", "int", "float",
    "vec2", "vec3", "vec4",
    "ivec2", "ivec3", "ivec4",
    "bvec2", "bvec3", "bvec4",
    "mat2", "mat3", "mat4",
    "sampler2D", "samplerCube",
)

QUALIFIERS = ("attribute", "uniform", "varying", "const", "in", "out", "inout", "highp", "mediump", "lowp", "flat", "smooth")

PRECISIONS = ("highp", "mediump", "lowp")

ASSIGNMENTS = ("=", "+=", "-=", "*=", "/=")

# Binary operators by precedence, the lowest first
BINARY_LEVELS = (
    ("||",),
    ("&&",),
    ("==", "!="),
    ("<", ">", "<=", ">="),
    ("+", "-"),
    ("*", "/"),
)

TOKEN_PATTERN = re.compile(
    r"\s*(?:(//[^\n]*|/\*.*?\*/)"
    r"|((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fF]?)"
    r"|([A-Za-z_][A-Za-z_0-9]*)"
    r"|(\+\+|--|\+=|-=|\*=|/=|==|!=|<=|>=|&&|\|\||.))",
    re.S,
)

##########################################
# Shader Code Function
##########################################

class CodeError(Exception):
    def __init__(self, message, line=0):
        if line:
            message = f"Line {line}: {message}"
        super().__init__(message)
        self.line = line

def read_pragma(line, number):
    match = re.match(r'\s*#\s*pragma\s+(\w+)\s+"([^"]*)"\s*$', line)
    if match is None or match.group(1) not in PRAGMAS:
        raise CodeError("#pragma vertex_program \"#NAME\" or #pragma combiner \"#NAME\" expected", number)

    return {"key": match.group(1), "value": match.group(2)}

def split_sections(text):
    """Cuts the file in its sections, a line outside a section is kept empty so the line numbers stay those of the file."""
    lines = text.split("\n")
    sections = {}
    pragmas = {}
    current = None
    in_comment = False

    for name in SECTIONS:
        sections[name] = None

    for index, line in enumerate(lines):
        number = index + 1
        stripped = line.strip()

        # A line inside a block comment is never a directive
        if in_comment:
            if "*/" in stripped:
                in_comment = False
            if current is not None:
                sections[current][index] = line
            continue

        if stripped.startswith("/*") and "*/" not in stripped:
            in_comment = True
            if current is not None:
                sections[current][index] = line
            continue

        directive = re.match(r"#\s*(\w+)\s*(\w*)", stripped)

        if directive is not None and directive.group(1) == "ifdef":
            if current is not None:
                raise CodeError("#ifdef inside a section, close it with #endif first", number)
            if directive.group(2) not in SECTIONS:
                raise CodeError("#ifdef VERTEX_SHADER or #ifdef FRAGMENT_SHADER expected", number)
            if sections[directive.group(2)] is not None:
                raise CodeError(f"The {directive.group(2)} section is written twice", number)

            current = directive.group(2)
            sections[current] = [""] * len(lines)
            continue

        if directive is not None and directive.group(1) == "endif":
            if current is None:
                raise CodeError("#endif without #ifdef", number)
            current = None
            continue

        if directive is not None and directive.group(1) == "pragma":
            pragma = read_pragma(stripped, number)
            pragmas[pragma["key"]] = pragma["value"]
            continue

        if current is not None:
            sections[current][index] = line
            continue

        if stripped and not stripped.startswith("//"):
            raise CodeError("Code outside of #ifdef VERTEX_SHADER / #ifdef FRAGMENT_SHADER", number)

    if current is not None:
        raise CodeError(f"The {current} section has no #endif", len(lines))

    result = {
        "pragmas": pragmas,
    }

    for name in SECTIONS:
        if sections[name] is None:
            result[name] = None
        else:
            result[name] = "\n".join(sections[name])

    return result

def preprocess(text):
    """The directives of a section: #define is a token replacement, #version / #extension / #line are dropped."""
    lines = text.split("\n")
    defines = {}

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        define = re.match(r"#\s*define\s+([A-Za-z_]\w*)\s*(.*)$", stripped)
        if define is not None:
            defines[define.group(1)] = define.group(2).strip()
        elif not re.match(r"#\s*(version|extension|line)\b", stripped):
            raise CodeError(f"Unknown directive {stripped.split()[0]}", index + 1)

        lines[index] = ""

    return {"text": "\n".join(lines), "defines": defines}

def get_line_starts(text):
    starts = [0]

    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)

    return starts

def tokenize(text):
    preprocessed = preprocess(text)
    text = preprocessed["text"]
    defines = preprocessed["defines"]
    line_starts = get_line_starts(text)
    tokens = []
    position = 0

    while position < len(text) and text[position:].strip():
        match = TOKEN_PATTERN.match(text, position)
        if match is None or match.end() == position:
            break

        position = match.end()
        line = bisect.bisect_right(line_starts, match.start(match.lastindex))

        comment = match.group(1)
        number = match.group(2)
        name = match.group(3)
        symbol = match.group(4)

        if comment is not None:
            continue

        if number is not None:
            is_int = "." not in number and "e" not in number.lower()
            tokens.append({"kind": "number", "text": number, "value": float(number.rstrip("fF")), "int": is_int, "line": line})
        elif name is not None and name in defines and defines[name]:
            for token in tokenize(defines[name]):
                if token["kind"] != "end":
                    token["line"] = line
                    tokens.append(token)
        elif name is not None:
            tokens.append({"kind": "name", "text": name, "line": line})
        else:
            tokens.append({"kind": "symbol", "text": symbol, "line": line})

    tokens.append({"kind": "end", "text": "", "line": len(line_starts)})

    return tokens

##########################################
# Parser
##########################################

class Parser:
    """Recursive descent over the tokens: declarations, functions, statements and expressions as dicts with their line."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def peek(self, offset=0):
        return self.tokens[min(self.position + offset, len(self.tokens) - 1)]

    def next(self):
        token = self.peek()
        self.position += 1
        return token

    def at(self, text, offset=0):
        token = self.peek(offset)
        return token["kind"] != "number" and token["text"] == text

    def accept(self, text):
        if self.at(text):
            return self.next()
        return None

    def expect(self, text):
        token = self.peek()
        if not self.at(text):
            found = token["text"]
            if token["kind"] == "end":
                found = "the end"
            raise CodeError(f"'{text}' expected, found '{found}'", token["line"])
        return self.next()

    def expect_name(self):
        token = self.peek()
        if token["kind"] != "name":
            raise CodeError(f"A name expected, found '{token['text']}'", token["line"])
        return self.next()

    def at_type(self, offset=0):
        token = self.peek(offset)
        return token["kind"] == "name" and token["text"] in TYPES

    def at_declaration(self):
        offset = 0
        while self.peek(offset)["kind"] == "name" and self.peek(offset)["text"] in QUALIFIERS:
            offset += 1
        return self.at_type(offset) and self.peek(offset + 1)["kind"] == "name"

    def parse(self):
        declarations = []
        functions = {}

        while self.peek()["kind"] != "end":
            if self.accept(";"):
                continue

            # precision mediump float;
            if self.at("precision"):
                self.next()
                self.next()
                self.next()
                self.expect(";")
                continue

            line = self.peek()["line"]
            qualifiers = self.read_qualifiers()

            if not self.at_type():
                token = self.peek()
                raise CodeError(f"A declaration or a function expected, found '{token['text']}'", token["line"])

            type_name = self.next()["text"]
            name = self.expect_name()["text"]

            if self.at("("):
                function = self.read_function(type_name, name, line)
                if function["body"] is not None:
                    functions[name] = function
                continue

            for declaration in self.read_declarators(qualifiers, type_name, name, line):
                declarations.append(declaration)

        return {
            "declarations": declarations,
            "functions": functions,
        }

    def read_qualifiers(self):
        qualifiers = []

        while self.peek()["kind"] == "name" and self.peek()["text"] in QUALIFIERS:
            text = self.next()["text"]
            if text not in PRECISIONS:
                qualifiers.append(text)

        return qualifiers

    def read_array_size(self):
        if not self.accept("["):
            return None

        token = self.next()
        if token["kind"] != "number" or not token["int"]:
            raise CodeError("The size of an array must be a whole number", token["line"])

        self.expect("]")

        return int(token["value"])

    def read_declarators(self, qualifiers, type_name, name, line):
        declarations = []

        while True:
            size = self.read_array_size()
            value = None

            if self.accept("="):
                value = self.read_assignment()

            declarations.append({
                "kind": "declare",
                "qualifiers": qualifiers,
                "type": type_name,
                "name": name,
                "size": size,
                "value": value,
                "line": line,
            })

            if not self.accept(","):
                break

            name = self.expect_name()["text"]

        self.expect(";")

        return declarations

    def read_function(self, return_type, name, line):
        self.expect("(")
        parameters = []

        if self.at("void") and self.at(")", 1):
            self.next()

        while not self.at(")"):
            qualifiers = self.read_qualifiers()
            if not self.at_type():
                token = self.peek()
                raise CodeError(f"A parameter type expected, found '{token['text']}'", token["line"])

            parameter_type = self.next()["text"]
            parameter_name = self.expect_name()["text"]
            size = self.read_array_size()
            parameters.append({"type": parameter_type, "name": parameter_name, "qualifiers": qualifiers, "size": size})

            if not self.accept(","):
                break

        self.expect(")")

        body = None
        if not self.accept(";"):
            body = self.read_block()

        return {
            "name": name,
            "type": return_type,
            "parameters": parameters,
            "body": body,
            "line": line,
        }

    def read_block(self):
        self.expect("{")
        statements = []

        while not self.at("}"):
            if self.peek()["kind"] == "end":
                raise CodeError("'}' expected before the end", self.peek()["line"])
            for statement in self.read_statement():
                statements.append(statement)

        self.expect("}")

        return statements

    def read_statement(self):
        """A list, a declaration of several names gives several statements."""
        token = self.peek()
        line = token["line"]

        if self.at("{"):
            return [{"kind": "block", "body": self.read_block(), "line": line}]

        if self.accept(";"):
            return []

        if self.accept("if"):
            self.expect("(")
            condition = self.read_expression()
            self.expect(")")
            then_body = self.read_body()
            else_body = []
            if self.accept("else"):
                else_body = self.read_body()
            return [{"kind": "if", "condition": condition, "then": then_body, "else": else_body, "line": line}]

        if self.accept("for"):
            self.expect("(")
            initial = []
            if not self.accept(";"):
                initial = self.read_simple_statement()
            condition = None
            if not self.at(";"):
                condition = self.read_expression()
            self.expect(";")
            step = None
            if not self.at(")"):
                step = self.read_expression()
            self.expect(")")
            body = self.read_body()
            return [{"kind": "for", "initial": initial, "condition": condition, "step": step, "body": body, "line": line}]

        if self.accept("return"):
            value = None
            if not self.at(";"):
                value = self.read_expression()
            self.expect(";")
            return [{"kind": "return", "value": value, "line": line}]

        if self.accept("discard"):
            raise CodeError("discard is not possible, the alpha test of the material (.atr) does it", line)

        if self.at("while") or self.at("do") or self.at("break") or self.at("continue"):
            raise CodeError(f"'{token['text']}' is not supported, only for loops with a fixed count", line)

        return self.read_simple_statement()

    def read_body(self):
        if self.at("{"):
            return self.read_block()
        return self.read_statement()

    def read_simple_statement(self):
        line = self.peek()["line"]

        if self.at_declaration():
            qualifiers = self.read_qualifiers()
            type_name = self.next()["text"]
            name = self.expect_name()["text"]
            return self.read_declarators(qualifiers, type_name, name, line)

        expression = self.read_expression()
        self.expect(";")

        return [{"kind": "expression", "value": expression, "line": line}]

    def read_expression(self):
        return self.read_assignment()

    def read_assignment(self):
        target = self.read_ternary()
        token = self.peek()

        if token["kind"] == "symbol" and token["text"] in ASSIGNMENTS:
            self.next()
            value = self.read_assignment()
            return {"kind": "assign", "operator": token["text"], "target": target, "value": value, "line": token["line"]}

        return target

    def read_ternary(self):
        condition = self.read_binary(0)
        token = self.peek()

        if self.accept("?"):
            first = self.read_assignment()
            self.expect(":")
            second = self.read_assignment()
            return {"kind": "ternary", "condition": condition, "first": first, "second": second, "line": token["line"]}

        return condition

    def read_binary(self, level):
        if level == len(BINARY_LEVELS):
            return self.read_unary()

        left = self.read_binary(level + 1)

        while True:
            token = self.peek()
            if token["kind"] != "symbol" or token["text"] not in BINARY_LEVELS[level]:
                return left

            self.next()
            right = self.read_binary(level + 1)
            left = {"kind": "binary", "operator": token["text"], "left": left, "right": right, "line": token["line"]}

    def read_unary(self):
        token = self.peek()

        if token["kind"] == "symbol" and token["text"] in ("-", "+", "!"):
            self.next()
            operand = self.read_unary()
            if token["text"] == "+":
                return operand
            return {"kind": "unary", "operator": token["text"], "operand": operand, "line": token["line"]}

        if token["kind"] == "symbol" and token["text"] in ("++", "--"):
            self.next()
            operand = self.read_unary()
            return {"kind": "increment", "operator": token["text"], "target": operand, "line": token["line"]}

        return self.read_postfix()

    def read_postfix(self):
        expression = self.read_primary()

        while True:
            token = self.peek()

            if self.accept("."):
                field = self.expect_name()["text"]
                expression = {"kind": "swizzle", "operand": expression, "components": field, "line": token["line"]}
            elif self.accept("["):
                index = self.read_expression()
                self.expect("]")
                expression = {"kind": "index", "operand": expression, "index": index, "line": token["line"]}
            elif token["kind"] == "symbol" and token["text"] in ("++", "--"):
                self.next()
                expression = {"kind": "increment", "operator": token["text"], "target": expression, "line": token["line"]}
            else:
                return expression

    def read_primary(self):
        token = self.next()

        if token["kind"] == "number":
            return {"kind": "number", "value": token["value"], "int": token["int"], "line": token["line"]}

        if token["kind"] == "name" and token["text"] in ("true", "false"):
            value = 0.0
            if token["text"] == "true":
                value = 1.0
            return {"kind": "number", "value": value, "int": False, "line": token["line"]}

        if token["kind"] == "name" and self.accept("("):
            arguments = []

            while not self.at(")"):
                arguments.append(self.read_assignment())
                if not self.accept(","):
                    break

            self.expect(")")

            return {"kind": "call", "name": token["text"], "arguments": arguments, "line": token["line"]}

        if token["kind"] == "name":
            return {"kind": "name", "name": token["text"], "line": token["line"]}

        if token["text"] == "(":
            expression = self.read_expression()
            self.expect(")")
            return expression

        found = token["text"]
        if token["kind"] == "end":
            found = "the end"

        raise CodeError(f"An expression expected, found '{found}'", token["line"])

def parse_section(text):
    return Parser(tokenize(text)).parse()

def parse_shader(text):
    """The pragmas and the parsed sections of a shader file, a missing section is None."""
    sections = split_sections(text)
    result = {
        "pragmas": sections["pragmas"],
        "vertex": None,
        "fragment": None,
    }

    if sections["VERTEX_SHADER"] is not None:
        result["vertex"] = parse_section(sections["VERTEX_SHADER"])

    if sections["FRAGMENT_SHADER"] is not None:
        result["fragment"] = parse_section(sections["FRAGMENT_SHADER"])

    if result["vertex"] is not None and "vertex_program" in result["pragmas"]:
        raise CodeError("The shader has a VERTEX_SHADER section and a #pragma vertex_program, keep one")

    if result["fragment"] is not None and "combiner" in result["pragmas"]:
        raise CodeError("The shader has a FRAGMENT_SHADER section and a #pragma combiner, keep one")

    return result

##########################################
# Writer
##########################################

def format_number(expression):
    if expression["int"]:
        return str(int(expression["value"]))

    text = repr(float(expression["value"]))
    if "e" in text or "." in text or "inf" in text or "nan" in text:
        return text
    return text + ".0"

def write_expression(expression, renames):
    """The code of an expression, every operation in parentheses, names renamed by the dict."""
    kind = expression["kind"]

    if kind == "number":
        return format_number(expression)

    if kind == "name":
        return renames.get(expression["name"], expression["name"])

    if kind == "binary":
        return f"({write_expression(expression['left'], renames)} {expression['operator']} {write_expression(expression['right'], renames)})"

    if kind == "unary":
        return f"({expression['operator']}{write_expression(expression['operand'], renames)})"

    if kind == "ternary":
        condition = write_expression(expression["condition"], renames)
        first = write_expression(expression["first"], renames)
        second = write_expression(expression["second"], renames)
        return f"({condition} ? {first} : {second})"

    if kind == "swizzle":
        return f"{write_expression(expression['operand'], renames)}.{expression['components']}"

    if kind == "index":
        return f"{write_expression(expression['operand'], renames)}[{write_expression(expression['index'], renames)}]"

    if kind == "assign":
        return f"{write_expression(expression['target'], renames)} {expression['operator']} {write_expression(expression['value'], renames)}"

    if kind == "increment":
        return f"{write_expression(expression['target'], renames)}{expression['operator']}"

    arguments = ", ".join(write_expression(argument, renames) for argument in expression["arguments"])
    return f"{renames.get(expression['name'], expression['name'])}({arguments})"

def write_declaration(statement, renames):
    text = ""
    if "const" in statement["qualifiers"]:
        text += "const "

    text += f"{statement['type']} {renames.get(statement['name'], statement['name'])}"

    if statement["size"] is not None:
        text += f"[{statement['size']}]"

    if statement["value"] is not None:
        text += f" = {write_expression(statement['value'], renames)}"

    return text + ";"

def write_statements(statements, renames, indent):
    lines = []
    space = "    " * indent

    for statement in statements:
        kind = statement["kind"]

        if kind == "declare":
            lines.append(space + write_declaration(statement, renames))
        elif kind == "expression":
            lines.append(space + write_expression(statement["value"], renames) + ";")
        elif kind == "return":
            if statement["value"] is None:
                lines.append(space + "return;")
            else:
                lines.append(space + f"return {write_expression(statement['value'], renames)};")
        elif kind == "block":
            lines.append(space + "{")
            lines += write_statements(statement["body"], renames, indent + 1)
            lines.append(space + "}")
        elif kind == "if":
            lines.append(space + f"if ({write_expression(statement['condition'], renames)}) {{")
            lines += write_statements(statement["then"], renames, indent + 1)
            if statement["else"]:
                lines.append(space + "} else {")
                lines += write_statements(statement["else"], renames, indent + 1)
            lines.append(space + "}")
        elif kind == "for":
            initial = ""
            if statement["initial"]:
                initial = write_statements(statement["initial"], renames, 0)[0].rstrip(";")
            condition = ""
            if statement["condition"] is not None:
                condition = write_expression(statement["condition"], renames)
            step = ""
            if statement["step"] is not None:
                step = write_expression(statement["step"], renames)
            lines.append(space + f"for ({initial}; {condition}; {step}) {{")
            lines += write_statements(statement["body"], renames, indent + 1)
            lines.append(space + "}")

    return lines

def write_function(function, renames, name=None):
    if name is None:
        name = renames.get(function["name"], function["name"])

    parameters = []
    for parameter in function["parameters"]:
        text = f"{parameter['type']} {parameter['name']}"
        if parameter["size"] is not None:
            text += f"[{parameter['size']}]"
        parameters.append(text)

    lines = [f"{function['type']} {name}({', '.join(parameters)}) {{"]
    lines += write_statements(function["body"], renames, 1)
    lines.append("}")

    return lines
