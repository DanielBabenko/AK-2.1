import json
from collections import namedtuple
from enum import Enum


class Opcode(str, Enum):
    """Opcode для инструкций."""

    INC = "inc"
    DEC = "dec"
    INPUT = "input"
    PRINT = "print"

    JMP = "jmp"
    JZ = "jz"
    JNZ = "jnz"
    JS = "js"
    JNS = "jns"

    HALT = "halt"

    MOV = "mov"

    ADD = "add"
    SUB = "sub"
    MOD = "mod"
    MUL = "mul"

    ADD_STR = "add_str"

    STORE = "store"

    RIGHT = "right"
    LEFT = "left"

    PRINT_CHAR = "print_char"

    CALL = "call"
    RET = "ret"

    WORD = "word"

    def __str__(self):
        return str(self.value)


class Term(namedtuple("Term", "line pos symbol")):
    """Описание выражения из исходного текста программы.

    Сделано через класс, чтобы был docstring.
    """

def write_code(filename, code):
    with open(filename, "w", encoding="utf-8") as file:
        buf = []
        for instr in code:
            buf.append(json.dumps(instr))
        file.write("[" + ",\n ".join(buf) + "]")


def read_code(filename: str) -> list:

    with open(filename, encoding="utf-8") as file:
        code = json.loads(file.read())

    for instr in code:
        if isinstance(instr, dict):
            instr["opcode"] = Opcode(instr["opcode"])

            if "term" in instr:
                assert len(instr["term"]) == 3
                instr["term"] = Term(instr["term"][0], instr["term"][1], instr["term"][2])

    return code
