#!/usr/bin/python3
"""Транслятор Asm в машинный код.
"""

import sys

from isa import Opcode, Term, write_code

def get_meaningful_token(line):
    return line.split(";", 1)[0].strip()


def translate_stage_1(text):
    opcodes_with_operand = [Opcode.JMP, Opcode.DEC, Opcode.INC, Opcode.PRINT_CHAR]
    opcodes_with_two_operands = [Opcode.JZ, Opcode.JNZ, Opcode.JS, Opcode.JNS, Opcode.ADD_STR, Opcode.STORE]
    opcodes_with_three_operands = [Opcode.MOV, Opcode.MOD, Opcode.MUL, Opcode.SUB, Opcode.ADD]
    opcodes_with_operands = opcodes_with_operand + opcodes_with_two_operands + opcodes_with_three_operands

    code = []
    labels = {}
    strings = {}
    data = []

    last_label = None

    for line_num, raw_line in enumerate(text.splitlines(), 1):
        token = get_meaningful_token(raw_line)
        if token == "":
            continue

        pc = len(code)

        if token.endswith(":"):
            label = token.strip(":")
            assert label not in labels, "Redefinition of label: {}".format(label)
            labels[label] = pc
            last_label = label

        elif " " in token:
            sub_tokens = token.split(" ")
            mnemonic, arg = sub_tokens
            arg = arg.split(",")
            opcode = Opcode(mnemonic)

            assert opcode in opcodes_with_operands, f"Instruction ({opcode}) doesn't take an argument"

            code.append({"index": pc, "opcode": opcode, "arg": arg, "term": Term(line_num, 0, token)})

            if opcode.value == Opcode.ADD_STR:
                assert last_label is not None, "There is no label before add_str"
                strings[last_label] = len(data)

                data.append(int(arg[0]))
                for let in arg[1][1:-1]:
                    data.append(ord(let))
            else:
                code.append({"index": pc, "opcode": opcode, "arg": arg, "term": Term(line_num, 0, token)})
        else:
            opcode = Opcode(token)
            code.append({"index": pc, "opcode": opcode, "term": Term(line_num, 0, token)})

    return labels, strings, code, data


def translate_stage_2(labels, strings, code):

    for instruction in code:
        if "arg" in instruction and instruction["opcode"].value in {
            Opcode.JZ.value,
            Opcode.JNZ.value,
            Opcode.JS.value,
            Opcode.JNS.value,
            Opcode.JMP.value
        }:
            label = instruction["arg"]
            if label[0].isdigit() or label[0][0] == "r" and label[0][1].isdigit():
                continue
            assert label in labels, "Label not defined: " + label
            instruction["arg"] = labels[label]
    return code


def translate(text):

    labels, strings, code, data  = translate_stage_1(text)
    code = translate_stage_2(labels, strings, code)

    # ruff: noqa: RET504
    return code


def main(source, target):
    """Функция запуска транслятора. Параметры -- исходный и целевой файлы."""
    with open(source, encoding="utf-8") as f:
        source = f.read()

    code = translate(source)

    write_code(target, code)
    print("source LoC:", len(source.split("\n")), "code instr:", len(code))


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Wrong arguments: translator_asm.py <input_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
