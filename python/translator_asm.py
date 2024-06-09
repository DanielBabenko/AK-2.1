#!/usr/bin/python3
"""Транслятор Asm в машинный код.
"""

import sys

from isa import Opcode, Term, write_code


def get_meaningful_token(line):
    """Извлекаем из строки содержательный токен (метка или инструкция), удаляем
    комментарии и пробелы в начале/конце строки.
    """
    return line.split(";", 1)[0].strip()


def translate_stage_1(text):
    """Первый проход транслятора. Преобразование текста программы в список
    инструкций и определение адресов меток.

    Особенность: транслятор ожидает, что в строке может быть либо 1 метка,
    либо 1 инструкция. Поэтому: `col` заполняется всегда 0, так как не несёт
    смысловой нагрузки.
    """
    opcodes_with_operand = [Opcode.JMP, Opcode.DEC, Opcode.INC, Opcode.PRINT_CHAR, Opcode.CALL]
    opcodes_with_two_operands = [Opcode.JZ, Opcode.JNZ, Opcode.JS, Opcode.JNS, Opcode.ADD_STR, Opcode.STORE]
    opcodes_with_three_operands = [Opcode.MOV, Opcode.MOD, Opcode.MUL, Opcode.SUB, Opcode.ADD]
    opcodes_with_operands = opcodes_with_operand + opcodes_with_two_operands + opcodes_with_three_operands

    code = []
    labels = {}

    for line_num, raw_line in enumerate(text.splitlines(), 1):
        token = get_meaningful_token(raw_line)
        if token == "":
            continue

        pc = len(code)

        if token.endswith(":"):
            label = token.strip(":")
            assert label not in labels, "Redefinition of label: {}".format(label)
            labels[label] = pc
        elif " " in token:
            sub_tokens = token.split(" ")
            assert len(sub_tokens) == 2, "Invalid instruction: {}".format(token)
            mnemonic, arg = sub_tokens
            opcode = Opcode(mnemonic)
            assert opcode == Opcode.JZ or opcode == Opcode.JMP, "Only `jz` and `jnz` instructions take an argument"
            code.append({"index": pc, "opcode": opcode, "arg": arg, "term": Term(line_num, 0, token)})
        else:
            opcode = Opcode(token)
            code.append({"index": pc, "opcode": opcode, "term": Term(line_num, 0, token)})

    return labels, code


def translate_stage_2(labels, code):

    for instruction in code:
        if "arg" in instruction:
            label = instruction["arg"]
            assert label in labels, "Label not defined: " + label
            instruction["arg"] = labels[label]
    return code


def translate(text):

    labels, code = translate_stage_1(text)
    code = translate_stage_2(labels, code)

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
