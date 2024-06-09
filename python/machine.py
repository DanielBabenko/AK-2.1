from __future__ import annotations

import logging
import sys
from typing import ClassVar

from isa import Opcode, read_code

class ALU:
    """Арифметико-логическое устройство"""

    operations: ClassVar[dict] = {
        Opcode.INC: lambda left, right: left + 1,
        Opcode.DEC: lambda left, right: left - 1,
        Opcode.MOV: lambda left, right: left,
        Opcode.ADD: lambda left, right: left + right,
        Opcode.SUB: lambda left, right: left - right,
        Opcode.MOD: lambda left, right: left % right,
        Opcode.MUL: lambda left, right: left * right,
    }
    zero = None
    sign = None
    res = None

    def calc(self, op: Opcode, left: int, right: int) -> int:
        assert op in self.operations, f"Unknown operation: {op}"
        self.res = self.operations.get(op)(left, right)
        self.zero = (self.res == 0)
        self.sign = (self.res < 0)
        return self.res

class DataPath:

    data_memory_size = None
    data_memory = None

    data_address = None
    acc = None

    input_buffer = None
    output_buffer = None

    def __init__(self, alu: ALU, data_memory_size, input_buffer):

        self.registers = {
            "r1": 0,
            "r2": 0,
            "r3": 0,
            "r4": 0,
            "rc": 0,  # "r5": 0,
            "rs": 0,  # "r6": 0,
            "r7": 0,
        }

        assert data_memory_size > 0, "Data_memory size should be non-zero"
        self.data_memory_size = data_memory_size
        self.data_memory = [0] * data_memory_size
        self.data_address = 0
        self.acc = 0
        self.alu = alu
        self.input_buffer = input_buffer
        self.output_buffer = []

    def signal_latch_data_addr(self, sel):

        assert sel in {Opcode.LEFT.value, Opcode.RIGHT.value}, "internal error, incorrect selector: {}".format(sel)

        if sel == Opcode.LEFT.value:
            self.data_address -= 1
        elif sel == Opcode.RIGHT.value:
            self.data_address += 1

        assert 0 <= self.data_address < self.data_memory_size, "out of memory: {}".format(self.data_address)

    def signal_latch_acc(self):
        self.acc = self.data_memory[self.data_address]

    def signal_alu_l(self, sel):
        if sel == "imml":
            self.alu_l_value = self.imml
        else:
            self.alu_l_value = self.input_register if sel == "ir" else self.registers.get(sel)

    def signal_alu_r(self, sel):
        if sel == "0":
            self.alu_r_value = 0
        elif sel == "immr":
            self.alu_r_value = self.immr
        else:
            self.alu_r_value = self.input_register if sel == "ir" else self.registers.get(sel)

    def signal_alu_op(self, sel):
        res = self.alu.calc(sel, self.alu_l_value, self.alu_r_value)
        assert res is not None, "unknown instruction"
        return res

    def signal_wr(self, sel):

        assert sel in {
            Opcode.INC.value,
            Opcode.DEC.value,
            Opcode.INPUT.value,
        }, "internal error, incorrect selector: {}".format(sel)

        if sel == Opcode.INC.value:
            self.data_memory[self.data_address] = self.acc + 1
            if self.data_memory[self.data_address] == 128:
                self.data_memory[self.data_address] = -128
        elif sel == Opcode.DEC.value:
            self.data_memory[self.data_address] = self.acc - 1
            if self.data_memory[self.data_address] == -129:
                self.data_memory[self.data_address] = 127
        elif sel == Opcode.INPUT.value:
            if len(self.input_buffer) == 0:
                raise EOFError()
            symbol = self.input_buffer.pop(0)
            symbol_code = ord(symbol)
            assert -128 <= symbol_code <= 127, "input token is out of bound: {}".format(symbol_code)
            self.data_memory[self.data_address] = symbol_code
            logging.debug("input: %s", repr(symbol))

    def signal_output(self):
        symbol = chr(self.acc)
        logging.debug("output: %s << %s", repr("".join(self.output_buffer)), repr(symbol))
        self.output_buffer.append(symbol)

    def zero(self):
        return self.acc == 0

    def sign(self):
        return self.acc < 0


class ControlUnit:

    program = None
    program_counter = None
    data_path = None
    _tick = None

    def __init__(self, program, data_path):
        self.program = program
        self.program_counter = 0
        self.data_path = data_path
        self._tick = 0

    def tick(self):
        self._tick += 1

    def current_tick(self):
        return self._tick

    def signal_latch_program_counter(self, sel_next):
        if sel_next:
            self.program_counter += 1
        else:
            instr = self.program[self.program_counter]
            assert "arg" in instr, "internal error"
            self.program_counter = instr["arg"]

    def decode_and_execute_control_flow_instruction(self, instr, opcode):

        if opcode is Opcode.HALT:
            raise StopIteration()

        if opcode is Opcode.JMP:
            addr = instr["arg"]
            self.program_counter = addr
            self.tick()

            return True

        if opcode is Opcode.JZ:
            addr = instr["arg"]

            self.data_path.signal_latch_acc()
            self.tick()

            if self.data_path.zero():
                self.signal_latch_program_counter(sel_next=False)
            else:
                self.signal_latch_program_counter(sel_next=True)
            self.tick()

            return True

        return False

    def decode_and_execute_instruction(self):
        instr = self.program[self.program_counter]
        opcode = instr["opcode"]

        if self.decode_and_execute_control_flow_instruction(instr, opcode):
            return

        if opcode in {Opcode.RIGHT, Opcode.LEFT}:
            self.data_path.signal_latch_data_addr(opcode.value)
            self.signal_latch_program_counter(sel_next=True)
            self.tick()

        elif opcode in {Opcode.INC, Opcode.DEC, Opcode.INPUT}:
            self.data_path.signal_latch_acc()
            self.tick()

            self.data_path.signal_wr(opcode.value)
            self.signal_latch_program_counter(sel_next=True)
            self.tick()

        elif opcode is Opcode.PRINT:
            self.data_path.signal_latch_acc()
            self.tick()

            self.data_path.signal_output()
            self.signal_latch_program_counter(sel_next=True)
            self.tick()

        if opcode in {Opcode.MOD, Opcode.MUL, Opcode.ADD, Opcode.SUB}:
            return self.execute_binary_operation(instr, opcode)

    def execute_binary_operation(self, instr, opcode):
        args: list[str]
        args = instr["arg"]
        a, b, c = args
        assert a in self.data_path.registers, "unknown register"

        if b.isdigit():
            self.data_path.imml = int(b)
            self.data_path.signal_alu_left("imml")
        else:
            self.data_path.signal_alu_left(b)

        if c.isdigit():
            self.data_path.immr = int(c)
            self.data_path.signal_alu_right("immr")
        else:
            self.data_path.signal_alu_right(c)

        self.data_path.signal_alu_op(instr["opcode"])

        value = self.data_path.alu.res

        self.data_path.signal_latch_r(a, value)

        self.data_path.signal_latch_program_counter(sel_next=True)
        self.tick()
        return True

    def __repr__(self):
        """Вернуть строковое представление состояния процессора."""
        state_repr = "TICK: {:3} PC: {:3} ADDR: {:3} MEM_OUT: {} ACC: {}".format(
            self._tick,
            self.program_counter,
            self.data_path.data_address,
            self.data_path.data_memory[self.data_path.data_address],
            self.data_path.acc,
        )

        instr = self.program[self.program_counter]
        opcode = instr["opcode"]
        instr_repr = str(opcode)

        if "arg" in instr:
            instr_repr += " {}".format(instr["arg"])

        if "term" in instr:
            term = instr["term"]
            instr_repr += "  ('{}'@{}:{})".format(term.symbol, term.line, term.pos)

        return "{} \t{}".format(state_repr, instr_repr)


def simulation(code, input_tokens, data_memory_size, limit):

    data_path = DataPath(data_memory_size, input_tokens)
    control_unit = ControlUnit(code, data_path)
    instr_counter = 0

    logging.debug("%s", control_unit)
    try:
        while instr_counter < limit:
            control_unit.decode_and_execute_instruction()
            instr_counter += 1
            logging.debug("%s", control_unit)
    except EOFError:
        logging.warning("Input buffer is empty!")
    except StopIteration:
        pass

    if instr_counter >= limit:
        logging.warning("Limit exceeded!")
    logging.info("output_buffer: %s", repr("".join(data_path.output_buffer)))
    return "".join(data_path.output_buffer), instr_counter, control_unit.current_tick()


def main(code_file, input_file):
    """Функция запуска модели процессора. Параметры -- имена файлов с машинным
    кодом и с входными данными для симуляции.
    """
    code = read_code(code_file)
    with open(input_file, encoding="utf-8") as file:
        input_text = file.read()
        input_token = []
        for char in input_text:
            input_token.append(char)

    output, instr_counter, ticks = simulation(
        code,
        input_tokens=input_token,
        data_memory_size=100,
        limit=1000,
    )

    print("".join(output))
    print("instr_counter: ", instr_counter, "ticks:", ticks)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Wrong arguments: machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)
