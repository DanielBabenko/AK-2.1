# Архитектура компьютера. Лабораторная работа №3

- Автор: Бабенко Даниил Александрович, P3234
- Вариант: Упрощённый
   - `asm | risc | harw | hw | instr | struct | stream | port | pstr | prob2`

## Язык программирования

Синтаксис в расширенной БНФ.

- `[ ... ]` -- вхождение 0 или 1 раз
- `{ ... }` -- повторение 0 или несколько раз
- `{ ... }-` -- повторение 1 или несколько раз

``` ebnf
program ::= { line }

line ::= label [ comment ] "\n"
       | instr [ comment ] "\n"
       | [ comment ] "\n"

label ::= label_name ":"

instr ::= op0
        | op1 arg
        | op2 arg,arg
        | op3 arg,arg,arg

op0 ::= "input"
      | "print"
      | "halt"
      | "left"
      | "right"

op1 ::= "inc"
      | "dec"
      | "jmp"
      | "print_char"
      
op2 ::= "jz"
      | "jnz"
      | "mov"
      | "load"
      | "store"
      | "add_str"
      
op3 ::= "add"
      | "sub"
      | "mod"
      | "mul"

arg ::= label_name | integer | register

register ::= "r" digit

integer ::= [ "-" ] { digit }-

digit ::= <any of "0-9">

label_name ::= <any of "a-z A-Z _"> { <any of "a-z A-Z 0-9 _"> }

comment ::= ";" <any symbols except "\n">
```

Поддерживаются однострочные комментарии, начинающиеся с `;`.

Команды:

- `inc r` -- увеличить значение в регистре r на 1.
- `dec r` -- уменьшить значение в регистре r на 1.
- `right` -- перейти к следующей ячейке.
- `left` -- перейти к предыдущей ячейке.
- `input` -- ввести извне значение и сохранить в текущей ячейке памяти данных.
- `print` -- напечатать значение из текущей ячейки памяти данных.

- `jmp addr` -- безусловный переход по заданному адресу или метке.
- `jz addr,r` -- условный переход по заданному адресу или метке, если значение регистра r ноль.
- `jnz addr,r` -- условный переход по заданному адресу или метке, если значение регистра r не ноль.
- `js addr,r` -- условный переход по заданному адресу или метке, если значение регистра r отрицательное.
- `jns addr,r` -- условный переход по заданному адресу или метке, если значение регистра r неотрицательное.
- `halt` -- остановка выполнения программы.

- `mov r, a` -- записать в регистр r значение.
- `add r, a, b` -- посчитать значение a + b и записать его в регистр r.
- `sub r, a, b` -- посчитать значение a - b и записать его в регистр r.
- `mul r, a, b` -- посчитать значение a * b и записать его в регистр r.
- `mov r, a, b` -- посчитать значение a % b и записать его в регистр r.
  - а и b в данных контекстах могут являться целыми числами и/или регистрами

- `load p, r` -- загрузить значение из ячейки памяти данных по адресу p в регистр r.
- `store r, p` -- записать значение из регистра r в ячейку памяти данных по адресу p.

- `add_str l,s` -- записать в память строку s длины l
- `print_char c` -- вывести символ из ячейки памяти данных c

Метки для переходов определяются на отдельных строчках:

``` asm
label:
    inc r1
```

И в другом месте (неважно, до или после определения) сослаться на эту метку:

``` asm
jmp label   ; --> `jmp 123`, где 123 - номер инструкции после объявления метки
```

Транслятор поставит на место использования метки адрес той инструкции, перед которой она определена.

В программе не может быть дублирующихся меток, определенных в разных местах с одним именем.


## Организация памяти

Память соответствует Гарвардской архитектуре: присутствуют отдельно память команд и память данных.

```text
      command memory
+------------------------+
| 001 : command1         |
| 002 : command2         |
| 003 : command3         |
|       ...              |
|  n  : command2         |
| n+1 : command3         |
| n+2 : hlt              | 
+------------------------+
```
- Команды в памяти располагаются последовательно.
- Последовательность команд всегда заканчивается командой `halt`.

```text
  data   memory
+----------------+
|   REGISTERS    |
+----------------+
|     DATA       | 
+----------------+
```

- Память данных делится на несколько "отсеков":
  1) ``Регистры``.
  2)  ``Data`` - непосредственно адресное пространство данных.
- Изнутри регистры представляют собой 16 ячеек памяти, отстоящих отдельно от основной памяти данных для более быстрого обращения к ним.
- Регистры ``r0 - r15`` являются такими же ячейками памяти, куда данные можно сохранять и работать с ними.
- Все регистры сохраняют состояние.

Поддерживаются 2 вида адресации:
1) Непосредственная загрузка данных в память;
2) Адресация в регистры (``r0 - r15, (r5 = rc, r6 = rs)``)

## Система команд

Особенности процессора:

- Машинное слово -- 32 бит, знаковое.
- Доступ к памяти данных осуществляется по адресу, хранящемуся в специальном регистре `data_address`.
- Установка адреса осуществляется путём инкрементирования или декрементирования инструкциями `left` и `right`.
- Обработка данных осуществляется по текущему адресу, а также через ввод/вывод.
- Поток управления:
  - После каждой инструкции `PC (program counter)` инкрементируется;
  - условные (`jz`, `js`, `jnz`, `jns`) и безусловный (`jmp`) переходы.

### Набор инструкций

Команды языка однозначно транслируюстя в инструкции.
Команда add_str выполняется на этапе трансляции.

| Инструкция | Кол-во тактов |
|:-----------|---------------|
| inc        | 1             |
| dec        | 1             |
| right      | 1             |
| left       | 1             |
| input      | 2             |
| print      | 2             |
| jmp        | 1             |
| jz         | 2             |
| jnz        | 2             |
| js         | 2             |
| jns        | 2             |
| halt       | 1             |
| mov        | 1             |
| add        | 1             |
| sub        | 1             |
| mod        | 1             |
| mul        | 1             |
| add_str    | 0             |
| print_char | 1             |

### Кодирование инструкций

- Машинный код сериализуется в список JSON.
- Один элемент списка -- одна инструкция.
- Индекс списка -- адрес инструкции. Используется для команд перехода.

Пример:

```json
[
  {
    "index": 2, 
    "opcode": "mov", 
    "arg": [
      "rs", 
      "r2"
    ], 
    "term": [
      7, 
      0, 
      "mov rs,r2"
    ]
  }
]
```

где:

- `index` -- адрес команды в памяти команд
- `opcode` -- строка с кодом операции;
- `arg` -- аргумент (может отсутствовать);
- `term` -- информация о связанном месте в исходном коде (если есть).

Типы данных в модуле isa, где:

- `Opcode` -- перечисление кодов операций;
- `Term` -- структура для описания значимого фрагмента кода исходной программы.

## Транслятор

Интерфейс командной строки: `translator.py <input_file> <target_file>`


Трансляция реализуется в два прохода:

Трансляция реализуется в два прохода:

1. Генерация машинного кода без адресов переходов и расчёт значений меток перехода.
  - Ассемблерные мнемоники один в один транслируются в машинные команды.
2. Подстановка меток перехода в инструкции.


## Модель процессора

Интерфейс командной строки: `machine.py <machine_code_file> <input_file>`


### DataPath
![alt text](python/images/DataPath2.png)

Реализован в классе `DataPath`.

Реализован в классе DataPath.

``data_memory`` -- однопортовая память, поэтому либо читаем, либо пишем.

Сигналы (обрабатываются за один такт, реализованы в виде методов класса):

- ``latch_data_addr`` -- защёлкнуть выбранное значение в data_addr;
- ``latch_registers`` -- защёлкнуть выбранное значение в регистры;
- ``alu_left / alu_right`` -- защёлкнуть значения на левом и правом входах АЛУ соответственно;
- ``latch_data_register`` -- защёлкнуть адрес команды в DR
- ``latch_proram_counter`` -- изменить счётчик команд (инкрементировать или взять необходимый адрес из DR);
- ``latch_output_register`` -- защёлкнуть слово из памяти в выходном регистре;
- ``wr``-- записать выбранное значение в память:
  - инкрементированное;
  - декрементированное;
  - с порта ввода ``input`` (обработка на Python):
    - извлечь из входного буфера значение и записать в память;
    - если буфер пуст -- выбрасывается исключение;
- ``output`` -- записать аккумулятор в порт вывода (обработка на Python).

Флаги:
- ``zero`` -- отражает наличие нулевого значения в ячейке.
- ``negative`` -- отражает наличие отрицательного значения в ячейке.

### ControlUnit

![alt text](python/images/ControlUnit.png)

Реализован в классе `ControlUnit`.

- Hardwired (реализовано полностью на Python).

Основная работа с данными происходит на уровне DataPath, а ControlUnit с помощью сигналов работает с этими данными. ControlUnit реализован как hardwired.

Особенности работы модели:

- Цикл симуляции осуществляется в функции `simulation`.
- Шаг моделирования соответствует одной инструкции с выводом состояния в журнал.
- Для журнала состояний процессора используется стандартный модуль `logging`.
- Количество инструкций для моделирования лимитировано.
- Остановка моделирования осуществляется при:
  - превышении лимита количества выполняемых инструкций;
  - исключении `EOFError` -- если нет данных для чтения из порта ввода;
  - исключении `StopIteration` -- если выполнена инструкция `halt`.

## Тестирование

Тестирование выполняется при помощи golden test-ов.



Запустить тесты: `poetry run pytest . -v`

Обновить конфигурацию golden tests:  `poetry run pytest . -v --update-goldens`

CI при помощи Github Action:

``` yaml
name: Python CI

on:
  push:
    branches:
      - master
    paths:
      - ".github/workflows/*"
      - "python/**"
  pull_request:
    branches:
      - master
    paths:
      - ".github/workflows/*"
      - "python/**"

defaults:
  run:
    working-directory: ./python

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install poetry
          poetry install

      - name: Run tests and collect coverage
        run: |
          poetry run coverage run -m pytest .
          poetry run coverage report -m
        env:
          CI: true

  lint:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install poetry
          poetry install

      - name: Run Ruff linters
        run: poetry run ruff check .
```

где:

- `poetry` -- управления зависимостями для языка программирования Python.
- `coverage` -- формирование отчёта об уровне покрытия исходного кода.
- `pytest` -- утилита для запуска тестов.
- `ruff` -- утилита для форматирования и проверки стиля кодирования.

Для алгоритмов реализованы golden-тесты:
- [cat](./python/golden/cat.yml) -- вывод введённого сообщения на экран
- [hello](./python/golden/hello.yml) -- вывод из памяти данных "Hello world!" на экран
- [hello_user](./python/golden/hello_user.yml) -- приветствие представившегося пользователя
- [simple](./python/golden/simple.yml) -- простой тест на работу арифметики
- [prob2](./python/golden/prob2.yml) -- ряд Фиббоначи

### Prob2

Код программы (asm)
``` text
  mov r1,1
  mov r2,2   ; first 2 Fibonacci arguments
  mov rs,r2  ; total sum
  mov rc,4000000 ; counter

  loop:
    sub r3,rc,r2
    js break,r3    ; if r3 < 0 --> break
    add r3,r1,r2
    mod r1,r3,2   ; r3 % 2
    jnz add_new_member,r1  ; if (r3 % 2) != 0 --> continue
  
    add rs,rs,r3 ; else sum += r3
  
    add_new_member:
      mov r1,r2
      mov r2,r3
  
    jmp loop

  break:
    halt
```

Транслированный машинный код:
```text
   {"index": 0, "opcode": "mov", "arg": ["r1", "1"], "term": [5, 0, "mov r1,1"]},
   {"index": 1, "opcode": "mov", "arg": ["r2", "2"], "term": [6, 0, "mov r2,2"]},
   {"index": 2, "opcode": "mov", "arg": ["rs", "r2"], "term": [7, 0, "mov rs,r2"]},
   {"index": 3, "opcode": "mov", "arg": ["rc", "4000000"], "term": [8, 0, "mov rc,4000000"]},
   {"index": 4, "opcode": "sub", "arg": ["r3", "rc", "r2"], "term": [11, 0, "sub r3,rc,r2"]},
   {"index": 5, "opcode": "js", "arg": [13, "r3"], "term": [12, 0, "js break,r3"]},
   {"index": 6, "opcode": "add", "arg": ["r3", "r1", "r2"], "term": [13, 0, "add r3,r1,r2"]},
   {"index": 7, "opcode": "mod", "arg": ["r1", "r3", "2"], "term": [14, 0, "mod r1,r3,2"]},
   {"index": 8, "opcode": "jnz", "arg": [10, "r1"], "term": [15, 0, "jnz add_new_member,r1"]},
   {"index": 9, "opcode": "add", "arg": ["rs", "rs", "r3"], "term": [17, 0, "add rs,rs,r3"]},
   {"index": 10, "opcode": "mov", "arg": ["r1", "r2"], "term": [20, 0, "mov r1,r2"]},
   {"index": 11, "opcode": "mov", "arg": ["r2", "r3"], "term": [21, 0, "mov r2,r3"]},
   {"index": 12, "opcode": "jmp", "arg": 4, "term": [23, 0, "jmp loop"]},
   {"index": 13, "opcode": "halt", "term": [26, 0, "halt"]}
```
Лог работы процессора:
``` shell
  DEBUG   machine:simulation    TICK:   0 PC:   0 ADDR:   0 MEM_OUT: 0 rs: 0 rc: 0 r1: 0 r2: 0 r3: 0 	mov ['r1', '1']  ('mov r1,1'@5:0)
  DEBUG   machine:simulation    TICK:   1 PC:   1 ADDR:   0 MEM_OUT: 0 rs: 0 rc: 0 r1: 1 r2: 0 r3: 0 	mov ['r2', '2']  ('mov r2,2'@6:0)
  DEBUG   machine:simulation    TICK:   2 PC:   2 ADDR:   0 MEM_OUT: 0 rs: 0 rc: 0 r1: 1 r2: 2 r3: 0 	mov ['rs', 'r2']  ('mov rs,r2'@7:0)
  DEBUG   machine:simulation    TICK:   3 PC:   3 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 0 r1: 1 r2: 2 r3: 0 	mov ['rc', '4000000']  ('mov rc,4000000'@8:0)
  DEBUG   machine:simulation    TICK:   4 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 0 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:   5 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3999998 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:   5 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3999998 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:   6 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3999998 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:   7 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:   8 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:   8 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:   9 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 2 r3: 3 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  10 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 2 r3: 3 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  11 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 3 r3: 3 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  12 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 3 r3: 3 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  13 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 3 r3: 3999997 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  13 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 3 r3: 3999997 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  14 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 3 r3: 3999997 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  15 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 2 r2: 3 r3: 5 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  16 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 3 r3: 5 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  16 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 3 r3: 5 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  17 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 1 r2: 3 r3: 5 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  18 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 3 r3: 5 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  19 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 5 r3: 5 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  20 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 5 r3: 5 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  21 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 5 r3: 3999995 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  21 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 5 r3: 3999995 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  22 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 5 r3: 3999995 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  23 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 3 r2: 5 r3: 8 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  24 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 0 r2: 5 r3: 8 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  24 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 0 r2: 5 r3: 8 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  25 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 2 rc: 4000000 r1: 0 r2: 5 r3: 8 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK:  26 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 0 r2: 5 r3: 8 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  27 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 5 r3: 8 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  28 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 8 r3: 8 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  29 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 8 r3: 8 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  30 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 8 r3: 3999992 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  30 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 8 r3: 3999992 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  31 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 8 r3: 3999992 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  32 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 5 r2: 8 r3: 13 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  33 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 1 r2: 8 r3: 13 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  33 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 1 r2: 8 r3: 13 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  34 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 1 r2: 8 r3: 13 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  35 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 8 r3: 13 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  36 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 13 r3: 13 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  37 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 13 r3: 13 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  38 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 13 r3: 3999987 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  38 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 13 r3: 3999987 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  39 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 13 r3: 3999987 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  40 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 8 r2: 13 r3: 21 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  41 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 1 r2: 13 r3: 21 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  41 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 1 r2: 13 r3: 21 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  42 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 1 r2: 13 r3: 21 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  43 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 13 r3: 21 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  44 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 21 r3: 21 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  45 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 21 r3: 21 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  46 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 21 r3: 3999979 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  46 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 21 r3: 3999979 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  47 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 21 r3: 3999979 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  48 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 13 r2: 21 r3: 34 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  49 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 0 r2: 21 r3: 34 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  49 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 0 r2: 21 r3: 34 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  50 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 10 rc: 4000000 r1: 0 r2: 21 r3: 34 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK:  51 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 0 r2: 21 r3: 34 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  52 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 21 r3: 34 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  53 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 34 r3: 34 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  54 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 34 r3: 34 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  55 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 34 r3: 3999966 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  55 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 34 r3: 3999966 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  56 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 34 r3: 3999966 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  57 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 21 r2: 34 r3: 55 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  58 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 1 r2: 34 r3: 55 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  58 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 1 r2: 34 r3: 55 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  59 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 1 r2: 34 r3: 55 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  60 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 34 r3: 55 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  61 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 55 r3: 55 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  62 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 55 r3: 55 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  63 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 55 r3: 3999945 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  63 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 55 r3: 3999945 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  64 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 55 r3: 3999945 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  65 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 34 r2: 55 r3: 89 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  66 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 1 r2: 55 r3: 89 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  66 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 1 r2: 55 r3: 89 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  67 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 1 r2: 55 r3: 89 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  68 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 55 r3: 89 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  69 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 89 r3: 89 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  70 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 89 r3: 89 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  71 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 89 r3: 3999911 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  71 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 89 r3: 3999911 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  72 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 89 r3: 3999911 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  73 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 55 r2: 89 r3: 144 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  74 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 0 r2: 89 r3: 144 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  74 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 0 r2: 89 r3: 144 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  75 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 44 rc: 4000000 r1: 0 r2: 89 r3: 144 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK:  76 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 0 r2: 89 r3: 144 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  77 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 89 r3: 144 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  78 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 144 r3: 144 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  79 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 144 r3: 144 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  80 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 144 r3: 3999856 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  80 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 144 r3: 3999856 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  81 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 144 r3: 3999856 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  82 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 89 r2: 144 r3: 233 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  83 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 1 r2: 144 r3: 233 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  83 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 1 r2: 144 r3: 233 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  84 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 1 r2: 144 r3: 233 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  85 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 144 r3: 233 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  86 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 233 r3: 233 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  87 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 233 r3: 233 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  88 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 233 r3: 3999767 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  88 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 233 r3: 3999767 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  89 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 233 r3: 3999767 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  90 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 144 r2: 233 r3: 377 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  91 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 1 r2: 233 r3: 377 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  91 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 1 r2: 233 r3: 377 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  92 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 1 r2: 233 r3: 377 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK:  93 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 233 r3: 377 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK:  94 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 377 r3: 377 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK:  95 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 377 r3: 377 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK:  96 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 377 r3: 3999623 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  96 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 377 r3: 3999623 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK:  97 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 377 r3: 3999623 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK:  98 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 233 r2: 377 r3: 610 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK:  99 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 0 r2: 377 r3: 610 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK:  99 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 0 r2: 377 r3: 610 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 100 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 188 rc: 4000000 r1: 0 r2: 377 r3: 610 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 101 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 0 r2: 377 r3: 610 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 102 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 377 r3: 610 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 103 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 610 r3: 610 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 104 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 610 r3: 610 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 105 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 610 r3: 3999390 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 105 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 610 r3: 3999390 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 106 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 610 r3: 3999390 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 107 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 377 r2: 610 r3: 987 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 108 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 1 r2: 610 r3: 987 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 108 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 1 r2: 610 r3: 987 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 109 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 1 r2: 610 r3: 987 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 110 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 610 r3: 987 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 111 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 987 r3: 987 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 112 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 987 r3: 987 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 113 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 987 r3: 3999013 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 113 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 987 r3: 3999013 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 114 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 987 r3: 3999013 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 115 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 610 r2: 987 r3: 1597 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 116 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 1 r2: 987 r3: 1597 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 116 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 1 r2: 987 r3: 1597 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 117 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 1 r2: 987 r3: 1597 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 118 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 987 r3: 1597 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 119 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 1597 r3: 1597 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 120 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 1597 r3: 1597 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 121 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 1597 r3: 3998403 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 121 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 1597 r3: 3998403 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 122 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 1597 r3: 3998403 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 123 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 987 r2: 1597 r3: 2584 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 124 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 0 r2: 1597 r3: 2584 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 124 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 0 r2: 1597 r3: 2584 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 125 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 798 rc: 4000000 r1: 0 r2: 1597 r3: 2584 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 126 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 0 r2: 1597 r3: 2584 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 127 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 1597 r3: 2584 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 128 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 2584 r3: 2584 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 129 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 2584 r3: 2584 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 130 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 2584 r3: 3997416 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 130 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 2584 r3: 3997416 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 131 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 2584 r3: 3997416 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 132 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1597 r2: 2584 r3: 4181 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 133 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1 r2: 2584 r3: 4181 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 133 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1 r2: 2584 r3: 4181 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 134 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1 r2: 2584 r3: 4181 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 135 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 2584 r3: 4181 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 136 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 4181 r3: 4181 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 137 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 4181 r3: 4181 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 138 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 4181 r3: 3995819 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 138 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 4181 r3: 3995819 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 139 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 4181 r3: 3995819 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 140 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 2584 r2: 4181 r3: 6765 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 141 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1 r2: 4181 r3: 6765 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 141 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1 r2: 4181 r3: 6765 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 142 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 1 r2: 4181 r3: 6765 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 143 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 4181 r3: 6765 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 144 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 6765 r3: 6765 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 145 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 6765 r3: 6765 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 146 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 6765 r3: 3993235 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 146 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 6765 r3: 3993235 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 147 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 6765 r3: 3993235 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 148 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 4181 r2: 6765 r3: 10946 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 149 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 0 r2: 6765 r3: 10946 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 149 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 0 r2: 6765 r3: 10946 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 150 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 3382 rc: 4000000 r1: 0 r2: 6765 r3: 10946 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 151 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 0 r2: 6765 r3: 10946 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 152 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 6765 r3: 10946 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 153 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 10946 r3: 10946 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 154 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 10946 r3: 10946 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 155 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 10946 r3: 3989054 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 155 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 10946 r3: 3989054 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 156 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 10946 r3: 3989054 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 157 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 6765 r2: 10946 r3: 17711 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 158 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 1 r2: 10946 r3: 17711 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 158 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 1 r2: 10946 r3: 17711 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 159 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 1 r2: 10946 r3: 17711 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 160 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 10946 r3: 17711 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 161 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 17711 r3: 17711 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 162 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 17711 r3: 17711 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 163 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 17711 r3: 3982289 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 163 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 17711 r3: 3982289 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 164 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 17711 r3: 3982289 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 165 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 10946 r2: 17711 r3: 28657 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 166 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 1 r2: 17711 r3: 28657 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 166 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 1 r2: 17711 r3: 28657 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 167 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 1 r2: 17711 r3: 28657 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 168 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 17711 r3: 28657 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 169 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 28657 r3: 28657 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 170 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 28657 r3: 28657 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 171 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 28657 r3: 3971343 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 171 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 28657 r3: 3971343 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 172 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 28657 r3: 3971343 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 173 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 17711 r2: 28657 r3: 46368 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 174 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 0 r2: 28657 r3: 46368 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 174 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 0 r2: 28657 r3: 46368 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 175 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 14328 rc: 4000000 r1: 0 r2: 28657 r3: 46368 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 176 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 0 r2: 28657 r3: 46368 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 177 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 28657 r3: 46368 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 178 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 46368 r3: 46368 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 179 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 46368 r3: 46368 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 180 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 46368 r3: 3953632 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 180 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 46368 r3: 3953632 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 181 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 46368 r3: 3953632 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 182 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 28657 r2: 46368 r3: 75025 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 183 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 1 r2: 46368 r3: 75025 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 183 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 1 r2: 46368 r3: 75025 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 184 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 1 r2: 46368 r3: 75025 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 185 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 46368 r3: 75025 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 186 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 75025 r3: 75025 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 187 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 75025 r3: 75025 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 188 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 75025 r3: 3924975 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 188 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 75025 r3: 3924975 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 189 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 75025 r3: 3924975 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 190 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 46368 r2: 75025 r3: 121393 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 191 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 1 r2: 75025 r3: 121393 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 191 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 1 r2: 75025 r3: 121393 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 192 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 1 r2: 75025 r3: 121393 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 193 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 75025 r3: 121393 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 194 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 121393 r3: 121393 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 195 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 121393 r3: 121393 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 196 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 121393 r3: 3878607 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 196 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 121393 r3: 3878607 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 197 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 121393 r3: 3878607 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 198 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 75025 r2: 121393 r3: 196418 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 199 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 0 r2: 121393 r3: 196418 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 199 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 0 r2: 121393 r3: 196418 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 200 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 60696 rc: 4000000 r1: 0 r2: 121393 r3: 196418 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 201 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 0 r2: 121393 r3: 196418 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 202 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 121393 r3: 196418 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 203 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 196418 r3: 196418 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 204 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 196418 r3: 196418 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 205 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 196418 r3: 3803582 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 205 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 196418 r3: 3803582 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 206 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 196418 r3: 3803582 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 207 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 121393 r2: 196418 r3: 317811 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 208 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 1 r2: 196418 r3: 317811 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 208 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 1 r2: 196418 r3: 317811 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 209 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 1 r2: 196418 r3: 317811 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 210 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 196418 r3: 317811 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 211 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 317811 r3: 317811 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 212 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 317811 r3: 317811 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 213 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 317811 r3: 3682189 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 213 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 317811 r3: 3682189 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 214 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 317811 r3: 3682189 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 215 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 196418 r2: 317811 r3: 514229 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 216 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 1 r2: 317811 r3: 514229 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 216 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 1 r2: 317811 r3: 514229 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 217 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 1 r2: 317811 r3: 514229 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 218 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 317811 r3: 514229 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 219 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 514229 r3: 514229 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 220 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 514229 r3: 514229 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 221 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 514229 r3: 3485771 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 221 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 514229 r3: 3485771 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 222 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 514229 r3: 3485771 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 223 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 317811 r2: 514229 r3: 832040 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 224 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 0 r2: 514229 r3: 832040 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 224 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 0 r2: 514229 r3: 832040 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 225 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 257114 rc: 4000000 r1: 0 r2: 514229 r3: 832040 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 226 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 0 r2: 514229 r3: 832040 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 227 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 514229 r3: 832040 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 228 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 832040 r3: 832040 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 229 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 832040 r3: 832040 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 230 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 832040 r3: 3167960 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 230 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 832040 r3: 3167960 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 231 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 832040 r3: 3167960 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 232 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 514229 r2: 832040 r3: 1346269 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 233 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1 r2: 832040 r3: 1346269 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 233 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1 r2: 832040 r3: 1346269 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 234 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1 r2: 832040 r3: 1346269 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 235 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 832040 r3: 1346269 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 236 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 1346269 r3: 1346269 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 237 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 1346269 r3: 1346269 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 238 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 1346269 r3: 2653731 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 238 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 1346269 r3: 2653731 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 239 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 1346269 r3: 2653731 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 240 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 832040 r2: 1346269 r3: 2178309 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 241 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1 r2: 1346269 r3: 2178309 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 241 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1 r2: 1346269 r3: 2178309 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 242 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1 r2: 1346269 r3: 2178309 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 243 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 1346269 r3: 2178309 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 244 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 2178309 r3: 2178309 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 245 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 2178309 r3: 2178309 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 246 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 2178309 r3: 1821691 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 246 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 2178309 r3: 1821691 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 247 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 2178309 r3: 1821691 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 248 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 1346269 r2: 2178309 r3: 3524578 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 249 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 0 r2: 2178309 r3: 3524578 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 249 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 0 r2: 2178309 r3: 3524578 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 250 PC:   9 ADDR:   0 MEM_OUT: 0 rs: 1089154 rc: 4000000 r1: 0 r2: 2178309 r3: 3524578 	add ['rs', 'rs', 'r3']  ('add rs,rs,r3'@17:0)
  DEBUG   machine:simulation    TICK: 251 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 0 r2: 2178309 r3: 3524578 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 252 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 2178309 r3: 3524578 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 253 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 3524578 r3: 3524578 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 254 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 3524578 r3: 3524578 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 255 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 3524578 r3: 475422 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 255 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 3524578 r3: 475422 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 256 PC:   6 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 3524578 r3: 475422 	add ['r3', 'r1', 'r2']  ('add r3,r1,r2'@13:0)
  DEBUG   machine:simulation    TICK: 257 PC:   7 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 2178309 r2: 3524578 r3: 5702887 	mod ['r1', 'r3', '2']  ('mod r1,r3,2'@14:0)
  DEBUG   machine:simulation    TICK: 258 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 1 r2: 3524578 r3: 5702887 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 258 PC:   8 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 1 r2: 3524578 r3: 5702887 	jnz [10, 'r1']  ('jnz add_new_member,r1'@15:0)
  DEBUG   machine:simulation    TICK: 259 PC:  10 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 1 r2: 3524578 r3: 5702887 	mov ['r1', 'r2']  ('mov r1,r2'@20:0)
  DEBUG   machine:simulation    TICK: 260 PC:  11 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 3524578 r2: 3524578 r3: 5702887 	mov ['r2', 'r3']  ('mov r2,r3'@21:0)
  DEBUG   machine:simulation    TICK: 261 PC:  12 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 3524578 r2: 5702887 r3: 5702887 	jmp 4  ('jmp loop'@23:0)
  DEBUG   machine:simulation    TICK: 262 PC:   4 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 3524578 r2: 5702887 r3: 5702887 	sub ['r3', 'rc', 'r2']  ('sub r3,rc,r2'@11:0)
  DEBUG   machine:simulation    TICK: 263 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 3524578 r2: 5702887 r3: -1702887 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 263 PC:   5 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 3524578 r2: 5702887 r3: -1702887 	js [13, 'r3']  ('js break,r3'@12:0)
  DEBUG   machine:simulation    TICK: 264 PC:  13 ADDR:   0 MEM_OUT: 0 rs: 4613732 rc: 4000000 r1: 3524578 r2: 5702887 r3: -1702887 	halt  ('halt'@26:0)
  INFO    machine:simulation    output_buffer: ''
```

## Аналитика
```text
| ФИО                           | алг        | LoC | code инстр. | инстр. | такт. |
| Бабенко Даниил Александрович  | cat        | 11  | 9           | 59     | 69    |
| Бабенко Даниил Александрович  | hello      | 16  | 8           | 63     | 63    |
| Бабенко Даниил Александрович  | hello_user | 60  | 35          | 204    | 210   |
| Бабенко Даниил Александрович  | prob2      | 26  | 14          | 264    | 264   |
| Бабенко Даниил Александрович  | simple     | 7   | 7           | 6      | 6     |
```
