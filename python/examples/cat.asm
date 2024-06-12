input
loop:
    jnz break,r7
    print
    input
    jmp loop
break:
    halt