# проверка prob2
previous = 1
next = 2
sum = 2

while next <= 4000000:
    tmp = previous + next
    if (tmp) % 2 == 0:
        sum += tmp
    previous = next
    next = tmp

print(sum)