

def nth_fib(n, l):
    if n <= l-1:
        return 0
    if n == l:
        return 1
    else:
        sum = 0
        for p in range(0, l+1):
            sum += nth_fib(n-p-1, l)
        return sum

m = int(input("period? "))

for x in range(1, 200):
    item = nth_fib(x, 2) % m
    print(f"{nth_fib(x, 2)}, {item}")
