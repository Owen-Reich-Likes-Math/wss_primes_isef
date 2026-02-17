# graphs the bound - simple script, unused

import matplotlib.pyplot as plt


def legendre_symbol(a, p):
    if a % p == 0: return 0
    res = pow(a, (p - 1) // 2, p)
    return 1 if res == 1 else -1


def calculate_bound(p):
    leg = legendre_symbol(5, p)
    exponent = 0.5 * (3 - leg)
    numerator = 2400 * p * (p ** exponent - 1)
    denominator = p - 1
    return numerator / denominator


def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0: return False
    return True


# Generate primes up to a reasonable limit for visualization
p_max = 500
primes = [p for p in range(2, p_max + 1) if is_prime(p) and p != 2 and p != 5]

p_plus = []
b_plus = []
p_minus = []
b_minus = []

for p in primes:
    leg = legendre_symbol(5, p)
    bound = calculate_bound(p)
    if leg == 1:
        p_plus.append(p)
        b_plus.append(bound)
    else:
        p_minus.append(p)
        b_minus.append(bound)

plt.figure(figsize=(12, 7))
plt.scatter(p_plus, b_plus, color='green', label=r'$(p/5) = 1$', alpha=0.7, edgecolors='k')
plt.scatter(p_minus, b_minus, color='red', label=r'$(p/5) = -1$', alpha=0.7, edgecolors='k')

plt.yscale('log')
plt.xlabel('Prime number (p)', fontsize=12)
plt.ylabel(r'Bound $\nu_p(\Xi)$ (Log Scale)', fontsize=12)
plt.title("Visualization of the divisibility bound given by Bugeaud-Laurent", fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.savefig('WSS_divisibility_bound.png')
plt.close()

# Providing a table of first few values to show the data structure
print(f"{'Prime':<10} | {'Legendre':<10} | {'Bound'}")
for p in primes[:10]:
    leg = legendre_symbol(5, p)
    print(f"{p:<10} | {leg:<10} | {calculate_bound(p):.2f}")
