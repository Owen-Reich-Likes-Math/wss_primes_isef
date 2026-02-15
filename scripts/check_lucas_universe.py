# checks a single universe for a high divisibility prime

import math

# -------------------------
# Basic number theory utils
# -------------------------

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def find_primes(n):
    return [i for i in range(2, n + 1) if is_prime(i)]


def v_p(n, p):
    """p-adic valuation of n"""
    if n == 0:
        return float("inf")
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


# -------------------------
# Legendre symbol (p / D)
# -------------------------

def legendre(p, D):
    """
    Legendre symbol (p / D), where D is an odd prime.
    """
    if D < 2 or D % 2 == 0 or not is_prime(D):
        raise ValueError("D must be an odd prime")

    if p % D == 0:
        return 0

    euler = pow(p, (D - 1) // 2, D)
    if euler == D - 1:
        return -1
    elif euler == 1:
        return 1
    else:
        return 0


# -------------------------
# Lucas sequence U_n(P, Q)
# -------------------------

def lucas_U(n, P, Q):
    """
    Computes the Lucas sequence U_n(P, Q) using integer recursion.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1

    U0, U1 = 0, 1
    for _ in range(2, n + 1):
        U0, U1 = U1, P * U1 - Q * U0
    return U1


# -------------------------
# Core computation
# -------------------------

def check_lucas_at_point(p, P, Q):
    """
    Returns v_p(U_{p - (p/D)}(P, Q))
    """
    D = P * P - 4 * Q

    # For this script we require D to be an odd prime
    if D <= 0 or D % 2 == 0 or not is_prime(D):
        return None

    chi = legendre(p, D)
    index = p - chi
    term = lucas_U(index, P, Q)

    return v_p(term, p)


def check_lucas_up_to(n, P, Q):
    xs, ys = [], []
    for p in find_primes(n):
        val = check_lucas_at_point(p, P, Q)
        if val is not None and val != float("inf"):
            xs.append(p)
            ys.append(val)
    return xs, ys
