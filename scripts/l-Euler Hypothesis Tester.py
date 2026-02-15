# lucas_wss_euler_test.py
# Tests whether prime exponents p from Euler-type counterexamples
# are (P,Q)-Wall--Sun--Sun primes by checking p^2 | U_{p-(D/p)}.
# basically checks my conjecture of connetion from other WSS primes to Euler's sum of powers

from typing import List, Tuple
import math

# --- helper: legendre symbol (a|p) for odd prime p ---
def legendre_symbol(a: int, p: int) -> int:
    a %= p
    if a == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    if ls == 1:
        return 1
    if ls == p - 1:
        return -1
    return 0  # should not happen for prime p

# --- 2x2 matrix exponentiation mod M ---
def mat_mul(A, B, mod):
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod, (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod, (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]
    ]

def mat_pow(M, e, mod):
    R = [[1,0],[0,1]]
    A = [row[:] for row in M]
    while e > 0:
        if e & 1:
            R = mat_mul(R, A, mod)
        A = mat_mul(A, A, mod)
        e >>= 1
    return R

# --- Lucas U_n(P,Q) via matrix [[P,-Q],[1,0]] ---
def lucas_U_mod(n: int, P: int, Q: int, mod: int) -> int:
    if n == 0:
        return 0 % mod
    if n == 1:
        return 1 % mod
    M = [[P % mod, (-Q) % mod], [1 % mod, 0]]
    Mexp = mat_pow(M, n-1, mod)
    # U_n is first component of M^(n-1) * [U_1, U_0] = M^(n-1) * [1,0]
    U_n = (Mexp[0][0] * 1 + Mexp[0][1] * 0) % mod
    return U_n

# --- test function for one (counterexample, P,Q) ---
def test_counterexample(a_list: List[int], z: int, p: int, P: int, Q: int, check_power_b: bool=True):
    b = len(a_list)
    D = P*P - 4*Q
    leg = legendre_symbol(D, p)  # (D/p)
    m = p - leg
    out = {"p": p, "P": P, "Q": Q, "D": D, "leg": leg, "m": m}
    mod_p2 = p*p
    U_m_mod_p2 = lucas_U_mod(m, P, Q, mod_p2)
    out["U_m_mod_p2"] = U_m_mod_p2
    out["p2_divides_U_m"] = (U_m_mod_p2 % mod_p2 == 0)
    if check_power_b:
        mod_pb = pow(p, b)
        U_m_mod_pb = lucas_U_mod(m, P, Q, mod_pb)
        out["p^b_divides_U_m"] = (U_m_mod_pb % mod_pb == 0)
    # also record p-adic valuation of U_m (how many p factors)
    val = lucas_U_mod(m, P, Q, pow(p, 10))  # safe upper cap; adjust if needed
    # crude valuation:
    v = 0
    tmp = val
    while tmp % p == 0 and tmp != 0:
        tmp //= p
        v += 1
    out["p_adic_valuation_approx"] = v
    return out

# --- a small driver for multiple (P,Q) and examples ---
def run_tests(countexamples: List[Tuple[List[int], int, int]], PQs: List[Tuple[int,int]]):
    results = []
    for a_list, z, p in countexamples:
        for P,Q in PQs:
            res = test_counterexample(a_list, z, p, P, Q)
            results.append((a_list, z, p, P, Q, res))
    return results

if __name__ == "__main__":
    # Known Euler counterexamples with prime exponent p=5 (a few)
    examples = [
        ([27,84,110,133], 144, 5),             # Lander-Parkin-Selfridge classic
        ([85282, 28969, 3183, 55], 85359, 5)   # another 4-term 5th-power example
        # add more examples here as needed
    ]
    # Choose a few (P,Q) Lucas families to search:
    PQ_choices = [
        (1, -1),   # Fibonacci (P=1,Q=-1), D=5
        (1, 1),    # Pell-like? (P=1,Q=1) - different discriminant
        (2, -1),   # k-Fibonacci with k=2 (P=2,Q=-1)
        (3, -1)    # k-Fibonacci k=3
    ]
    results = run_tests(examples, PQ_choices)
    for a_list, z, p, P, Q, r in results:
        print(f"example a={a_list} = {z}^{p} with exponent p={p}, testing (P,Q)=({P},{Q}):")
        print(f"  D={r['D']}, leg(D/p)={r['leg']}, m={r['m']}")
        print(f"  U_m mod p^2 = {r['U_m_mod_p2']}, p^2 | U_m? {r['p2_divides_U_m']}")
        print(f"  approx p-adic valuation of U_m (capped): {r['p_adic_valuation_approx']}")
        print()
