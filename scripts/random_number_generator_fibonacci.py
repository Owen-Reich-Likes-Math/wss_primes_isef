# toy random number generator, also not used
# experiment with connections to cryptography

import sys
import random

j = 3
k = 7
modval = 10
val = "8675309"


def conv(val):
    arr = []
    for i in range(len(val)):
        arr.append(int(val[i]))
    return arr


# Read command-line overrides
if len(sys.argv) > 1:
    val = str(sys.argv[1])
if len(sys.argv) > 2:
    modval = int(sys.argv[2])

s = conv(val)

if len(s) < k:
    print("Value needs to be larger than 7")
    sys.exit()

results = []

for n in range(20):
    out = (s[j - 1] + s[k - 1]) % modval

    # shift left
    for i in range(len(s) - 1):
        s[i] = s[i + 1]

    s[len(s) - 1] = out
    results.append(out)

print(results)
