import math

def ln(x):
    """
    ln(x) = lim n(n^(1/n) -1) ; n = float("inf")
    """
    n = 1000000000000.0
    return n * ((x ** (1/n)) - 1)

print(ln(math.e), math.log(math.e))
print(ln(0.5), math.log(0.5))
print(ln(100.0), math.log(100.0))
