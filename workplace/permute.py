a = 0x9E3779B97F4A7C15
b = 0x1D3F84A5B7C29E3
m = (1 << 71) - 1
range_start = 1 << 71

def bit_reverse_71(num):
    r = 0
    for _ in range(71):
        r = (r << 1) | (num & 1)
        num >>= 1
    return r

def perm_with_reverse(i):
    y = (a * i + b) % m
    x = bit_reverse_71(y)
    k = range_start + x
    return f"{k:019X}"

def perm_without_reverse(i):
    y = (a * i + b) % m
    k = range_start + y
    return f"{k:019X}"

i_base = 0xB38AF1800
stride = 256  # example thread stride

results_with = []
results_without = []

for j in range(10):
    i = i_base + j * stride
    k_with = perm_with_reverse(i)
    k_without = perm_without_reverse(i)
    results_with.append((i, k_with))
    results_without.append((i, k_without))

print("With reverse:")
for i, k in results_with:
    print(f"i={i:016X}, k={k}")

print("\nWithout reverse:")
for i, k in results_without:
    print(f"i={i:016X}, k={k}")