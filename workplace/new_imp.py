import random

# Constants
MODULUS_P = [
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
]
_1_CONSTANT = [0, 0, 0, 0, 0, 0, 0, 1]
P_INT = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# Generalized raw_add and raw_sub for variable limbs

def raw_add(result, lhs, rhs, num_limbs=8):
    carry = 0
    for i in range(num_limbs - 1, -1, -1):
        temp = lhs[i] + rhs[i] + carry
        result[i] = temp & 0xFFFFFFFF
        carry = temp >> 32
    return carry

def raw_sub(result, lhs, rhs, num_limbs=8):
    borrow = 0
    for i in range(num_limbs - 1, -1, -1):
        temp = lhs[i] - rhs[i] - borrow
        result[i] = temp & 0xFFFFFFFF
        borrow = 1 if temp < 0 else 0
    return borrow

# Other helpers same

def is_zero(a):
    return all(x == 0 for x in a)

def set_zero(a):
    for i in range(8):
        a[i] = 0

def copy_int(src, dest):
    for i in range(8):
        dest[i] = src[i]

def is_one(a):
    return (a[7] == 1) and all(a[i] == 0 for i in range(6))

def is_even(a):
    return (a[7] & 1) == 0

def halve(a):
    carry = 0
    for i in range(8):
        temp = a[i]
        a[i] = (temp >> 1) | (carry << 31)
        carry = temp & 1

def is_greater_or_equal(a, b):
    for i in range(8):
        if a[i] > b[i]:
            return True
        if a[i] < b[i]:
            return False
    return True

def amp(a, b, c):
    carry = raw_add(c, a, b)
    gt = False
    for i in range(8):
        if c[i] > MODULUS_P[i]:
            gt = True
            break
        elif c[i] < MODULUS_P[i]:
            break
    if carry or gt:
        raw_sub(c, c, MODULUS_P)

def smp(a, b, c):
    borrow = raw_sub(c, a, b)
    if borrow:
        raw_add(c, c, MODULUS_P)

# Fixed halve_odd with general raw_add

def halve_odd(x):
    num_limbs = 9
    tmp_ext = [0] * num_limbs
    for i in range(8):
        tmp_ext[i+1] = x[i]
    p_ext = [0] + MODULUS_P[:]
    carry = raw_add(tmp_ext, tmp_ext, p_ext, num_limbs)
    if carry:
        print("Carry out from add in halve_odd")
    # Halve the 9-limb number
    carry = 0
    for i in range(num_limbs):
        temp = tmp_ext[i]
        tmp_ext[i] = (temp >> 1) | (carry << 31)
        carry = temp & 1

    # Check extra high
    if tmp_ext[0] != 0:
        print("Warning: extra high after halve_odd: ", tmp_ext[0])
    return tmp_ext[1:]

# IMP sim same

def imp_sim(value, trace=False):
    if is_zero(value):
        set_zero(value)
        return []

    u = [0]*8
    v = [0]*8
    x1 = [0]*8
    x2 = [0]*8
    tmp = [0]*8

    copy_int(value, u)
    copy_int(MODULUS_P, v)
    copy_int(_1_CONSTANT, x1)
    set_zero(x2)

    steps = []
    if trace:
        steps.append(f"Initial: u={u}, v={v}, x1={x1}, x2={x2}")

    loop_count = 0
    while not is_one(u) and not is_one(v):
        loop_count += 1
        if loop_count > 10000:
            steps.append("Infinite loop detected")
            return steps

        while is_even(u):
            halve(u)
            if is_even(x1):
                halve(x1)
            else:
                x1[:] = halve_odd(x1)
            if trace:
                steps.append(f"u even halve: u={u}, x1={x1}")

        while is_even(v):
            halve(v)
            if is_even(x2):
                halve(x2)
            else:
                x2[:] = halve_odd(x2)
            if trace:
                steps.append(f"v even halve: v={v}, x2={x2}")

        if is_greater_or_equal(u, v):
            smp(u, v, u)
            smp(x1, x2, x1)
            if trace:
                steps.append(f"u >= v: u={u}, x1={x1}")
        else:
            smp(v, u, v)
            smp(x2, x1, x2)
            if trace:
                steps.append(f"v > u: v={v}, x2={x2}")

        if trace:
            steps.append(f"End iteration: u={u}, v={v}, x1={x1}, x2={x2}")

    if is_one(u):
        copy_int(x1, value)
    else:
        copy_int(x2, value)

    return steps if trace else None

# Conversion

def limbs_to_int(limbs):
    val = 0
    for w in limbs:
        val = (val << 32) | w
    return val

def int_to_limbs(n):
    n %= P_INT
    limbs = [0]*8
    for i in range(7, -1, -1):
        limbs[i] = n & 0xFFFFFFFF
        n >>= 32
    return limbs

# Test with val = 2

val_int = 2
val = int_to_limbs(val_int)
val_copy = val[:]

steps = imp_sim(val, trace=False)

inv_int = limbs_to_int(val)
check = (val_int * inv_int) % P_INT == 1

print(f"val={val_int}, inv={inv_int}, check={check}")
print(f"inv limbs={val}")
print(f"expected inv2 = {(P_INT + 1) // 2}")

# For comparison, correct inv2 limbs
correct_inv2 = int_to_limbs((P_INT + 1) // 2)
print(f"correct inv2 limbs={correct_inv2}")