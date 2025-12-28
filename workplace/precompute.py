import ecdsa

curve = ecdsa.SECP256k1
G = curve.generator

# MSB
def int_to_words_32(x):
    """Return 8 x 32-bit words, MSB first"""
    hex_str = format(x, '064x')  # 256-bit, zero-padded
    words = [int(hex_str[i:i+8], 16) for i in range(0, 64, 8)]
    return words

# LSB
# def int_to_words_32(x):
#     """Return 8 x 32-bit words, LSB first (low word at index 0)"""
#     hex_str = format(x, '064x')  # 256-bit, zero-padded
#     # Take slices from end to start (reverse order)
#     words = [int(hex_str[i:i+8], 16) for i in range(56, -8, -8)]
#     return words
    
def print_cuda_array(name, values):
    print(f"__constant__ unsigned int {name}[32][8] = {{")
    for i, words in enumerate(values):
        word_str = ", ".join(f"0x{w:08x}" for w in words)
        print(f"    {{ {word_str} }}, // {i}G")
    print("};\n")

precomp_x = []
precomp_y = []

for i in range(32):  # 0 to 31 for w=5
    P = i * G
    if i == 0:
        precomp_x.append([0]*8)
        precomp_y.append([0]*8)
    else:
        precomp_x.append(int_to_words_32(P.x()))
        precomp_y.append(int_to_words_32(P.y()))

print_cuda_array("PRECOMP_X", precomp_x)
print_cuda_array("PRECOMP_Y", precomp_y)
