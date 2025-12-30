#include <stdint.h>

// BitCrack Library
namespace sha256 {
    #include "sha256.cuh"
}

// BitCrack Library
namespace ripemd160 {
    #include "ripemd160.cuh"
}


// ==================================================================================================


__constant__ bool DEBUG_TEST_MODE = true;

// priv key 1
// __constant__ uint8_t TARGET_H160[20] = {
//     0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94,
//     0x1c, 0x45, 0xd1, 0xb3, 0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6
// };

// priv key 100
// __constant__ uint8_t TARGET_H160[20] = {
//     0x10, 0x18, 0x85, 0x36, 0x70, 0xf9, 0xf3, 0xb0, 0x58, 0x2c,
//     0x5b, 0x9e, 0xe8, 0xce, 0x93, 0x76, 0x4a, 0xc3, 0x2b, 0x93
// };

// Private key decimal 1_000
// __constant__ uint8_t TARGET_H160[20] = {
//     0x46, 0xce, 0xee, 0xd7, 0x97, 0xcd, 0x8c, 0x7b, 0x6e, 0x7f,
//     0x52, 0x1b, 0x07, 0x5a, 0xd2, 0xfb, 0x1e, 0x5d, 0x7e, 0x42
// };

// Private key decimal 10_000
__constant__ uint8_t TARGET_H160[20] = {
    0xdb, 0x6e, 0xb1, 0x82, 0x09, 0x4f, 0xca, 0x54, 0x94, 0x78,
    0xa2, 0xe5, 0x8a, 0x16, 0xcf, 0x13, 0xb4, 0xdb, 0x3e, 0x2e
};

// Private key decimal 100_000
// __constant__ uint8_t TARGET_H160[20] = {
//     0x96, 0xa9, 0xd7, 0x6a, 0x7f, 0x3f, 0x96, 0x1e, 0x30, 0x82,
//     0xc1, 0x3b, 0xe7, 0x0d, 0x09, 0x19, 0x0e, 0x4f, 0xc3, 0x4b
// };

// Private key decimal 1_000_000
// __constant__ uint8_t TARGET_H160[20] = {
//     0x08, 0x50, 0xa3, 0xb1, 0x7f, 0xb3, 0x0e, 0x2d, 0x9b, 0x0d,
//     0xc8, 0xc6, 0xaa, 0x50, 0x75, 0x2e, 0x2e, 0x41, 0x63, 0x35
// };

// Puzzle #72 priv key?
// __constant__ uint8_t TARGET_H160[20] = {
//     0xbf, 0x74, 0x13, 0xe8, 0xdf, 0x4e, 0x7a, 0x34, 0xce, 0x9d,
//     0xc1, 0x3e, 0x2f, 0x26, 0x48, 0x78, 0x3e, 0xc5, 0x4a, 0xdb
// };


// ==================================================================================================


/* ---------------------------------------------------------
    This header provides low-level CUDA PTX helpers for
    multi-precision arithmetic with explicit carry handling.

    Naming convention:
    - HI / LO : upper or lower 32 bits of a 64-bit multiplication
    - CC      : sets the carry flag
    - C       : consumes the carry flag
--------------------------------------------------------- */
#define PTX_MADC_LO_CC(result, multiplicand, multiplier, addend) \
    asm volatile (                                               \
        "madc.lo.cc.u32 %0, %1, %2, %3;\n\t"                     \
        : "=r"(result)                                           \
        : "r"(multiplicand), "r"(multiplier), "r"(addend))

#define PTX_MAD_LO_CC(result, multiplicand, multiplier, addend) \
    asm volatile (                                              \
        "mad.lo.cc.u32 %0, %1, %2, %3;\n\t"                     \
        : "=r"(result)                                          \
        : "r"(multiplicand), "r"(multiplier), "r"(addend))


/* ---------------------------------------------------------
   Multiply-Add with Carry (High 32 bits)
   result = high32(multiplicand * multiplier + addend + carry)
--------------------------------------------------------- */
#define PTX_MADC_HI_CC(result, multiplicand, multiplier, addend) \
    asm volatile (                                               \
        "madc.hi.cc.u32 %0, %1, %2, %3;\n\t"                     \
        : "=r"(result)                                           \
        : "r"(multiplicand), "r"(multiplier), "r"(addend))

/* Multiply-Add (High 32 bits), sets carry flag */
#define PTX_MAD_HI_CC(result, multiplicand, multiplier, addend) \
    asm volatile (                                              \
        "mad.hi.cc.u32 %0, %1, %2, %3;\n\t"                     \
        : "=r"(result)                                          \
        : "r"(multiplicand), "r"(multiplier), "r"(addend))


/* ---------------------------------------------------------
    Addition with Carry
--------------------------------------------------------- */
/* result = lhs + rhs + carry */
#define PTX_ADDC(result, lhs, rhs)  \
    asm volatile (                  \
        "addc.u32 %0, %1, %2;\n\t"  \
        : "=r"(result)              \
        : "r"(lhs), "r"(rhs))

/* result = lhs + rhs, sets carry flag */
#define PTX_ADD_CC(result, lhs, rhs)    \
    asm volatile (                      \
        "add.cc.u32 %0, %1, %2;\n\t"    \
        : "=r"(result)                  \
        : "r"(lhs), "r"(rhs))

/* result = lhs + rhs + carry, sets carry */
#define PTX_ADDC_CC(result, lhs, rhs)      \
    asm volatile (                         \
        "addc.cc.u32 %0, %1, %2;\n\t"      \
        : "=r"(result)                     \
        : "r"(lhs), "r"(rhs))


/* ---------------------------------------------------------
   Subtraction with Borrow
--------------------------------------------------------- */
/* result = lhs - rhs, sets borrow flag */
#define PTX_SUB_CC(result, lhs, rhs)     \
    asm volatile (                       \
        "sub.cc.u32 %0, %1, %2;\n\t"     \
        : "=r"(result)                   \
        : "r"(lhs), "r"(rhs))

/* result = lhs - rhs - borrow, sets borrow */
#define PTX_SUBC_CC(result, lhs, rhs)        \
    asm volatile (                           \
        "subc.cc.u32 %0, %1, %2;\n\t"        \
        : "=r"(result)                       \
        : "r"(lhs), "r"(rhs))

/* result = lhs - rhs - borrow */
#define PTX_SUBC(result, lhs, rhs)     \
    asm volatile (                     \
        "subc.u32 %0, %1, %2;\n\t"     \
        : "=r"(result)                 \
        : "r"(lhs), "r"(rhs))


// ==================================================================================================


__constant__ unsigned int MODULUS_P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

__constant__ unsigned int _X_CONSTANT[8] = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
    0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

__constant__ unsigned int _Y_CONSTANT[8] = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
    0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};

__constant__ unsigned int _1_CONSTANT[8] = {
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000001
};


// ==================================================================================================


// __device__ __forceinline__ uint32_t rotate_left_32(uint32_t value, uint32_t shift) {
//     shift &= 31;
// 
//     return (value << shift) | (value >> (32 - shift));
// }
// 
// __device__ __forceinline__ uint64_t bit_reverse_71(uint64_t x) {
//     uint64_t r = 0;
// 
//     #pragma unroll
//     for (int i = 0; i < 71; ++i) {
//         r = (r << 1) | (x & 1);
// 
//         x >>= 1;
//     }
// 
//     return r;
// }

__device__ bool isZero(const unsigned int a[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0u) return false;
    }

    return true;
}

__device__ void setZero(unsigned int a[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) a[i] = 0;
}

__device__ __forceinline__ void copyInt(const unsigned int src[8], unsigned int dest[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        dest[i] = src[i];
    }
}

// ==================================================================================================


__device__ __forceinline__ bool areAllWordsMaxUInt32(const unsigned int words[8]) {
    bool allMax = true;

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        if (words[i] != 0xFFFFFFFFu)
        {
            allMax = false;

            break;
        }
    }

    return allMax;
}

__device__ bool isGreaterOrEqual(const unsigned int a[8], const unsigned int b[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[7-i] > b[7-i]) return true;

        if (a[7-i] < b[7-i]) return false;
    }

    return true;
}


// ==================================================================================================


__device__ unsigned int rawSub(unsigned int result[8], const unsigned int lhs[8], const unsigned int rhs[8]) {
    PTX_SUB_CC(result[7], lhs[7], rhs[7]);
    PTX_SUBC_CC(result[6], lhs[6], rhs[6]);
    PTX_SUBC_CC(result[5], lhs[5], rhs[5]);
    PTX_SUBC_CC(result[4], lhs[4], rhs[4]);
    PTX_SUBC_CC(result[3], lhs[3], rhs[3]);
    PTX_SUBC_CC(result[2], lhs[2], rhs[2]);
    PTX_SUBC_CC(result[1], lhs[1], rhs[1]);
    PTX_SUBC_CC(result[0], lhs[0], rhs[0]);

    unsigned int borrowOut = 0;
    PTX_SUBC(borrowOut, 0u, 0u);

    return borrowOut;
}

__device__ void rawAdd(unsigned int result[8], const unsigned int lhs[8], const unsigned int rhs[8]) {
    PTX_ADD_CC(result[7], lhs[7], rhs[7]);
    PTX_ADDC_CC(result[6], lhs[6], rhs[6]);
    PTX_ADDC_CC(result[5], lhs[5], rhs[5]);
    PTX_ADDC_CC(result[4], lhs[4], rhs[4]);
    PTX_ADDC_CC(result[3], lhs[3], rhs[3]);
    PTX_ADDC_CC(result[2], lhs[2], rhs[2]);
    PTX_ADDC_CC(result[1], lhs[1], rhs[1]);
    PTX_ADDC_CC(result[0], lhs[0], rhs[0]);
}


// ==================================================================================================


__device__ static void SMP(const unsigned int lhs[8],const unsigned int rhs[8],unsigned int result[8]) {  
    bool needModularReduction = !isGreaterOrEqual(lhs, rhs);

    PTX_SUB_CC(result[7], lhs[7], rhs[7]);   // MS limb
    PTX_SUBC_CC(result[6], lhs[6], rhs[6]);
    PTX_SUBC_CC(result[5], lhs[5], rhs[5]);
    PTX_SUBC_CC(result[4], lhs[4], rhs[4]);
    PTX_SUBC_CC(result[3], lhs[3], rhs[3]);
    PTX_SUBC_CC(result[2], lhs[2], rhs[2]);
    PTX_SUBC_CC(result[1], lhs[1], rhs[1]);
    PTX_SUBC_CC(result[0], lhs[0], rhs[0]);   // LS limb

    unsigned int borrowOut = 0;
    PTX_SUBC(borrowOut, 0u, 0u);

    if (borrowOut)
    {
        unsigned int carryFlag = 0; // warning#550-D: variable "carryFlag" was set but never used unsigned int carryFlag = 0; ???

        PTX_ADD_CC (result[7], result[7], MODULUS_P[7]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[6], result[6], MODULUS_P[6]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[5], result[5], MODULUS_P[5]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[4], result[4], MODULUS_P[4]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[3], result[3], MODULUS_P[3]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[2], result[2], MODULUS_P[2]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[1], result[1], MODULUS_P[1]);
        PTX_ADDC(carryFlag, 0u, 0u);

        PTX_ADDC_CC(result[0], result[0], MODULUS_P[0]);
    } 
}

__device__ static void AMP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8]) {
	PTX_ADD_CC(c[7], a[7], b[7]);
	PTX_ADDC_CC(c[6], a[6], b[6]);
	PTX_ADDC_CC(c[5], a[5], b[5]);
	PTX_ADDC_CC(c[4], a[4], b[4]);
	PTX_ADDC_CC(c[3], a[3], b[3]);
	PTX_ADDC_CC(c[2], a[2], b[2]);
	PTX_ADDC_CC(c[1], a[1], b[1]);
	PTX_ADDC_CC(c[0], a[0], b[0]);

	unsigned int carry = 0;
	PTX_ADDC(carry, 0, 0);

	bool gt = false;
	for(int i = 0; i < 8; i++) {
		if(c[i] > MODULUS_P[i]) {
			gt = true;
			break;
		} else if(c[i] < MODULUS_P[i]) {
			break;
		}
	}

	if(carry || gt) {
		PTX_SUB_CC(c[7], c[7], MODULUS_P[7]);
		PTX_SUBC_CC(c[6], c[6], MODULUS_P[6]);
		PTX_SUBC_CC(c[5], c[5], MODULUS_P[5]);
		PTX_SUBC_CC(c[4], c[4], MODULUS_P[4]);
		PTX_SUBC_CC(c[3], c[3], MODULUS_P[3]);
		PTX_SUBC_CC(c[2], c[2], MODULUS_P[2]);
		PTX_SUBC_CC(c[1], c[1], MODULUS_P[1]);
		PTX_SUBC(c[0], c[0], MODULUS_P[0]);
	}
}

__device__ static void MMP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8]) {
    unsigned int high[8] = { 0 };
    unsigned int t = a[7];

    for(int i = 7; i >= 0; i--) {
        c[i] = t * b[i];
    }
    
    unsigned int dummy;   // warnin#550-D: variable "dummy" was set but never used ????
    
    PTX_ADD_CC(dummy, 0u, 0u);
    PTX_MAD_HI_CC(c[6], t, b[7], c[6]);
    PTX_MADC_HI_CC(c[5], t, b[6], c[5]);
    PTX_MADC_HI_CC(c[4], t, b[5], c[4]);
    PTX_MADC_HI_CC(c[3], t, b[4], c[3]);
    PTX_MADC_HI_CC(c[2], t, b[3], c[2]);
    PTX_MADC_HI_CC(c[1], t, b[2], c[1]);
    PTX_MADC_HI_CC(c[0], t, b[1], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[0], high[7]);

    t = a[6];
    PTX_MAD_LO_CC(c[6], t, b[7], c[6]);
    PTX_MADC_LO_CC(c[5], t, b[6], c[5]);
    PTX_MADC_LO_CC(c[4], t, b[5], c[4]);
    PTX_MADC_LO_CC(c[3], t, b[4], c[3]);
    PTX_MADC_LO_CC(c[2], t, b[3], c[2]);
    PTX_MADC_LO_CC(c[1], t, b[2], c[1]);
    PTX_MADC_LO_CC(c[0], t, b[1], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[0], high[7]);
    PTX_ADDC(high[6], high[6], 0);
    PTX_MAD_HI_CC(c[5], t, b[7], c[5]);
    PTX_MADC_HI_CC(c[4], t, b[6], c[4]);
    PTX_MADC_HI_CC(c[3], t, b[5], c[3]);
    PTX_MADC_HI_CC(c[2], t, b[4], c[2]);
    PTX_MADC_HI_CC(c[1], t, b[3], c[1]);
    PTX_MADC_HI_CC(c[0], t, b[2], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[1], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[0], high[6]);

    t = a[5];
    PTX_MAD_LO_CC(c[5], t, b[7], c[5]);
    PTX_MADC_LO_CC(c[4], t, b[6], c[4]);
    PTX_MADC_LO_CC(c[3], t, b[5], c[3]);
    PTX_MADC_LO_CC(c[2], t, b[4], c[2]);
    PTX_MADC_LO_CC(c[1], t, b[3], c[1]);
    PTX_MADC_LO_CC(c[0], t, b[2], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[1], high[7]);
    PTX_MADC_LO_CC(high[6], t, b[0], high[6]);
    PTX_ADDC(high[5], high[5], 0);
    PTX_MAD_HI_CC(c[4], t, b[7], c[4]);
    PTX_MADC_HI_CC(c[3], t, b[6], c[3]);
    PTX_MADC_HI_CC(c[2], t, b[5], c[2]);
    PTX_MADC_HI_CC(c[1], t, b[4], c[1]);
    PTX_MADC_HI_CC(c[0], t, b[3], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[2], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[1], high[6]);
    PTX_MADC_HI_CC(high[5], t, b[0], high[5]);

    t = a[4];
    PTX_MAD_LO_CC(c[4], t, b[7], c[4]);
    PTX_MADC_LO_CC(c[3], t, b[6], c[3]);
    PTX_MADC_LO_CC(c[2], t, b[5], c[2]);
    PTX_MADC_LO_CC(c[1], t, b[4], c[1]);
    PTX_MADC_LO_CC(c[0], t, b[3], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[2], high[7]);
    PTX_MADC_LO_CC(high[6], t, b[1], high[6]);
    PTX_MADC_LO_CC(high[5], t, b[0], high[5]);
    PTX_ADDC(high[4], high[4], 0);
    PTX_MAD_HI_CC(c[3], t, b[7], c[3]);
    PTX_MADC_HI_CC(c[2], t, b[6], c[2]);
    PTX_MADC_HI_CC(c[1], t, b[5], c[1]);
    PTX_MADC_HI_CC(c[0], t, b[4], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[3], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[2], high[6]);
    PTX_MADC_HI_CC(high[5], t, b[1], high[5]);
    PTX_MADC_HI_CC(high[4], t, b[0], high[4]);

    t = a[3];
    PTX_MAD_LO_CC(c[3], t, b[7], c[3]);
    PTX_MADC_LO_CC(c[2], t, b[6], c[2]);
    PTX_MADC_LO_CC(c[1], t, b[5], c[1]);
    PTX_MADC_LO_CC(c[0], t, b[4], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[3], high[7]);
    PTX_MADC_LO_CC(high[6], t, b[2], high[6]);
    PTX_MADC_LO_CC(high[5], t, b[1], high[5]);
    PTX_MADC_LO_CC(high[4], t, b[0], high[4]);
    PTX_ADDC(high[3], high[3], 0);
    PTX_MAD_HI_CC(c[2], t, b[7], c[2]);
    PTX_MADC_HI_CC(c[1], t, b[6], c[1]);
    PTX_MADC_HI_CC(c[0], t, b[5], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[4], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[3], high[6]);
    PTX_MADC_HI_CC(high[5], t, b[2], high[5]);
    PTX_MADC_HI_CC(high[4], t, b[1], high[4]);
    PTX_MADC_HI_CC(high[3], t, b[0], high[3]);

    t = a[2];
    PTX_MAD_LO_CC(c[2], t, b[7], c[2]);
    PTX_MADC_LO_CC(c[1], t, b[6], c[1]);
    PTX_MADC_LO_CC(c[0], t, b[5], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[4], high[7]);
    PTX_MADC_LO_CC(high[6], t, b[3], high[6]);
    PTX_MADC_LO_CC(high[5], t, b[2], high[5]);
    PTX_MADC_LO_CC(high[4], t, b[1], high[4]);
    PTX_MADC_LO_CC(high[3], t, b[0], high[3]);
    PTX_ADDC(high[2], high[2], 0);
    PTX_MAD_HI_CC(c[1], t, b[7], c[1]);
    PTX_MADC_HI_CC(c[0], t, b[6], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[5], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[4], high[6]);
    PTX_MADC_HI_CC(high[5], t, b[3], high[5]);
    PTX_MADC_HI_CC(high[4], t, b[2], high[4]);
    PTX_MADC_HI_CC(high[3], t, b[1], high[3]);
    PTX_MADC_HI_CC(high[2], t, b[0], high[2]);

    t = a[1];
    PTX_MAD_LO_CC(c[1], t, b[7], c[1]);
    PTX_MADC_LO_CC(c[0], t, b[6], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[5], high[7]);
    PTX_MADC_LO_CC(high[6], t, b[4], high[6]);
    PTX_MADC_LO_CC(high[5], t, b[3], high[5]);
    PTX_MADC_LO_CC(high[4], t, b[2], high[4]);
    PTX_MADC_LO_CC(high[3], t, b[1], high[3]);
    PTX_MADC_LO_CC(high[2], t, b[0], high[2]);
    PTX_ADDC(high[1], high[1], 0);
    PTX_MAD_HI_CC(c[0], t, b[7], c[0]);
    PTX_MADC_HI_CC(high[7], t, b[6], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[5], high[6]);
    PTX_MADC_HI_CC(high[5], t, b[4], high[5]);
    PTX_MADC_HI_CC(high[4], t, b[3], high[4]);
    PTX_MADC_HI_CC(high[3], t, b[2], high[3]);
    PTX_MADC_HI_CC(high[2], t, b[1], high[2]);
    PTX_MADC_HI_CC(high[1], t, b[0], high[1]);

    t = a[0];
    PTX_MAD_LO_CC(c[0], t, b[7], c[0]);
    PTX_MADC_LO_CC(high[7], t, b[6], high[7]);
    PTX_MADC_LO_CC(high[6], t, b[5], high[6]);
    PTX_MADC_LO_CC(high[5], t, b[4], high[5]);
    PTX_MADC_LO_CC(high[4], t, b[3], high[4]);
    PTX_MADC_LO_CC(high[3], t, b[2], high[3]);
    PTX_MADC_LO_CC(high[2], t, b[1], high[2]);
    PTX_MADC_LO_CC(high[1], t, b[0], high[1]);
    PTX_ADDC(high[0], high[0], 0);
    PTX_MAD_HI_CC(high[7], t, b[7], high[7]);
    PTX_MADC_HI_CC(high[6], t, b[6], high[6]);
    PTX_MADC_HI_CC(high[5], t, b[5], high[5]);
    PTX_MADC_HI_CC(high[4], t, b[4], high[4]);
    PTX_MADC_HI_CC(high[3], t, b[3], high[3]);
    PTX_MADC_HI_CC(high[2], t, b[2], high[2]);
    PTX_MADC_HI_CC(high[1], t, b[1], high[1]);
    PTX_MADC_HI_CC(high[0], t, b[0], high[0]);

    const unsigned int s = 977;
    unsigned int high7 = high[7];
    unsigned int high6 = high[6];

    PTX_ADD_CC(dummy, 0u, 0u);
    PTX_ADD_CC(c[6], high[7], c[6]);
    PTX_ADDC_CC(c[5], high[6], c[5]);
    PTX_ADDC_CC(c[4], high[5], c[4]);
    PTX_ADDC_CC(c[3], high[4], c[3]);
    PTX_ADDC_CC(c[2], high[3], c[2]);
    PTX_ADDC_CC(c[1], high[2], c[1]);
    PTX_ADDC_CC(c[0], high[1], c[0]);
    PTX_ADDC_CC(high[7], high[0], 0);
    PTX_ADDC(high[6], 0, 0);
    PTX_ADD_CC(dummy, 0u, 0u);
    PTX_MAD_LO_CC(c[7], high7, s, c[7]);
    PTX_MADC_LO_CC(c[6], high6, s, c[6]);
    PTX_MADC_LO_CC(c[5], high[5], s, c[5]);
    PTX_MADC_LO_CC(c[4], high[4], s, c[4]);
    PTX_MADC_LO_CC(c[3], high[3], s, c[3]);
    PTX_MADC_LO_CC(c[2], high[2], s, c[2]);
    PTX_MADC_LO_CC(c[1], high[1], s, c[1]);
    PTX_MADC_LO_CC(c[0], high[0], s, c[0]);
    PTX_ADDC_CC(high[7], high[7], 0);
    PTX_ADDC(high[6], high[6], 0);
    PTX_MAD_HI_CC(c[6], high7, s, c[6]);
    PTX_MADC_HI_CC(c[5], high6, s, c[5]);
    PTX_MADC_HI_CC(c[4], high[5], s, c[4]);
    PTX_MADC_HI_CC(c[3], high[4], s, c[3]);
    PTX_MADC_HI_CC(c[2], high[3], s, c[2]);
    PTX_MADC_HI_CC(c[1], high[2], s, c[1]);
    PTX_MADC_HI_CC(c[0], high[1], s, c[0]);
    PTX_MADC_HI_CC(high[7], high[0], s, high[7]);
    PTX_ADDC(high[6], high[6], 0);

    high7 = high[7];
    high6 = high[6];
    PTX_ADD_CC(dummy, 0u, 0u);
    PTX_ADD_CC(c[6], high[7], c[6]);
    PTX_ADDC_CC(c[5], high[6], c[5]);
    PTX_ADDC_CC(c[4], c[4], 0);
    PTX_ADDC_CC(c[3], c[3], 0);
    PTX_ADDC_CC(c[2], c[2], 0);
    PTX_ADDC_CC(c[1], c[1], 0);
    PTX_ADDC_CC(c[0], c[0], 0);
    PTX_ADDC(high[7], 0, 0);
    PTX_ADD_CC(dummy, 0u, 0u);
    PTX_MAD_LO_CC(c[7], high7, s, c[7]);
    PTX_MADC_LO_CC(c[6], high6, s, c[6]);
    PTX_ADDC_CC(c[5], c[5], 0);
    PTX_ADDC_CC(c[4], c[4], 0);
    PTX_ADDC_CC(c[3], c[3], 0);
    PTX_ADDC_CC(c[2], c[2], 0);
    PTX_ADDC_CC(c[1], c[1], 0);
    PTX_ADDC_CC(c[0], c[0], 0);
    PTX_ADDC(high[7], high[7], 0);
    PTX_MAD_HI_CC(c[6], high7, s, c[6]);
    PTX_MADC_HI_CC(c[5], high6, s, c[5]);
    PTX_ADDC_CC(c[4], c[4], 0);
    PTX_ADDC_CC(c[3], c[3], 0);
    PTX_ADDC_CC(c[2], c[2], 0);
    PTX_ADDC_CC(c[1], c[1], 0);
    PTX_ADDC_CC(c[0], c[0], 0);
    PTX_ADDC(high[7], high[7], 0);
    
    bool overflow = (high[7] != 0 || high[6] != 0);
    
    unsigned int borrow = rawSub(c, c, MODULUS_P);
    
    if(overflow) {
        if(!borrow) {
            rawSub(c, c, MODULUS_P);
        }
    } else {
        if(borrow) {
            rawAdd(c, c, MODULUS_P);
        }
    }
}

__device__ static void SMP(const unsigned int a[8], unsigned int b[8]) {
	MMP(a, a, b);
}

__device__ static void SMP(unsigned int x[8]) {
	unsigned int tmp[8];
	SMP(x, tmp);
	copyInt(tmp, x);
}

__device__ static void IMP(unsigned int value[8]) {
    unsigned int base[8];
    copyInt(value, base);
    unsigned int result[8] = { 0, 0, 0, 0, 0, 0, 0, 1 };
    unsigned int temp_res[8];

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);
    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);
    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);
    SMP(base);
    SMP(base);
    SMP(base);
    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);

    #pragma unroll
    for(int i = 0; i < 20; i++) {
        MMP(result, base, temp_res);
        copyInt(temp_res, result);
        SMP(base);
    }

    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    SMP(base);

    #pragma unroll
    for(int i = 0; i < 219; i++) {
        MMP(result, base, temp_res);
        copyInt(temp_res, result);
        SMP(base);
    }

    MMP(result, base, temp_res);
    copyInt(temp_res, result);

    copyInt(result, value);
}


// ==================================================================================================


__device__ void jacobianDouble(
    unsigned int X3[8],
    unsigned int Y3[8],
    unsigned int Z3[8],
    const unsigned int X1[8],
    const unsigned int Y1[8],
    const unsigned int Z1[8]
) {
    if (isZero(Z1) || isZero(Y1)) {
        setZero(X3);
        setZero(Y3);
        setZero(Z3);
        return;
    }

    unsigned int A[8], B[8], C[8], D[8], E[8], F[8], tmp[8];

    // A = X1^2
    SMP(X1, A);

    // B = Y1^2
    SMP(Y1, B);

    // C = B^2
    SMP(B, C);

    // D = 2*((X1+B)^2 - A - C)
    AMP(X1, B, tmp);
    unsigned int tmp_sq[8];
    SMP(tmp, tmp_sq);
    copyInt(tmp_sq, tmp);
    SMP(tmp, A, tmp);
    SMP(tmp, C, tmp);
    AMP(tmp, tmp, D);

    // E = 3*A
    AMP(A, A, E);
    AMP(E, A, E);

    // F = E^2
    SMP(E, F);

    // X3 = F - 2*D
    AMP(D, D, tmp);
    SMP(F, tmp, X3);

    // Y3 = E*(D - X3) - 8*C
    SMP(D, X3, tmp);
    unsigned int tmp_mul[8];
    MMP(E, tmp, tmp_mul);
    AMP(C, C, C);
    AMP(C, C, C);
    AMP(C, C, C); // C = 8*C
    SMP(tmp_mul, C, Y3);

    // Z3 = 2*Y1*Z1
    MMP(Y1, Z1, Z3);
    AMP(Z3, Z3, Z3);
}

__device__ void jacobianAdd(
    unsigned int X3[8],
    unsigned int Y3[8],
    unsigned int Z3[8],
    const unsigned int X1[8],
    const unsigned int Y1[8],
    const unsigned int Z1[8],
    const unsigned int X2[8],
    const unsigned int Y2[8],
    const unsigned int Z2[8]
) {
    if (isZero(Z1)) {
        copyInt(X2, X3);
        copyInt(Y2, Y3);
        copyInt(Z2, Z3);
        return;
    }
    if (isZero(Z2)) {
        copyInt(X1, X3);
        copyInt(Y1, Y3);
        copyInt(Z1, Z3);
        return;
    }

    unsigned int Z1Z1[8], Z2Z2[8], U1[8], U2[8], S1[8], S2[8];
    unsigned int H[8], R[8], H2[8], H3[8], tmp[8];

    SMP(Z1, Z1Z1);
    SMP(Z2, Z2Z2);

    MMP(X1, Z2Z2, U1);
    MMP(X2, Z1Z1, U2);

    MMP(Z2, Z2Z2, tmp);
    MMP(Y1, tmp, S1);

    MMP(Z1, Z1Z1, tmp);
    MMP(Y2, tmp, S2);

    SMP(U2, U1, H);
    SMP(S2, S1, R);

    if (isZero(H)) {
        if (isZero(R)) {
            jacobianDouble(X3, Y3, Z3, X1, Y1, Z1);
            return;
        } else {
            setZero(X3);
            setZero(Y3);
            setZero(Z3);
            return;
        }
    }

    SMP(H, H2);
    MMP(H, H2, H3);

    MMP(U1, H2, tmp);

    SMP(R, X3);
    SMP(X3, H3, X3);
    SMP(X3, tmp, X3);
    SMP(X3, tmp, X3);

    SMP(tmp, X3, tmp);
    MMP(R, tmp, Y3);
    MMP(S1, H3, tmp);
    SMP(Y3, tmp, Y3);

    MMP(Z1, Z2, Z3);
    unsigned int tmp_z[8];
    MMP(Z3, H, tmp_z);
    copyInt(tmp_z, Z3);
}

__device__ void jacobianMixedAdd(
    unsigned int X3[8],
    unsigned int Y3[8],
    unsigned int Z3[8],
    const unsigned int X1[8],
    const unsigned int Y1[8],
    const unsigned int Z1[8],
    const unsigned int X2[8],
    const unsigned int Y2[8]
) {
    if (isZero(Z1)) {
        copyInt(X2, X3);
        copyInt(Y2, Y3);
        copyInt(_1_CONSTANT, Z3);
        return;
    }

    unsigned int Z1Z1[8], U2[8], S2[8], H[8], HH[8], V[8], H3[8], tmp[8],R[8];
    SMP(Z1, Z1Z1);
    MMP(X2, Z1Z1, U2);
    MMP(Y2, Z1Z1, tmp);
    MMP(tmp, Z1, S2);
    SMP(U2, X1, H);

    if (isZero(H)) {
        SMP(S2, Y1, tmp);

        if (isZero(tmp)) {
            jacobianDouble(X3, Y3, Z3, X1, Y1, Z1);
            return;
        } else {
            setZero(X3);
            setZero(Y3);
            setZero(Z3);
            return;
        }
    }

    SMP(H, HH);  // HH = H²
    MMP(H, HH, H3); // H3 = H³
    MMP(X1, HH, V); // V = X1 * H²
    SMP(S2, Y1, R); // R = S2 - Y1
    SMP(R, X3);  // X3 = R²
    SMP(X3, H3, X3); // X3 -= H³
    SMP(X3, V, tmp);
    SMP(tmp, V, X3); // X3 -= 2*V
    SMP(V, X3, tmp); // tmp = V - X3
    MMP(R, tmp, Y3); // Y3 = R * (V - X3)
    MMP(Y1, H3, tmp); // tmp = Y1 * H³
    SMP(Y3, tmp, Y3); // Y3 -= Y1 * H³
    MMP(Z1, H, Z3);
}


// ==================================================================================================


__device__ void scalarMultiplication(unsigned int pubX[8], unsigned int pubY[8], const uint8_t scalar[32]) {
    // Initialize point to generator point in Jacobian coordinates
    unsigned int X[8], Y[8], Z[8];
    setZero(X);
    setZero(Y);
    setZero(Z);

    // Perform scalar multiplication
    for (int byte_idx = 0; byte_idx < 32; byte_idx++) {
        uint8_t byte = scalar[31 - byte_idx];
        for (int bit_idx = 7; bit_idx >= 0; bit_idx--) {
            // Double point
            unsigned int X2[8], Y2[8], Z2[8];
            jacobianDouble(X2, Y2, Z2, X, Y, Z);
            copyInt(X2, X);
            copyInt(Y2, Y);
            copyInt(Z2, Z);

            // Add point if bit is set
            bool bit = (byte >> bit_idx) & 1;
            if (bit) {
                unsigned int tempX[8], tempY[8], tempZ[8];
                jacobianMixedAdd(tempX, tempY, tempZ, X, Y, Z, _X_CONSTANT, _Y_CONSTANT);
                copyInt(tempX, X);
                copyInt(tempY, Y);
                copyInt(tempZ, Z);
            }
        }
    }

    // Convert to affine
    if (isZero(Z)) {
        setZero(pubX);
        setZero(pubY);
        return;
    }

    unsigned int denominator[8], inv_z[8];
    SMP(Z, denominator); // denom = Z²
    copyInt(Z, inv_z);
    IMP(inv_z); // inv_z = 1/Z
    IMP(denominator); // denom = 1/Z²
    MMP(X, denominator, pubX); // x = X / Z²

    unsigned int temp[8];
    MMP(Y, denominator, temp); // temp = Y / Z²
    MMP(temp, inv_z, pubY); // y = Y / Z³
}

__device__ void getCompressedPubKey(uint8_t *output, const unsigned int x[8], const unsigned int y[8]) {
    // HSB
    output[0] = (y[7] & 1u) ? 0x03u : 0x02u;
    for (int i = 0; i < 8; ++i) {
        uint32_t word = x[i];
        int base = 1 + i * 4;
        output[base + 0] = static_cast<uint8_t>(word >> 24);
        output[base + 1] = static_cast<uint8_t>(word >> 16);
        output[base + 2] = static_cast<uint8_t>(word >> 8);
        output[base + 3] = static_cast<uint8_t>(word);
    }
}

extern "C" __global__ void generate_and_check_keys(
    bool search_mode,
    uint64_t start_i_low, 
    uint64_t start_i_high, 
    uint64_t count, 
    uint64_t a_low, 
    uint64_t a_high, 
    uint64_t b_low, 
    uint64_t b_high, 
    uint64_t range_start_low, 
    uint64_t range_start_high, 
    unsigned long long *out_found_index
) {
    const bool debug = DEBUG_TEST_MODE;

    uint64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    // ------------------------------------------------------------------
    // DEBUG MODE: force only thread 0 in block 0 to run (single key)
    // ------------------------------------------------------------------
    if (debug) {
        tid    = 0;                     // only this virtual thread runs
        stride = 1;                     // no striding → processes offset 0 only
    }

    __uint128_t base_i      = ((__uint128_t)start_i_high << 64) | start_i_low;
    __uint128_t range_start = ((__uint128_t)range_start_high << 64) | range_start_low;

    // ------------------------------------------------------------------
    // Main loop – identical behaviour in both modes
    // ------------------------------------------------------------------
    #pragma unroll
    for (uint64_t offset = tid; offset < count; offset += stride) {

        if (*out_found_index != ULLONG_MAX) return;

        __uint128_t i = base_i + offset;

        __uint128_t k;
    
        // ======= 0. SEARCH MODE =======
        if (search_mode) {
            k = range_start + i;
        } else {
            // PCG mixing
            __uint128_t a = ((__uint128_t)a_high << 64) | a_low;
            __uint128_t b = ((__uint128_t)b_high << 64) | b_low;
            __uint128_t y = a * i + b;
            y &= (((__uint128_t)0x7FULL << 64) | 0xFFFFFFFFFFFFFFFFULL);

            uint64_t high = (uint64_t)(y >> 64);
            uint64_t low  = (uint64_t)y;
            low ^= high;
            low *= 0x9E3779B97F4A7C15ULL;
            high = ((high << 32) | (low >> 32)) ^ low;
            high *= 0xBDD1F3B71727E72BULL;
            y = ((__uint128_t)high << 64) | low;
            y ^= y >> 67;

            // Bound back to mod 2^71
            y &= (((__uint128_t)0x7FULL << 64) | 0xFFFFFFFFFFFFFFFFULL);

            k = range_start + y;
        }
    
    
        // ======= 1. Private key =======
        uint8_t priv[32] = {0};
        
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            priv[i] = (k >> (8 * i)) & 0xFF;
        }
    
        if (debug) {
            printf("\n--- Processing key index=%llu ---\n", (unsigned long long)(uint64_t)i);
            printf("Private key (big-endian hex): ");
            for (int b = 31; b >= 0; b--) printf("%02x", priv[b]);
            printf("\n");
        }
    
    
        // ======= 2. Public key =======
        unsigned int pubX[8], pubY[8];
        scalarMultiplication(pubX, pubY, priv);
    
        uint8_t pub_compressed[33];
        getCompressedPubKey(pub_compressed, pubX, pubY);
    
        if (debug) {
            printf("Compressed pubkey: ");
            for (int b = 0; b < 33; ++b) printf("%02x", pub_compressed[b]);
            printf("\n");    
        }
    
        // ======= 3. SHA256(publicKey) =======
        unsigned int sha_digest[8];
        unsigned int yParity = (pub_compressed[0] == 0x03) ? 1 : 0;
        sha256::sha256PublicKeyCompressed(pubX, yParity, sha_digest);
    
        uint8_t sha_out[32];
    
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t word = sha_digest[i];
            int off = i * 4;
            sha_out[off + 0] = (word >> 24) & 0xff;
            sha_out[off + 1] = (word >> 16) & 0xff;
            sha_out[off + 2] = (word >>  8) & 0xff;
            sha_out[off + 3] =  word        & 0xff;
        }
    
        if (debug) {
            printf("SHA256 digest: ");
            for (int b = 0; b < 32; ++b) printf("%02x", sha_out[b]);
            printf("\n");
        }
    
        unsigned int sha_words[8];
    
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int off = i * 4;
            sha_words[i] =
                (uint32_t)sha_out[off + 0] |
                (uint32_t)sha_out[off + 1] << 8 |
                (uint32_t)sha_out[off + 2] << 16 |
                (uint32_t)sha_out[off + 3] << 24;
        }
        
        // ======= 4. RIPEMD160(SHA256) =======
        unsigned int ripe_words[5];
        ripemd160::ripemd160sha256(sha_words, ripe_words);
    
        if (debug) {
            printf("ripe_words (LE words): ");
            for (int i = 0; i < 5; i++)
               printf("%08x ", ripe_words[i]);
            printf("\n");
        }
    
        uint8_t hash160[20];
    
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            uint32_t w = ripe_words[i];
            int off = i * 4;
    
            hash160[off + 0] =  w        & 0xff;
            hash160[off + 1] = (w >>  8) & 0xff;
            hash160[off + 2] = (w >> 16) & 0xff;
            hash160[off + 3] = (w >> 24) & 0xff;
        }   
    
        if (debug) {
            printf("Address hash: ");
            for (int b = 0; b < 20; ++b) printf("%02x", hash160[b]);
            printf("\n");
        }
    
    
        // ======= CHECK =======
        bool match = true;
        #pragma unroll
        for (int j = 0; j < 20; ++j) {
            if (hash160[j] != TARGET_H160[j]) {
                match = false;
                break;
            }
        }
    
    
        // ======= CONGRATULATION =======
        if (match) {
            atomicExch(out_found_index, i);
            return;
        }
    }

}