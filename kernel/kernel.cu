#include <stdint.h>


struct JacobianPoint {
    unsigned int X[8];
    unsigned int Y[8];
    unsigned int Z[8];
};


// BitCrack Library
namespace sha256 {
    #include "sha256.cuh"
}

// BitCrack Library
namespace ripemd160 {
    #include "ripemd160.cuh"
}


// ==================================================================================================


__constant__ bool DEBUG_TEST_MODE = false;

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
// __constant__ uint8_t TARGET_H160[20] = {
//     0xdb, 0x6e, 0xb1, 0x82, 0x09, 0x4f, 0xca, 0x54, 0x94, 0x78,
//     0xa2, 0xe5, 0x8a, 0x16, 0xcf, 0x13, 0xb4, 0xdb, 0x3e, 0x2e
// };

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
__constant__ uint8_t TARGET_H160[20] = {
    0xbf, 0x74, 0x13, 0xe8, 0xdf, 0x4e, 0x7a, 0x34, 0xce, 0x9d,
    0xc1, 0x3e, 0x2f, 0x26, 0x48, 0x78, 0x3e, 0xc5, 0x4a, 0xdb
};


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

__constant__ unsigned int PRECOMP_X[64][8] = {
    { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, // 0G
    { 0x79be667e, 0xf9dcbbac, 0x55a06295, 0xce870b07, 0x029bfcdb, 0x2dce28d9, 0x59f2815b, 0x16f81798 }, // 1G
    { 0xc6047f94, 0x41ed7d6d, 0x3045406e, 0x95c07cd8, 0x5c778e4b, 0x8cef3ca7, 0xabac09b9, 0x5c709ee5 }, // 2G
    { 0xf9308a01, 0x9258c310, 0x49344f85, 0xf89d5229, 0xb531c845, 0x836f99b0, 0x8601f113, 0xbce036f9 }, // 3G
    { 0xe493dbf1, 0xc10d80f3, 0x581e4904, 0x930b1404, 0xcc6c1390, 0x0ee07584, 0x74fa94ab, 0xe8c4cd13 }, // 4G
    { 0x2f8bde4d, 0x1a072093, 0x55b4a725, 0x0a5c5128, 0xe88b84bd, 0xdc619ab7, 0xcba8d569, 0xb240efe4 }, // 5G
    { 0xfff97bd5, 0x755eeea4, 0x20453a14, 0x355235d3, 0x82f6472f, 0x8568a18b, 0x2f057a14, 0x60297556 }, // 6G
    { 0x5cbdf064, 0x6e5db4ea, 0xa398f365, 0xf2ea7a0e, 0x3d419b7e, 0x0330e39c, 0xe92bdded, 0xcac4f9bc }, // 7G
    { 0x2f01e5e1, 0x5cca351d, 0xaff3843f, 0xb70f3c2f, 0x0a1bdd05, 0xe5af888a, 0x67784ef3, 0xe10a2a01 }, // 8G
    { 0xacd484e2, 0xf0c7f653, 0x09ad178a, 0x9f559abd, 0xe0979697, 0x4c57e714, 0xc35f110d, 0xfc27ccbe }, // 9G
    { 0xa0434d9e, 0x47f3c862, 0x35477c7b, 0x1ae6ae5d, 0x3442d49b, 0x1943c2b7, 0x52a68e2a, 0x47e247c7 }, // 10G
    { 0x774ae7f8, 0x58a9411e, 0x5ef4246b, 0x70c65aac, 0x5649980b, 0xe5c17891, 0xbbec1789, 0x5da008cb }, // 11G
    { 0xd01115d5, 0x48e7561b, 0x15c38f00, 0x4d734633, 0x687cf441, 0x9620095b, 0xc5b0f470, 0x70afe85a }, // 12G
    { 0xf28773c2, 0xd975288b, 0xc7d1d205, 0xc3748651, 0xb075fbc6, 0x610e58cd, 0xdeeddf8f, 0x19405aa8 }, // 13G
    { 0x499fdf9e, 0x895e719c, 0xfd64e67f, 0x07d38e32, 0x26aa7b63, 0x678949e6, 0xe49b241a, 0x60e823e4 }, // 14G
    { 0xd7924d4f, 0x7d43ea96, 0x5a465ae3, 0x095ff411, 0x31e5946f, 0x3c85f79e, 0x44adbcf8, 0xe27e080e }, // 15G
    { 0xe60fce93, 0xb59e9ec5, 0x3011aabc, 0x21c23e97, 0xb2a31369, 0xb87a5ae9, 0xc44ee89e, 0x2a6dec0a }, // 16G
    { 0xdefdea4c, 0xdb677750, 0xa420fee8, 0x07eacf21, 0xeb9898ae, 0x79b97687, 0x66e4faa0, 0x4a2d4a34 }, // 17G
    { 0x5601570c, 0xb47f238d, 0x2b0286db, 0x4a990fa0, 0xf3ba28d1, 0xa319f5e7, 0xcf55c2a2, 0x444da7cc }, // 18G
    { 0x2b4ea0a7, 0x97a443d2, 0x93ef5cff, 0x444f4979, 0xf06acfeb, 0xd7e86d27, 0x74756561, 0x38385b6c }, // 19G
    { 0x4ce119c9, 0x6e2fa357, 0x200b559b, 0x2f7dd5a5, 0xf02d5290, 0xaff74b03, 0xf3e471b2, 0x73211c97 }, // 20G
    { 0x352bbf4a, 0x4cdd1256, 0x4f93fa33, 0x2ce33330, 0x1d9ad402, 0x71f81071, 0x81340aef, 0x25be59d5 }, // 21G
    { 0x421f5fc9, 0xa2106544, 0x5c96fdb9, 0x1c0c1e2f, 0x2431741c, 0x72713b4b, 0x99ddcb31, 0x6f31e9fc }, // 22G
    { 0x2fa2104d, 0x6b38d11b, 0x02300105, 0x59879124, 0xe42ab8df, 0xeff5ff29, 0xdc9cdadd, 0x4ecacc3f }, // 23G
    { 0xfe72c435, 0x413d33d4, 0x8ac09c91, 0x61ba8b09, 0x68321543, 0x9d62b794, 0x0502bda8, 0xb202e6ce }, // 24G
    { 0x9248279b, 0x09b4d68d, 0xab21a9b0, 0x66edda83, 0x263c3d84, 0xe09572e2, 0x69ca0cd7, 0xf5453714 }, // 25G
    { 0x6687cdb5, 0xb650d558, 0xf40cbdef, 0xc8e40997, 0xc03fe1b2, 0xabb84088, 0x5e5cad81, 0x710c4c8a }, // 26G
    { 0xdaed4f2b, 0xe3a8bf27, 0x8e70132f, 0xb0beb752, 0x2f570e14, 0x4bf615c0, 0x7e996d44, 0x3dee8729 }, // 27G
    { 0x55eb67d7, 0xb7238a70, 0xa7fa6f64, 0xd5dc3c82, 0x6b31536d, 0xa6eb344d, 0xc39a66f9, 0x04f97968 }, // 28G
    { 0xc44d12c7, 0x065d812e, 0x8acf28d7, 0xcbb19f90, 0x11ecd9e9, 0xfdf281b0, 0xe6a3b5e8, 0x7d22e7db }, // 29G
    { 0x6d2b085e, 0x9e382ed1, 0x0b69fc31, 0x1a03f864, 0x1ccfff21, 0x574de092, 0x7513a49d, 0x9a688a00 }, // 30G
    { 0x6a245bf6, 0xdc698504, 0xc89a20cf, 0xded60853, 0x152b6953, 0x36c28063, 0xb61c65cb, 0xd269e6b4 }, // 31G
    { 0xd30199d7, 0x4fb5a22d, 0x47b6e054, 0xe2f378ce, 0xdacffcb8, 0x9904a61d, 0x75d0dbd4, 0x07143e65 }, // 32G
    { 0x1697ffa6, 0xfd9de627, 0xc077e3d2, 0xfe541084, 0xce13300b, 0x0bec1146, 0xf95ae57f, 0x0d0bd6a5 }, // 33G
    { 0x1be68a5a, 0x028f2601, 0xd0e80d46, 0x8c344ba3, 0x31d611b9, 0x6c358b60, 0x32e8b4da, 0x0547fc11 }, // 34G
    { 0x605bdb01, 0x9981718b, 0x986d0f07, 0xe834cb0d, 0x9deb8360, 0xffb7f61d, 0xf982345e, 0xf27a7479 }, // 35G
    { 0xe0392cfa, 0x338aaf2f, 0x0b56c563, 0xe3e5e67a, 0x5d5fefe3, 0x388f85d9, 0x0c899da2, 0x0f0198f9 }, // 36G
    { 0x62d14dab, 0x4150bf49, 0x7402fdc4, 0x5a215e10, 0xdcb01c35, 0x4959b10c, 0xfe31c7e9, 0xd87ff33d }, // 37G
    { 0xb699a30e, 0x6e184cdf, 0xa88ac16c, 0x7d80bffd, 0x38e2e1fc, 0x705821ea, 0x69cd5fdf, 0x1691fff7 }, // 38G
    { 0x80c60ad0, 0x040f27da, 0xde5b4b06, 0xc408e56b, 0x2c50e9f5, 0x6b9b8b42, 0x5e555c2f, 0x86308b6f }, // 39G
    { 0x91de2f6b, 0xb67b1113, 0x9f0e2120, 0x3041bf08, 0x0eacf59a, 0x33d99cd9, 0xf1929141, 0xbb0b4d0b }, // 40G
    { 0x7a9375ad, 0x6167ad54, 0xaa74c634, 0x8cc54d34, 0x4cc5dc94, 0x87d84704, 0x9d5eabb0, 0xfa03c8fb }, // 41G
    { 0xfe8d1eb1, 0xbcb3432b, 0x1db5833f, 0xf5f2226d, 0x9cb5e65c, 0xee430558, 0xc18ed3a3, 0xc86ce1af }, // 42G
    { 0xd528ecd9, 0xb696b54c, 0x907a9ed0, 0x45447a79, 0xbb408ec3, 0x9b68df50, 0x4bb51f45, 0x9bc3ffc9 }, // 43G
    { 0x5d045857, 0x332d5b9e, 0x54151473, 0x1622af8d, 0x60c18016, 0x5d971a61, 0xe06b70a9, 0xb3834765 }, // 44G
    { 0x049370a4, 0xb5f43412, 0xea25f514, 0xe8ecdad0, 0x5266115e, 0x4a7ecb13, 0x87231808, 0xf8b45963 }, // 45G
    { 0xf8b0b03d, 0x44112259, 0xf903b3d1, 0x00e3950d, 0x980fdde9, 0xc7e85701, 0xc16baedc, 0x90235717 }, // 46G
    { 0x77f23093, 0x6ee88cbb, 0xd73df930, 0xd64702ef, 0x881d811e, 0x0e1498e2, 0xf1c13eb1, 0xfc345d74 }, // 47G
    { 0x6eca335d, 0x9645307d, 0xb441656e, 0xf4e65b4b, 0xfc579b27, 0x452bebc1, 0x9bd870aa, 0x1118e5c3 }, // 48G
    { 0xf2dac991, 0xcc4ce4b9, 0xea44887e, 0x5c7c0bce, 0x58c80074, 0xab9d4dba, 0xeb28531b, 0x7739f530 }, // 49G
    { 0x29757774, 0xcc6f3be1, 0xd5f1774a, 0xefa8f02e, 0x50bc6440, 0x4230e7a6, 0x7e8fde79, 0xbd559a9a }, // 50G
    { 0x463b3d9f, 0x662621fb, 0x1b4be8fb, 0xbe252012, 0x5a216cdf, 0xc9dae3de, 0xbcba4850, 0xc690d45b }, // 51G
    { 0x2b22efda, 0x32491a9e, 0x0294339c, 0xa3da761f, 0x7d36cfc8, 0x814c1b29, 0xca731921, 0x025ff695 }, // 52G
    { 0xf16f8042, 0x44e46e2a, 0x09232d4a, 0xff3b5997, 0x6b98fac1, 0x4328a2d1, 0xa32496b4, 0x9998f247 }, // 53G
    { 0x4fdcb8fa, 0x639cee44, 0x1c8331fd, 0x47a2e5ff, 0x3447be24, 0x500ca7a5, 0x24997106, 0x7c1d506b }, // 54G
    { 0xcaf75427, 0x2dc84563, 0xb0352b7a, 0x14311af5, 0x5d245315, 0xace27c65, 0x369e15f7, 0x151d41d1 }, // 55G
    { 0xbce74de6, 0xd5f98dc0, 0x27740c2b, 0xbff05b6a, 0xafe5fd8d, 0x103f827e, 0x48894a2b, 0xd3460117 }, // 56G
    { 0x2600ca4b, 0x282cb986, 0xf85d0f17, 0x09979d8b, 0x44a09c07, 0xcb86d7c1, 0x24497bc8, 0x6f082120 }, // 57G
    { 0x45562f03, 0x3698faca, 0x1540cbc9, 0xbf962cf4, 0x764c1ef4, 0x094ee4b6, 0x742b761c, 0x49b46d3b }, // 58G
    { 0x7635ca72, 0xd7e8432c, 0x338ec53c, 0xd12220bc, 0x01c48685, 0xe24f7dc8, 0xc602a774, 0x6998e435 }, // 59G
    { 0x01257e93, 0xa78a5b7d, 0x8fe0cf28, 0xff1d8822, 0x350c778a, 0xc8a30e57, 0xd2acfc4d, 0x5fb8c192 }, // 60G
    { 0x754e3239, 0xf325570c, 0xdbbf4a87, 0xdeee8a66, 0xb7f2b334, 0x79d468fb, 0xc1a50743, 0xbf56cc18 }, // 61G
    { 0x108443b9, 0x48d15535, 0x84a27133, 0x3f7fbd04, 0x3c4d66a9, 0x1706edec, 0xbf07f689, 0x4c04f299 }, // 62G
    { 0xe3e6bd10, 0x71a1e96a, 0xff57859c, 0x82d570f0, 0x33080066, 0x1d1c952f, 0x9fe26946, 0x91d9b9e8 }, // 63G
};

__constant__ unsigned int PRECOMP_Y[64][8] = {
    { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, // 0G
    { 0x483ada77, 0x26a3c465, 0x5da4fbfc, 0x0e1108a8, 0xfd17b448, 0xa6855419, 0x9c47d08f, 0xfb10d4b8 }, // 1G
    { 0x1ae168fe, 0xa63dc339, 0xa3c58419, 0x466ceaee, 0xf7f63265, 0x3266d0e1, 0x236431a9, 0x50cfe52a }, // 2G
    { 0x388f7b0f, 0x632de814, 0x0fe337e6, 0x2a37f356, 0x6500a999, 0x34c2231b, 0x6cb9fd75, 0x84b8e672 }, // 3G
    { 0x51ed993e, 0xa0d455b7, 0x5642e209, 0x8ea51448, 0xd967ae33, 0xbfbdfe40, 0xcfe97bdc, 0x47739922 }, // 4G
    { 0xd8ac2226, 0x36e5e3d6, 0xd4dba9dd, 0xa6c9c426, 0xf788271b, 0xab0d6840, 0xdca87d3a, 0xa6ac62d6 }, // 5G
    { 0xae12777a, 0xacfbb620, 0xf3be9601, 0x7f45c560, 0xde80f0f6, 0x518fe4a0, 0x3c870c36, 0xb075f297 }, // 6G
    { 0x6aebca40, 0xba255960, 0xa3178d6d, 0x861a54db, 0xa813d0b8, 0x13fde7b5, 0xa5082628, 0x087264da }, // 7G
    { 0x5c4da8a7, 0x41539949, 0x293d082a, 0x132d13b4, 0xc2e213d6, 0xba5b7617, 0xb5da2cb7, 0x6cbde904 }, // 8G
    { 0xcc338921, 0xb0a7d9fd, 0x64380971, 0x763b61e9, 0xadd888a4, 0x375f8e0f, 0x05cc262a, 0xc64f9c37 }, // 9G
    { 0x893aba42, 0x5419bc27, 0xa3b6c7e6, 0x93a24c69, 0x6f794c2e, 0xd877a159, 0x3cbee53b, 0x037368d7 }, // 10G
    { 0xd984a032, 0xeb6b5e19, 0x0243dd56, 0xd7b7b365, 0x372db1e2, 0xdff9d6a8, 0x301d74c9, 0xc953c61b }, // 11G
    { 0xa9f34ffd, 0xc815e0d7, 0xa8b64537, 0xe17bd815, 0x79238c5d, 0xd9a86d52, 0x6b051b13, 0xf4062327 }, // 12G
    { 0x0ab0902e, 0x8d880a89, 0x758212eb, 0x65cdaf47, 0x3a1a06da, 0x521fa91f, 0x29b5cb52, 0xdb03ed81 }, // 13G
    { 0xcac2f6c4, 0xb54e8551, 0x90f044e4, 0xa7b3d464, 0x464279c2, 0x7a3f95bc, 0xc65f40d4, 0x03a13f5b }, // 14G
    { 0x581e2872, 0xa86c72a6, 0x83842ec2, 0x28cc6def, 0xea40af2b, 0xd896d3a5, 0xc504dc9f, 0xf6a26b58 }, // 15G
    { 0xf7e35073, 0x99e59592, 0x9db99f34, 0xf5793710, 0x1296891e, 0x44d23f0b, 0xe1f32cce, 0x69616821 }, // 16G
    { 0x4211ab06, 0x94635168, 0xe997b0ea, 0xd2a93dae, 0xced1f4a0, 0x4a95c0f6, 0xcfb199f6, 0x9e56eb77 }, // 17G
    { 0xc136c1dc, 0x0cbeb930, 0xe9e29804, 0x3589351d, 0x81d8e0bc, 0x736ae2a1, 0xf5192e5e, 0x8b061d58 }, // 18G
    { 0x85e89bc0, 0x37945d93, 0xb343083b, 0x5a1c8613, 0x1a01f60c, 0x50269763, 0xb570c854, 0xe5c09b7a }, // 19G
    { 0x12ba26dc, 0xb10ec162, 0x5da61fa1, 0x0a844c67, 0x61629482, 0x71d96967, 0x450288ee, 0x9233dc3a }, // 20G
    { 0x321eb407, 0x5348f534, 0xd59c1825, 0x9dda3e1f, 0x4a1b3b2e, 0x71b1039c, 0x67bd3d8b, 0xcf81998c }, // 21G
    { 0x2b90f16d, 0x11dabdb6, 0x16f6db7e, 0x225d1e14, 0x743034b3, 0x7b223115, 0xdb20717a, 0xd1cd6781 }, // 22G
    { 0x02de1068, 0x295dd865, 0xb6456933, 0x5bd5dd80, 0x181d70ec, 0xfc882648, 0x423ba76b, 0x532b7d67 }, // 23G
    { 0x6851de06, 0x7ff24a68, 0xd3ab47e0, 0x9d729981, 0x01dc88e3, 0x6b4a9d22, 0x978ed2fb, 0xcf58c5bf }, // 24G
    { 0x73016f7b, 0xf234aade, 0x5d1aa71b, 0xdea2b1ff, 0x3fc0de2a, 0x887912ff, 0xe54a32ce, 0x97cb3402 }, // 25G
    { 0x3fd502b3, 0x111178b1, 0x1a1fa873, 0x825c7200, 0x0ef8e529, 0xf033f272, 0xb32e83b2, 0x5c83ad64 }, // 26G
    { 0xa69dce4a, 0x7d6c98e8, 0xd4a1aca8, 0x7ef8d700, 0x3f83c230, 0xf3afa726, 0xab40e522, 0x90be1c55 }, // 27G
    { 0x7d916a47, 0xb2b58140, 0x0b1e718b, 0xf4042585, 0x40973bce, 0x1c95052d, 0xd0689f2f, 0x493be3c8 }, // 28G
    { 0x2119a460, 0xce326cdc, 0x76c45926, 0xc982fdac, 0x0e106e86, 0x1edf61c5, 0xa039063f, 0x0e0e6482 }, // 29G
    { 0xacb82eb9, 0x3309ad1c, 0xc739ddfa, 0x33604a83, 0x776238aa, 0x0bd5ff24, 0x8dbac47a, 0x17f388fb }, // 30G
    { 0xe022cf42, 0xc2bd4a70, 0x8b3f5126, 0xf16a24ad, 0x8b33ba48, 0xd0423b6e, 0xfd5e6348, 0x100d8a82 }, // 31G
    { 0x95038d9d, 0x0ae3d5c3, 0xb3d6dec9, 0xe9838065, 0x1f760cc3, 0x64ed8196, 0x05b3ff1f, 0x24106ab9 }, // 32G
    { 0xb9c398f1, 0x86806f5d, 0x27561506, 0xe4557433, 0xa2cf1500, 0x9e498ae7, 0xadee9d63, 0xd01b2396 }, // 33G
    { 0xbebc4751, 0x1ade7308, 0xb3ca6265, 0xf9400779, 0xc076329c, 0x75146bc6, 0xff1822f5, 0xd1f30e79 }, // 34G
    { 0x02972d2d, 0xe4f8d206, 0x81a78d93, 0xec96fe23, 0xc26bfae8, 0x4fb14db4, 0x3b01e1e9, 0x056b8c49 }, // 35G
    { 0x76d45864, 0x2a2c93ad, 0xee7a347a, 0x5e4681f9, 0xbb5b10f4, 0xbd8aa51e, 0xdfd6e3f5, 0x0e7da3ac }, // 36G
    { 0x80fc06bd, 0x8cc5b010, 0x98088a19, 0x50eed0db, 0x01aa1329, 0x67ab4722, 0x35f56424, 0x83b25eaf }, // 37G
    { 0xd505700c, 0x51d860ce, 0x5a096ee6, 0x37ebed3b, 0xd9d72681, 0x26c76a16, 0xb745bc31, 0x8a51ab04 }, // 38G
    { 0x1c38303f, 0x1cc5c30f, 0x26e66bad, 0x7fe72f70, 0xa65eed4c, 0xbe7024eb, 0x1aa01f56, 0x430bd57a }, // 39G
    { 0xeb9ef6c0, 0x31eed31d, 0xe34e7a10, 0x09f87251, 0x55b03158, 0x202a9d3e, 0x9a9a2e83, 0x124a7899 }, // 40G
    { 0x0d0e3fa9, 0xeca87269, 0x09559e0d, 0x79269046, 0xbdc59ea1, 0x0c70ce2b, 0x02d499ec, 0x224dc7f7 }, // 41G
    { 0x07b158f2, 0x44cd0de2, 0x134ac7c1, 0xd371cffb, 0xfae4db40, 0x801a2572, 0xe531c573, 0xcda9b5b4 }, // 42G
    { 0xeecf4125, 0x3136e5f9, 0x9966f218, 0x81fd656e, 0xbc434540, 0x5c520dbc, 0x063465b5, 0x21409933 }, // 43G
    { 0xdb2ba972, 0x802d45fd, 0x2decbab8, 0xd098a8c2, 0xa1d1f347, 0x61c6cf26, 0x1879a7ca, 0xbf06fb68 }, // 44G
    { 0x758f3f41, 0xafd6ed42, 0x8b3081b0, 0x512fd62a, 0x54c3f3af, 0xbb5b6764, 0xb653052a, 0x12949c9a }, // 45G
    { 0xbd8e9dc3, 0x01d9adc9, 0x6be1883b, 0x362f123b, 0xd0a98692, 0x8ac79972, 0x517ab5c2, 0x46242203 }, // 46G
    { 0x958ef42a, 0x7886b640, 0x0a08266e, 0x9ba1b378, 0x96c95330, 0xd97077cb, 0xbe8eb3c7, 0x671c60d6 }, // 47G
    { 0xd50123b5, 0x7a7a0710, 0x592f5790, 0x74b875a0, 0x3a496a3a, 0x3bf8ec34, 0x498a2f78, 0x05a08668 }, // 48G
    { 0xe0dedc9b, 0x3b2f8dad, 0x4da1f32d, 0xec2531df, 0x9eb5fbeb, 0x0598e4fd, 0x1a117dba, 0x703a3c37 }, // 49G
    { 0xc39d0733, 0x7ddc9268, 0xa0eba45a, 0x7a41876d, 0x151b423e, 0xac4033b5, 0x50bd28c1, 0x7c470134 }, // 50G
    { 0x5ed430d7, 0x8c296c35, 0x43114306, 0xdd8622d7, 0xc622e27c, 0x970a1de3, 0x1cb377b0, 0x1af7307e }, // 51G
    { 0x7ed52032, 0x7080a9fa, 0x4c16662f, 0xc134fadc, 0xc7048846, 0xd46ade00, 0x30b83fd1, 0x9adc87cd }, // 52G
    { 0xcedabd9b, 0x82203f7e, 0x13d206fc, 0xdf4e33d9, 0x2a6c53c2, 0x6e5cce26, 0xd6579962, 0xc4e31df6 }, // 53G
    { 0x25a5208b, 0x674bfd4c, 0xae4d91eb, 0x555010aa, 0x422cc824, 0x09d50796, 0x90f3743d, 0x00fdaefb }, // 54G
    { 0xcb474660, 0xef35f5f2, 0xa41b643f, 0xa5e46057, 0x5f4fa9b7, 0x962232a5, 0xc32f9083, 0x18a04476 }, // 55G
    { 0x5bea1fa1, 0x7a41b115, 0x525a3e7d, 0xbf0d8d5a, 0x4f7ce5c6, 0xfc73a6f4, 0xf2165124, 0x17c9f6b4 }, // 56G
    { 0x4119b887, 0x53c15bd6, 0xa693b03f, 0xcddbb45d, 0x5ac6be74, 0xab5f0ef4, 0x4b0be947, 0x5a7e4b40 }, // 57G
    { 0x9403d11a, 0x2b419eda, 0xacf931bf, 0xbd9c32a2, 0x64558508, 0x362bc5fc, 0x99025ec6, 0x2b034e02 }, // 58G
    { 0x091b6496, 0x09489d61, 0x3d1d5e59, 0x0f78e6d7, 0x4ecfc061, 0xd57048ba, 0xd9e76f30, 0x2c5b9c61 }, // 59G
    { 0x1124ec11, 0xc77d356e, 0x042dad15, 0x4e1116ed, 0xa7cc6924, 0x4f295166, 0xb54e3d34, 0x1904a1a7 }, // 60G
    { 0x0673fb86, 0xe5bda30f, 0xb3cd0ed3, 0x04ea49a0, 0x23ee33d0, 0x197a695d, 0x0c5d9809, 0x3c536683 }, // 61G
    { 0x4e7b5dab, 0xa34fbcf9, 0xf055520d, 0x4db8c49f, 0xd60282d3, 0x2adfca55, 0x5b04403d, 0xb9581a9f }, // 62G
    { 0x59c9e0bb, 0xa394e76f, 0x40c0aa58, 0x379a3cb6, 0xa5a22839, 0x93e90c41, 0x67002af4, 0x920e37f5 }, // 63G
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

__device__ __forceinline__ bool isZero(const unsigned int a[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0u) return false;
    }

    return true;
}

__device__ __forceinline__ void setZero(unsigned int a[8]) {
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

__device__ __forceinline__ bool isGreaterOrEqual(const unsigned int a[8], const unsigned int b[8]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a[7-i] > b[7-i]) return true;

        if (a[7-i] < b[7-i]) return false;
    }

    return true;
}


// ==================================================================================================


__device__ __forceinline__ unsigned int rawSub(unsigned int result[8], const unsigned int lhs[8], const unsigned int rhs[8]) {
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

__device__ __forceinline__ void rawAdd(unsigned int result[8], const unsigned int lhs[8], const unsigned int rhs[8]) {
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


__device__ __forceinline__ void jacobianDouble(
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

__device__ __forceinline__ void jacobianAdd(
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

__device__ __forceinline__ void jacobianMixedAdd(
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


__device__ __forceinline__ void scalarMultiplication(unsigned int pubX[8], unsigned int pubY[8], const uint8_t scalar[32]) {
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
                jacobianMixedAdd(tempX, tempY, tempZ, X, Y, Z, PRECOMP_X[1], PRECOMP_Y[1]);
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

__device__ __forceinline__ void getCompressedPubKey(uint8_t *output, const unsigned int x[8], const unsigned int y[8]) {
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

__device__ __forceinline__ void jacobianAddG(JacobianPoint &P) {
    unsigned int ZZ[8];
    MMP(P.Z, P.Z, ZZ);

    unsigned int U2[8];
    MMP(PRECOMP_X[1], ZZ, U2);

    unsigned int ZZZ[8];
    MMP(ZZ, P.Z, ZZZ);

    unsigned int S2[8];
    MMP(PRECOMP_Y[1], ZZZ, S2);

    unsigned int H[8];
    SMP(U2, P.X, H);

    if (isZero(H)) {
        // fallback to affine double using precomp[2]
        copyInt(PRECOMP_X[2], P.X);
        copyInt(PRECOMP_Y[2], P.Y);
        copyInt(_1_CONSTANT, P.Z);
        return;
    }

    unsigned int HH[8];
    MMP(H, H, HH);

    unsigned int HHH[8];
    MMP(HH, H, HHH);

    unsigned int V[8];
    MMP(P.X, HH, V);

    unsigned int r[8];
    SMP(S2, P.Y, r);

    // X3
    unsigned int X3[8];
    MMP(r, r, X3);
    SMP(X3, HHH, X3);
    unsigned int twoV[8];
    AMP(V, V, twoV);
    SMP(X3, twoV, X3);

    // Y3 – the important fix is here
    unsigned int temp[8];               // V - X3
    unsigned int Y3[8];    
    SMP(V, X3, temp);
    MMP(r, temp, Y3);                   // Y3 = r * (V - X3)

    unsigned int Y1HHH[8];
    MMP(P.Y, HHH, Y1HHH);               // Y1 * H^3

    // Compute -Y1HHH mod p
    unsigned int negY1HHH[8];
    if (isZero(Y1HHH)) {
        // -0 = 0
        // set all limbs to 0
        #pragma unroll
        for (int i = 0; i < 8; ++i) negY1HHH[i] = 0u;
    } else {
        // -Y1HHH = p - Y1HHH
        SMP(MODULUS_P, Y1HHH, negY1HHH);
    }

    // Y3 = r*(V - X3) + (-Y1HHH)  mod p
    AMP(Y3, negY1HHH, Y3);

    // Z3
    unsigned int Z3[8];
    MMP(P.Z, H, Z3);

    copyInt(X3, P.X);
    copyInt(Y3, P.Y);
    copyInt(Z3, P.Z);
}

__device__ __forceinline__ void jacobian_to_affine(const JacobianPoint &P, unsigned int outX[8], unsigned int outY[8]) {
    // Zinv = Z^-1
    unsigned int Zinv[8];
    copyInt(P.Z, Zinv);
    IMP(Zinv);   // in-place modular inverse

    // Zinv2 = Zinv^2
    unsigned int Zinv2[8];
    MMP(Zinv, Zinv, Zinv2);

    // Zinv3 = Zinv^3
    unsigned int Zinv3[8];
    MMP(Zinv2, Zinv, Zinv3);

    // x = X / Z^2
    MMP(P.X, Zinv2, outX);

    // y = Y / Z^3
    MMP(P.Y, Zinv3, outY);
}

__device__ __forceinline__ __uint128_t compute_private_key(
    bool search_mode,
    __uint128_t index,
    __uint128_t range_start,
    uint64_t a_low, uint64_t a_high,
    uint64_t b_low, uint64_t b_high
) {
    if (search_mode) {
        return range_start + index;
    } else {
        __uint128_t a = ((__uint128_t)a_high << 64) | a_low;
        __uint128_t b = ((__uint128_t)b_high << 64) | b_low;
        __uint128_t y = a * index + b;
        y &= (((__uint128_t)0x7FULL << 64) | 0xFFFFFFFFFFFFFFFFULL);

        uint64_t high = (uint64_t)(y >> 64);
        uint64_t low  = (uint64_t)y;
        low ^= high;
        low *= 0x9E3779B97F4A7C15ULL;
        high = ((high << 32) | (low >> 32)) ^ low;
        high *= 0xBDD1F3B71727E72BULL;
        y = ((__uint128_t)high << 64) | low;
        y ^= y >> 67;
        y &= (((__uint128_t)0x7FULL << 64) | 0xFFFFFFFFFFFFFFFFULL);

        return range_start + y;
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
    constexpr uint32_t KEYS_PER_THREAD = 1;

    uint64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;

    // ------------------------------------------------------------------
    // DEBUG MODE: force only thread 0 in block 0 to run (single key)
    // ------------------------------------------------------------------
    if (debug) {
        tid    = 0;                     // only this virtual thread runs
    }

    __uint128_t base_i      = ((__uint128_t)start_i_high << 64) | start_i_low;
    __uint128_t range_start = ((__uint128_t)range_start_high << 64) | range_start_low;
    __uint128_t i0 = base_i + (__uint128_t)tid * KEYS_PER_THREAD;

    // Out of range exit early
    if (i0 >= base_i + count) return;

    __uint128_t current_k;
    if (search_mode) {
        current_k = i0;  // sequence: start exactly at range_start
    } else {
        current_k = compute_private_key(false, i0, range_start, a_low, a_high, b_low, b_high);  // PCG: permute i0
    }

    // ======= 1. Private key =======
    uint8_t priv[32] = {0};

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        priv[i] = (current_k >> (8 * i)) & 0xFF;
    }

    if (debug) {
        printf("\n--- Thread tid=%llu processing starting index i0=%llu ---\n",
               (unsigned long long)tid, (unsigned long long)i0);
        printf("Private key (dec): %llu\n", (unsigned long long)current_k);
        printf("Private key (hex BE): ");
        for (int b = 31; b >= 0; --b) printf("%02x", priv[b]);
        printf("\n");
    }


    // ======= 2. Public key =======
    unsigned int pubX[8], pubY[8];
    scalarMultiplication(pubX, pubY, priv);

    uint8_t pub_compressed[33];
    getCompressedPubKey(pub_compressed, pubX, pubY);

    // if (debug) {
    //     printf("Compressed pubkey: ");
    //     for (int b = 0; b < 33; ++b) printf("%02x", pub_compressed[b]);
    //     printf("\n");
    // }

    // ======= 3. Add to jacob cords =======
    JacobianPoint P;
    copyInt(pubX, P.X);
    copyInt(pubY, P.Y);
    copyInt(_1_CONSTANT, P.Z);
    

    for (uint32_t j = 0; j < KEYS_PER_THREAD; ++j) {
        __uint128_t current_index = i0 + j;

        if (current_index >= base_i + count) return;
        if (*out_found_index != ULLONG_MAX) return;

        unsigned int affX[8], affY[8];
        jacobian_to_affine(P, affX, affY);
        
        // unsigned int yParity = (P.Y[7] & 1u) ^ (P.Z[7] & 1u);
        unsigned int yParity = affY[7] & 1u;

        // ======= 4. SHA256 =======
        unsigned int sha_digest[8];
        sha256::sha256PublicKeyCompressed(affX, yParity, sha_digest);

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

        uint8_t temp_comp[33];
        getCompressedPubKey(temp_comp, affX, affY);
        
        // if (debug) {
        //     printf("Point %llu compressed: ", (unsigned long long)(current_k + j));
        //     for (int b = 0; b < 33; ++b) printf("%02x", temp_comp[b]);
        //     printf("\n");
        // }

        // if (debug) {
        //     printf("SHA256 digest:      ");
        //     for (int b = 0; b < 32; ++b) printf("%02x", sha_out[b]);
        //     printf("\n\n");
        // }

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


        // ======= 5. RIPEMD160 =======
        unsigned int ripe_words[5];
        ripemd160::ripemd160sha256(sha_words, ripe_words);
    
        // if (debug) {
        //     printf("ripe_words (LE words): ");
        //     for (int i = 0; i < 5; i++)
        //        printf("%08x ", ripe_words[i]);
        //     printf("\n");
        // }
    
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
    
        // if (debug) {
        //     printf("Address hash: ");
        //     for (int b = 0; b < 20; ++b) printf("%02x", hash160[b]);
        //     printf("\n");
        // }
    
    
        // ======= CHECK =======
        bool match = true;
        #pragma unroll
        for (int k = 0; k < 20; ++k) {
            if (hash160[k] != TARGET_H160[k]) {
                match = false;
                break;
            }
        }
    
    
        // ======= CONGRATULATION =======
        if (match) {
            atomicExch(out_found_index, (unsigned long long)(current_k + j));
            return;
        }

        if (j + 1 < KEYS_PER_THREAD) {
            jacobianAddG(P);
        }
    }
}


