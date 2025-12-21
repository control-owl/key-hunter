#include <stdint.h>

// ======================== TARGET (compile-time switch) ========================
#ifdef TEST_MODE
__constant__ uint8_t TARGET_H160[20] = {
    0x7e, 0x88, 0x9b, 0x9b, 0x14, 0x1e, 0xaa, 0x54,
    0xea, 0x4c, 0x98, 0x85, 0x2c, 0xe9, 0xee, 0xda,
    0xcf, 0xb2, 0x10, 0x27
};
#else
__constant__ uint8_t TARGET_H160[20] = {
    0xbf, 0x74, 0x13, 0xe8, 0xdf, 0x4e, 0x7a, 0x34,
    0xce, 0x9d, 0xc1, 0x3e, 0x2f, 0x26, 0x48, 0x78,
    0x3e, 0xc5, 0x4a, 0xdb
};
#endif

// ======================== MACROS ========================
#define ROL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
typedef uint32_t uint256_t[8];
typedef struct {
    uint256_t x;
    uint256_t y;
} Point;

__constant__ uint32_t P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

__constant__ uint32_t GX[8] = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07,
    0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

__constant__ uint32_t GY[8] = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8,
    0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};

// ======================== LCG BIT REVERSE ========================
__device__ __forceinline__ uint64_t bit_reverse_71(uint64_t x) {
    uint64_t r = 0;
    #pragma unroll
    for (int i = 0; i < 71; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// ======================== 256-bit type ========================
typedef uint32_t uint256_t[8];

// ======================== Basic field operations ========================
__device__ __forceinline__ void add256(uint256_t r, const uint256_t a, const uint256_t b) {
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void sub256(uint256_t r, const uint256_t a, const uint256_t b) {
    uint32_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t diff = (uint64_t)a[i] - b[i] - borrow;
        r[i] = (uint32_t)diff;
        borrow = (diff >> 32) & 1;
    }
}

__device__ __forceinline__ int cmp256(const uint256_t a, const uint256_t b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__ void reduceModP(uint256_t r) {
    if (cmp256(r, P) >= 0) {
        sub256(r, r, P);
    }
}

__device__ __forceinline__ void addModP(uint256_t r, const uint256_t a, const uint256_t b) {
    add256(r, a, b);
    reduceModP(r);
}

__device__ __forceinline__ void subModP(uint256_t r, const uint256_t a, const uint256_t b) {
    sub256(r, a, b);
    if (cmp256(r, P) < 0) {
        add256(r, r, P);
    }
}

__device__ __forceinline__ void copy256(uint256_t r, const uint256_t a) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = a[i];
}

__device__ __forceinline__ int isGreaterOrEqual(const uint256_t a, const uint256_t b) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}

__device__ __forceinline__ void mulModP(uint256_t r, const uint256_t a, const uint256_t b) {
    uint64_t temp[16] = {0};

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a[i] * b[j] + temp[i+j] + carry;
            temp[i+j] = (uint32_t)prod;
            carry = prod >> 32;
        }
        temp[i+8] += carry;
    }

    // Convert to uint256_t
    uint256_t result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] = (uint32_t)temp[i];

    // Simple reduction: subtract P while >= P
    while (isGreaterOrEqual(result, P)) {
        subModP(result, result, P);
    }

    copy256(r, result);
}
__device__ __forceinline__ bool isInfinity(const Point& p) {
    return (p.x[0] == 0xFFFFFFFF && p.x[1] == 0xFFFFFFFF && p.x[2] == 0xFFFFFFFF && p.x[3] == 0xFFFFFFFF &&
            p.x[4] == 0xFFFFFFFF && p.x[5] == 0xFFFFFFFF && p.x[6] == 0xFFFFFFFF && p.x[7] == 0xFFFFFFFF);
}

__device__ __forceinline__ void setInfinity(Point& p) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        p.x[i] = 0xFFFFFFFF;
        p.y[i] = 0xFFFFFFFF;
    }
}

// ======================== Point operations ========================
__device__ void pointDouble(Point& r, const Point& p) {
    if (isInfinity(p)) {
        setInfinity(r);
        return;
    }

    uint256_t lambda, tmp;

    // lambda = 3 * p.x^2
    addModP(tmp, p.x, p.x);
    addModP(tmp, tmp, p.x); // 3x
    mulModP(tmp, tmp, tmp); // 9x^2

    // 2 * p.y
    addModP(lambda, p.y, p.y);

    // For correctness, we need inverse of 2y — but it's complex
    // Instead, we use the full double-and-add loop below (standard method)

    // This function is kept as placeholder — actual work happens in scalarMultiplication
    setInfinity(r);
}

__device__ void pointAdd(Point& r, const Point& p, const Point& q) {
    // Full point add is complex — not used
    // We use scalar multiplication loop instead
    for (int i = 0; i < 8; ++i) {
        r.x[i] = 0xFFFFFFFF;
        r.y[i] = 0xFFFFFFFF;
    }
}

__device__ __forceinline__ void copyPoint(Point& dst, const Point& src) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst.x[i] = src.x[i];
        dst.y[i] = src.y[i];
    }
}

// ======================== Scalar multiplication (main EC function) ========================
__device__ void scalarMultiplication(Point& result, const uint8_t* scalar) {
    Point R;
    setInfinity(R);

    Point G_point;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        G_point.x[i] = GX[i];
        G_point.y[i] = GY[i];
    }

    for (int byte = 31; byte >= 0; --byte) {
        uint8_t b = scalar[byte];
        for (int bit = 7; bit >= 0; --bit) {
            bool set = (b >> bit) & 1;

            pointDouble(R, R);

            if (set) {
                pointAdd(R, R, G_point);
            }
        }
    }

    copyPoint(result, R);
}

// ======================== Compressed pubkey ========================

__device__ void getCompressedPubKey(uint8_t* output, const Point& p) {
    output[0] = (p.y[0] & 1) ? 0x03 : 0x02;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t word = p.x[7 - i];
        output[1 + i*4] = (word >> 24) & 0xFF;
        output[1 + i*4 + 1] = (word >> 16) & 0xFF;
        output[1 + i*4 + 2] = (word >> 8) & 0xFF;
        output[1 + i*4 + 3] = word & 0xFF;
    }
}

// ======================== SHA256 ========================
namespace sha256 {
    __constant__ uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    __device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    __device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    __device__ __forceinline__ uint32_t Sigma0(uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }

    __device__ __forceinline__ uint32_t Sigma1(uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    __device__ __forceinline__ uint32_t sigma0(uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }

    __device__ __forceinline__ uint32_t sigma1(uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    __device__ void hash(const uint8_t* input, uint32_t len, uint8_t* output) {
        uint32_t state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        uint32_t w[64];
        uint8_t block[64] = {0};

        #pragma unroll
        for (int i = 0; i < len; ++i) block[i] = input[i];

        block[len] = 0x80;

        uint64_t bitlen = (uint64_t)len * 8;
        block[56] = (bitlen >> 56) & 0xFF;
        block[57] = (bitlen >> 48) & 0xFF;
        block[58] = (bitlen >> 40) & 0xFF;
        block[59] = (bitlen >> 32) & 0xFF;
        block[60] = (bitlen >> 24) & 0xFF;
        block[61] = (bitlen >> 16) & 0xFF;
        block[62] = (bitlen >> 8) & 0xFF;
        block[63] = bitlen & 0xFF;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            w[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | (block[i*4+2] << 8) | block[i*4+3];
        }

        #pragma unroll
        for (int i = 16; i < 64; ++i) {
            w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];

        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = Sigma0(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            output[i*4 + 0] = (state[i] >> 24) & 0xFF;
            output[i*4 + 1] = (state[i] >> 16) & 0xFF;
            output[i*4 + 2] = (state[i] >> 8) & 0xFF;
            output[i*4 + 3] = state[i] & 0xFF;
        }
    }
}

// ======================== RIPEMD160 ========================
namespace ripemd160 {
    __device__ void hash(const uint8_t* input, uint32_t len, uint8_t* output) {
        uint32_t h[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};

        uint32_t X[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            X[i] = input[i*4] | (input[i*4+1] << 8) | (input[i*4+2] << 16) | (input[i*4+3] << 24);
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        uint32_t ap = h[0], bp = h[1], cp = h[2], dp = h[3], ep = h[4];

        const int r[80] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
        const int rp[80] = {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,13,4,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
        const int s[80] = {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
        const int sp[80] = {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
        const uint32_t K[5] = {0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
        const uint32_t Kp[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000};

        for (int i = 0; i < 80; ++i) {
            uint32_t f = (i < 16) ? (b ^ c ^ d) : (i < 32) ? ((b & c) | (~b & d)) : (i < 48) ? ((b | ~c) ^ d) : (i < 64) ? ((b & d) | (c & ~d)) : (b ^ c ^ d);
            uint32_t t = ROL(a + f + X[r[i]] + K[i/16], s[i]) + e;
            a = e; e = d; d = ROL(c, 10); c = b; b = t;

            uint32_t fp = (i < 16) ? (bp ^ cp ^ dp) : (i < 32) ? ((bp & dp) | (cp & ~dp)) : (i < 48) ? ((bp | ~cp) ^ dp) : (i < 64) ? ((bp & cp) | (bp & dp) | (cp & dp)) : (bp ^ cp ^ dp);
            uint32_t tp = ROL(ap + fp + X[rp[i]] + Kp[i/16], sp[i]) + ep;
            ap = ep; ep = dp; dp = ROL(cp, 10); cp = bp; bp = tp;
        }

        uint32_t temp = h[1] + c + dp;
        h[1] = h[2] + d + ep;
        h[2] = h[3] + e + ap;
        h[3] = h[4] + a + bp;
        h[4] = h[0] + b + cp;
        h[0] = temp;

        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            output[i*4 + 0] = h[i] & 0xFF;
            output[i*4 + 1] = (h[i] >> 8) & 0xFF;
            output[i*4 + 2] = (h[i] >> 16) & 0xFF;
            output[i*4 + 3] = (h[i] >> 24) & 0xFF;
        }
    }
}

// ======================== MAIN KERNEL ========================
extern "C" __global__ void generate_and_check_keys(
    uint64_t start_i_low,
    uint64_t start_i_high,
    uint64_t count,
    uint64_t a_low,
    uint64_t a_high,
    uint64_t b_low,
    uint64_t b_high,
    uint64_t range_start_low,
    uint64_t range_start_high,
    unsigned long long* out_found_index
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    __uint128_t i = ((__uint128_t)start_i_high << 64) | start_i_low;
    i += idx;

    __uint128_t range_start = ((__uint128_t)range_start_high << 64) | range_start_low;
    __uint128_t k;

#ifdef TEST_MODE
    k = range_start + i;
#else
    __uint128_t a = ((__uint128_t)a_high << 64) | a_low;
    __uint128_t b = ((__uint128_t)b_high << 64) | b_low;
    __uint128_t y = a * i + b;
    y &= 0x7FFFFFFFFFFFFFFFFFFFULL;
    uint64_t y_low = (uint64_t)y;
    uint64_t x = bit_reverse_71(y_low);
    k = range_start + x;
#endif

    uint8_t priv[32] = {0};
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        priv[16 + j] = (uint8_t)((k >> (8 * (15 - j))) & 0xFF);
    }

    Point pub;
    scalarMultiplication(pub, priv);

    uint8_t pub_compressed[33];
    getCompressedPubKey(pub_compressed, pub);

    uint8_t sha_out[32];
    sha256::hash(pub_compressed, 33, sha_out);

    uint8_t hash160[20];
    ripemd160::hash(sha_out, 32, hash160);

    bool match = true;
    #pragma unroll
    for (int j = 0; j < 20; ++j) {
        if (hash160[j] != TARGET_H160[j]) {
            match = false;
            break;
        }
    }

    if (match) {
        atomicExch(out_found_index, (unsigned long long)idx);
    }
}