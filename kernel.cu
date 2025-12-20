#include <stdint.h>

// __constant__ uint8_t TARGET_H160[20] = {
//     0xbf, 0x74, 0x13, 0xe8, 0xdf, 0x4e, 0x7a, 0x34,
//     0xce, 0x9d, 0xc1, 0x3e, 0x2f, 0x26, 0x48, 0x78,
//     0x3e, 0xc5, 0x4a, 0xdb
// };

// TEST
__constant__ uint8_t TARGET_H160[20] = {
    0x7e, 0x88, 0x9b, 0x9b, 0x14, 0x1e, 0xaa, 0x54, 0xea, 0x4c, 0x98, 0x85, 0x2c, 0xe9, 0xee, 0xda,
    0xcf, 0xb2, 0x10, 0x27
};


__device__ __forceinline__ uint64_t bit_reverse_71(uint64_t x) {
    uint64_t r = 0;
    #pragma unroll
    for (int i = 0; i < 71; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

__device__ void sha256_cuda_block(const uint8_t* input, uint8_t* output) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint32_t w[64];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)input[i*4] << 24) |
               ((uint32_t)input[i*4+1] << 16) |
               ((uint32_t)input[i*4+2] << 8) |
               input[i*4+3];
    }

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = (w[i-15] >> 7 | w[i-15] << 25) ^ (w[i-15] >> 18 | w[i-15] << 14) ^ (w[i-15] >> 3);
        uint32_t s1 = (w[i-2] >> 17 | w[i-2] << 15) ^ (w[i-2] >> 19 | w[i-2] << 13) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = (e >> 6 | e << 26) ^ (e >> 11 | e << 21) ^ (e >> 25 | e << 7);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = (a >> 2 | a << 30) ^ (a >> 13 | a << 19) ^ (a >> 22 | a << 10);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        output[i*4]     = state[i] >> 24;
        output[i*4 + 1] = state[i] >> 16;
        output[i*4 + 2] = state[i] >> 8;
        output[i*4 + 3] = state[i];
    }
}

__device__ void sha256_cuda(const uint8_t* input, uint8_t* output) {
    uint8_t block[64] = {0};

    // Copy 32-byte input
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        block[i] = input[i];

    // SHA-256 padding
    block[32] = 0x80; // append '1' bit

    // Length in bits (32 bytes = 256 bits) as 64-bit big-endian
    uint64_t bit_len = 32ULL * 8ULL;
    block[56] = (bit_len >> 56) & 0xFF;
    block[57] = (bit_len >> 48) & 0xFF;
    block[58] = (bit_len >> 40) & 0xFF;
    block[59] = (bit_len >> 32) & 0xFF;
    block[60] = (bit_len >> 24) & 0xFF;
    block[61] = (bit_len >> 16) & 0xFF;
    block[62] = (bit_len >> 8) & 0xFF;
    block[63] = bit_len & 0xFF;

    sha256_cuda_block(block, output);
}

// ---------------- RIPEMD-160 ----------------
__device__ uint32_t ROL(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }

__device__ void ripemd160_cuda(const uint8_t* input, uint8_t* output) {
    uint32_t h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        X[i] = (uint32_t)input[i*4] |
               ((uint32_t)input[i*4+1] << 8) |
               ((uint32_t)input[i*4+2] << 16) |
               ((uint32_t)input[i*4+3] << 24);
    }

    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    uint32_t ap = h0, bp = h1, cp = h2, dp = h3, ep = h4;

    // Full 80 rounds constants
    const int r[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    const int rp[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,13,4,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };
    const int s[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    const int sp[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };
    const uint32_t K[5] = {0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
    const uint32_t Kp[5] = {0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};

    for(int i=0;i<80;i++){
        uint32_t t = ROL(a + ((i<16)?(b^c^d):(i<32)?((b&c)|(~b&d)):(i<48)?((b|~c)^d):(i<64)?((b&d)|(c&~d)):(b^c^d)) + X[r[i]] + K[i/16], s[i]) + e;
        a = e; e = d; d = ROL(c,10); c = b; b = t;
        t = ROL(ap + ((i<16)?(bp^cp^dp):(i<32)?((bp&dp)|(cp&~dp)):(i<48)?((bp|~cp)^dp):(i<64)?((bp&cp)|(bp&dp)|(cp&dp)):(bp^cp^dp)) + X[rp[i]] + Kp[i/16], sp[i]) + ep;
        ap = ep; ep = dp; dp = ROL(cp,10); cp = bp; bp = t;
    }

   uint32_t temp = h1 + c + dp;
    h1 = h2 + d + ep;
    h2 = h3 + e + ap;
    h3 = h4 + a + bp;
    h4 = h0 + b + cp;
    h0 = temp;

    // === FIXED: Little-endian output ===
    uint32_t vals[5] = { h0, h1, h2, h3, h4 };
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        uint32_t v = vals[i];
        output[i*4 + 0] = v & 0xFF;
        output[i*4 + 1] = (v >> 8) & 0xFF;
        output[i*4 + 2] = (v >> 16) & 0xFF;
        output[i*4 + 3] = (v >> 24) & 0xFF;
    }
}

// ---------------- Kernel ----------------
extern "C" __global__ void generate_and_check_keys(
    uint64_t start_i,
    uint64_t count,
    uint64_t a,
    uint64_t b,
    uint64_t range_start,
    unsigned long long* out_found_index
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t i = start_i + idx;

    // === FIXED: Correct 71-bit mask ===
    uint64_t y = (a * i + b) & 0x7FFFFFFFFFFFFFFFULL;

    uint64_t x = bit_reverse_71(y);
    uint64_t k = range_start + x;

    uint8_t key[32] = {0};
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        key[16 + j] = (k >> (8 * (15 - j))) & 0xFF;

    uint8_t sha256_out[32];
    sha256_cuda(key, sha256_out);

    uint8_t hash160[20];
    ripemd160_cuda(sha256_out, hash160);

    bool match = true;
    #pragma unroll
    for (int j = 0; j < 20; ++j)
        if (hash160[j] != TARGET_H160[j]) { match = false; break; }

    if (match)
        atomicExch(out_found_index, (unsigned long long)(start_i + idx));
}