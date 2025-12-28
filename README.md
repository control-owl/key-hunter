# Bitcoin Puzzle #72 Key Hunter

This is a **brute-force** attempt for **Bitcoin Puzzle #72**.

### What is Bitcoin Puzzle #72?

- Part of the famous ~1000 BTC challenge created in 2015.
- Puzzle #72 has **71 bits** of entropy (search space: 2^71 keys ≈ 2.36 quintillion).
- Target address: **1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR**
- Current reward: **7.2 BTC** (as of December 2025, still **unsolved**).

The goal: find the private key in the range starting with 8 (hex) that matches the address.

### Disclaimer

This code is intended **only** for solving this public puzzle.  
Do **not** use it to brute-force keys or addresses that are not yours!  
That would be stealing and is illegal.

### Features
- **CPU solver** (multi-threaded using all cores)
- **GPU solver** (CUDA-based, optimized for NVIDIA GeForce GTX 1650 (4GB GDDR5 memory))
- ~~Combined CPU + GPU mode~~
- **Live dashboard** showing progress, speed, current key
- **Crash-safe**: progress saved to `status/` folder, resumes automatically
- **No overlaps or skipped keys** (shared allocator with logging)


# Third party libraries:

## BitCrack's:
- sha256.cuh
- ripemd160.cuh