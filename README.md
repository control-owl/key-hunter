# Bitcoin Puzzle #72 Key Hunter

## Key-Hunter PRO (SOON)

- For Christmas 2026 I will release Key-Hunter PRO version.
- PRO has fancy GUI. 
- PRO supports **all** unsolved Bitcoin puzzles, not just #72.
- PRO version is not MIT. It is copyrighted.
- PRO has ALL mode (round-robin all unsolved pozzles)
- PRO has machine unique PCG constants (everybody has their own magic numbers)
- PRO supports multi GPU systems with different nVidia arch
- One unique .OWL file contains all kernels
- PRO has CPU and GPU solvers. CPU solver is free for use. 
- Kernel packs for GPU will be sold. Crypto only.
- A blog post will be released soon to explain more. 


https://github.com/control-owl/key-hunter-pro


# This project will be archived when PRO goes live

---

This was a **brute-force** attempt for **Bitcoin Puzzle #72**.

Right now I am not uploading my code. It is working fine on my pc. I even created a GUI in egui. And support for endomorphism.
My version supports multiple kernels. A lot of changes in few months. 

I really want to solve this, maybe thinking about small GPU farm. Will see. 

Since there is no incentive for me, no more public updates. As soon as I solve the puzzle, I will upload my latest version. Or in some special cases. For now its me against the world.

For more information follow my blog post: [control-owl.github.io](https://control-owl.github.io/blog/post/2026/10/09/key-hunter.html)

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
- **Combined CPU + GPU mode**
- **Live dashboard** showing progress, speed, current key
- **Crash-safe**: progress saved to `status/` folder, resumes automatically
- **No overlaps or skipped keys** (shared allocator with logging)
- **Sequence** and **PCG** (Permuted Congruential Generator) search


# Third party libraries:

## BitCrack's:
- sha256.cuh
- ripemd160.cuh
