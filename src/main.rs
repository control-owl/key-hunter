// authors = ["Control Owl <c0ntr01-_-0w1[at]r-o0-t[dot]wtf>"]
// license = "CC-BY-NC-ND-4.0 [2025] Control Owl"
//
// Target Bitcoin address for Puzzle #72 (compressed P2PKH)
// Puzzle #72 uses exactly 71 bits of entropy (after the leading '8' bit)
const TARGET_ADDRESS: &str = "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR";
const TARGET_ENCODED: [u8; 20] = [
    191, 116, 19, 232, 223, 78, 122, 52, 206, 157, 193, 62, 47, 38, 72, 120, 62, 197, 74, 219,
];

// Define only the start and the bit-size; derive end at runtime (Option 1)
const RANGE_START_HEX: &str = "800000000000000000"; // 19 hex digits
const N_BITS: u32 = 71; // domain size = 2^71
const N_MASK: u128 = (1u128 << N_BITS) - 1; // mask for 71 bits

// Linear Congruential Generator (LCG) parameters for permutation
// A_CONST: Multiplier — must be odd and co-prime with 2^71 for full cycle
// Using golden ratio conjugate (φ-1) gives excellent statistical properties
// Increasing/decreasing: minor effect on distribution, negligible on speed
const A_CONST: u128 = 0x9E3779B97F4A7C15u128;
// B_CONST: Increment — arbitrary constant, improves avalanche
const B_CONST: u128 = 0x1D3F84A5B7C29E3u128;

// Paths & persistence
const STATUS_DIR: &str = "status";
const FOUND_FILE: &str = "status/address.txt"; // save found result here
const GLOBAL_NEXT_FILE: &str = "status/GLOBAL_NEXT"; // next unassigned index (hex)
const ALLOC_LOG_FILE: &str = "status/ALLOC.log"; // assignment log

// Dashboard & persistence cadence
const DASHBOARD_UPDATE_INTERVAL_MS: u128 = 500; // per-thread display update
const PROGRESS_SAVE_INTERVAL_SEC: u64 = 5; // per-thread checkpoint

// Batch parallelization inside a thread iteration
// Tune PARALLEL_KEYS to your CPU cache; 64 is a good default
const PARALLEL_KEYS: usize = 64;

// Chunk size claimed atomically per assignment. Larger chunks reduce allocator contention
// but increase potential rework on crash; we use assignment log to avoid skips.
const CHUNK_SIZE: u128 = (PARALLEL_KEYS as u128) * 1024; // 64 * 1024 = 65,536 keys per claim

use hex::encode as hex_encode;
use rayon::prelude::*;
use ripemd::Ripemd160;
use secp256k1::Secp256k1;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::panic;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant};

static TOTAL_CHECKED: AtomicU64 = AtomicU64::new(0);

lazy_static::lazy_static! {
    static ref DASHBOARD: Arc<Mutex<Vec<ThreadStatus>>> = Arc::new(Mutex::new(Vec::new()));
}

thread_local! {
    static LOCAL_SECP: Secp256k1<secp256k1::All> = Secp256k1::new();
}

#[derive(Clone)]
struct ThreadStatus {
    checked: u64,
    last_i: u128,    // last domain index processed (0..2^71-1)
    key_hex: String, // last key hex shown
    speed_kps: f64,
    last_update: Instant,
    last_checked: u64,
}

impl Default for ThreadStatus {
    fn default() -> Self {
        Self {
            checked: 0,
            last_i: 0,
            key_hex: String::from(""),
            speed_kps: 0.0,
            last_update: Instant::now(),
            last_checked: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Chunk {
    start_i: u128, // domain index start (0..TOTAL_KEYS)
    len: u128,     // number of indices in this chunk
}

struct AllocState {
    next_i: u128,             // next unassigned domain index
    pending: VecDeque<Chunk>, // recovered chunks to re-run
}

fn main() {
    fs::create_dir_all(STATUS_DIR).expect("Failed to create status directory");
    let num_threads = num_cpus::get();

    let start = u128::from_str_radix(RANGE_START_HEX, 16).unwrap();
    let total_keys_u128: u128 = 1u128 << N_BITS; // authoritative domain size
    let end = start + total_keys_u128 - 1; // derived end consistent with N_BITS

    println!("Searching Bitcoin Puzzle #72 (CPU-only)");
    println!("Target address: {}", TARGET_ADDRESS);
    println!("Target address (h160): {:?}", TARGET_ENCODED);
    println!("Range : {:019X} to {:019X}", start, end);
    println!("Total keys : {} (2^{})", total_keys_u128, N_BITS);
    println!("Using {} CPU threads", num_threads);
    println!("Traversal: key = start + rev71((A*i + B) mod 2^71), i = 0..2^71-1");
    println!("A (odd) = {:X}", A_CONST & N_MASK);
    println!("B = {:X}", B_CONST & N_MASK);

    let global_start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    // Initialize dashboard
    {
        let mut dash = DASHBOARD.lock().unwrap();
        dash.clear();
        dash.extend((0..num_threads).map(|_| ThreadStatus::default()));
    }

    // Load allocator state: recovered chunks + next_i
    let (pending_from_log, farthest_end_from_log) = load_assignment_log();
    let next_i_from_file = load_next_i();
    let initial_next_i = std::cmp::max(farthest_end_from_log, next_i_from_file);

    let alloc = Arc::new(Mutex::new(AllocState {
        next_i: initial_next_i.min(total_keys_u128),
        pending: pending_from_log,
    }));

    // Dashboard thread
    let dash_starts = start;
    let dash_ends = end;
    let dash_shutdown = shutdown.clone();

    let dash_thread = spawn(move || {
        while !dash_shutdown.load(Ordering::Relaxed) {
            print_dashboard(dash_starts, dash_ends, global_start, num_threads);
            sleep(Duration::from_secs(2));
        }
        print_dashboard(dash_starts, dash_ends, global_start, num_threads);
    });

    // Panic hook
    {
        let alloc_for_panic = alloc.clone();
        panic::set_hook(Box::new(move |_info| {
            save_alloc_state(&alloc_for_panic);
            eprintln!("\n\nPanic detected! Progress saved to status files.");
        }));
    }

    (0..num_threads).into_par_iter().for_each(|t| {
        let t_usize = t;
        let mut checked: u64 = 0;
        let mut last_stat = Instant::now();
        let mut last_save = Instant::now();

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let maybe_chunk = {
                let mut a = alloc.lock().unwrap();

                if let Some(ch) = a.pending.pop_front() {
                    Some(ch)
                } else if a.next_i < total_keys_u128 {
                    let start_i = a.next_i;
                    let len = CHUNK_SIZE.min(total_keys_u128 - start_i);

                    a.next_i += len;

                    save_next_i(a.next_i);
                    append_log_assigned(start_i, len);

                    Some(Chunk { start_i, len })
                } else {
                    None
                }
            };

            let chunk = match maybe_chunk {
                Some(c) => c,

                None => break,
            };

            let mut i = chunk.start_i;
            let end_i = chunk.start_i + chunk.len;

            while i < end_i {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                // Build batch of PARALLEL_KEYS
                let mut key_bytes_batch = [[0u8; 32]; PARALLEL_KEYS];
                let mut k_batch = [0u128; PARALLEL_KEYS];
                let mut first_k_hex: Option<String> = None;

                let batch_len_u128 = (end_i - i).min(PARALLEL_KEYS as u128);
                let batch_len = batch_len_u128 as usize;

                for p in 0..batch_len {
                    let current_i = i + p as u128;
                    let x = permute_index(current_i);
                    let k = start + x; // derived key inside fixed 2^71 domain

                    k_batch[p] = k;

                    if first_k_hex.is_none() {
                        first_k_hex = Some(format!("{:019X}", k));
                    }

                    let be = k.to_be_bytes();
                    key_bytes_batch[p][16..].copy_from_slice(&be);
                }

                // Check all keys in batch
                for p in 0..batch_len {
                    let k = k_batch[p];
                    let key_bytes: [u8; 32] = key_bytes_batch[p];

                    let is_match = LOCAL_SECP.with(|secp| {
                        if let Ok(sk) = secp256k1::SecretKey::from_byte_array(key_bytes) {
                            let pk = secp256k1::PublicKey::from_secret_key(secp, &sk);
                            let compressed: [u8; 33] = pk.serialize();

                            let h160 = Ripemd160::digest(Sha256::digest(compressed));
                            h160.as_slice() == TARGET_ENCODED
                        } else {
                            false
                        }
                    });

                    if is_match {
                        save_found_key(t as u64, &key_bytes, k, TARGET_ADDRESS.to_string());
                        save_alloc_state(&alloc);

                        shutdown.store(true, Ordering::Relaxed);

                        return; // exit thread
                    }
                }

                // Exact accounting for partial batches
                checked += batch_len as u64;

                TOTAL_CHECKED.fetch_add(batch_len as u64, Ordering::Relaxed);

                let now = Instant::now();

                if now.duration_since(last_stat).as_millis() >= DASHBOARD_UPDATE_INTERVAL_MS {
                    {
                        let mut dash = DASHBOARD.lock().unwrap();
                        let status = &mut dash[t_usize];
                        let elapsed = now.duration_since(status.last_update).as_secs_f64();

                        if elapsed > 0.1 {
                            status.speed_kps =
                                (checked - status.last_checked) as f64 / elapsed / 1_000.0;
                        }

                        status.checked = checked;
                        status.last_i = i;
                        status.key_hex = first_k_hex.unwrap_or_else(|| String::from(""));
                        status.last_update = now;
                        status.last_checked = checked;
                    }

                    last_stat = now;
                }

                if now.duration_since(last_save).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC {
                    save_thread_checkpoint(t as u64, i);

                    last_save = now;
                }

                i += PARALLEL_KEYS as u128;
            }

            // Mark chunk finished in the assignment log
            append_log_finished(chunk.start_i);
        }

        // Final per-thread checkpoint
        save_thread_checkpoint(t as u64, 0);

        let mut dash = DASHBOARD.lock().unwrap();
        dash[t_usize].checked = checked;
    });

    shutdown.store(true, Ordering::Relaxed);
    dash_thread.join().unwrap();

    print_dashboard(start, end, global_start, num_threads);

    if !shutdown.load(Ordering::Relaxed) {
        println!("\nSearch completed across active threads. No key found.");
    }
}

#[inline(always)]
fn permute_index(i: u128) -> u128 {
    // LCG on 2^71 plus 71-bit bit-reversal for high-quality permutation
    let a = A_CONST & N_MASK;
    let b = B_CONST & N_MASK;
    let y = (a.wrapping_mul(i) + b) & N_MASK;

    bit_reverse_71(y)
}

#[inline(always)]
fn bit_reverse_71(mut x: u128) -> u128 {
    let mut r: u128 = 0;

    for _ in 0..N_BITS {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }

    r
}

// #[inline(always)]
// fn decode_target_h160(addr: &str) -> Option<[u8; 20]> {
//     // Decode Base58Check, expect 25 bytes: [version=0x00][h160=20 bytes][checksum=4 bytes]
//     if let Ok(bytes) = bs58::decode(addr).into_vec() {
//         if bytes.len() == 25 && bytes[0] == 0x00 {
//             let mut h = [0u8; 20];
//             h.copy_from_slice(&bytes[1..21]);
//
//             let checksum_calc = &Sha256::digest(Sha256::digest(&bytes[..21]))[..4];
//             if &bytes[21..25] == checksum_calc {
//                 return Some(h);
//             }
//         }
//     }
//     None
// }

fn print_dashboard(start: u128, end: u128, global_start: Instant, num_threads: usize) {
    let dash = DASHBOARD.lock().unwrap();
    let total_checked_u64 = TOTAL_CHECKED.load(Ordering::Relaxed);
    let elapsed = global_start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    let overall_speed_kps = if elapsed_secs > 0.1 {
        (total_checked_u64 as f64) / elapsed_secs / 1_000.0
    } else {
        0.0
    };

    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;

    print!("\x1B[2J\x1B[H");
    println!("╔═════════════════════════════════════════╗");
    println!("║     KEY HUNTER - BITCOIN PUZZLE #72     ║");
    println!("╚═════════════════════════════════════════╝");
    println!("Target Address: {}", TARGET_ADDRESS);
    println!("Range : {:019X} ──▶ {:019X}", start, end);
    println!("Threads : {}", num_threads);
    println!("Parallel tasks: {}\n", PARALLEL_KEYS);

    for t in 0..num_threads {
        let status = &dash[t];
        let idx_hex = format!("{:019X}", status.last_i);
        println!(
            "CPU {:2} Index: {} Checked: {:8}K Current key: {} Speed: {:6.1} K/s",
            t,
            idx_hex,
            status.checked / 1_000,
            status.key_hex,
            status.speed_kps
        );
    }

    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ TOTAL CHECKED: {:9}K keys │ OVERALL SPEED: {:7.1} K/s │ Elapsed: {:3}h {:3}m {:3}s ║",
        (total_checked_u64) / 1_000,
        overall_speed_kps,
        h,
        m,
        s
    );
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let _ = io::stdout().flush();
}

fn save_thread_checkpoint(thread_id: u64, index: u128) {
    let path = format!("{}/CPU{}", STATUS_DIR, thread_id);
    let hex = format!("{:X}", index);

    if let Err(e) = std::fs::write(&path, hex) {
        eprintln!("Failed to save CPU{} progress: {}", thread_id, e);
    }
}

fn save_alloc_state(alloc_arc: &Arc<Mutex<AllocState>>) {
    let a = alloc_arc.lock().unwrap();

    save_next_i(a.next_i);
}

fn load_next_i() -> u128 {
    let path = GLOBAL_NEXT_FILE;

    if let Ok(s) = std::fs::read_to_string(path) {
        if let Ok(v) = u128::from_str_radix(s.trim(), 16) {
            return v;
        }
    }
    0
}

fn save_next_i(v: u128) {
    let path = GLOBAL_NEXT_FILE;

    if let Err(e) = std::fs::write(path, format!("{:X}", v)) {
        eprintln!("Failed to save GLOBAL_NEXT: {}", e);
    }
}

fn append_log_assigned(start_i: u128, len: u128) {
    append_log_line(format!("ASSIGNED {:X} {:X}\n", start_i, len));
}

fn append_log_finished(start_i: u128) {
    append_log_line(format!("FINISHED {:X}\n", start_i));
}

fn append_log_line(line: String) {
    let path = ALLOC_LOG_FILE;

    if let Err(e) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(line.as_bytes()))
    {
        eprintln!("Failed to append to {}: {}", path, e);
    }
}

fn load_assignment_log() -> (VecDeque<Chunk>, u128) {
    use std::collections::{HashMap, HashSet};
    use std::io::{self, BufRead};

    let mut assigned: HashMap<u128, u128> = HashMap::new();
    let mut finished: HashSet<u128> = HashSet::new();
    let mut farthest_end: u128 = 0;
    let mut pending = VecDeque::new();
    let path = ALLOC_LOG_FILE;

    if let Ok(file) = File::open(path) {
        for line in io::BufReader::new(file).lines().map_while(Result::ok) {
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "ASSIGNED" if parts.len() == 3 => {
                    if let (Ok(s), Ok(l)) = (
                        u128::from_str_radix(parts[1], 16),
                        u128::from_str_radix(parts[2], 16),
                    ) {
                        assigned.insert(s, l);
                        farthest_end = farthest_end.max(s + l);
                    }
                }

                "FINISHED" if parts.len() == 2 => {
                    if let Ok(s) = u128::from_str_radix(parts[1], 16) {
                        finished.insert(s);
                    }
                }

                _ => {
                    // Unknown line; safely ignore
                }
            }
        }
    }

    // Any ASSIGNED not FINISHED is pending work (to avoid skip after crash)
    for (s, l) in assigned.into_iter() {
        if !finished.contains(&s) {
            pending.push_back(Chunk { start_i: s, len: l });
        }
    }

    (pending, farthest_end)
}

fn save_found_key(thread_id: u64, key_bytes: &[u8; 32], decimal: u128, address: String) {
    let magic_key = format!(
        "Private Key (hex): {}\nPrivate Key (dec): {}\nAddress: {}\n",
        hex_encode(key_bytes),
        decimal,
        address
    );

    if let Err(e) = std::fs::write(FOUND_FILE, magic_key) {
        eprintln!("Failed to save found key by CPU{}: {}", thread_id, e);
    }

    println!("\n\n!!! PRIVATE KEY FOUND BY THREAD {} !!!", thread_id);
    println!("Private key (hex): {}", hex_encode(key_bytes));
    println!("Private key (dec): {}", decimal);
    println!("Address: {}", address);
    println!("Saved to {}", FOUND_FILE);
}
