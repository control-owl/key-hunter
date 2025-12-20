// authors = ["Control Owl <c0ntr01-_-0w1[at]r-o0-t[dot]wtf>"]
// license = "CC-BY-NC-ND-4.0 [2025] Control Owl"

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

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
// B_CONST: Increment — arbitrary constant, improves avalanche
const A_CONST: u128 = 0x9E3779B97F4A7C15u128;
const B_CONST: u128 = 0x1D3F84A5B7C29E3u128;

// Paths & persistence
const BACKUPS_DIR: &str = "backups";
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
const CHUNK_SIZE: u128 = (PARALLEL_KEYS as u128) * 1024 * 10; // 64 * 1024 * 10 = 655,360 keys per claim

// Check if key derivation is working
// Correct hash160 for this key (verified with ecdsa + sha256 + ripemd160)
const TEST_MODE: bool = true;
const TEST_TARGET_ADDRESS: &str = "1L2GMvQ6nHXrXvNxiBVNDRtz6aov2e2Qv7";
const TEST_TARGET_ENCODED: [u8; 20] = [
    0x7e, 0x88, 0x9b, 0x9b, 0x14, 0x1e, 0xaa, 0x54, 0xea, 0x4c, 0x98, 0x85, 0x2c, 0xe9, 0xee, 0xda,
    0xcf, 0xb2, 0x10, 0x27,
];

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

use cust::{init, prelude::*};
use hex::encode as hex_encode;
use rayon::prelude::*;
use ripemd::Ripemd160;
use secp256k1::Secp256k1;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::BufRead;
use std::io::{self, Write};
use std::panic;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant, SystemTime};

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

static TOTAL_CHECKED: AtomicU64 = AtomicU64::new(0);
static FOUND: AtomicBool = AtomicBool::new(false);

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

struct GpuSolver {
    _ctx: Context,
    module: Module,
    stream: Stream,
    found_buffer: DeviceBuffer<u64>,
}

impl GpuSolver {
    fn new() -> Result<Self, Box<dyn Error>> {
        init(CudaFlags::empty())?;

        let device_count = Device::num_devices()?;
        println!("CUDA devices found: {}", device_count);
        if device_count == 0 {
            panic!("No CUDA device found!");
        }

        let device = Device::get_device(0)?;
        let name = device.name()?;
        println!("Using GPU: {}", name);

        let ctx = Context::new(device)?;

        let ptx = include_str!("../kernel.ptx");
        println!("PTX embedded: {} bytes", ptx.len());

        let module = Module::from_ptx(ptx, &[])?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let mut found_buffer = DeviceBuffer::zeroed(1)?;
        found_buffer.copy_from(&[u64::MAX])?;

        Ok(Self {
            _ctx: ctx,
            module,
            stream,
            found_buffer,
        })
    }

    fn search_batch(&self, start_i: u64) -> Result<Option<u128>, Box<dyn Error>> {
        let func = self.module.get_function("generate_and_check_keys")?;

        let block = 256;
        let grid = ((CHUNK_SIZE as u64 + block as u64 - 1) / block as u64) as u32;

        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                start_i,
                CHUNK_SIZE as u64,
                A_CONST as u64,
                B_CONST as u64,
                u128::from_str_radix(RANGE_START_HEX, 16).unwrap() as u64,
                self.found_buffer.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        let mut host_found = [0u64; 1];
        self.found_buffer.copy_to(&mut host_found)?;

        if host_found[0] == u64::MAX {
            Ok(None)
        } else {
            let found_i = host_found[0] as u128;
            let x = permute_index(found_i);
            let key = u128::from_str_radix(RANGE_START_HEX, 16).unwrap() + x;
            Ok(Some(key))
        }
    }
}

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

#[inline(always)]
fn permute_index(i: u128) -> u128 {
    if TEST_MODE {
        return i; // Sequential: no LCG, no reversal
    }
    let y = (A_CONST.wrapping_mul(i) + B_CONST) & N_MASK;
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
        "\n╔═══════════════════════════════════════════════════════════════════════════════════════╗"
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

fn backup_and_clean() {
    if !Path::new(BACKUPS_DIR).exists() {
        fs::create_dir(BACKUPS_DIR).expect("Failed to create backups directory");
    }

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let timestamp = format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        2025,
        12,
        20,
        (now / 3600) % 24,
        (now / 60) % 60,
        now % 60
    );
    let backup_path = format!("{}/{}", BACKUPS_DIR, timestamp);

    println!("Creating backup of previous session: {}", backup_path);

    if Path::new(STATUS_DIR).exists() {
        copy_dir_all(STATUS_DIR, &backup_path).expect("Failed to create backup");
    } else {
        fs::create_dir(&backup_path).expect("Failed to create empty backup dir");
    }

    println!("Backup completed.");

    fs::create_dir_all(STATUS_DIR).expect("Failed to recreate status directory");
}

fn copy_dir_all(src: impl AsRef<Path>, dst: &str) -> io::Result<()> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = format!("{}/{}", dst, path.file_name().unwrap().to_string_lossy());

        if path.is_dir() {
            copy_dir_all(&path, &target)?;
        } else {
            fs::copy(&path, target)?;
        }
    }

    Ok(())
}

fn find_latest_backup_alloc_log() -> String {
    let mut latest_time = 0u64;
    let mut latest_path = String::new();

    if Path::new(BACKUPS_DIR).exists() {
        for entry in fs::read_dir(BACKUPS_DIR).unwrap() {
            let path = entry.unwrap().path();

            if path.is_dir() {
                if let Some(alloc_path) = path.join("ALLOC.log").to_str() {
                    if Path::new(alloc_path).exists() {
                        if let Ok(metadata) = fs::metadata(&path) {
                            if let Ok(modified) = metadata.modified() {
                                if let Ok(time) = modified.duration_since(SystemTime::UNIX_EPOCH) {
                                    let secs = time.as_secs();

                                    if secs > latest_time {
                                        latest_time = secs;
                                        latest_path = alloc_path.to_string();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    latest_path
}

fn load_assignment_log_from_path(path: &str) -> (VecDeque<Chunk>, u128) {
    let mut assigned: std::collections::HashMap<u128, u128> = std::collections::HashMap::new();
    let mut finished: std::collections::HashSet<u128> = std::collections::HashSet::new();

    let mut farthest_end: u128 = 0;
    let mut pending = VecDeque::new();

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

                _ => {}
            }
        }
    }

    for (s, l) in assigned {
        if !finished.contains(&s) {
            pending.push_back(Chunk { start_i: s, len: l });
        }
    }

    fs::remove_file(&path).ok();

    (pending, farthest_end)
}

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

fn main() {
    println!("╔════════════════════════════════════════╗");
    println!("║     BITCOIN PUZZLE #72 SOLVER          ║");
    println!("╚════════════════════════════════════════╝");
    println!("Target: {}", TARGET_ADDRESS);
    println!(
        "Range : {:019X} ... (2^71 keys)",
        u128::from_str_radix(RANGE_START_HEX, 16).unwrap()
    );
    println!();

    loop {
        print!("Select mode:\n  [1] CPU only\n  [2] GPU only (not implemented yet)\n  [3] CPU + GPU (not implemented yet)\n\nChoice (1/2/3): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let choice = input.trim();

        match choice {
            "1" | "cpu" => {
                println!("\nStarting CPU-only mode...\n");
                run_cpu_solver();
                break;
            }
            "2" | "gpu" => {
                println!("\nStarting GPU-only mode...\n");
                run_gpu_solver();
                break;
            }
            "3" | "both" | "cpu+gpu" => {
                println!("\nCPU+GPU combined mode not implemented yet.\n");
                // Future: run_combined_solver();
            }
            _ => {
                println!("Invalid choice. Please enter 1, 2 or 3.\n");
            }
        }
    }
}

fn run_cpu_solver() {
    // === Backup & cleanup old session ===
    let target_encoded = if TEST_MODE {
        TEST_TARGET_ENCODED
    } else {
        TARGET_ENCODED
    };

    let target_address = if TEST_MODE {
        TEST_TARGET_ADDRESS
    } else {
        TARGET_ADDRESS
    };

    if Path::new(STATUS_DIR).exists() && fs::read_dir(STATUS_DIR).unwrap().count() > 0 {
        backup_and_clean();
    } else {
        fs::create_dir_all(STATUS_DIR).expect("Failed to create status directory");
        println!("First run detected — starting fresh.");
    }

    let num_threads = num_cpus::get();

    let start = u128::from_str_radix(RANGE_START_HEX, 16).unwrap();
    let total_keys_u128: u128 = 1u128 << N_BITS;
    let end = start + total_keys_u128 - 1;
    let global_start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    // Initialize dashboard
    {
        let mut dash = DASHBOARD.lock().unwrap();
        dash.clear();
        dash.extend((0..num_threads).map(|_| ThreadStatus::default()));
    }

    // Recover progress
    let alloc_log_path = if Path::new(ALLOC_LOG_FILE).exists() {
        ALLOC_LOG_FILE.to_string()
    } else {
        find_latest_backup_alloc_log()
    };

    let (pending_from_log, farthest_end_from_log) = if !alloc_log_path.is_empty() {
        load_assignment_log_from_path(&alloc_log_path)
    } else {
        (VecDeque::new(), 0)
    };

    let next_i_from_file = load_next_i();
    let initial_next_i = if TEST_MODE {
        0u128
    } else {
        std::cmp::max(farthest_end_from_log, next_i_from_file)
    };

    let alloc = Arc::new(Mutex::new(AllocState {
        next_i: initial_next_i.min(total_keys_u128),
        pending: pending_from_log.clone(),
    }));

    let _ = fs::remove_file(ALLOC_LOG_FILE);
    File::create(ALLOC_LOG_FILE).expect("Failed to create new ALLOC.log");

    println!("Recovered {} unfinished chunks.", pending_from_log.len());

    // Dashboard thread
    let dash_thread = {
        let dash_shutdown = shutdown.clone();
        spawn(move || {
            while !dash_shutdown.load(Ordering::Relaxed) {
                print_dashboard(start, end, global_start, num_threads);
                sleep(Duration::from_secs(2));
            }
            print_dashboard(start, end, global_start, num_threads);
        })
    };

    // Panic hook
    {
        let alloc_for_panic = alloc.clone();
        panic::set_hook(Box::new(move |_info| {
            save_alloc_state(&alloc_for_panic);
            eprintln!("\n\nPanic detected! Progress saved.");
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

                    LOCAL_SECP.with(|secp| {
                        if let Ok(sk) = secp256k1::SecretKey::from_byte_array(key_bytes) {
                            let pk = secp256k1::PublicKey::from_secret_key(secp, &sk);
                            let compressed: [u8; 33] = pk.serialize();

                            let h160 = Ripemd160::digest(Sha256::digest(compressed));

                            if h160.as_slice() == target_encoded {
                                save_found_key(t as u64, &key_bytes, k, target_address.to_string());
                                save_alloc_state(&alloc);

                                shutdown.store(true, Ordering::Relaxed);

                                return; // exit thread
                            }
                        }
                    });
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

        save_thread_checkpoint(t as u64, 0);
        let mut dash = DASHBOARD.lock().unwrap();
        if t_usize < dash.len() {
            dash[t_usize].checked = checked;
        }
    });

    shutdown.store(true, Ordering::Relaxed);
    dash_thread.join().unwrap();

    print_dashboard(start, end, global_start, num_threads);

    if !shutdown.load(Ordering::Relaxed) {
        println!("\nSearch completed across active threads. No key found.");
    }
}

fn run_gpu_solver() {
    if Path::new(STATUS_DIR).exists() && fs::read_dir(STATUS_DIR).unwrap().count() > 0 {
        backup_and_clean();
    } else {
        fs::create_dir_all(STATUS_DIR).expect("Failed to create status directory");
        println!("First run (GPU) — starting fresh.");
    }

    let start = u128::from_str_radix(RANGE_START_HEX, 16).unwrap();
    let total_keys = 1u128 << N_BITS;
    let end = start + total_keys - 1;

    let solver = match GpuSolver::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GPU initialization failed: {}", e);
            return;
        }
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    let global_start = Instant::now();

    // Dashboard: show single "GPU" worker
    {
        let mut dash = DASHBOARD.lock().unwrap();
        dash.clear();
        dash.push(ThreadStatus {
            key_hex: String::from("initializing..."),
            ..Default::default()
        });
    }

    // Recover progress
    let alloc_log_path = if Path::new(ALLOC_LOG_FILE).exists() {
        ALLOC_LOG_FILE.to_string()
    } else {
        find_latest_backup_alloc_log()
    };

    let (pending, farthest) = if !alloc_log_path.is_empty() {
        load_assignment_log_from_path(&alloc_log_path)
    } else {
        (VecDeque::new(), 0)
    };

    let next_i = if TEST_MODE {
        0u128
    } else {
        std::cmp::max(farthest, load_next_i()).min(total_keys)
    };

    let mut current_i = next_i;
    let mut recovered = pending;

    let _ = fs::remove_file(ALLOC_LOG_FILE);
    File::create(ALLOC_LOG_FILE).unwrap();

    // Dashboard thread
    let dash_thread = {
        let shutdown = shutdown.clone();
        spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                print_dashboard(start, end, global_start, 1); // 1 = GPU "thread"
                sleep(Duration::from_secs(2));
            }
            print_dashboard(start, end, global_start, 1);
        })
    };

    // Panic hook
    panic::set_hook(Box::new(move |_| {
        eprintln!("\nPanic! Saving progress...");
        save_next_i(current_i);
    }));

    println!("Starting GPU search from index: {:X}", current_i);

    // First process recovered chunks
    while let Some(chunk) = recovered.pop_front() {
        if FOUND.load(Ordering::Relaxed) {
            break;
        }

        println!(
            "Processing recovered chunk: {:X}..{:X}",
            chunk.start_i,
            chunk.start_i + chunk.len
        );

        if let Some(key) = solver.search_batch(chunk.start_i as u64).unwrap_or(None) {
            FOUND.store(true, Ordering::Relaxed);
            save_found_key_gpu(key);
            break;
        }

        TOTAL_CHECKED.fetch_add(chunk.len as u64, Ordering::Relaxed);
        current_i = chunk.start_i + chunk.len;
        save_next_i(current_i);
        append_log_finished(chunk.start_i);

        update_gpu_dashboard(current_i, chunk.len as u64);
    }

    // Main search loop
    while current_i < total_keys && !FOUND.load(Ordering::Relaxed) {
        let len = CHUNK_SIZE.min(total_keys - current_i);
        let chunk_start = current_i;

        append_log_assigned(chunk_start, len);
        save_next_i(current_i + len);

        if let Some(key) = solver.search_batch(current_i as u64).unwrap_or(None) {
            FOUND.store(true, Ordering::Relaxed);
            save_found_key_gpu(key);
            break;
        }

        TOTAL_CHECKED.fetch_add(len as u64, Ordering::Relaxed);
        current_i += len;
        append_log_finished(chunk_start);

        update_gpu_dashboard(current_i, len as u64);
    }

    shutdown.store(true, Ordering::Relaxed);
    dash_thread.join().unwrap();

    if FOUND.load(Ordering::Relaxed) {
        println!("\nKEY FOUND BY GPU! Saved to {}", FOUND_FILE);
    } else {
        println!("\nGPU search completed. No key found.");
    }
}

fn update_gpu_dashboard(current_i: u128, checked_this_batch: u64) {
    let now = Instant::now();
    let mut dash = DASHBOARD.lock().unwrap();

    if let Some(status) = dash.get_mut(0) {
        let elapsed = now.duration_since(status.last_update).as_secs_f64();

        if elapsed > 0.5 {
            status.speed_kps = checked_this_batch as f64 / elapsed / 1_000.0;
        }

        status.checked += checked_this_batch;
        status.last_i = current_i;

        let approx_key = u128::from_str_radix(RANGE_START_HEX, 16).unwrap()
            + permute_index(current_i.saturating_sub(1));

        status.key_hex = format!("{:032x}", approx_key);
        status.last_update = now;
        status.last_checked = status.checked;
    }
}

fn save_found_key_gpu(private_key: u128) {
    let key_hex = format!("{:032x}", private_key);
    let content = format!(
        "PRIVATE KEY FOUND BY GPU!\nAddress: {}\nPrivate key (hex): {}\nPrivate key (dec): {}\n",
        TARGET_ADDRESS, key_hex, private_key
    );

    let _ = fs::write(FOUND_FILE, content);

    println!("\n!!! PRIVATE KEY FOUND BY GPU !!!\nHex: {}", key_hex);
}
