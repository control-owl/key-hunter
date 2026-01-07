// authors = ["Control Owl <c0ntr01-_-0w1[at]r-o0-t[dot]wtf>"]
// license = "MIT [2025] Control Owl"

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

// Target Bitcoin address for Puzzle #72 (compressed P2PKH)
// Puzzle #72 uses exactly 71 bits of entropy (after the leading '8' bit)
const TARGET_ADDRESS: &str = "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR";
const TARGET_ENCODED: [u8; 20] = [
    191, 116, 19, 232, 223, 78, 122, 52, 206, 157, 193, 62, 47, 38, 72, 120, 62, 197, 74, 219,
];

// Search mode: Sequence or PCG
const SEQUENCE_MODE: bool = true;

// Define only the start and the bit-size; derive end at runtime
const RANGE_START: u128 = 0x800000000000000000;
const N_BITS: u32 = 71;
const N_MASK: u128 = (1u128 << N_BITS) - 1;

// Linear Congruential Generator (LCG) parameters for permutation
// A_CONST: Multiplier — must be odd and co-prime with 2^71 for full cycle
// Using golden ratio conjugate (φ-1) gives excellent statistical properties
// Increasing/decreasing: minor effect on distribution, negligible on speed
// B_CONST: Increment — arbitrary constant, improves avalanche
const A_CONST: u128 = 0x9E3779B97F4A7C15u128;
const B_CONST: u128 = 0x1D3F84A5B7C29E3u128;

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

// Paths & persistence
const BACKUPS_DIR: &str = "backups";
const STATUS_DIR: &str = "status";
const FOUND_FILE: &str = "status/address.txt"; // save found result here
const GLOBAL_NEXT_FILE: &str = "status/GLOBAL_NEXT"; // next unassigned index (hex)
const ALLOC_LOG_FILE: &str = "status/ALLOC.log"; // assignment log

// Dashboard & persistence cadence
const DASHBOARD_UPDATE_INTERVAL_MS: u128 = 500; // per-thread display update
const PROGRESS_SAVE_INTERVAL_SEC: u64 = 5; // per-thread checkpoint

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

// Batch parallelization inside a thread iteration
const CPU_PARALLEL_KEYS: usize = 64;
// Chunk size claimed atomically per assignment. Larger chunks reduce allocator contention
// but increase potential rework on crash; we use assignment log to avoid skips.
const CPU_CHUNK_SIZE: u128 = (CPU_PARALLEL_KEYS as u128) * 1024; // 64 * 1024 = 65,536 keys per claim

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

const GPU_TEST_MODE: bool = false;
const GPU_PARALLEL_KEYS: u64 = 4;
// Host (CPU)
//  └── GPU_CHUNK_SIZE   ← how much work you give the GPU per launch
//       └── GRID        ← how many blocks exist concurrently
//            └── BLOCK  ← how many threads cooperate per block
//                 └── GPU_PARALLEL_KEYS ← how much work each thread does

// Number of Streaming Multiprocessors (SMs) available on the GPU.
// Each SM executes multiple warps concurrently and represents a unit of parallel execution.
// This value is hardware-specific and directly impacts total parallel throughput.
const GPU_SM_COUNT: u32 = 16; // NVIDIA GeForce GTX 1650 Mobile

// Total number of keys processed by the GPU in a single kernel launch.
// Larger values reduce kernel launch and synchronization overhead,
// but excessively large chunks can distort benchmarks and increase latency.
// This value should be large enough to amortize overhead, but small enough
// to complete within a few seconds for accurate throughput measurement.
const GPU_CHUNK_SIZE: u128 = 1024 * 12 * GPU_SM_COUNT as u128;

// Total number of CUDA blocks launched per kernel.
// Should be a multiple of the SM count to ensure full device saturation.
// Increasing beyond SM saturation provides no additional performance benefit.
const GPU_GRID_SIZE: u32 = GPU_SM_COUNT * 8;

// Number of threads per CUDA block.
// Must be a multiple of 32 (warp size).
// Controls occupancy, register pressure, and SM utilization.
// Values between 128 and 256 provide the best balance for EC-heavy workloads.
const GPU_BLOCK_SIZE: u32 = GPU_SM_COUNT * 16;

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

use cust::{init, prelude::*};
use rayon::prelude::*;
use ripemd::Ripemd160;
use secp256k1::Secp256k1;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::BufRead;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};
use std::{panic, thread};

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

static TOTAL_CHECKED_LOW: AtomicU64 = AtomicU64::new(0);
static TOTAL_CHECKED_HIGH: AtomicU64 = AtomicU64::new(0);
static FOUND: AtomicBool = AtomicBool::new(false);
static PANIC_OCCURRED: AtomicBool = AtomicBool::new(false);

lazy_static::lazy_static! {
    static ref DASHBOARD: Arc<Mutex<Vec<ThreadStatus>>> = Arc::new(Mutex::new(Vec::new()));
}

thread_local! {
    static LOCAL_SECP: Secp256k1<secp256k1::All> = Secp256k1::new();
}

#[derive(Clone)]
struct ThreadStatus {
    worker_type: WorkerType,
    checked: u128,
    last_i: u128,    // last domain index processed (0..2^71-1)
    key_hex: String, // last key hex shown
    speed_kps: f64,
    last_update: Instant,
    last_checked: u128,
}

impl Default for ThreadStatus {
    fn default() -> Self {
        Self {
            worker_type: WorkerType::CPU,
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

#[derive(Clone, Copy)]
enum WorkerType {
    CPU,
    GPU,
}

enum AllocCommand {
    RequestChunk {
        chunk_size: u128,
        reply: Sender<Option<Chunk>>,
    },

    ReportFinished {
        start_i: u128,
        len: u128,
    },
    SaveNow,
    Stop,
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

    fn search_batch(&self, start_i: u128) -> Result<Option<u128>, Box<dyn Error>> {
        let func = self.module.get_function("generate_and_check_keys")?;

        let search_mode = SEQUENCE_MODE;

        let range_start: u128 = if GPU_TEST_MODE { 1 } else { RANGE_START };
        let block: u32 = if GPU_TEST_MODE { 1 } else { GPU_BLOCK_SIZE };
        let grid: u32 = if GPU_TEST_MODE { 1 } else { GPU_GRID_SIZE };
        let parallel_keys: u64 = if GPU_TEST_MODE { 1 } else { GPU_PARALLEL_KEYS };

        let a_val: u128 = A_CONST;
        let b_val: u128 = B_CONST;

        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                search_mode,
                parallel_keys,
                start_i as u64,               // low
                (start_i >> 64) as u64,       // high
                GPU_CHUNK_SIZE as u64,
                a_val as u64,                 // low
                (a_val >> 64) as u64,         // high
                b_val as u64,                 // low
                (b_val >> 64) as u64,         // high
                range_start as u64,           // low
                (range_start >> 64) as u64,   // high
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

            let actual_key = if search_mode {
                found_i
            } else {
                let x = permute_index(found_i);
                RANGE_START + x
            };

            Ok(Some(actual_key))
        }
    }
}

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

#[inline(always)]
fn permute_index(i: u128) -> u128 {
    if SEQUENCE_MODE {
        i
    } else {
        let mut y = A_CONST.wrapping_mul(i).wrapping_add(B_CONST) & N_MASK;

        // PCG mixing (match GPU)
        let high: u64 = (y >> 64) as u64;
        let mut low: u64 = y as u64;
        low ^= high;
        low = low.wrapping_mul(0x9E3779B97F4A7C15);
        let mut mixed_high: u64 = ((high << 32) | (low >> 32)) ^ low;
        mixed_high = mixed_high.wrapping_mul(0xBDD1F3B71727E72B);
        y = ((mixed_high as u128) << 64) | (low as u128);
        y ^= y >> 67;

        // Bound back to mod 2^71
        y &= N_MASK;

        y
    }
}

// fn print_dashboard(mode: &str, start: u128, end: u128, global_start: Instant, num_threads: usize) {
//     let dash = DASHBOARD.lock().unwrap();
//
//     let low = TOTAL_CHECKED_LOW.load(Ordering::Relaxed);
//     let high = TOTAL_CHECKED_HIGH.load(Ordering::Relaxed);
//     let total_checked_u128 = (high as u128) << 64 | (low as u128);
//
//     let elapsed = global_start.elapsed();
//     let elapsed_secs = elapsed.as_secs_f64();
//
//     let overall_speed_kps = if elapsed_secs > 0.1 {
//         total_checked_u128 as f64 / elapsed_secs / 1_000.0
//     } else {
//         0.0
//     };
//
//     let h = elapsed.as_secs() / 3600;
//     let m = (elapsed.as_secs() % 3600) / 60;
//     let s = elapsed.as_secs() % 60;
//
//     let total_keys = end - start + 1;
//     let progress = if total_keys > 0 {
//         total_checked_u128 as f64 / total_keys as f64
//     } else {
//         0.0
//     };
//     let progress_percent = progress * 100.0;
//
//     // progress bar 50 chars wide
//     let bar_width = 50;
//     let filled = (progress * bar_width as f64).min(bar_width as f64).max(0.0) as usize;
//     let mut bar = String::new();
//     for i in 0..bar_width {
//         if i < filled {
//             bar.push('█');
//         } else {
//             bar.push('░');
//         }
//     }
//
//     // ANSI colors
//     let reset = "\x1b[0m";
//     let cpu_color = "\x1b[34m"; // blue
//     let gpu_color = "\x1b[32m"; // green
//
//     print!("\x1B[2J\x1B[H");
//     println!("╔═════════════════════════════════════════╗");
//     println!("║     KEY HUNTER - BITCOIN PUZZLE #72     ║");
//     println!("╚═════════════════════════════════════════╝");
//     println!("Target Address: {}", TARGET_ADDRESS);
//     println!("Range : {:019X} ──▶ {:019X}", start, end);
//     println!("Threads : {}", num_threads);
//     println!(
//         "Parallel tasks: {}\n",
//         if mode == "CPU" {
//             CPU_PARALLEL_KEYS
//         } else {
//             GPU_PARALLEL_KEYS as usize
//         }
//     );
//
//     for t in 0..num_threads {
//         let status = &dash[t];
//         let idx_hex = format!("{:019X}", status.last_i);
//
//         let label = match status.worker_type {
//             WorkerType::CPU => "CPU",
//             WorkerType::GPU => "GPU",
//         };
//
//         println!(
//             "{} {:2} Index: {} Current key: {} Checked: {:8}K keys Speed: {:6.1}K keys/s",
//             label,
//             t,
//             idx_hex,
//             status.key_hex,
//             status.checked / 1_000,
//             status.speed_kps
//         );
//
//         // println!(
//         //     "{} {:2} Index: {} Current key: {} Checked: {:8}K keys Speed: {:6.1}K keys/s",
//         //     mode,
//         //     t,
//         //     idx_hex,
//         //     status.key_hex,
//         //     status.checked / 1_000,
//         //     status.speed_kps
//         // );
//     }
//
//     println!(
//         "\n╔════════════════════════════════════════════════════════════════════════════════════════════════╗"
//     );
//     println!(
//         "║ TOTAL CHECKED: {:12}K keys │ OVERALL SPEED: {:8.1}K keys/s │ Elapsed: {:4}h {:3}m {:3}s ║",
//         (total_checked_u128) / 1_000,
//         overall_speed_kps,
//         h,
//         m,
//         s
//     );
//     println!(
//         "╚════════════════════════════════════════════════════════════════════════════════════════════════╝"
//     );
//
//     let _ = io::stdout().flush();
// }

// fn save_thread_checkpoint(thread_id: u64, index: u128) {
//     let path = format!("{}/CPU{}", STATUS_DIR, thread_id);
//     let hex = format!("{:X}", index);
//
//     if let Err(e) = std::fs::write(&path, hex) {
//         eprintln!("Failed to save CPU{} progress: {}", thread_id, e);
//     }
// }

fn save_alloc_state(alloc_arc: &Arc<Mutex<AllocState>>) {
    let a = alloc_arc.lock().unwrap();

    let path = GLOBAL_NEXT_FILE;

    if let Err(e) = std::fs::write(path, format!("{:X}", a.next_i)) {
        eprintln!("Failed to save GLOBAL_NEXT: {}", e);
    }
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

// fn save_found_key(thread_id: u64, key_bytes: &[u8; 32], decimal: u128, address: String) {
//     let magic_key = format!(
//         "Private Key (hex): {}\nPrivate Key (dec): {}\nAddress: {}\n",
//         hex_encode(key_bytes),
//         decimal,
//         address
//     );
//
//     if let Err(e) = std::fs::write(FOUND_FILE, magic_key) {
//         eprintln!("Failed to save found key by CPU{}: {}", thread_id, e);
//     }
//
//     println!("\n\n!!! PRIVATE KEY FOUND BY THREAD {} !!!", thread_id);
//     println!("Private key (hex): {}", hex_encode(key_bytes));
//     println!("Private key (dec): {}", decimal);
//     println!("Address: {}", address);
//     println!("Saved to {}", FOUND_FILE);
// }

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

    fs::remove_file(path).ok();

    (pending, farthest_end)
}

// -.-. --- .--. -.-- .-. .. --. .... - / -.-. --- -. - .-. --- .-.. / --- .-- .-..

fn main() {
    println!("╔════════════════════════════════════════╗");
    println!("║     BITCOIN PUZZLE #{} SOLVER          ║", N_BITS + 1);
    println!("╚════════════════════════════════════════╝");
    println!("Target: {}", TARGET_ADDRESS);
    println!("Range : {}", RANGE_START);
    println!();

    loop {
        print!("Select mode:\n  [1] CPU\n  [2] GPU\n  [3] CPU+GPU\n\nChoice (1/2/3): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let choice = input.trim().to_lowercase();

        match choice.as_str() {
            "1" | "cpu" => {
                println!("\nStarting CPU-only mode...\n");
                start_solver("CPU");
                break;
            }
            "2" | "gpu" => {
                println!("\nStarting GPU-only mode...\n");
                start_solver("GPU");
                break;
            }
            "3" | "both" | "cpu+gpu" => {
                println!("\nStarting CPU+GPU mode...\n");
                start_solver("Both");
                break;
            }
            _ => {
                println!("Invalid choice. Please enter 1, 2 or 3.\n");
            }
        }
    }
}

fn start_solver(mode: &str) {
    let start: u128 = RANGE_START;
    let total_keys: u128 = 1u128 << N_BITS;
    let end: u128 = start + total_keys - 1;

    let shutdown = Arc::new(AtomicBool::new(false));
    let global_start = Instant::now();

    // 1. Create status directory
    setup_status_dir();

    // 2. Recover allocator state
    let alloc = Arc::new(Mutex::new(recover_allocator(total_keys)));

    // 3. Install panic hook
    install_panic_hook(alloc.clone());

    // 4. Dashboard initialization
    let cpu_threads = num_cpus::get();
    let (workers, gpu_dash_index) = match mode {
        "CPU" => (cpu_threads, None),
        "GPU" => (1, Some(0)),
        "Both" => (cpu_threads + 1, Some(cpu_threads)),
        _ => panic!("Invalid mode"),
    };

    {
        let mut dash = DASHBOARD.lock().unwrap();
        dash.clear();
        dash.resize(workers, ThreadStatus::default());

        match mode {
            "CPU" => {
                // all are CPU by default
            }
            "GPU" => {
                if !dash.is_empty() {
                    dash[0].worker_type = WorkerType::GPU;
                }
            }
            "Both" => {
                // first cpu_threads entries = CPU, last one = GPU
                let cpu_threads = num_cpus::get();
                if workers > cpu_threads {
                    dash[cpu_threads].worker_type = WorkerType::GPU;
                }
            }
            _ => {}
        }
    }

    // 5. Allocator thread + command channel
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<AllocCommand>();

    let allocator_handle = spawn_allocator_thread(
        alloc.clone(),
        cmd_rx,
        total_keys,
        shutdown.clone(),
        ready_tx,
    );

    let _ = ready_rx.recv();

    // 6. Dashboard thread
    let mode_string = mode.to_string();
    let dash_thread = spawn_dashboard_thread(
        mode_string.clone(),
        shutdown.clone(),
        start,
        end,
        global_start,
        workers,
    );

    // 7. Start workers according to mode
    let mut worker_handles: Vec<JoinHandle<()>> = Vec::new();

    match mode {
        "CPU" => {
            // CPU-only
            worker_handles.push(spawn_cpu_workers(
                cmd_tx.clone(),
                shutdown.clone(),
                start,
                cpu_threads,
            ));
        }
        "GPU" => {
            // GPU-only
            worker_handles.push(spawn_gpu_worker(
                cmd_tx.clone(),
                shutdown.clone(),
                start,
                0, // dashboard index 0
            ));
        }
        "Both" => {
            // CPU workers
            worker_handles.push(spawn_cpu_workers(
                cmd_tx.clone(),
                shutdown.clone(),
                start,
                cpu_threads,
            ));

            // GPU worker (last dashboard slot)
            if let Some(idx) = gpu_dash_index {
                worker_handles.push(spawn_gpu_worker(
                    cmd_tx.clone(),
                    shutdown.clone(),
                    start,
                    idx,
                ));
            }
        }
        _ => unreachable!(),
    }

    // 8. Wait for workers to finish
    for handle in worker_handles {
        let _ = handle.join();
    }

    // 9. Tell allocator to stop and wait
    let _ = cmd_tx.send(AllocCommand::Stop);
    let _ = allocator_handle.join();

    // 10. Stop dashboard
    shutdown.store(true, Ordering::Relaxed);
    let _ = dash_thread.join();
}

fn setup_status_dir() {
    if Path::new(STATUS_DIR).exists() && fs::read_dir(STATUS_DIR).unwrap().count() > 0 {
        backup_and_clean();
    } else {
        fs::create_dir_all(STATUS_DIR).expect("Failed to create status directory");
    }
}

fn recover_allocator(total_keys: u128) -> AllocState {
    let alloc_log_path = if Path::new(ALLOC_LOG_FILE).exists() {
        ALLOC_LOG_FILE.to_string()
    } else {
        find_latest_backup_alloc_log()
    };

    let (pending, farthest_end_from_log) = if !alloc_log_path.is_empty() {
        load_assignment_log_from_path(&alloc_log_path)
    } else {
        (VecDeque::new(), 0)
    };

    let next_i = {
        let next_i_from_file = load_next_i();
        let resume_point = std::cmp::max(farthest_end_from_log, next_i_from_file);

        if SEQUENCE_MODE {
            // Sequence mode: start from 1 only if no progress exists
            if resume_point == 0 {
                1u128
            } else {
                resume_point
            }
        } else {
            // Normal mode: resume from farthest progress
            resume_point
        }
    }
    .min(total_keys);

    let _ = fs::remove_file(ALLOC_LOG_FILE);
    File::create(ALLOC_LOG_FILE).expect("Failed to create new ALLOC.log");

    println!(
        "Recovered {} unfinished chunks. Starting from index: {:X}",
        pending.len(),
        next_i
    );

    AllocState { next_i, pending }
}

fn spawn_allocator_thread(
    alloc: Arc<Mutex<AllocState>>,
    rx: Receiver<AllocCommand>,
    total_keys: u128,
    shutdown: Arc<AtomicBool>,
    ready_tx: Sender<()>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut last_save = Instant::now();
        let mut finished_chunks: BTreeMap<u128, u128> = BTreeMap::new();

        {
            let _state = alloc.lock().unwrap();
        }
        let _ = ready_tx.send(());

        while let Ok(cmd) = rx.recv() {
            match cmd {
                AllocCommand::RequestChunk { chunk_size, reply } => {
                    let mut state = alloc.lock().unwrap();

                    // Do pending first
                    if let Some(chunk) = state.pending.pop_front() {
                        let _ = reply.send(Some(chunk));
                        continue;
                    }

                    // Then allocate fresh
                    if state.next_i >= total_keys {
                        let _ = reply.send(None);
                        continue;
                    }

                    let start_i = state.next_i;
                    let len = std::cmp::min(chunk_size, total_keys - start_i);
                    state.next_i += len;

                    append_log_assigned(start_i, len);

                    // Periodic save
                    let now = Instant::now();
                    if now.duration_since(last_save).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC {
                        save_alloc_state(&alloc);
                        last_save = now;
                    }

                    let _ = reply.send(Some(Chunk { start_i, len }));
                }

                AllocCommand::ReportFinished { start_i, len } => {
                    append_log_finished(start_i);

                    let end_i = start_i + len;

                    if let Some((&prev_start, &prev_len)) =
                        finished_chunks.range(..start_i).next_back()
                    {
                        let prev_end = prev_start + prev_len;
                        if start_i < prev_end {
                            eprintln!(
                                "[WARN] Finished chunk overlap: [{:X}..{:X}) overlaps with previous [{:X}..{:X})",
                                start_i, end_i, prev_start, prev_end
                            );
                        }
                    }

                    if let Some((&next_start, &next_len)) =
                        finished_chunks.range(start_i + 1..).next()
                    {
                        let next_end = next_start + next_len;
                        if end_i > next_start {
                            eprintln!(
                                "[WARN] Finished chunk overlap: [{:X}..{:X}) overlaps with next [{:X}..{:X})",
                                start_i, end_i, next_start, next_end
                            );
                        }
                    }

                    finished_chunks.insert(start_i, len);

                    let now = Instant::now();
                    if now.duration_since(last_save).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC {
                        save_alloc_state(&alloc);
                        last_save = now;
                    }
                }

                AllocCommand::SaveNow => {
                    save_alloc_state(&alloc);
                    last_save = Instant::now();
                }

                AllocCommand::Stop => {
                    save_alloc_state(&alloc);

                    let state = alloc.lock().unwrap();
                    let completed_full_range = !FOUND.load(Ordering::Relaxed)
                        && state.next_i >= total_keys
                        && state.pending.is_empty();

                    if completed_full_range && !finished_chunks.is_empty() {
                        let mut iter = finished_chunks.iter();
                        if let Some((&first_start, &first_len)) = iter.next() {
                            let current_start = first_start;
                            let mut current_end = first_start + first_len;

                            if current_start > 0 {
                                eprintln!(
                                    "[WARN] Gap at beginning: finished starts at {:X}, expected 0",
                                    current_start
                                );
                            }

                            for (&s, &l) in iter {
                                if s > current_end {
                                    eprintln!(
                                        "[WARN] Gap detected: previous end {:X}, next start {:X}",
                                        current_end, s
                                    );
                                } else if s < current_end {
                                    eprintln!(
                                        "[WARN] Overlap detected while merging: [{:X}..{:X}) with [{:X}..{:X})",
                                        current_start,
                                        current_end,
                                        s,
                                        s + l
                                    );
                                }

                                let new_end = s + l;
                                if new_end > current_end {
                                    current_end = new_end;
                                }
                            }

                            if current_end < total_keys {
                                eprintln!(
                                    "[WARN] Gap at end: coverage ends at {:X}, expected {:X}",
                                    current_end, total_keys
                                );
                            }
                        }
                    }

                    shutdown.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
    })
}

fn install_panic_hook(alloc: Arc<Mutex<AllocState>>) {
    std::panic::set_hook(Box::new(move |info| {
        if PANIC_OCCURRED.swap(true, Ordering::Relaxed) {
            return;
        }

        eprintln!("PANIC: {}", info);
        let _ = std::panic::catch_unwind(|| {
            save_alloc_state(&alloc);
            backup_and_clean();
            eprintln!("Progress saved and backup done.");
        });
    }));
}

fn spawn_dashboard_thread(
    mode: String,
    shutdown: Arc<AtomicBool>,
    start: u128,
    end: u128,
    global_start: Instant,
    workers: usize,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            print_dashboard(&mode, start, end, global_start, workers);
            thread::sleep(Duration::from_secs(2));
        }
        print_dashboard(&mode, start, end, global_start, workers);
    })
}

fn spawn_cpu_workers(
    cmd_tx: Sender<AllocCommand>,
    shutdown: Arc<AtomicBool>,
    start: u128,
    threads: usize,
) -> JoinHandle<()> {
    thread::spawn(move || {
        (0..threads).into_par_iter().for_each(|t| {
            let thread_id = t;
            let mut checked: u128 = 0;
            let mut last_stat = Instant::now();
            let mut last_save_request = Instant::now();

            loop {
                if shutdown.load(Ordering::Relaxed) || FOUND.load(Ordering::Relaxed) {
                    break;
                }

                // Request a chunk for CPU
                let (reply_tx, reply_rx) = mpsc::channel();
                let _ = cmd_tx.send(AllocCommand::RequestChunk {
                    chunk_size: CPU_CHUNK_SIZE,
                    reply: reply_tx,
                });

                let chunk = match reply_rx.recv() {
                    Ok(Some(c)) => c,
                    _ => break, // No more work
                };

                let mut i = chunk.start_i;
                let end_i = chunk.start_i + chunk.len;

                while i < end_i {
                    if shutdown.load(Ordering::Relaxed) || FOUND.load(Ordering::Relaxed) {
                        break;
                    }

                    let batch_len_u128 = (end_i - i).min(CPU_PARALLEL_KEYS as u128);
                    let batch_len = batch_len_u128 as usize;

                    let mut key_bytes_batch = [[0u8; 32]; CPU_PARALLEL_KEYS];
                    let mut k_batch = [0u128; CPU_PARALLEL_KEYS];
                    let mut first_k_hex: Option<String> = None;

                    // Build batch
                    for p in 0..batch_len {
                        let current_i = i + p as u128;
                        let x = permute_index(current_i);
                        let k = start + x;
                        k_batch[p] = k;

                        if first_k_hex.is_none() {
                            first_k_hex = Some(format!("{:019X}", k));
                        }

                        let be = k.to_be_bytes();
                        key_bytes_batch[p][16..].copy_from_slice(&be);
                    }

                    // Check batch
                    for p in 0..batch_len {
                        let k = k_batch[p];
                        let key_bytes = key_bytes_batch[p];

                        LOCAL_SECP.with(|secp| {
                            if let Ok(sk) = secp256k1::SecretKey::from_byte_array(key_bytes) {
                                let pk = secp256k1::PublicKey::from_secret_key(secp, &sk);
                                let compressed: [u8; 33] = pk.serialize();
                                let h160 = Ripemd160::digest(Sha256::digest(compressed));

                                if h160.as_slice() == TARGET_ENCODED {
                                    save_found_key(k);
                                    FOUND.store(true, Ordering::Relaxed);
                                    shutdown.store(true, Ordering::Relaxed);
                                }
                            }
                        });
                    }

                    checked += batch_len as u128;
                    add_to_total_checked(batch_len as u128);

                    let now = Instant::now();

                    // Dashboard update
                    if now.duration_since(last_stat).as_millis() >= DASHBOARD_UPDATE_INTERVAL_MS {
                        let mut dash = DASHBOARD.lock().unwrap();
                        if thread_id < dash.len() {
                            let status = &mut dash[thread_id];
                            let elapsed = now.duration_since(status.last_update).as_secs_f64();
                            if elapsed > 0.1 {
                                status.speed_kps =
                                    (checked - status.last_checked) as f64 / elapsed / 1_000.0;
                            }
                            status.checked = checked;
                            status.last_i = i + RANGE_START;

                            let approx_key =
                                RANGE_START + permute_index(chunk.start_i.saturating_sub(1));
                            status.key_hex = format!("{:019X}", approx_key);

                            // status.key_hex = first_k_hex.clone().unwrap_or_default();
                            status.last_update = now;
                            status.last_checked = checked;
                        }
                        last_stat = now;
                    }

                    if now.duration_since(last_save_request).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC
                    {
                        let _ = cmd_tx.send(AllocCommand::SaveNow);
                        last_save_request = now;
                    }

                    i += CPU_PARALLEL_KEYS as u128;
                }

                let _ = cmd_tx.send(AllocCommand::ReportFinished {
                    start_i: chunk.start_i,
                    len: chunk.len,
                });
            }
        });
    })
}

fn spawn_gpu_worker(
    cmd_tx: Sender<AllocCommand>,
    shutdown: Arc<AtomicBool>,
    start: u128,
    dash_index: usize,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let solver = match GpuSolver::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("GPU initialization failed: {}", e);
                return;
            }
        };

        let mut last_save_request = Instant::now();

        while !shutdown.load(Ordering::Relaxed) && !FOUND.load(Ordering::Relaxed) {
            // Request GPU-sized chunk
            let (reply_tx, reply_rx) = mpsc::channel();
            let _ = cmd_tx.send(AllocCommand::RequestChunk {
                chunk_size: GPU_CHUNK_SIZE,
                reply: reply_tx,
            });

            let chunk = match reply_rx.recv() {
                Ok(Some(c)) => c,
                _ => break, // no more work
            };

            // --- GPU KERNEL TIMING START ---
            let gpu_start = Instant::now();
            let result = solver.search_batch(chunk.start_i);
            let gpu_end = Instant::now();
            let gpu_elapsed = gpu_end.duration_since(gpu_start).as_secs_f64();
            // --- GPU KERNEL TIMING END ---

            match result {
                Ok(Some(key)) => {
                    save_found_key(key);
                    FOUND.store(true, Ordering::Relaxed);
                    shutdown.store(true, Ordering::Relaxed);

                    let _ = cmd_tx.send(AllocCommand::ReportFinished {
                        start_i: chunk.start_i,
                        len: chunk.len,
                    });
                    break;
                }
                Ok(None) => {
                    add_to_total_checked(chunk.len);
                    let _ = cmd_tx.send(AllocCommand::ReportFinished {
                        start_i: chunk.start_i,
                        len: chunk.len,
                    });
                }
                Err(e) => {
                    eprintln!("GPU error: {}", e);
                    break;
                }
            }

            // --- DASHBOARD UPDATE ---
            let mut dash = DASHBOARD.lock().unwrap();
            if dash_index < dash.len() {
                let status = &mut dash[dash_index];

                // Compute speed ONLY from GPU kernel time
                if gpu_elapsed > 0.0 {
                    status.speed_kps = (chunk.len as f64) / gpu_elapsed / 1_000.0;
                }

                status.checked += chunk.len;
                status.last_i = start + chunk.start_i;
                status.last_update = gpu_end;
                status.last_checked = status.checked;
                let approx_key = RANGE_START + permute_index(chunk.start_i.saturating_sub(1));

                status.key_hex = format!("{:019X}", approx_key);
            }

            // --- PERIODIC SAVE ---
            let now = Instant::now();
            if now.duration_since(last_save_request).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC {
                let _ = cmd_tx.send(AllocCommand::SaveNow);
                last_save_request = now;
            }
        }

        if FOUND.load(Ordering::Relaxed) {
            println!("\nKEY FOUND BY GPU! Saved to {}", FOUND_FILE);
        } else {
            println!("\nGPU search completed. No key found.");
        }
    })
}

fn print_dashboard(mode: &str, start: u128, end: u128, global_start: Instant, num_threads: usize) {
    let dash = DASHBOARD.lock().unwrap();

    let low = TOTAL_CHECKED_LOW.load(Ordering::Relaxed);
    let high = TOTAL_CHECKED_HIGH.load(Ordering::Relaxed);
    let total_checked_u128 = (high as u128) << 64 | (low as u128);

    let elapsed = global_start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    let overall_speed_kps = if elapsed_secs > 0.1 {
        total_checked_u128 as f64 / elapsed_secs / 1_000.0
    } else {
        0.0
    };

    let h = elapsed.as_secs() / 3600;
    let m = (elapsed.as_secs() % 3600) / 60;
    let s = elapsed.as_secs() % 60;

    let total_keys = end - start + 1;
    let progress = if total_keys > 0 {
        total_checked_u128 as f64 / total_keys as f64
    } else {
        0.0
    };
    let progress_percent = progress * 100.0;

    // progress bar 50 chars wide
    let bar_width = 60;
    let filled = (progress * bar_width as f64).min(bar_width as f64).max(0.0) as usize;
    let mut bar = String::new();
    for i in 0..bar_width {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }

    // ANSI colors
    let reset = "\x1b[0m";
    let cpu_color = "\x1b[34m"; // blue
    let gpu_color = "\x1b[32m"; // green

    // Clear screen
    print!("\x1B[2J\x1B[H");
    println!("╔═════════════════════════════════════════╗");
    println!("║     KEY HUNTER - BITCOIN PUZZLE #72     ║");
    println!("╚═════════════════════════════════════════╝");
    println!("Target Address: {}", TARGET_ADDRESS);
    println!("Range : {:019X} ──▶ {:019X}", start, end);
    println!("Threads : {}", num_threads);
    if mode == "CPU" {
        println!("Parallel tasks: {}\n", CPU_PARALLEL_KEYS);
    } else if mode == "GPU" {
        println!("Parallel tasks: {}\n", GPU_PARALLEL_KEYS);
    } else {
    };

    // Separate counters for CPU and GPU display indices
    let mut cpu_idx = 0usize;
    let mut gpu_idx = 0usize;

    for t in 0..num_threads {
        let status = &dash[t];
        let idx_hex = format!("{:019X}", status.last_i);

        match status.worker_type {
            WorkerType::CPU => {
                print!("{}", cpu_color);
                println!(
                    "CPU {:2} Index: {} Current key: {} Checked: {:8}K keys Speed: {:6.1}K keys/s",
                    cpu_idx,
                    idx_hex,
                    status.key_hex,
                    status.checked / 1_000,
                    status.speed_kps
                );
                cpu_idx += 1;
            }
            WorkerType::GPU => {
                print!("{}", gpu_color);
                println!(
                    "GPU {:2} Index: {} Current key: {} Checked: {:8}K keys Speed: {:6.1}K keys/s",
                    gpu_idx,
                    idx_hex,
                    status.key_hex,
                    status.checked / 1_000,
                    status.speed_kps
                );
                gpu_idx += 1;
            }
        }

        print!("{}", reset);
    }

    println!(
        "\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  CHECKED: {:13}K keys │ SPEED: {:8.1}K keys/s │ PROGRESS: {:>8.12}% │ TIME: {:4}h {:3}m {:3}s  ║",
        (total_checked_u128) / 1_000,
        overall_speed_kps,
        progress_percent,
        h,
        m,
        s,
    );

    // println!("║ PROGRESS: {:>8.15}% │ [{}] ║", progress_percent, bar);
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    let _ = io::stdout().flush();
}

// #######################################

// fn run_cpu_solver() {
//     let mode = "CPU";
//     if Path::new(STATUS_DIR).exists() && fs::read_dir(STATUS_DIR).unwrap().count() > 0 {
//         backup_and_clean();
//     } else {
//         fs::create_dir_all(STATUS_DIR).expect("Failed to create status directory");
//         println!("First run detected — starting fresh.");
//     }
//
//     let num_threads = num_cpus::get();
//
//     let start: u128 = RANGE_START;
//     let total_keys_u128: u128 = 1u128 << N_BITS;
//     let end: u128 = start + total_keys_u128 - 1;
//
//     let global_start = Instant::now();
//     let shutdown = Arc::new(AtomicBool::new(false));
//
//     // Initialize dashboard
//     {
//         let mut dash = DASHBOARD.lock().unwrap();
//         dash.clear();
//         dash.extend((0..num_threads).map(|_| ThreadStatus::default()));
//     }
//
//     // Recover progress
//     let alloc_log_path = if Path::new(ALLOC_LOG_FILE).exists() {
//         ALLOC_LOG_FILE.to_string()
//     } else {
//         find_latest_backup_alloc_log()
//     };
//
//     let (pending_from_log, farthest_end_from_log) = if !alloc_log_path.is_empty() {
//         load_assignment_log_from_path(&alloc_log_path)
//     } else {
//         (VecDeque::new(), 0)
//     };
//
//     let initial_next_i = if SEQUENCE_MODE {
//         1u128
//     } else {
//         let next_i_from_file = load_next_i();
//         std::cmp::max(farthest_end_from_log, next_i_from_file)
//     };
//
//     let alloc = Arc::new(Mutex::new(AllocState {
//         next_i: initial_next_i.min(total_keys_u128),
//         pending: pending_from_log.clone(),
//     }));
//
//     let _ = fs::remove_file(ALLOC_LOG_FILE);
//     File::create(ALLOC_LOG_FILE).expect("Failed to create new ALLOC.log");
//
//     println!("Recovered {} unfinished chunks.", pending_from_log.len());
//
//     // Dashboard thread
//     let dash_thread = {
//         let dash_shutdown = shutdown.clone();
//         spawn(move || {
//             while !dash_shutdown.load(Ordering::Relaxed) {
//                 print_dashboard(mode, start, end, global_start, num_threads);
//                 sleep(Duration::from_secs(2));
//             }
//             print_dashboard(mode, start, end, global_start, num_threads);
//         })
//     };
//
//     // Panic hook
//     {
//         let alloc_for_panic = alloc.clone();
//         panic::set_hook(Box::new(move |_info| {
//             save_alloc_state(&alloc_for_panic);
//             eprintln!("\n\nPanic detected! Progress saved.");
//             backup_and_clean();
//             eprintln!("Backup done.");
//         }));
//     }
//
//     (0..num_threads).into_par_iter().for_each(|t| {
//         let t_usize = t;
//         let mut checked: u128 = 0;
//         let mut last_stat = Instant::now();
//         let mut last_save = Instant::now();
//
//         loop {
//             if shutdown.load(Ordering::Relaxed) {
//                 break;
//             }
//
//             let maybe_chunk = {
//                 let mut a = alloc.lock().unwrap();
//
//                 if let Some(ch) = a.pending.pop_front() {
//                     Some(ch)
//                 } else if a.next_i < total_keys_u128 {
//                     let start_i = a.next_i;
//                     let len = CPU_CHUNK_SIZE.min(total_keys_u128 - start_i);
//
//                     a.next_i += len;
//
//                     save_alloc_state(&alloc);
//                     append_log_assigned(start_i, len);
//
//                     Some(Chunk { start_i, len })
//                 } else {
//                     None
//                 }
//             };
//
//             let chunk = match maybe_chunk {
//                 Some(c) => c,
//                 None => break,
//             };
//
//             let mut i = chunk.start_i;
//             let end_i = chunk.start_i + chunk.len;
//
//             while i < end_i {
//                 if shutdown.load(Ordering::Relaxed) {
//                     break;
//                 }
//
//                 // Build batch of CPU_PARALLEL_KEYS
//                 let mut key_bytes_batch = [[0u8; 32]; CPU_PARALLEL_KEYS];
//                 let mut k_batch = [0u128; CPU_PARALLEL_KEYS];
//                 let mut first_k_hex: Option<String> = None;
//
//                 let batch_len_u128 = (end_i - i).min(CPU_PARALLEL_KEYS as u128);
//                 let batch_len = batch_len_u128 as usize;
//
//                 for p in 0..batch_len {
//                     let current_i = i + p as u128;
//                     let x = permute_index(current_i);
//                     let k = start + x; // derived key inside fixed 2^71 domain
//
//                     k_batch[p] = k;
//
//                     if first_k_hex.is_none() {
//                         first_k_hex = Some(format!("{:019X}", k));
//                     }
//
//                     let be = k.to_be_bytes();
//                     key_bytes_batch[p][16..].copy_from_slice(&be);
//                 }
//
//                 // Check all keys in batch
//                 for p in 0..batch_len {
//                     let k = k_batch[p];
//                     let key_bytes: [u8; 32] = key_bytes_batch[p];
//
//                     LOCAL_SECP.with(|secp| {
//                         if let Ok(sk) = secp256k1::SecretKey::from_byte_array(key_bytes) {
//                             let pk = secp256k1::PublicKey::from_secret_key(secp, &sk);
//                             let compressed: [u8; 33] = pk.serialize();
//
//                             let h160 = Ripemd160::digest(Sha256::digest(compressed));
//
//                             if h160.as_slice() == TARGET_ENCODED {
//                                 save_found_key(t as u64, &key_bytes, k, TARGET_ADDRESS.to_string());
//                                 save_alloc_state(&alloc);
//
//                                 shutdown.store(true, Ordering::Relaxed);
//                             }
//                         }
//                     });
//                 }
//
//                 // Exact accounting for partial batches
//                 checked += batch_len as u128;
//
//                 add_to_total_checked(batch_len as u128);
//
//                 let now = Instant::now();
//
//                 if now.duration_since(last_stat).as_millis() >= DASHBOARD_UPDATE_INTERVAL_MS {
//                     {
//                         let mut dash = DASHBOARD.lock().unwrap();
//                         let status = &mut dash[t_usize];
//                         let elapsed = now.duration_since(status.last_update).as_secs_f64();
//
//                         if elapsed > 0.1 {
//                             status.speed_kps =
//                                 (checked - status.last_checked) as f64 / elapsed / 1_000.0;
//                         }
//
//                         status.checked = checked;
//                         status.last_i = i + RANGE_START;
//                         status.key_hex = first_k_hex.unwrap_or_else(|| String::from(""));
//                         status.last_update = now;
//                         status.last_checked = checked;
//                     }
//
//                     last_stat = now;
//                 }
//
//                 if now.duration_since(last_save).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC {
//                     save_alloc_state(&alloc);
//
//                     last_save = now;
//                 }
//
//                 i += CPU_PARALLEL_KEYS as u128;
//             }
//
//             // Mark chunk finished in the assignment log
//             append_log_finished(chunk.start_i);
//         }
//
//         save_alloc_state(&alloc);
//         let mut dash = DASHBOARD.lock().unwrap();
//         if t_usize < dash.len() {
//             dash[t_usize].checked = checked;
//         }
//     });
//
//     shutdown.store(true, Ordering::Relaxed);
//     dash_thread.join().unwrap();
//
//     print_dashboard(mode, start, end, global_start, num_threads);
//
//     if !shutdown.load(Ordering::Relaxed) {
//         println!("\nSearch completed across active threads. No key found.");
//     }
// }
//
// fn run_gpu_solver() {
//     let mode = "GPU";
//
//     if Path::new(STATUS_DIR).exists() && fs::read_dir(STATUS_DIR).unwrap().count() > 0 {
//         backup_and_clean();
//     } else {
//         fs::create_dir_all(STATUS_DIR).expect("Failed to create status directory");
//         println!("First run (GPU) — starting fresh.");
//     }
//
//     let start: u128 = RANGE_START;
//     let total_keys_u128: u128 = 1u128 << N_BITS;
//     let end: u128 = start + total_keys_u128 - 1;
//
//     let solver = match GpuSolver::new() {
//         Ok(s) => s,
//         Err(e) => {
//             eprintln!("GPU initialization failed: {}", e);
//             return;
//         }
//     };
//
//     let shutdown = Arc::new(AtomicBool::new(false));
//     let global_start = Instant::now();
//
//     // Dashboard: show single "GPU" worker
//     {
//         let mut dash = DASHBOARD.lock().unwrap();
//         dash.clear();
//         dash.push(ThreadStatus {
//             key_hex: String::from("initializing..."),
//             ..Default::default()
//         });
//     }
//
//     // Recover progress
//     let alloc_log_path = if Path::new(ALLOC_LOG_FILE).exists() {
//         ALLOC_LOG_FILE.to_string()
//     } else {
//         find_latest_backup_alloc_log()
//     };
//
//     let (pending, farthest) = if !alloc_log_path.is_empty() {
//         load_assignment_log_from_path(&alloc_log_path)
//     } else {
//         (VecDeque::new(), 0)
//     };
//
//     let next_i = if SEQUENCE_MODE {
//         1u128
//     } else {
//         std::cmp::max(farthest, load_next_i()).min(total_keys_u128)
//     };
//
//     let mut current_i = next_i;
//     let mut recovered = pending;
//
//     let _ = fs::remove_file(ALLOC_LOG_FILE);
//     File::create(ALLOC_LOG_FILE).unwrap();
//
//     let alloc = Arc::new(Mutex::new(AllocState {
//         next_i: current_i.min(total_keys_u128),
//         pending: recovered.clone(),
//     }));
//
//     // Dashboard thread
//     let dash_thread = {
//         let shutdown = shutdown.clone();
//         spawn(move || {
//             while !shutdown.load(Ordering::Relaxed) {
//                 print_dashboard(mode, start, end, global_start, 1); // 1 = GPU "thread"
//                 sleep(Duration::from_secs(2));
//             }
//             print_dashboard(mode, start, end, global_start, 1);
//         })
//     };
//
//     let panic_alloc = alloc.clone();
//     // Panic hook
//     panic::set_hook(Box::new(move |_| {
//         eprintln!("\nPanic! Saving progress...");
//         save_alloc_state(&panic_alloc);
//         backup_and_clean();
//         eprintln!("Backup done.");
//     }));
//
//     println!("Starting GPU search from index: {:X}", current_i);
//
//     // First process recovered chunks
//     while let Some(chunk) = recovered.pop_front() {
//         if FOUND.load(Ordering::Relaxed) {
//             break;
//         }
//
//         println!(
//             "Processing recovered chunk: {:X}..{:X}",
//             chunk.start_i,
//             chunk.start_i + chunk.len
//         );
//
//         if let Some(key) = solver.search_batch(chunk.start_i).unwrap_or(None) {
//             FOUND.store(true, Ordering::Relaxed);
//             save_found_key_gpu(key);
//             break;
//         }
//
//         // add_to_total_checked(batch_len as u128);
//         current_i = chunk.start_i + chunk.len;
//         save_alloc_state(&alloc);
//         append_log_finished(chunk.start_i);
//
//         update_gpu_dashboard(current_i, chunk.len);
//     }
//
//     // Main search loop
//     let mut last_log_time = Instant::now();
//     let mut pending_chunks: Vec<(u128, u128)> = Vec::new(); // (start, len)
//
//     while current_i < total_keys_u128 && !FOUND.load(Ordering::Relaxed) {
//         let len = GPU_CHUNK_SIZE.min(total_keys_u128 - current_i);
//         let chunk_start = current_i;
//
//         // Remember chunk
//         pending_chunks.push((chunk_start, len));
//
//         // Perform GPU search
//         if let Some(key) = solver.search_batch(current_i).unwrap_or(None) {
//             FOUND.store(true, Ordering::Relaxed);
//             save_found_key_gpu(key);
//             break;
//         }
//
//         add_to_total_checked(len);
//         current_i += len;
//         update_gpu_dashboard(current_i, len);
//
//         // 4. Flush pending logs only every 5 seconds
//         let now = Instant::now();
//         if now.duration_since(last_log_time).as_secs() >= PROGRESS_SAVE_INTERVAL_SEC {
//             // 1. Update next_i
//             save_alloc_state(&alloc);
//
//             // Write all pending assigned chunks
//             for &(start, length) in &pending_chunks {
//                 append_log_assigned(start, length);
//             }
//
//             // Write all as finished
//             for &(start, _) in &pending_chunks {
//                 append_log_finished(start);
//             }
//
//             pending_chunks.clear();
//             last_log_time = now;
//         }
//     }
//
//     // Final flush on exit or found
//     if !pending_chunks.is_empty() {
//         for &(start, length) in &pending_chunks {
//             append_log_assigned(start, length);
//         }
//         for &(start, _) in &pending_chunks {
//             append_log_finished(start);
//         }
//     }
//
//     shutdown.store(true, Ordering::Relaxed);
//     dash_thread.join().unwrap();
//
//     if FOUND.load(Ordering::Relaxed) {
//         println!("\nKEY FOUND BY GPU! Saved to {}", FOUND_FILE);
//     } else {
//         println!("\nGPU search completed. No key found.");
//     }
// }

// fn update_gpu_dashboard(current_i: u128, checked_this_batch: u128) {
//     let now = Instant::now();
//     let mut dash = DASHBOARD.lock().unwrap();
//
//     if let Some(status) = dash.get_mut(0) {
//         let elapsed = now.duration_since(status.last_update).as_millis();
//
//         if elapsed >= 500 {
//             status.speed_kps = (checked_this_batch / elapsed * 1_000) as f64;
//         }
//
//         status.checked += checked_this_batch;
//         status.last_i = RANGE_START + current_i;
//
//         let approx_key = RANGE_START + permute_index(current_i.saturating_sub(1));
//
//         status.key_hex = format!("{:019X}", approx_key);
//         status.last_update = now;
//         status.last_checked = status.checked;
//     }
// }

fn save_found_key(private_key: u128) {
    let key_hex = format!("{:064x}", private_key);

    let content = format!(
        "PRIVATE KEY FOUND BY GPU!\nAddress: {}\nPrivate key (hex): {}",
        TARGET_ADDRESS, key_hex
    );

    let _ = fs::write(FOUND_FILE, content);

    println!(
        "\n!!! PRIVATE KEY FOUND BY GPU !!!\nHex: {}\nDec: {}",
        key_hex, private_key
    );
}

fn add_to_total_checked(added: u128) {
    let added_low = added as u64;
    let added_high = (added >> 64) as u64;

    let old_low = TOTAL_CHECKED_LOW.fetch_add(added_low, Ordering::Relaxed);
    let mut carry = if old_low > u64::MAX - added_low { 1 } else { 0 };
    carry += added_high; // Add high part + any low carry

    if carry > 0 {
        TOTAL_CHECKED_HIGH.fetch_add(carry, Ordering::Relaxed);
    }
}
