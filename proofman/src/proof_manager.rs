use std::sync::{Arc, Mutex, Condvar};
use std::collections::VecDeque;
use proofman_starks_lib_c::{register_proof_done_callback_c};
use proofman_common::{ProofCtx, ProofType};
use p3_field::Field;

pub struct Counter {
    counter: Mutex<usize>,
    cvar: Condvar,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { counter: Mutex::new(0), cvar: Condvar::new() }
    }

    pub fn increment(&self) -> usize {
        let mut count = self.counter.lock().unwrap();
        *count += 1;
        *count
    }

    pub fn decrement(&self) {
        let mut count = self.counter.lock().unwrap();
        *count -= 1;
        if *count == 0 {
            self.cvar.notify_all();
        }
    }

    pub fn wait_until_zero(&self) {
        let mut count = self.counter.lock().unwrap();
        while *count > 0 {
            count = self.cvar.wait(count).unwrap();
        }
    }

    pub fn wait_until_zero_and_check_streams<F: FnMut()>(&self, mut check_streams: F) {
        let mut count = self.counter.lock().unwrap();
        while *count > 0 {
            check_streams();

            let (c, _) = self.cvar.wait_timeout(count, std::time::Duration::from_micros(100)).unwrap();
            count = c;
        }
    }
}

#[derive(Debug)]
pub enum WitnessType {
    Basic(usize),
    Compressor(usize, usize, usize),
    Recursive1(usize, usize, usize),
    Recursive2(usize, usize, usize),
}

impl WitnessType {
    fn priority(&self) -> u8 {
        match self {
            WitnessType::Recursive2(..) => 3,
            WitnessType::Recursive1(..) | WitnessType::Compressor(..) => 2,
            WitnessType::Basic(..) => 1,
        }
    }
}

#[derive(Default)]
pub struct FastQueue {
    queue: Mutex<WitnessQueues>,
    condvar: Condvar,
}

#[derive(Default)]
struct WitnessQueues {
    recursive2: VecDeque<WitnessType>,
    recursive1: VecDeque<WitnessType>,
    basic: VecDeque<WitnessType>,
}

impl FastQueue {
    pub fn push(&self, witness_type: WitnessType) {
        let mut witness = self.queue.lock().unwrap();
        match witness_type.priority() {
            3 => witness.recursive2.push_back(witness_type),
            2 => witness.recursive1.push_back(witness_type),
            1 => witness.basic.push_back(witness_type),
            _ => unreachable!(),
        }
        self.condvar.notify_one();
    }

    pub fn pop(&self) -> WitnessType {
        let mut queue = self.queue.lock().unwrap();
        loop {
            if let Some(witness_type) = queue.recursive2.pop_front() {
                return witness_type;
            }
            if let Some(witness_type) = queue.recursive1.pop_front() {
                return witness_type;
            }
            if let Some(witness_type) = queue.basic.pop_front() {
                return witness_type;
            }
            queue = self.condvar.wait(queue).unwrap();
        }
    }
}

#[derive(Debug)]
struct BufferData {
    items: VecDeque<(usize, usize)>,
    capacity: usize,
    basic_count: usize, // NEW: Track only Basic items
    closed: bool,
}

#[derive(Debug)]
pub struct WitnessBuffer {
    queue: Mutex<BufferData>,
    condvar: Condvar,
}

impl WitnessBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            queue: Mutex::new(BufferData {
                items: VecDeque::with_capacity(cap),
                capacity: cap,
                basic_count: 0,
                closed: false,
            }),
            condvar: Condvar::new(),
        }
    }

    pub fn push(&self, item: (usize, usize)) -> bool {
        let mut data = self.queue.lock().unwrap();
        while item.1 == ProofType::Basic as usize && data.basic_count >= data.capacity {
            if data.closed {
                return false;
            }
            data = self.condvar.wait(data).unwrap();
        }

        if item.1 == ProofType::Basic as usize {
            data.basic_count += 1;
        }

        data.items.push_back(item);
        self.condvar.notify_all();
        true
    }

    pub fn pop(&self) -> Option<(usize, usize)> {
        let mut data = self.queue.lock().unwrap();

        loop {
            let (mut basic, mut recursive1, mut recursive2, mut compressor) = (0, 0, 0, 0);
            for &(_, proof_type) in data.items.iter() {
                match proof_type {
                    x if x == ProofType::Basic as usize => basic += 1,
                    _ => {}
                }
            }

            println!("[pop_recursive] Queue: basic={}", basic,);

            if let Some((id, proof_type)) = data.items.pop_front() {
                if proof_type == ProofType::Basic as usize {
                    data.basic_count = data.basic_count.saturating_sub(1);
                }
                self.condvar.notify_all();
                return Some((id, proof_type));
            }

            if data.closed {
                return None;
            }

            data = self.condvar.wait(data).unwrap();
        }
    }

    pub fn pop_recursive(&self) -> Option<(usize, usize)> {
        let mut data = self.queue.lock().unwrap();

        loop {
            if data.closed && data.items.is_empty() {
                return None;
            }

            let (mut basic, mut recursive1, mut recursive2, mut compressor) = (0, 0, 0, 0);
            for &(_, proof_type) in data.items.iter() {
                match proof_type {
                    x if x == ProofType::Basic as usize => basic += 1,
                    x if x == ProofType::Recursive1 as usize => recursive1 += 1,
                    x if x == ProofType::Recursive2 as usize => recursive2 += 1,
                    x if x == ProofType::Compressor as usize => compressor += 1,
                    _ => {}
                }
            }

            println!(
                "[pop_recursive] Queue: basic={}, recursive1={}, recursive2={}, compressor={}",
                basic, recursive1, recursive2, compressor
            );

            if let Some(pos) =
                data.items.iter().position(|&(_, proof_type)| proof_type == ProofType::Recursive2 as usize)
            {
                let (id, proof_type) = data.items.remove(pos).unwrap();
                self.condvar.notify_all();
                return Some((id, proof_type));
            }

            if let Some(pos) = data.items.iter().position(|&(_, proof_type)| proof_type != ProofType::Basic as usize) {
                let (id, proof_type) = data.items.remove(pos).unwrap();
                self.condvar.notify_all();
                return Some((id, proof_type));
            }

            if let Some((id, proof_type)) = data.items.pop_front() {
                if proof_type == ProofType::Basic as usize {
                    data.basic_count = data.basic_count.saturating_sub(1);
                }
                self.condvar.notify_all();
                return Some((id, proof_type));
            }

            data = self.condvar.wait(data).unwrap();
        }
    }

    pub fn close(&self) {
        let mut data = self.queue.lock().unwrap();
        data.closed = true;
        self.condvar.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.queue.lock().unwrap().closed
    }
}

pub fn proofs_done_listener<F: Field>(
    pctx: Arc<ProofCtx<F>>,
    pending_witness: Arc<FastQueue>,
    proofs_counter: Arc<Counter>,
    aggregation: bool,
) -> std::thread::JoinHandle<()> {
    let (tx, rx) = crossbeam_channel::unbounded::<(u64, String)>();

    register_proof_done_callback_c(tx);

    std::thread::spawn(move || {
        let pctx_clone = pctx.clone();
        while let Ok((instance_id, proof_type)) = rx.recv() {
            let p: ProofType = proof_type.parse().unwrap();
            if aggregation {
                let new_proof_type = if p == ProofType::Basic {
                    let instances = pctx_clone.dctx_get_instances();
                    let (airgroup_id, air_id, _) = instances[instance_id as usize];
                    if pctx_clone.global_info.get_air_has_compressor(airgroup_id, air_id) {
                        ProofType::Compressor as usize
                    } else {
                        ProofType::Recursive1 as usize
                    }
                } else if p == ProofType::Compressor {
                    ProofType::Recursive1 as usize
                } else {
                    ProofType::Recursive2 as usize
                };
                if new_proof_type == ProofType::Recursive2 as usize {
                    pending_witness.push(WitnessType::Recursive2(instance_id as usize, p as usize, new_proof_type));
                } else if new_proof_type == ProofType::Recursive1 as usize {
                    pending_witness.push(WitnessType::Recursive1(instance_id as usize, p as usize, new_proof_type));
                } else {
                    pending_witness.push(WitnessType::Compressor(instance_id as usize, p as usize, new_proof_type));
                }
            } else {
                proofs_counter.decrement();
            }
        }
    })
}
