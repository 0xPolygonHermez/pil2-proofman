#[cfg(feature = "distributed")]
use mpi::collective::CommunicatorCollectives;
#[cfg(feature = "distributed")]
use mpi::traits::Communicator;
#[cfg(feature = "distributed")]
use mpi::environment::Universe;

/// Represents the context of distributed computing
pub struct DistributionCtx {
    pub rank: i32,
    pub n_processes: i32,
    #[cfg(feature = "distributed")]
    pub universe: Universe,
    #[cfg(feature = "distributed")]
    pub world: mpi::topology::SimpleCommunicator,
    pub n_instances: usize,
    pub my_instances: Vec<usize>,
    pub instances: Vec<(usize, usize)>,
}

impl DistributionCtx {
    pub fn new() -> Self {
        #[cfg(feature = "distributed")]
        {
            let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Multiple).unwrap();
            let world = universe.world();
            DistributionCtx {
                rank: world.rank(),
                n_processes: world.size(),
                universe,
                world,
                n_instances: 0,
                my_instances: Vec::new(),
                instances: Vec::new(),
            }
        }
        #[cfg(not(feature = "distributed"))]
        {
            DistributionCtx { rank: 0, n_processes: 1, n_instances: 0, my_instances: Vec::new(), instances: Vec::new() }
        }
    }

    #[inline]
    pub fn barrier(&self) {
        #[cfg(feature = "distributed")]
        {
            self.world.barrier();
        }
    }

    #[inline]
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    #[inline]
    pub fn is_distributed(&self) -> bool {
        self.n_processes > 1
    }

    #[inline]
    pub fn is_my_instance(&self, instance_idx: usize) -> bool {
        instance_idx % self.n_processes as usize == self.rank as usize
    }

    #[inline]
    pub fn owner(&self, instance_idx: usize) -> usize {
        instance_idx % self.n_processes as usize
    }

    #[inline]
    pub fn add_instance(&mut self, airgroup_id: usize, air_id: usize, _size: usize) -> (bool, usize) {
        let mut is_mine = false;
        if self.is_my_instance(self.n_instances) {
            self.my_instances.push(self.n_instances);
            is_mine = true;
        }
        self.n_instances += 1;
        self.instances.push((airgroup_id, air_id));
        (is_mine, self.n_instances - 1)
    }

    pub fn add_reduce_multiplicity(&self, _multiplicity: &mut Vec<u64>, _owner: usize) {
        #[cfg(feature = "distributed")]
        {
            //assert that I can operate with u32
            assert!(_multiplicity.len() < std::u32::MAX as usize);

            if _owner != self.rank as usize {
                //pack multiplicities in a sparce vector
                let mut packed_multiplicity = Vec::new();
                for (idx, &m) in _multiplicity.iter().enumerate() {
                    if m != 0 {
                        assert!(m < std::u32::MAX as usize);
                        packed_multiplicity.push(idx as u32);
                        packed_multiplicity.push(m as u32);
                    }
                }
                self.world.process_at_rank(_owner as i32).send(&packed_multiplicity[..]);
            } else {
                for i in 0..self.n_processes {
                    if i != _owner as i32 {
                        let mut packed_multiplicity = Vec::new();
                        self.world.process_at_rank(i).receive(&mut packed_multiplicity);
                        for j in (0..packed_multiplicity.len()).step_by(2) {
                            let idx = packed_multiplicity[j] as usize;
                            let m = packed_multiplicity[j + 1] as f64;
                            _multiplicity[idx] += m;
                        }
                    }
                }
            }
        }
    }
    pub fn add_reduce_multiplicities(&self, _multiplicities: &mut Vec<Vec<u64>>, _owner: usize) {
        #[cfg(feature = "distributed")]
        {
            // Ensure that each multiplicity vector can be operated with u32
            for multiplicity in _multiplicities.iter() {
                assert!(multiplicity.len() < std::u32::MAX as usize);
            }

            let n_columns = _multiplicities.len();
            if _owner != self.rank as usize {
                // Pack multiplicities in a sparse vector
                let mut packed_multiplicities = vec![0u32; n_columns];
                for (col_idx, multiplicity) in _multiplicities.iter().enumerate() {
                    for (idx, &m) in multiplicity.iter().enumerate() {
                        if m != 0 {
                            assert!(m < std::u32::MAX as usize);
                            packed_multiplicities[col_idx] += 1;
                            packed_multiplicities.push(idx as u32);
                            packed_multiplicities.push(m as u32);
                        }
                    }
                }
                self.world.process_at_rank(_owner as i32).send(&packed_multiplicities[..]);
            } else {
                for i in 0..self.n_processes {
                    if i != _owner as i32 {
                        let mut packed_multiplicities = Vec::new();
                        self.world.process_at_rank(i).receive(&mut packed_multiplicities);

                        // Read counters
                        let mut counters = vec![0usize; n_columns];
                        for col_idx in 0..n_columns {
                            counters[col_idx] = packed_multiplicities[col_idx] as usize;
                        }

                        // Unpack multiplicities
                        let mut idx = n_columns;
                        for col_idx in 0..n_columns {
                            for _ in 0..counters[col_idx] {
                                let row_idx = packed_multiplicities[idx] as usize;
                                let m = packed_multiplicities[idx + 1] as u64;
                                _multiplicities[col_idx][row_idx] += m;
                                idx += 2;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Default for DistributionCtx {
    fn default() -> Self {
        DistributionCtx::new()
    }
}
unsafe impl Send for DistributionCtx {}
unsafe impl Sync for DistributionCtx {}
