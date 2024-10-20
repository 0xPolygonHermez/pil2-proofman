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
    pub fn add_instance(&mut self, airgroup_id: usize, air_id: usize, _size: usize) -> bool {
        let mut is_mine = false;
        if self.is_my_instance(self.n_instances) {
            self.my_instances.push(self.n_instances);
            is_mine = true;
        }
        self.n_instances += 1;
        self.instances.push((airgroup_id, air_id));
        is_mine
    }
}

impl Default for DistributionCtx {
    fn default() -> Self {
        DistributionCtx::new()
    }
}
unsafe impl Send for DistributionCtx {}
unsafe impl Sync for DistributionCtx {}
