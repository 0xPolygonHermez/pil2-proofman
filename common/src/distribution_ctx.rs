#[cfg(feature = "distributed")]
use mpi::traits::Communicator;
#[cfg(feature = "distributed")]
use mpi::collective::CommunicatorCollectives;

/// Represents the context of distributed computing
#[derive(Default)]
pub struct DistributionCtx {
    pub rank: i32,
    pub n_processes: i32,
    #[cfg(feature = "distributed")]
    pub world: mpi::topology::SimpleCommunicator,
    #[cfg(not(feature = "distributed"))]
    pub world: i32,
    pub n_instances: i32,
    pub my_instances: Vec<usize>,
    pub instances: Vec<(usize, usize)>,
}

impl DistributionCtx {
    pub fn new() -> Self {
        let mut ctx = DistributionCtx {
            rank: 0,
            n_processes: 1,
            #[cfg(feature = "distributed")]
            world: mpi::topology::SimpleCommunicator::null(),
            #[cfg(not(feature = "distributed"))]
            world: -1,
            n_instances: 0,
            my_instances: Vec::new(),
            instances: Vec::new(),
        };
        ctx.init();
        ctx
    }

    pub fn init(&mut self) {
        #[cfg(feature = "distributed")]
        {
            let (universe, _threading) = mpi::initialize_with_threading(mpi::Threading::Multiple).unwrap();
            self.world = universe.world();
            self.rank = self.world.rank();
            self.n_processes = self.world.size();
            self.n_instances = 0;
            self.my_instances = Vec::new();
        }
        #[cfg(not(feature = "distributed"))]
        {
            self.rank = 0;
            self.n_processes = 1;
            self.world = -1;
            self.n_instances = 0;
            self.my_instances = Vec::new();
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
    pub fn add_instance(&mut self, airgroup_id: usize, air_id: usize, instance_idx: usize, _size: usize) {
        self.n_instances += 1;
        if self.is_my_instance(instance_idx) {
            self.my_instances.push(instance_idx);
        }
        self.instances.push((airgroup_id, air_id));
    }
}
