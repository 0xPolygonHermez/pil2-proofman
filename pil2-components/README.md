# PIL2 Components

PIL2 Components hosts reusable components for PIL2-based proving systems
- **Standard PIL library**
  - Reusable constraints and expressions you can import from other PILs (e.g. generalized sum “gsum” utilities).
  - Organized to keep constraint degrees bounded and reusable across machines.

- **Rust helpers**
  - Utilities for witness generation and orchestration.
  - Virtual table management (e.g. `StdVirtualTable` and `VirtualTableAir`) to collect multiplicities and map logical table IDs into concrete trace regions efficiently.