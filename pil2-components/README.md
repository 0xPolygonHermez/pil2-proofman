# PIL2 Components

PIL2 Components hosts reusable components for PIL2-based proving systems
- **Standard PIL library**
  - Reusable constraints and expressions you can import from other PILs (e.g. generalized sum “gsum” utilities).
  - Organized to keep constraint degrees bounded and reusable across machines.
  - On Goldilocks the grand sum is a cyclic accumulator with no `__L1__` fixed column, so an AIR whose only
    constraints are std sums has no fixed columns at all. The grand product, and both on BN128, keep `__L1__`.

- **Rust helpers**
  - Utilities for witness generation and orchestration.
  - Virtual table management (e.g. `StdVirtualTable` and `VirtualTableAir`) to collect multiplicities and map logical table IDs into concrete trace regions efficiently.