Currently pil2-components tests can be launched with the following commands:

------------------------------------
SIMPLE

```bash
rm -rf ./pil2-components/test/simple/build/ \
&& mkdir -p ./pil2-components/test/simple/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/simple/simple.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/simple/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/simple/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/simple/build/build.pilout \
     -u ./pil2-components/test/simple/build/fixed \
     -b ./pil2-components/test/simple/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/simple/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/simple/build/build.pilout \
     --path ./pil2-components/test/simple/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libsimple.so \
     --proving-key ./pil2-components/test/simple/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/libsimple.so \
     --proving-key ./pil2-components/test/simple/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/simple/build/proofs
```

------------------------------------
CONNECTION

```bash
rm -rf ./pil2-components/test/connection/build/ \
&& mkdir -p ./pil2-components/test/connection/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/connection/connection.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/connection/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/connection/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/connection/build/build.pilout \
     -u ./pil2-components/test/connection/build/fixed \
     -b ./pil2-components/test/connection/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/connection/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/connection/build/build.pilout \
     --path ./pil2-components/test/connection/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libconnection.so \
     --proving-key ./pil2-components/test/connection/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/libconnection.so \
     --proving-key ./pil2-components/test/connection/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/connection/build/proofs
```

------------------------------------
DIFF BUSES

```bash
rm -rf ./pil2-components/test/diff_buses/build/ \
&& mkdir -p ./pil2-components/test/diff_buses/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/diff_buses/diff_buses.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/diff_buses/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/diff_buses/build/diff_buses.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/diff_buses/build/diff_buses.pilout \
     -u ./pil2-components/test/diff_buses/build/fixed \
     -b ./pil2-components/test/diff_buses/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/diff_buses/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/diff_buses/build/diff_buses.pilout \
     --path ./pil2-components/test/diff_buses/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libdiff_buses.so \
     --proving-key ./pil2-components/test/diff_buses/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/libdiff_buses.so \
     --proving-key ./pil2-components/test/diff_buses/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/diff_buses/build/proofs
```

------------------------------------
DIRECT UPDATES

```bash
rm -rf ./pil2-components/test/direct_update/build/ \
&& mkdir -p ./pil2-components/test/direct_update/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/direct_update/direct_update.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/direct_update/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/direct_update/build/direct_update.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/direct_update/build/direct_update.pilout \
     -u ./pil2-components/test/direct_update/build/fixed \
     -b ./pil2-components/test/direct_update/build \
&& cargo run --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/direct_update/build/provingKey \
&& cargo run --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/direct_update/build/direct_update.pilout \
     --path ./pil2-components/test/direct_update/rs/src -o \
&& cargo build \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libdirect_update.so \
     --proving-key ./pil2-components/test/direct_update/build/provingKey \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libdirect_update.so \
     --proving-key ./pil2-components/test/direct_update/build/provingKey \
     --output-dir ./pil2-components/test/direct_update/build/proofs -y
```

------------------------------------
LOOKUP

```bash
rm -rf ./pil2-components/test/lookup/build/ \
&& mkdir -p ./pil2-components/test/lookup/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/lookup/lookup.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/lookup/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/lookup/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/lookup/build/build.pilout \
     -u ./pil2-components/test/lookup/build/fixed \
     -b ./pil2-components/test/lookup/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/lookup/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/lookup/build/build.pilout \
     --path ./pil2-components/test/lookup/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/liblookup.so \
     --proving-key ./pil2-components/test/lookup/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/liblookup.so \
     --proving-key ./pil2-components/test/lookup/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/lookup/build/proofs
```

------------------------------------
ONE INSTANCE

```bash
rm -rf ./pil2-components/test/one_instance/build/ \
&& mkdir -p ./pil2-components/test/one_instance/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/one_instance/one_instance.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/one_instance/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/one_instance/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/one_instance/build/build.pilout \
     -u ./pil2-components/test/one_instance/build/fixed \
     -b ./pil2-components/test/one_instance/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/one_instance/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/one_instance/build/build.pilout \
     --path ./pil2-components/test/one_instance/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libone_instance.so \
     --proving-key ./pil2-components/test/one_instance/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/libone_instance.so \
     --proving-key ./pil2-components/test/one_instance/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/one_instance/build/proofs
```

------------------------------------
PERMUTATION

```bash
rm -rf ./pil2-components/test/permutation/build/ \
&& mkdir -p ./pil2-components/test/permutation/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/permutation/permutation.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/permutation/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/permutation/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/permutation/build/build.pilout \
     -u ./pil2-components/test/permutation/build/fixed \
     -b ./pil2-components/test/permutation/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/permutation/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/permutation/build/build.pilout \
     --path ./pil2-components/test/permutation/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libpermutation.so \
     --proving-key ./pil2-components/test/permutation/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/libpermutation.so \
     --proving-key ./pil2-components/test/permutation/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/permutation/build/proofs
```

------------------------------------
RANGE CHECKS

```bash
rm -rf ./pil2-components/test/range_check/build/ \
&& mkdir -p ./pil2-components/test/range_check/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/range_check/build.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/range_check/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/range_check/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/range_check/build/build.pilout \
     -u ./pil2-components/test/range_check/build/fixed \
     -b ./pil2-components/test/range_check/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/range_check/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/range_check/build/build.pilout \
     --path ./pil2-components/test/range_check/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/librange_check.so \
     --proving-key ./pil2-components/test/range_check/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/librange_check.so \
     --proving-key ./pil2-components/test/range_check/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/range_check/build/proofs
```

------------------------------------
VIRTUAL TABLES

```bash
rm -rf ./pil2-components/test/virtual_tables/build/ \
&& mkdir -p ./pil2-components/test/virtual_tables/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/virtual_tables/virtual_tables.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/virtual_tables/build/fixed -O fixed-to-file \
     -o ./pil2-components/test/virtual_tables/build/build.pilout \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/virtual_tables/build/build.pilout \
     -u ./pil2-components/test/virtual_tables/build/fixed \
     -b ./pil2-components/test/virtual_tables/build \
&& cargo run  --bin proofman-cli check-setup \
     --proving-key ./pil2-components/test/virtual_tables/build/provingKey \
&& cargo run  --bin proofman-cli pil-helpers \
     --pilout ./pil2-components/test/virtual_tables/build/build.pilout \
     --path ./pil2-components/test/virtual_tables/rs/src -o \
&& cargo build  \
&& cargo run  --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libvirtual_tables.so \
     --proving-key ./pil2-components/test/virtual_tables/build/provingKey \
&& cargo run  --bin proofman-cli prove \
     --witness-lib ./target/debug/libvirtual_tables.so \
     --proving-key ./pil2-components/test/virtual_tables/build/provingKey \
     --verify-proofs \
     --output-dir ./pil2-components/test/virtual_tables/build/proofs
```

------------------------------------
SPECIAL

```bash
rm -rf ./pil2-components/test/special/build/ \
&& mkdir -p ./pil2-components/test/special/build/ \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/special/array_size.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/special/build/fixed_array_size -O fixed-to-file \
     -o ./pil2-components/test/special/build/array_size.pilout \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/special/direct_optimizations.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/special/build/fixed_direct_optimizations -O fixed-to-file \
     -o ./pil2-components/test/special/build/direct_optimizations.pilout \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/special/expr_optimizations.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/special/build/fixed_expr_optimizations -O fixed-to-file \
     -o ./pil2-components/test/special/build/expr_optimizations.pilout \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/special/intermediate_prods.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/special/build/fixed_intermediate_prods -O fixed-to-file \
     -o ./pil2-components/test/special/build/intermediate_prods.pilout \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/special/intermediate_sums.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/special/build/fixed_intermediate_sums -O fixed-to-file \
     -o ./pil2-components/test/special/build/intermediate_sums.pilout \
&& node ../pil2-compiler/src/pil.js ./pil2-components/test/special/table.pil \
     -I ./pil2-components/lib/std/pil \
     -u ./pil2-components/test/special/build/fixed_table -O fixed-to-file \
     -o ./pil2-components/test/special/build/table.pilout \
&& node --stack-size=1500 ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/special/build/array_size.pilout \
     -u ./pil2-components/test/special/build/fixed_array_size \
     -b ./pil2-components/test/special/build \
     -t ./pil2-stark/build/bctree \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/special/build/direct_optimizations.pilout \
     -u ./pil2-components/test/special/build/fixed_direct_optimizations \
     -b ./pil2-components/test/special/build \
     -t ./pil2-stark/build/bctree \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/special/build/expr_optimizations.pilout \
     -u ./pil2-components/test/special/build/fixed_expr_optimizations \
     -b ./pil2-components/test/special/build \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/special/build/intermediate_prods.pilout \
     -u ./pil2-components/test/special/build/fixed_intermediate_prods \
     -b ./pil2-components/test/special/build \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/special/build/intermediate_sums.pilout \
     -u ./pil2-components/test/special/build/fixed_intermediate_sums \
     -b ./pil2-components/test/special/build \
&& node ../pil2-proofman-js/src/main_setup.js \
     -a ./pil2-components/test/special/build/table.pilout \
     -u ./pil2-components/test/special/build/fixed_table \
     -b ./pil2-components/test/special/build
```