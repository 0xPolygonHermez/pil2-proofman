#!/bin/sh
# Downloads official powers-of-tau files of the Hermez ceremony, the ones of the table in snarkjs's
# README (https://github.com/iden3/snarkjs#7-prepare-phase-2), and checks each against the
# README's blake2b-512, pinned below (pilfflonk/docs/README.md#tests):
#
#     pilfflonk/tests/download-ptau.sh [<power>...]
#
# The file of power k is powersOfTau28_hez_final_<k>.ptau (powersOfTau28_hez_final.ptau for 28)
# and holds 2^(k+1) - 1 powers [τ^i]₁. They go to $PILFFLONK_PTAU_DIR, by default
# ${XDG_CACHE_HOME:-$HOME/.cache}/pilfflonk/ptau, where the tests look for them; the tests never
# download anything. Without arguments, the powers the tests read: 12 (8191 powers, 4.8 MB) and
# 19 (1,048,575, 604 MB, for the wraps' keys). The CI job test-pilfflonk-wrap reads 22 (8,388,607,
# 4.8 GB), for the key of fibonacci-square's wrap.
#
# A file already there is checked and kept if its hash is the README's; otherwise it is
# downloaded again. A download goes to a file of its own, renamed into place once checked.
set -eu

url=https://circom.info
dir="${PILFFLONK_PTAU_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pilfflonk/ptau}"

# The README's blake2b-512 of the file of each power.
blake2b() {
    case "$1" in
    8) echo d6a8fb3a04feb600096c3b791f936a578c4e664d262e4aa24beed1b7a9a96aa5eb72864d628db247e9293384b74b36ffb52ca8d148d6e1b8b51e279fdf57b583 ;;
    9) echo 94f108a80e81b5d932d8e8c9e8fd7f46cf32457e31462deeeef37af1b71c2c1b3c71fb0d9b59c654ec266b042735f50311f9fd1d4cadce47ab234ad163157cb5 ;;
    10) echo 6cfeb8cda92453099d20120bdd0e8a5c4e7706c2da9a8f09ccc157ed2464d921fd0437fb70db42104769efd7d6f3c1f964bcf448c455eab6f6c7d863e88a5849 ;;
    11) echo 47c282116b892e5ac92ca238578006e31a47e7c7e70f0baa8b687f0a5203e28ea07bbbec765a98dcd654bad618475d4661bfaec3bd9ad2ed12e7abc251d94d33 ;;
    12) echo ded2694169b7b08e898f736d5de95af87c3f1a64594013351b1a796dbee393bd825f88f9468c84505ddd11eb0b1465ac9b43b9064aa8ec97f2b73e04758b8a4a ;;
    13) echo 58efc8bf2834d04768a3d7ffcd8e1e23d461561729beaac4e3e7a47829a1c9066d5320241e124a1a8e8aa6c75be0ba66f65bc8239a0542ed38e11276f6fdb4d9 ;;
    14) echo eeefbcf7c3803b523c94112023c7ff89558f9b8e0cf5d6cdcba3ade60f168af4a181c9c21774b94fbae6c90411995f7d854d02ebd93fb66043dbb06f17a831c1 ;;
    15) echo 982372c867d229c236091f767e703253249a9b432c1710b4f326306bfa2428a17b06240359606cfe4d580b10a5a1f63fbed499527069c18ae17060472969ae6e ;;
    16) echo 6a6277a2f74e1073601b4f9fed6e1e55226917efb0f0db8a07d98ab01df1ccf43eb0e8c3159432acd4960e2f29fe84a4198501fa54c8dad9e43297453efec125 ;;
    17) echo 6247a3433948b35fbfae414fa5a9355bfb45f56efa7ab4929e669264a0258976741dfbe3288bfb49828e5df02c2e633df38d2245e30162ae7e3bcca5b8b49345 ;;
    18) echo 7e6a9c2e5f05179ddfc923f38f917c9e6831d16922a902b0b4758b8e79c2ab8a81bb5f29952e16ee6c5067ed044d7857b5de120a90704c1d3b637fd94b95b13e ;;
    19) echo bca9d8b04242f175189872c42ceaa21e2951e0f0f272a0cc54fc37193ff6648600eaf1c555c70cdedfaf9fb74927de7aa1d33dc1e2a7f1a50619484989da0887 ;;
    20) echo 89a66eb5590a1c94e3f1ee0e72acf49b1669e050bb5f93c73b066b564dca4e0c7556a52b323178269d64af325d8fdddb33da3a27c34409b821de82aa2bf1a27b ;;
    21) echo 9aef0573cef4ded9c4a75f148709056bf989f80dad96876aadeb6f1c6d062391f07a394a9e756d16f7eb233198d5b69407cca44594c763ab4a5b67ae73254678 ;;
    22) echo 0d64f63dba1a6f11139df765cb690da69d9b2f469a1ddd0de5e4aa628abb28f787f04c6a5fb84a235ec5ea7f41d0548746653ecab0559add658a83502d1cb21b ;;
    23) echo 3063a0bd81d68711197c8820a92466d51aeac93e915f5136d74f63c394ee6d88c5e8016231ea6580bec02e25d491f319d92e77f5c7f46a9caa8f3b53c0ea544f ;;
    24) echo fa404d140d5819d39984833ca5ec3632cd4995f81e82db402371a4de7c2eae8687c62bc632a95b0c6aadba3fb02680a94e09174b7233ccd26d78baca2647c733 ;;
    25) echo 0377d860cdb09a8a31ea1b0b8c04335614c8206357181573bf294c25d5ca7dff72387224fbd868897e6769f7805b3dab02854aec6d69d7492883b5e4e5f35eeb ;;
    26) echo 418dee4a74b9592198bd8fd02ad1aea76f9cf3085f206dfd7d594c9e264ae919611b1459a1cc920c2f143417744ba9edd7b8d51e44be9452344a225ff7eead19 ;;
    27) echo 10ffd99837c512ef99752436a54b9810d1ac8878d368fb4b806267bdd664b4abf276c9cd3c4b9039a1fa4315a0c326c0e8e9e8fe0eb588ffd4f9021bf7eae1a1 ;;
    28) echo 55c77ce8562366c91e7cda394cf7b7c15a06c12d8c905e8b36ba9cf5e13eb37d1a429c589e8eaba4c591bc4b88a0e2828745a53e170eac300236f5c1a326f41a ;;
    *) return 1 ;;
    esac
}

name() {
    if [ "$1" = 28 ]; then
        echo powersOfTau28_hez_final.ptau
    else
        printf 'powersOfTau28_hez_final_%02d.ptau\n' "$1"
    fi
}

checks() {
    [ "$(b2sum "$1" | cut -d' ' -f1)" = "$2" ]
}

[ $# -ge 1 ] || set -- 12 19
mkdir -p "$dir"
for power in "$@"; do
    power=${power#0}
    expected=$(blake2b "$power") || {
        echo "download-ptau.sh: no file of power $power in the README's table (8 to 28)" >&2
        exit 1
    }
    file="$dir/$(name "$power")"
    if [ -f "$file" ] && checks "$file" "$expected"; then
        echo "$file: present, blake2b checked"
        continue
    fi
    partial="$file.$$.partial"
    trap 'rm -f "$partial"' EXIT
    echo "downloading $url/$(name "$power") to $file"
    curl -fsSL --retry 3 -o "$partial" "$url/$(name "$power")"
    if ! checks "$partial" "$expected"; then
        echo "download-ptau.sh: $url/$(name "$power") does not have the README's blake2b" >&2
        exit 1
    fi
    mv "$partial" "$file"
    trap - EXIT
    echo "$file: downloaded, blake2b checked"
done
