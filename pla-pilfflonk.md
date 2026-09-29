# Pla d'execució incremental: pilfflonk

**Data:** 29-09-2026
**Deriva de:** `spec-seed.md` v4 (29-09-2026). Si hi ha cap discrepància, mana l'especificació. Les dues que va trobar aquest pla (N1 i N4, secció 8) ja estan corregides a l'especificació.

**Prioritat:** primer una versió CPU funcional. El tall vertical és a M20, i la versió completa arriba al final de la Fase 3. Després, de manera incremental, la GPU (Fase 5) i les millores. Abans de copiar res de pil-fflonk, es comprova que no existeixi ja en aquest repositori o en alguna dependència.

**En resum:**
- **Primer objectiu.** Un tall vertical amb el Fibonacci de l'Annex G portat a PIL2: una AIR, una instància, només l'stage 1, cap bus, `Q` sense partir i un *layout* amb `k = 1`. Es compila a BN254, se'n fa el setup amb `setup-pilfflonk`, es prova i es verifica amb el verificador JS (D8), com el FFLONK existent. Ha d'acceptar la prova bona i rebutjar-ne una de manipulada.
- **Com s'hi arriba.** Hi ha 21 fites (M0–M20) repartides en quatre pistes paral·leles. Són unes 5 setmanes amb 3–4 persones. El camí crític és LDE → SRS/KZG → SHPLONK → prover (C++) → verificador (JS). No queda cap decisió pendent.
- **Després del tall.** Fites més grans porten fins als criteris de sortida de les fases 1 a 5, sense tocar cap format normatiu de l'Annex A.

**Convencions:**
- **Mida**, per a una persona: S = 1–2 dies, M = 3–4 dies, L = 5 dies.
- **Pistes:** A = C++ a `pil2-stark`; B = setup en Rust; C = orquestrador i CLI; D = fixtures i oracle.
- **[SUPÒSIT X]** marca una decisió pendent de l'usuari (secció 8).
- ***(proposta)*** marca un nom de fitxer que no surt a l'especificació.
- Les rutes segueixen la convenció de l'especificació.

---

## 0. Com treballen els agents d'implementació (obligatori)

Aquesta secció és per als agents que implementen el pla. Es treballa a la branca `feature/pilfflonk`.

**A cada pas** (cada fita, i cada subpas que deixi el codi en un estat nou):
1. **Auditar el que s'acaba de fer, abans de continuar:**
   - rellegir el diff sencer i contrastar-lo amb els lliurables i el criteri de fet de la fita, i amb la secció corresponent de `spec-seed.md`;
   - comprovar la regla de reutilització: no s'ha copiat de pil-fflonk res que ja existeixi en aquest repositori o en una dependència (rapidsnark, ffiasm, snarkjs);
   - comprovar que el codi segueix els patrons del voltant i les convencions de §5.4 de l'especificació (errors, prefix `pilfflonk`, sense `panic!` ni `exit()` en biblioteques);
   - comprovar que no s'ha tocat res fora de l'abast del pas.
2. **Testejar que s'ha obtingut el que s'esperava:**
   - escriure i executar els tests que demostren el criteri de fet; que compili no n'hi ha prou;
   - executar també la compilació afectada (`make` a `pil2-stark`, `cargo build`), `cargo fmt --check`, `cargo clippy -D warnings` i els tests existents que el canvi pot afectar;
   - guardar l'evidència: les comandes i la sortida rellevant.
3. **No passar al pas següent** fins que l'auditoria i els tests passin. Si alguna cosa falla, es diu tal com és, amb la sortida. No s'amaga, no es desactiva ni s'esborra cap test.
4. **Informe de cada fita:** què s'ha fet, els fitxers tocats, l'evidència dels tests, les desviacions respecte al pla i els punts oberts.

**Auditoria independent.** Quan un agent acaba una fita, un agent diferent en revisa el diff, el criteri de fet i els tests, i els torna a executar. La fita no es dona per tancada fins que aquesta auditoria passa.

**Restriccions:**
- cap *commit* ni *push*;
- no fer servir com a referència la branca `feat/pil2-fflonk` ni el directori `pil2-fflonk/`;
- no modificar altres repositoris: `../pil2-compiler` es fa servir amb `PIL2C_EXEC` des de la branca local `develop-0.14.0-pil2-fflonk`, i `../pil-fflonk` i `../pil-stark` només es llegeixen;
- no baixar cap `ptau`, i no modificar ffiasm ni snarkjs;
- si una decisió no és a l'especificació ni al pla, s'atura i es pregunta.

---

## 1. Estat actual (29-09-2026)

| Element | Estat |
|---|---|
| Especificació | `spec-seed.md` v4 a `pil2-proofman`, branca `pre-develop-1.4.0-alpha` (`ff0ff959`). Sense commit. |
| Compilador | Branca local `develop-0.14.0-pil2-fflonk`, creada des de `503862c`, amb C1–C4 fets (§4.1 de l'especificació):<br>- **Tests nous:** `test/bn254_fixed.js` i `test/bn254/big_fixed.pil` (19 tests). La suite dona 24 correctes i 1 error, el de `test/basic.js`, que ja fallava abans.<br>- **Validació:** els 10 programes de la CI compilen a BN254, i a Goldilocks surten idèntics byte a byte.<br>- **Git:** sense commit ni push. Tots els canvis són a l'índex i l'arbre de treball hi coincideix (comprovat). |
| Fixtures PIL1 | `pil-fflonk/pil/`: 14 exemples, amb els generadors `sm_*.js` i un `README.md`. Git no els segueix (`??`). |
| Referències JS | - La còpia retallada de pil-stark i shplonkjs és a `../pil-stark`. **Només es llegeix**, per adaptar-ne el verificador (D8) i portar-ne l'agrupació.<br>- El JS de pil2-proofman serà el compilador PIL2 (`pil2com`) i, a partir de M8, el verificador `pilfflonk/js/`. `ffjavascript` 0.3.1 i `@noble/hashes` ja són a `setup/pil2-stark/node_modules`, a través de snarkjs 0.7.6. |
| Codi pilfflonk | Branca `feature/pilfflonk`, sense commit. Vegeu la taula de progrés de sota.<br>- Al *workspace* no hi ha cap crate de Keccak.<br>- `num-bigint` i `thiserror` sí que hi són; `tera` només hi és a través de `pil2-stark-recurser`. |
| Entorn | `nvcc` és a `/usr/local/cuda/bin/nvcc` (CUDA 12.9), fora del `PATH`, i no hi ha driver de GPU: el codi CUDA compila i enllaça però no s'executa. Per això `provers/starks-lib-c/build.rs` tria GPU per defecte; `--features cpu-only` enllaça `libstarks.a`. |

**Progrés** (cada fita, amb auditoria independent passada):

| Fita | Estat | Notes |
|---|---|---|
| M0 | Feta | `setup/golden/{generate,check,lib}.sh`, `manifest.sha256` (499 entrades) i el job de CI `golden-setup`. Generada des d'una exportació de `ff0ff959`; dues execucions (`SETUP_JOBS=1` i per defecte) donen el mateix manifest. La CI fa servir `-r` a `fibonacci-square`, però la part recursiva queda fora del *golden*. |
| M1 | Feta | API C amb `guard` i últim error per fil, `pilfflonk_fr_check_canonical`, *bindings*, `pilfflonk_test` al Makefile i els crates `pil-info`, `pilfflonk-setup` i `proofman-pilfflonk`. Enllaça a CPU i a GPU. |
| M13 (part 1) | Feta | `compile-pil -P, --config` (sense `-P`, sortida idèntica a l'anterior) i `pilfflonk/tests/fixtures/fibonacci/` (PIL2 i `bn254.json`). Queden el generador de witness i `FileWitnessSource` (després de M12). |
| M4 | Feta | `pilfflonk_transcript_{new,free,absorb,squeeze}` sobre el `Keccak256Transcript` de rapidsnark, sense tocar-lo; `guardNew` (N4). L'API rebutja escalars i coordenades no canòniques, punts fora de la corba, el punt a l'infinit i punts amb coordenades `< 2^192` (troballa: `RawFq::toRprBE`, spec §4.4). Els reptes coincideixen amb un càlcul independent d'A.4. El *golden* STARK continua verd (499/499). |
| M5 | Feta | `PilFflonk::Lde` (`pilfflonk_lde.*`): INTT amb espai per al blinding (`Polynomial::fromEvaluations`), extensió al *coset* `g·H'` amb `g = 5` i la inversa, sobre la FFT d'ffiasm; en paral·lel per columnes quan n'hi ha prou. Coincideix amb Horner i les anades i tornades són exactes. Pendent per a M17/M18: mapar `std::invalid_argument` a `PILFFLONK_ERR_INVALID_ARGUMENT`. Nota de rendiment: cada FFT obre diverses regions OpenMP i, amb 256 fils en una màquina carregada, els tests triguen uns 2 minuts. |
| M6 | Feta | `pilfflonk_srs.*` (lector del `ptau` amb `BinFile`, només seccions 1–3; `pilfflonk.srs.bin` tipus `"pfsr"`), `pilfflonk_commit.*` (commit KZG amb `multiMulByScalar` i escalars canònics, empaquetat amb `CPolynomial`, `commitFixed`). API C: `pilfflonk_srs_{from_ptau,load,free}`, `pilfflonk_commit_fixed` en memòria i `pilfflonk_last_status`. Ajudant de test que genera un `ptau` amb `τ` fixa (N13). Comprovat també amb un `ptau` real de snarkjs de `2^15`. Defectes de rapidsnark/ffiasm trobats: spec Annex F.9. |
| M7 | Feta (decisió pendent) | `PilFflonk::ShplonkProver` (`pilfflonk_shplonk_prover.*`): arrels d'A.2.5 amb *offsets* amb signe, avaluacions dels components, `r_i`, `W`, `W'` amb el transcript de M4 i `Srs::commit`. La identitat d'A.5 es compleix en escalars i en G1 amb `τ` coneguda, per a `k` ∈ {1,2,3,4,6,12} i *offsets* `{0}`, `{0,1}`, `{−1,0,1,2}`. Genera *fixtures* JSON per a M8 (`PILFFLONK_SHPLONK_FIXTURES`). **Pendent:** les funcions de `Polynomial` de rapidsnark que fa servir tenen fuites (spec Annex F.9); ASan passa amb tres supressions. |
| M9 | Feta | Extracció mecànica a `setup/pil-info`: la majoria de fitxers es mouen sense cap canvi; el STARK conserva `StarkStruct`, la validació, la fita de grau, `get_prover_memory` i el resum. `pil-info` no depèn de `pil2-stark-setup`. *Golden* 499/499. S'ha fet en un *worktree* propi i s'ha portat a la branca amb `cherry-pick`. |
| M10 | En curs (*worktree*) | |
| M12 | En curs (*worktree*) | |
| Llicències (context de P7) | - `pil2-stark/LICENSE` és AGPL-3.0, però el *workspace* declara `MIT OR Apache-2.0`.<br>- `pil-fflonk` té `LICENSE` AGPL-3.0 i, des de `0132359`, també `LICENSE-APACHE` i `LICENSE-MIT`. |

---

## 2. Camí crític cap al primer tall

### 2.1 Què és el primer tall

**La fixture.** És el Fibonacci de `pil-fflonk/pil/sm_fibonacci/fibonacci.pil`, portat a PIL2:
- columnes fixes `L1` i `LLAST`, i columnes de witness `l1` i `l2`;
- publics `in1`, `in2` i `out`;
- 5 restriccions, totes `everyRow` (N1);
- grau màxim 3: `qDeg = 2` i, amb D = 9, no calen im pols;
- *offsets* `{0, 1}`.

**Criteri de fet**, escrit com a runbook. Les comandes són les de §4.2 i §4.4 de l'especificació, i alguns arguments són [SUPÒSIT N9, N13, N14].

```
node ../pil2-compiler/src/pil.js pilfflonk/tests/fixtures/fibonacci/fibonacci.pil \
     -P pilfflonk/tests/fixtures/bn254.json -o build/fibonacci.pilout
proofman-setup setup-pilfflonk -a build/fibonacci.pilout -b build --powers-of-tau $PILFFLONK_TEST_PTAU --no-packing
<generador de la fixture> --in1 1 --in2 2 -o build/witness
proofman-cli pilfflonk prove  -k build/provingKey --witness build/witness -o build/proof
proofman-cli pilfflonk verify -k build/provingKey -p build/proof/proof.json -u build/proof/publics.json   # codi 0
# qualsevol camp de proof.json o de publics.json canviat → codi ≠ 0
```

### 2.2 Peces de l'especificació: què entra al tall i què pot esperar

| Peça (secció de l'especificació) | Al tall? | Fita |
|---|---|---|
| C1–C4 del compilador (§4.1) | Sí | Fet |
| Std BN254: `bn254.pil` i `ACTIVE_FIELD` (§4.1) | No | M29 (P10) |
| Validació del `pilout` (§4.2.1) | Sí. És barata i es fa sencera; al tall, tots els *hints* de prover es rebutgen. | M15, M16 |
| Extracció de `pil-info` i `PilInfoCfg`: mòdul, `neg`, dimensió, política de grau, constants grans, ganxo FRI (§4.2.2) | Sí | M9, M10 |
| `Result` en lloc de `panic!` i truncacions `u64` als escriptors STARK (§4.2.2) | No | M27 |
| Plegat amb `std_vc` i *zerofier* `everyRow` (§4.2.3, A.1) | Sí | M10, M16, M17 |
| Selecció d'im pols (§4.2.3) | La passada ja existeix, però el tall no l'exercita | M23 |
| Dominis `firstRow`, `lastRow` i `everyFrame` (A.1) | No: el compilador no els emet (N1) | M24 |
| Partició de `Q` (§4.2.3, A.1) | No | M33 |
| Agrupació fflonk (§4.2.4, A.2) | No: al tall, *layout* amb `k = 1` (drecera R1) | M21, M22 |
| Bytecode, `verifierinfo.json`, SRS, commits fixos i *digest* (§4.2.5) | Sí | M11, M15, M16 |
| Restriccions globals | No: `globalConstraints.json` té 0 entrades a la v1 (D2) | Després de la v1 |
| `provingKey/` complet (§4.2.6, A.6) | Sí: tots els camps, amb llistes buides quan toca | M12, M15, M16 |
| `WitnessSource` sobre fitxer (§4.3) | Sí | M13 |
| Witness en `Fr` (D4) | No | M38 |
| Stage 1, blinding, `Q` sobre el *coset*, avaluacions i SHPLONK (§4.4, A.3) | Sí | M17, M18 |
| Stages ≥ 2 i *hints* (§4.4) | No | M30, M31 |
| `pilfflonk check` (§4.4) | No: mentrestant, oracle | M25 |
| Transcript d'A.4, amb el nombre d'instàncies i sense air values | Sí | M4, M18 |
| SHPLONK i ordre global (A.5) | Sí | M7, M8, M19 |
| Verificador JS amb *pairing* (§4.5) | Sí | M8, M19 |
| Solidity (§4.5, Fase 4) | No | M40–M42 |
| Regressió del *wrap* abans de tocar ffiasm (Fase 0) | No cal: ffiasm no es toca (D3) | — |
| *Golden* del STARK abans de l'extracció (Fase 1) | Sí | M0 |

### 2.3 Seqüència i durada

**Camí crític:** M1 → M5 → M6 → M7 → M18 → M19 → M20, unes 24 jornades amb 3–4 persones.

El camí és gairebé pla. Hi ha dues cadenes amb només 1–3 dies de marge:
- **setup:** M0 → M9 → M10 → M16;
- **verificador JS:** M4 + M7 → M8 → M19.

**Estimació fins a M20:**
- 3–4 persones: unes 5 setmanes;
- 2 persones: 7–8 setmanes;
- 1 persona: 11–13 setmanes.

### 2.4 El setup del tall: D1 per etapes

Hi ha tres maneres d'arribar-hi:

1. **Parametritzar les passades on són, darrere d'un `cfg`, i extreure-les després.** Descartada. `proofman-setup` executa STARK i pilfflonk en el mateix binari, i per tant el paràmetre ha de ser de temps d'execució (`PilInfoCfg`). Un `cfg` s'hauria de desfer, i obligaria a validar el *golden* amb dues compilacions.
2. **Parametritzar les passades on són, amb `PilInfoCfg`, i posar `setup-pilfflonk` provisionalment com a mòdul de `pil2-stark-setup`.** Estalvia uns 2 dies de moure codi, però crea una ubicació provisional que s'ha de desfer, i fa l'extracció més tardana i més grossa.
3. **Primer una extracció mecànica (només moure codi, M9) i després la parametrització dins de `pil-info` (M10).** Triada, per quatre motius:
   - el tall és net: les passades (`pil/`, `expr/`, `types/pilout_info.rs`) només importen `types::output`, `types::pilout_info` i `StarkStruct`, i `StarkStruct` només apareix a `pil/prepare.rs:69-80` i `pil/info.rs:61, 372`;
   - moure codi és barat, i el compilador de Rust troba tots els camins trencats;
   - el *golden* valida cada pas per separat;
   - la pista B no és al camí crític, de manera que fer-ho bé no retarda la primera prova.

   M10 només fa el mínim que necessita el tall. El `Result` i les truncacions queden per a M27.

---

## 3. Pistes i graf de dependències

| Pista | Contingut | Fites fins al tall |
|---|---|---|
| A: C++ (`pil2-stark`) | LDE (FFT d'ffiasm), transcript, SRS/KZG, SHPLONK (orquestració de `ShPlonkProver`), intèrpret `Fr`, prover, API C | M1, M4–M7, M17, M18 |
| B: setup Rust | *Golden*, extracció de `pil-info`, `PilInfoCfg`, bytecode, escriptors del `provingKey/`, *digest*. L'agrupació va fora del camí crític. | M0, M9–M11, M15, M16 (M21) |
| C: orquestrador i verificador | `proofman-pilfflonk` (tipus, càrrega, bucle d'stages, `WitnessSource`, prova), la CLI i el verificador JS (`pilfflonk/js/`) | M8, M12, M18, M19 |
| D: fixtures | Port del Fibonacci, generador de witness, oracle Rust (només per a tests) | M13, M14 |

**Graf fins al primer tall.** Una fletxa vol dir "cal abans de".

```
M0 ──► M9 ──► M10 ──┐
       M9 ──► M11 ──┼──► M16 ──┐
M12 + M6 ──► M15 ───┘          │
M1 ──► M5 ──► M6 ──┐           │
M1 ──► M4 ─────────┴──► M7 ────┼──► M18 ──► M19 ──► M20
M5 + M11 ──► M17 ──────────────┤              ▲
M1 ──► M12 ──► M13 ────────────┘              │
M4 + M7 ──► M8 (JS) ──────────────────────────┘
M13 ──► M14   (recomanat abans de M18)
```

---

## 4. Fites fins al primer tall (M0–M20)

### M0 · Referències *golden* del setup STARK (B)

- **Objectiu.** Congelar totes les sortides del setup STARK dels 10 programes de la CI abans de tocar cap passada.
- **Lliurables** [SUPÒSIT N2]:
  - `setup/golden/generate.sh` *(proposta)*.
    - Compila els 10 programes (`fibonacci-square` i els 9 de `pil2-components/test`) amb el `pil2-compiler` fixat a `setup/pil2-stark/package.json`.
    - Hi executa `proofman-setup setup` amb les opcions de `.github/workflows/ci.yaml`. A `fibonacci-square` això és `-u … --hash Poseidon1`. La CI hi afegeix `-r`, però la part recursiva queda fora del *golden*.
  - `setup/golden/manifest.sha256`, amb els sha256 de:
    - cada `.pilout`, per fixar l'entrada;
    - `starkinfo`, `expressionsinfo`, `verifierinfo`, `.bin` i `.verifier.bin`;
    - `pilout.globalInfo.json` i `pilout.globalConstraints.{json,bin}`;
    - `.const` i `verkey.json`, perquè M27 toca `io/fixed_cols.rs`.
  - `setup/golden/check.sh`, que **falla**, sense saltar-se res, si falta una entrada o un hash no quadra, i diu quin fitxer és.
  - Un job de CI `golden-setup`, o un pas dins del job de tests, que ja té Node.
- **Dependències:** cap. La base és `ff0ff959`.
- **Criteri de fet:**
  - dues execucions seguides a `ff0ff959` donen el mateix manifest, amb `SETUP_JOBS=1` i amb el valor per defecte;
  - un canvi deliberat en una branca de prova (per exemple, un literal de `pil/codegen.rs`) fa fallar `check.sh`.
- **Mida:** S.
- **Riscos:** el no-determinisme de l'ordre o del paral·lelisme. El setup amb `-r` queda fora de la base: demana circom i és lent. Es pot afegir com a job nocturn.
- **En paral·lel:** M1, M13, M21.

### M1 · Bastida C++/FFI i esquelets de crates (A, C)

- **Objectiu.** Cridar un símbol `pilfflonk_*` des de Rust abans d'escriure cap algorisme.
- **Lliurables:**
  - **C++:**
    - `pil2-stark/src/pilfflonk/`, amb el namespace `PilFflonk` i fitxers `pilfflonk_*.{hpp,cpp,c.hpp}`;
    - `pil2-stark/src/api/pilfflonk_api.{hpp,cpp}`: `try/catch`, codis d'estat i cap crida a `exitProcess`.
  - **Makefile:** `./src/api/pilfflonk_api.*` i `./src/pilfflonk` a les tres llistes `SRCS_STARKS_LIB*` (`pil2-stark/Makefile:228, 231, 234`).
  - ***Bindings*:**
    - `provers/starks-lib-c/bindings_pilfflonk.rs`;
    - `src/ffi_pilfflonk.rs`, amb embolcalls que retornen `Result`;
    - `mod ffi_pilfflonk; pub use ffi_pilfflonk::*;` a `src/lib.rs`.
  - **Crates** registrats al *workspace*: `setup/pil-info` (buit fins a M9), `setup/pilfflonk` (`pilfflonk-setup`) i `pilfflonk/` (`proofman-pilfflonk`), tots amb `thiserror`.
  - **Arnès de tests C++:** un target `pilfflonk_test` al Makefile [SUPÒSIT N3].
  - **API C:** el refinament de l'esbós de §5.3 [SUPÒSIT N4].
- **Dependències:** cap.
- **Criteri de fet:**
  - `cargo test -p proofman-starks-lib-c` crida `pilfflonk_*` i rep un codi d'error per una entrada invàlida, sense que el procés avorti;
  - `make -C pil2-stark starks_lib` enllaça, i `starks_lib_gpu` també, a una màquina amb `nvcc`;
  - `clippy -D warnings` i `fmt` surten nets.
- **Mida:** S.
- **Riscos:** col·lisions d'*includes* (`Makefile:150`), que es mitiguen amb el prefix `pilfflonk_`. No cal tocar `build.rs`.
- **En paral·lel:** M0, M13.

### M2 · Eliminada

ffiasm no es toca (D3), i per tant no cal cap línia base de regressió d'ffiasm ni del *wrap*. El criteri de M20 inclou un `git diff` buit a `pil2-stark/src/bn128/src/ffiasm/`.

### M3 · Eliminada

No hi ha verificador natiu (D3). El *pairing* el fa `ffjavascript` dins del verificador JS (M8).

### M4 · Accés al transcript (A)

- **Lliurables.** `pilfflonk_transcript_{new,free,absorb,squeeze}` a l'API C, sobre el `Keccak256Transcript<AltBn128::Engine>` de `pil2-stark/src/rapidsnark/`. És la mateixa classe que el FFLONK existent (P6) i **no es modifica**:
  - `squeeze` = `getChallenge()`, i després `reset()` + `addScalar(repte)`, com a `fflonk_prover.c.hpp:849-851`.
- **Dependències:** M1.
- **Criteri de fet:** per a seqüències que barregen `Fr` i G1, coincideix amb un càlcul independent: la codificació d'A.4 feta a mà, el `keccak_wrapper` de C++ i la reducció mòdul `r`. No es fa servir cap vector JS.
- **Mida:** S (menys d'1 dia).
- **Riscos:** cap. El punt zero i el VLA no afecten els casos reals (§4.4 de l'especificació).
- **En paral·lel:** M5.

### M5 · LDE sobre *coset* amb la FFT existent (A)

- **Lliurables:** `pilfflonk_lde.{hpp,cpp}`, sobre el `Polynomial`/`Evaluations` de rapidsnark i la FFT d'ffiasm, columna a columna i en paral·lel per columnes:
  - INTT, deixant espai per al blinding (`fromEvaluations(…, blindLength)`);
  - LDE sobre el *coset*: s'escalen els coeficients pels poders del desplaçament abans de la FFT;
  - la inversa: coeficients a partir del *coset*.

  No es porta la NTT multicolumna de pil-fflonk: no cal per a la versió de CPU.
- **Dependències:** M1.
- **Criteri de fet:**
  - l'LDE coincideix amb l'avaluació de Horner als punts `g·ω_ext^i`;
  - l'anada i tornada entre coeficients i *coset* és exacta;
  - si `nBitsExt > 28`, dona un error.
- **Mida:** S.
- **Riscos:** el rendiment, que es millorarà a la Fase 5 amb la NTT de GPU que ja existeix.
- **En paral·lel:** M4.

### M6 · SRS, commit KZG i `pilfflonk_commit_fixed` (A)

- **Lliurables:**
  - el lector del `ptau`, reutilitzant `BinFile` com fa `rapidsnark/fflonk_setup.cpp:46-49, 527-531`;
  - el lector de `pilfflonk.srs.bin` [SUPÒSIT N6];
  - el commit KZG: `multiMulByScalar` amb escalars canònics, convertits des de Montgomery just abans de cada MSM;
  - l'empaquetat `f_i(X) = Σ_j p_j(X^k)·X^j` amb el `CPolynomial` de rapidsnark, que ja ho fa per a qualsevol `n`;
  - `pilfflonk_commit_fixed`, escrit sobre `CPolynomial` i `multiMulByScalar`, com fa rapidsnark. No es porta `computeFCommitments` de pil-fflonk.
- **Dependències:** M1, M5.
- **Criteri de fet:**
  - amb un SRS de test de τ conegut, el commit de `p` és `[p(τ)]₁`;
  - el commit empaquetat és el commit dels coeficients intercalats;
  - una entrada malformada retorna un error.
- **Mida:** S.
- **Riscos:** el format de les seccions del `ptau`.
- **En paral·lel:** M4, M11, M12.

### M7 · SHPLONK: la banda del prover (A)

- **Lliurables:** `pilfflonk_shplonk_prover.{hpp,cpp}`. Només s'adapta l'orquestració genèrica de `ShPlonkProver` (`pil-fflonk/src/shplonk.cpp`), que és l'única part que no existeix aquí. S'escriu sobre `lagrangePolynomialInterpolation`, `zerofierPolynomial`, `divByMonic` i `CPolynomial` de rapidsnark:
  - en lloc de `PilFflonkZkey`, rep una descripció de l'obertura en memòria: la llista global de `f_i` (A.5), amb el seu `k`, els *offsets* amb signe i les arrels (A.2.5);
  - `R`, `Z_T`, `L`, `W` i `W'`, i la inversa en lot;
  - `divByMonic(1, β)` (`rapidsnark/polynomial/polynomial.c.hpp:423`) en lloc de `divByXSubValue`;
  - el transcript de M4;
  - sense els defectes de C.3.8.
- **Dependències:** M4, M6.
- **Criteri de fet:**
  - amb polinomis aleatoris, *offsets* `{0}`, `{0,1}` i `{−1,0,1,2}`, `k` ∈ {1, 2, 3, 4, 6, 12} i arrels repetides, la identitat d'A.5 es compleix en escalars amb τ conegut, sense *pairing*: `F − E − J + y·W' = τ·W'`;
  - ASan i UBSan no troben res.
- **Mida:** L.
- **Riscos:** les arrels amb `s < 0` (`ω_{kN}^s`). Aquesta fita fixa l'ordre global dels `f_i`.
- **En paral·lel:** la pista B.

### M8 · SHPLONK: verificació en JS (C). Tanca la Fase 0

- **Lliurables:** la base de `pilfflonk/js/`:
  - `package.json`, amb `ffjavascript` i `@noble/hashes` a les versions de snarkjs 0.7.6, que s'instal·la amb `node_deps::ensure_node_deps` com snarkjs;
  - `transcript.js`, amb la semàntica del `Keccak256Transcript` de rapidsnark, com el `Keccak256Transcript.js` de snarkjs;
  - `shplonk.js`: `F`, `E`, `J` i el *pairing* amb l'estructura de `snarkjs/src/fflonk_verify.js` (`computeF`, `computeE`, `computeJ`, `isValidPairing`), i `r_i(y)` i `q_i` generalitzats com a `verifyOpenings` (`shplonkjs/src/helpers/verifier.js:107-138`). Els commitments fixos arriben per un paràmetre a part (la regla d'A.5).
- **Dependències:** M4, M7.
- **Criteri de fet:**
  - el transcript JS i el de M4 donen els mateixos reptes (validació 4 de la Fase 0);
  - accepta les obertures de M7;
  - les rebutja si es canvia qualsevol bit de `W`, `W'`, d'un commitment o d'una avaluació (validació 5);
  - M4, M5 i M8 passen. Amb això, i amb el compilador fet, **es tanca la Fase 0**.
- **Mida:** M.
- **Riscos:**
  - la convenció de `Z_{T∖T_0}` quan hi ha repeticions;
  - la conversió de Montgomery i de l'ordre dels bytes entre C++ i JS.
- **En paral·lel:** la pista B.

### M9 · Extracció mecànica de `pil-info` (B) [SUPÒSIT D1(a), N7]

- **Objectiu.** Moure les passades a `setup/pil-info` sense canviar-ne el comportament.
- **Lliurables:**
  - **Es mouen a `pil-info`:**
    - `setup/pil2-stark/src/{pil,expr}/` i `types/pilout_info.rs`;
    - els tipus de `types/output.rs` que fan servir les passades (`CodeEntry`, `CodeRef` i companyia);
    - `io/bin_file_writer.rs`;
    - l'assignació de temporals (`io/parser_args.rs:85-205`);
    - `build_global_constraints_json` (`output/global_info.rs:245`);
    - els constructors de `expressionsinfo`/`verifierinfo` (`output/stark_info.rs:306-560`);
    - els constructors de la part comuna del `globalInfo` (`:180`, `:203`).
  - **Es queden a `pil2-stark-setup`:** `StarkStruct`, la validació de `pil/prepare.rs:69-80`, `get_prover_memory` (`pil/info.rs:372`) i el resum que s'imprimeix.
  - `pil2-stark-setup` reexporta el que calgui.
  - S'esborra el fitxer orfe `setup/pil2-stark/src/pilout_info.rs`.
- **Dependències:** M0.
- **Criteri de fet:**
  - `setup/golden/check.sh` passa;
  - `cargo test --workspace`, `clippy -D warnings` i `fmt` surten nets;
  - `cargo tree -p pil-info` no mostra `pil2-stark-setup`.
- **Mida:** M.
- **Riscos:** separar la part STARK del resum de `pil_info`.
- **En paral·lel:** la pista A, M12, M13.

### M10 · `PilInfoCfg`: les passades parametritzades pel camp (B)

- **Lliurables:**
  - `PilInfoCfg { modulus, ext_dim, degree_policy }`, amb els constructors `::goldilocks()` i `::bn254()`;
  - **`FIELD_EXTENSION` se substitueix** a `expr/helpers.rs`, `types/pilout_info.rs`, `pil/*.rs` i a la còpia privada de `io/parser_args.rs:7`;
  - **`NEG_ONE` se substitueix** (`expr/helpers.rs:5`);
  - `buf_to_bigint_string` passa a `num-bigint` (`types/pilout_info.rs:126-135`);
  - la política de grau: `FromBlowup` per al STARK (`pil/info.rs:61`) i `Search { max: D }` per a pilfflonk (A.1);
  - un ganxo d'obertura a `gen_code`: el polinomi FRI, `queryVerifier` i `friExp` (`gen_code.rs:344-522`) només es generen per al STARK.
- **Dependències:** M9.
- **Criteri de fet:**
  - el *golden* STARK surt idèntic;
  - amb el `pilout` del Fibonacci (M13) i `bn254()`: tots els `dim` valen 1, `neg` es representa com a `r−1`, les constants surten intactes i no hi ha cap expressió FRI.
- **Mida:** M.
- **Riscos:** hi ha 3 implícits que són propis del STARK i s'hi han de quedar (`output/stark_info.rs:278`, `verifier_hashes.rs:294`).
- **En paral·lel:** M11, M12.

### M11 · Format i codificador del bytecode `Fr` (B) [SUPÒSIT N8]

- **Lliurables:**
  - **La nota de format**, a la capçalera del mòdul, per a `<air>.bin`:
    - contenidor `"chps"` (el `BinFileWriter` de `pil-info`) amb una versió pròpia;
    - els codis d'operació del STARK, amb dimensió 1;
    - constants de 32 bytes *little-endian*;
    - seccions d'expressions (im pols i `Q`), de depuració de restriccions (per a M25) i de *hints*, que a la Fase 1 queda buida.
  - **El codificador**, a `setup/pilfflonk/src/bytecode.rs` *(proposta)*.
- **Dependències:** M9. La nota de format pot sortir abans; la integració amb dimensió 1 necessita M10.
- **Criteri de fet:**
  - una anada i tornada en Rust (codificar i descodificar) retorna els mateixos `CodeEntry`;
  - un test creuat amb el lector C++ de M17.
- **Mida:** M.
- **Riscos:** si el format no s'acorda abans amb M17, caldrà refer feina. L'ABI dels buffers del STARK no es copia (§4.2.5).
- **En paral·lel:** M10, M12 i la pista A.

### M12 · Tipus del `provingKey/` i de la prova (C) [SUPÒSIT D7]

- **Lliurables** (a `pilfflonk/src/`, *proposta*):
  - els tipus `PilfflonkGlobalInfo`, `PilfflonkInfo` (amb `layout` i `qDim = 1`), `AirVerkey`, `Vkey` (la vkey autocontinguda), `Proof` i `Publics`, amb tots els camps d'A.6;
  - JSON determinista: ordre de camps fix, sense `HashMap`;
  - el lector C++ de `pilfflonkinfo.json` amb `nlohmann/json` (`pilfflonk_info.{hpp,cpp}`).
- **Dependències:** M1.
- **Criteri de fet:**
  - una anada i tornada Rust → fitxer → C++;
  - serialitzar dues vegades dona els mateixos bytes;
  - `common::GlobalInfo::from_file` rebutja el `globalInfo` pilfflonk.
- **Mida:** M.
- **Riscos:** canvis d'esquema tardans, que mitiga el fet de tenir un sol propietari.
- **En paral·lel:** tot.

### M13 · Fixture Fibonacci, generador de witness i `WitnessSource` (D, C) [SUPÒSIT N1, N9, N14]

- **Lliurables:**
  - `pilfflonk/tests/fixtures/fibonacci/fibonacci.pil` *(proposta)*:
    - un `airtemplate` amb `col fixed L1 = [1,0...]` i `LLAST` com a columna fixa (la sintaxi, per exemple `[0:(N-1),1]`, és per confirmar);
    - `col witness l1, l2`;
    - `public in1, in2, out`;
    - les 5 restriccions de l'Annex G;
  - `bn254.json`, amb `prime` com a cadena;
  - un generador de test en Rust amb `num-bigint`: el port de `sm_fibonacci.js`, amb entrades `[1, 2]`;
  - `FileWitnessSource`, que implementa `WitnessSource`.
  - l'opció `-P, --config <json>` de `compile-pil` (P9), igual que la de `pil2com`, que per defecte no canvia res.
- **Dependències:** M12, només per al `WitnessSource`.
- **Criteri de fet:**
  - `proofman-cli pilout inspect` mostra `baseField = r`, 1 AIR, 2 columnes fixes, 2 de witness, 3 publics, 5 restriccions `everyRow` i cap *hint*;
  - `out` coincideix amb el tercer públic de `pil-fflonk/runtime/public.json` (la mateixa fixture: `N = 2^8`, entrades `[1, 2]`).
- **Mida:** S.
- **Riscos:** la sintaxi de `LLAST`.
- **En paral·lel:** tot.

### M14 · Oracle Rust (D, només per a tests)

- **Lliurables:** un mòdul de tests de `proofman-pilfflonk`, amb `num-bigint` a les `dev-dependencies`, que llegeix `pil2-pilout` directament i:
  - avalua les expressions del `pilout` fila a fila;
  - avalua columnes en punts arbitraris per interpolació baricèntrica;
  - calcula `Q(z)` segons A.1.
- **Dependències:** M13.
- **Criteri de fet:**
  - amb el witness bo, tots els numeradors valen 0;
  - amb una cel·la mutada, falla exactament la fila esperada.
- **Mida:** M.
- **Riscos:** cap per al tall, que no depèn d'aquesta fita; només cost d'oportunitat.
- **En paral·lel:** tot. Es recomana tenir-la abans de M18.

### M15 · `setup-pilfflonk`, part 1: validació i fitxers independents de les passades (B)

- **Lliurables:**
  - `SetupPilfflonkOptions` i el subcomandament `setup-pilfflonk` a `setup/pil2-stark/src/main.rs`, amb els arguments de §4.2;
  - la validació de §4.2.1: tots els *hints* de prover es rebutgen, i els de witness i de depuració s'ignoren;
  - `<air>.const`, en 32 bytes *little-endian*, amb descodificació `num-bigint`;
  - `pilfflonk.srs.bin`, amb `pilfflonk_srs_from_ptau` de M6; per a l'`X_2` de la vkey cal un accessor C de `[τ]₂` en forma canònica (per exemple, `pilfflonk_srs_g2`), que M6 no té;
  - `<air>.verkey.json`, per mitjà de `pilfflonk_commit_fixed`;
  - `pilout.globalInfo.json`;
  - la funció de *digest* d'A.6, sobre el JSON canònic de la vkey [SUPÒSIT N10].
- **Dependències:** M12, i M6 per al commit fix, que s'integra al final.
- **Criteri de fet:**
  - un test per a cada error de §4.2.1, amb `pilout` sintètics fets amb `prost`. Entre ells, un `pilout` de Goldilocks, que és el que surt si `-P bn254.json` es fa servir sense `PIL2C_EXEC`, perquè el compilador fixat ignora `prime` sense avisar (troballa de M13);
  - el *digest* canvia si canvia qualsevol camp de la vkey, i Rust i JS en calculen el mateix sobre el JSON canònic.
- **Mida:** M.
- **Riscos:** la vkey necessita dades de M16 (el `qVerifier`, el *layout*), i per això s'escriu al final de l'orquestració.
- **En paral·lel:** M10, M11, M17.

### M16 · `setup-pilfflonk`, part 2: passades, *layout* i bytecode (B) [SUPÒSIT D5]

- **Lliurables:**
  - `pil_info` amb `bn254()` i `Search { max: D }`;
  - els polinomis compromesos (stage, fita en coeficients i `O`), derivats de l'`evMap`. Les columnes que no s'obren no es comprometen, amb un avís;
  - el *layout* sense empaquetar (R1);
  - `nBitsExt` segons A.1, que també ha de cobrir `N + |O|_max + 1` (troballa de M5);
  - la comprovació que l'SRS té almenys `max grau(f_i) + 1` punts;
  - els fitxers `<air>.pilfflonkinfo.json`, `expressionsinfo.json`, `verifierinfo.json` (sense `queryVerifier`; el llegeix el verificador JS), `.bin` i `pilout.globalConstraints.json`;
  - `pilfflonk.vkey.json`, amb el *digest*, al final de tot.
- **Dependències:** M10, M11, M15.
- **Criteri de fet:**
  - `setup-pilfflonk … --no-packing` genera el `provingKey/` de §4.2.6;
  - dues execucions donen bytes idèntics;
  - el *golden* STARK continua verd.
- **Mida:** M.
- **Riscos:** la fita de grau de `Q` amb `|O|max` (C.3.2).
- **En paral·lel:** M17.

### M17 · Intèrpret `Fr` (A)

- **Lliurables:**
  - el lector del format de M11 (`pilfflonk_expressions_bin.{hpp,cpp}`);
  - l'intèrpret sobre `FrElement` (`pilfflonk_expressions.{hpp,c.hpp}`), amb dos modes:
    - **prover:** per blocs del *coset* estès, amb els operands `cm`, `const`, `public`, `challenge`, `number`, `tmp` i el *zerofier* invers de cada domini;
    - **verificador:** sobre les avaluacions a `ξ·ω^s` i `Z_D(ξ)`.
- **Dependències:** M11, M5.
- **Criteri de fet:**
  - els bins de M11 amb expressions petites donen els mateixos valors que un càlcul amb `num-bigint`;
  - en mode verificador, `Q(z)` coincideix amb l'oracle (M14).
- **Mida:** M.
- **Riscos:** la semàntica de `Zi` i de `x`, que es copia del STARK (`expressions_pack.hpp`, `setup_ctx.hpp`) però sense el possible error de `lastRow` (F.8).
- **En paral·lel:** M15, M16.

### M18 · Prover d'una instància i orquestrador (A, C) [SUPÒSIT N4, N9]

- **Lliurables:**
  - **C++:**
    - `pilfflonk_ctx_new`: carrega el `provingKey/` i recalcula els coeficients de les columnes fixes;
    - `pilfflonk_instance_new`;
    - `pilfflonk_commit_stage`: INTT (`fromEvaluations`), blinding d'A.3 (`blindCoefficients`), empaquetat (`CPolynomial`) i MSM;
    - `pilfflonk_commit_q`: LDE, *zerofiers*, intèrpret, coeficients i commit;
    - `pilfflonk_evaluate`: les columnes fixes, un cop per AIR;
    - `pilfflonk_open`, amb M7;
    - el RNG de libsodium, amb una llavor que es pot injectar.
  - **Rust:**
    - la seqüència d'A.4 per a `nStages = 1`: *digest*, nombre d'instàncies i publics → commits → `std_vc` → commits de `Q` → `xiSeed` → avaluacions → SHPLONK;
    - `proof.json` i `publics.json`.
  - **CLI:** `proofman-cli pilfflonk prove`, com a subcomandament niat (el patró de `cli/src/commands/pilout/mod.rs`).
- **Dependències:** M7, M12, M13, M16, M17.
- **Criteri de fet:**
  - amb la mateixa llavor, dues execucions donen la mateixa prova;
  - els coeficients de `Q` per sobre de la fita d'A.1 valen 0; si no, el prover avisa que el witness no compleix les restriccions;
  - `Q(z)` coincideix amb l'oracle.
- **Mida:** L.
- **Riscos:** és la fita més gran del tall. La memòria del *coset* no és un problema amb la fixture.
- **En paral·lel:** preparar M19.

### M19 · Verificador JS i `pilfflonk verify` (C)

- **Lliurables:**
  - `pilfflonk/js/verify.js`, amb `snarkjs/src/fflonk_verify.js` com a plantilla (interfície i passos) i les parts genèriques de `fflonk_verify.js` de pil-stark. Fa aquests passos (§4.5 de l'especificació):
    - comprova les longituds, que els escalars siguin `< r` i que els punts siguin a la corba;
    - recalcula el *digest* de la vkey i el compara;
    - refà el transcript d'A.4;
    - calcula `Q(ξ)` amb el `qVerifier` de la vkey;
    - fa la comprovació SHPLONK de M8 i el *pairing*, amb els commitments fixos de la vkey;
    - surt amb un codi diferent de 0 quan rebutja;
  - `proofman-cli pilfflonk verify`, que el crida amb `node` igual que `verify_snark_proof` crida snarkjs (`proofman/src/snark_wrapper.rs:580-638`), amb tres arguments com `snarkjs fflonk verify`: la vkey, els publics i la prova.
- **Dependències:** M8, M16, M18.
- **Criteri de fet:** accepta la prova de M18, i la rebutja si canvia:
  - un commitment, una avaluació, `W` o `W'`;
  - els publics;
  - un camp de la vkey (el *digest* no quadra).
- **Mida:** M–L.
- **Riscos:** que el JS llegeixi els fitxers d'una manera diferent de com els escriu Rust. L'E2E de M20 ho cobreix.
- **En paral·lel:** cap.

### M20 · Primer tall vertical (totes les pistes)

- **Lliurables:**
  - el test E2E `pilfflonk/tests/e2e_fibonacci.rs` *(proposta)*, marcat `#[ignore]` (R9);
  - quatre tests de rebuig: prova, publics, witness mutat i `provingKey/` modificat;
  - el runbook de §2.1.
- **Dependències:** M18, M19. Es recomana tenir M14.
- **Criteri de fet:**
  - `PIL2C_EXEC=../pil2-compiler/src/pil.js PILFFLONK_TEST_PTAU=… cargo test -p proofman-pilfflonk --test e2e_fibonacci -- --ignored` surt verd;
  - el *golden* STARK continua verd, i `pil2-stark/src/bn128/src/ffiasm/` no té cap canvi;
  - `clippy` i `fmt` surten nets.
- **Mida:** S.

---

## 5. Dreceres temporals permeses

Cap d'aquestes dreceres canvia el transcript, els formats ni el *digest* de l'Annex A.

| # | Drecera | Què guanya | Per què no contradiu l'especificació | Es retira a |
|---|---|---|---|---|
| R1 | *Layout* amb `k = 1` (`--no-packing`) al tall | Treu M21 (L) del camí crític | `--no-packing` ja és una opció de test (§4.2), i el *layout* és una dada de `pilfflonkinfo` | M22: l'empaquetat passa a ser el comportament per defecte, i `--no-packing` es manté com a opció de test |
| R2 | Només l'stage 1; stages ≥ 2 i *hints* de prover donen un error clar | No cal res de la std | És l'abast de la Fase 1, i el setup ja ha de fallar amb el que no suporta | M30, M31 |
| R3 | `Q` sense partir; `--max-q-degree > 0` dona error | Treu la partició de `Q` del tall | És l'abast de la Fase 1 | M33 |
| R4 | Una sola instància; més d'una dona error | Treu la complexitat de diverses instàncies | El transcript ja absorbeix el nombre d'instàncies i la prova en porta la llista (A.4, A.6) | M35 |
| R5 | `globalConstraints.json` amb 0 entrades | No cal res de restriccions globals | A la v1 no n'hi ha (D2) | Després de la v1 (M37) |
| R6 | Els `panic!` de les passades i les truncacions `u64` dels escriptors STARK es queden | Treu M27 del camí crític | No afecta cap format, i pilfflonk té escriptors propis | M27, abans de tancar la Fase 1 |
| R7 | Sense `check`; es depura amb l'oracle | Una fita menys al tall | És una eina de depuració | M25 |
| R8 | Oracle Rust amb `num-bigint`, només a `dev-dependencies` | Depuració més ràpida i la validació 5 | És només de test i no entra al camí de producció (principi 2) | Es queda com a test |
| R9 | E2E `#[ignore]` amb `PIL2C_EXEC`, sense cap `pilout` versionat | No depèn de P8 | Respecta que els `pilout` no es versionen | M28 |
| R10 | Blinding fix (llavor fixa) als tests i a la CI | Proves reproduïbles | És la decisió D6 | Es queda |
| R11 | Test d'enllaç GPU ajornat si no hi ha màquina amb `nvcc` | Cap bloqueig | Les entrades del Makefile ja hi són des de M1 | M28 |

**Dreceres descartades:**
- desactivar el blinding, perquè canvia els graus i el *layout* (D6);
- saltar-se el *digest* o la validació d'entrada del verificador;
- agafar els commitments fixos de la prova;
- versionar formats provisionals.

---

## 6. Després del primer tall

```
M20 ──► M22 ──► M23 ──┐
M21 ────┘             │
M20 ──► M24 ──────────┤
M20 ──► M25 ──► M26 ──┼──► M28  FASE 1  [P8, N12]
M14 ────────────┘     │
M10 ──► M27 ──────────┘

M28 ──► M30 ──► M31 ─────────┬──► M34  FASE 2
        M30 ──► M33 ──────────┤
M29 [P10] ────────────────────┘   (M29 és PIL pur: es pot fer en qualsevol moment)
M34 ──► M38 [D4] ──► M39  FASE 3
M39 ──► M35 ──► M36 ──► M37  DESPRÉS DE LA v1 (D2)
M28 ──► M40 ──► M41 ──► M42  FASE 4  [P1]  (el criteri de sortida necessita M39)
M39 ──► M43  FASE 5 (GPU, després de la versió CPU)
```

M21 només depèn de M1 i dels fitxers que ja hi ha a `pil-fflonk/config/`, i és una funció pura: es pot començar la segona setmana.

### 6.1 Fins als criteris de sortida de la Fase 1

| Fita | Objectiu i lliurables | Dependències | Criteri de fet | Mida |
|---|---|---|---|---|
| M21 Agrupació | `group()` a `setup/pilfflonk/src/grouping.rs` *(proposta)*, amb les regles 1–6 d'A.2.<br>**Golden:** sense executar JS. L'entrada i el resultat de l'exemple `all` ja existeixen (`pil-fflonk/config/pilfflonk.fflonkinfo.json` i `pilfflonk.shkey.json`), i es fan servir tal com són. | M1 | **Golden:** coincideixen les classes, les particions, l'ordre i les arrels, amb `powerW` numèric (C.3.7).<br>**Propietats:** cada parella (polinomi, offset) està coberta, cada `f` és d'un sol stage, el resultat és determinista i es compleix `kN \| r−1`. | L |
| M22 Empaquetat per defecte | El setup fa servir `group()`, amb `--extra-muls`, `powerW` i `ξ = xiSeed^powerW` | M20, M21 | La prova E2E verifica amb i sense `--no-packing` (validació 1) | S–M |
| M23 Im pols i *offsets* amb signe | Una AIR sintètica amb *offsets* `{−1,0,1,2}` i grau ≥ 4. El prover calcula els im pols amb el bytecode abans de comprometre l'stage 1 (§4.2.3). | M22 | E2E verd amb im pols triats, també amb un `--max-constraint-degree` baix | M |
| M24 Dominis | `pilout` sintètics fets amb `prost` (N1) per a `firstRow`, `lastRow` i `everyFrame`, i *zerofiers* al prover i al verificador; `δ_i` a `qDeg` | M20 | E2E per a cada domini, i rebuig si la restricció es viola a la fila de la vora | M |
| M25 `pilfflonk check` | Recorregut fila a fila amb la secció de depuració de `<air>.bin` i l'intèrpret de M17 | M20 | Amb un witness mutat, diu quina restricció i quina fila fallen (validació 3) | S–M |
| M26 Enduriment | - validació d'entrada completa;<br>- tests de manipulació de la prova, dels publics i del `provingKey/`;<br>- les tres implementacions: bytecode del prover contra l'oracle a totes les files, i el `qVerifier` del verificador JS contra l'oracle en punts aleatoris;<br>- ASan i UBSan sobre l'E2E;<br>- repàs de la llista C.3 | M14, M25 | Validacions 3, 4 i 5 de la Fase 1 | M |
| M27 `pil-info` sense `panic!` | - les passades retornen `Result` (hi ha uns 30 `panic!`/`unwrap`/`expect`);<br>- les truncacions `u64` es fan només en escriure (`types/stark_info.rs:254, 425, 524-593`; `io/bin_file.rs:268, 338, 365`; `output/global_constraints.rs:108, 137`; `io/fixed_cols.rs:297-305`) | M10 | Golden STARK idèntic (validació 2) | M |
| M28 CI i tancament de la Fase 1 | - job de CI: compilació BN254, `setup-pilfflonk`, `prove`, `verify` i els tests de rebuig;<br>- job `golden-setup`;<br>- test d'enllaç GPU | M22–M27, P8, N12, N13 | Les validacions 1–6 de la Fase 1 passen a la CI | S |

### 6.2 Fase 2: busos de la std en una instància

| Fita | Contingut | Criteri de fet | Mida |
|---|---|---|---|
| M29 Std BN254 | `pil2-components/lib/std/pil/bn254.pil`, amb `Bn254_Gen[i] = 5^((r−1)/2^i)` per a `i ≤ 28` i `Bn254_k`; `FIELD_BN254` i `ACTIVE_FIELD` derivat de `PRIME` (`std_constants.pil:1, 131-153`) | Els `pilout` Goldilocks dels 10 programes tenen el mateix sha256, i el *golden* continua verd | S |
| M30 Stage 2 | - els reptes de l'stage 2 surten de `numChallenges`;<br>- *hints* `gsum_col` i `gprod_col` en C++ sobre `Fr`, amb la semàntica de `calculateWitnessSTD` (`gen_proof.hpp:57`);<br>- el setup accepta aquests *hints* | Les columnes de l'stage 2 coincideixen amb una referència seqüencial ingènua (una extensió de l'oracle) | L |
| M31 `im_col` | `calculateImHints` (`gen_proof.hpp:27`) per als `im_col`. Els `im_airval` fan servir air values, i el setup els rebutja a la v1 (D2). | E2E amb `im_col` | M |
| M33 Partició de `Q` | `--max-q-degree`:<br>- `m` trossos, amb blinding PLONK a les fronteres;<br>- les avaluacions `Q_i(ξ)` a la prova;<br>- la comprovació `Σ ξ^(i·M·N)·Q_i(ξ) = Q(ξ)`;<br>- sense el defecte C.3.4 | E2E amb `Q` partit | M |
| M34 Fixtures i tancament | Plookup, Permutation i Connection de l'Annex G amb la std (`STD_MODE_ONE_INSTANCE`), un range check, en variant de suma i de producte, i l'exemple `all` en una sola AIR | Validacions 1–3 de la Fase 2 | L |

### 6.3 Fase 3: witness en `Fr` i rendiment

| Fita | Contingut | Mida |
|---|---|---|
| M38 Witness en `Fr` (D4) | Un tipus `Fr` al crate `fields` i una biblioteca de witness que alimenti `WitnessSource` | L (potser dues fites) |
| M39 Rendiment i tancament | Informe de temps i memòria fins al límit de P2 (`N ≤ 2^24`), i avaluació per blocs si cal | M |

### 6.4 Fase 4: Solidity (prioritat segons P1)

| Fita | Contingut | Mida |
|---|---|---|
| M40 | Plantilles `tera` a `setup/pilfflonk` (el patró de `templates.rs`) i l'opció `--solidity`. Com a referència estructural, `PilFflonkVerifier` i `ShPlonkVerifier`. | L |
| M41 | El codificador de calldata (inverses auxiliars), i Foundry acceptant les proves de les fases 1–3 | M |
| M42 | *Fuzzing* diferencial entre el verificador JS i el Solidity, i un informe de gas | M |

### 6.5 Fase 5: GPU (necessària, després de la versió CPU; prioritats segons M39)

| Fita | Contingut | Mida |
|---|---|---|
| M43 | MSM i NTT de `pil2-stark/src/bn128/src/{msm,ntt}` amb `--gpu` (la MSM amb `mont=true`), i l'intèrpret a GPU si cal. Els resultats han de ser idèntics bit a bit als de CPU. | L+ |

### 6.6 Després de la v1: diverses AIRs i instàncies (D2)

| Fita | Contingut | Mida |
|---|---|---|
| M35 M instàncies d'una AIR | - ordre canònic, un transcript i una obertura;<br>- les columnes fixes s'obren una sola vegada per AIR;<br>- rebuig si s'elimina, es duplica o es reordena una instància | M–L |
| M36 Diverses AIRs | - `N` per AIR i `powerW` com a mínim comú múltiple de tots els `k`;<br>- la variant BN254 de `fibonacci-square` **sense** el custom commit `rom` (N15) | M–L |
| M37 Valors, agregació i restriccions globals | Air, airgroup i proof values a la prova i al transcript (A.4, pas 2.2); SUM i PROD; la std en el mode per defecte; el verificador JS agrega els airgroup values i avalua `pilout.globalConstraints.json` | M–L |

---

## 7. Riscos del pla

| Risc | Impacte | Mitigació |
|---|---|---|
| Divergència C++ ↔ JS (transcript, codificació) | La prova no verifica i costa trobar per què | Test creuat del transcript a M8 abans de M18, i els mateixos vectors a totes dues bandes |
| Camí crític gairebé pla | Un retard a M5, M6 o M7 retarda el tall | Començar la pista A el primer dia, i tenir l'oracle (M14) abans de M18 |
| El format del bytecode no quadra entre M11 i M17 | Refer feina | Tancar N8 abans de M11, i fer un test creuat tan aviat com hi hagi el primer `.bin` |
| No-determinisme del setup STARK | Falsos positius al *golden* | M0 comprova el determinisme abans de res |
| Una sola persona | Unes 12 setmanes fins al tall | Ordre recomanat: M0, M1, M5, M6, M4, M7, M9, M10, M11, M12, M13, M15, M16, M17, M18, M8, M19, M20. M14 i M21 als forats. |

---

## 8. Coses pendents de decidir

La columna "Avançar?" diu si es pot continuar treballant assumint la recomanació.

| # | Què és | Recomanació | Bloqueja | Avançar? | Termini |
|---|---|---|---|---|---|
| ~~P7~~ | **Decidit (29-09-2026):** la llicència és l'actual | — | — | — | — |
| ~~D1~~ | **Decidit (29-09-2026):** el codi compartit és una dependència. Crate `pil-info`: primer una extracció mecànica i després la parametrització (§2.4). | — | M9 | — | — |
| ~~D2~~ | **Decidit (29-09-2026):** la v1 reprodueix pil-fflonk amb PIL2: una AIR, una instància, una prova; sense air/airgroup/proof values ni restriccions globals (std en `STD_MODE_ONE_INSTANCE`). Diverses AIRs o instàncies (M35), en una versió futura. | — | M18, M35 | — | — |
| ~~D3~~ | **Decidit (29-09-2026):** no hi ha verificador natiu, com el FFLONK existent. Cap *pairing* a C++; ffiasm no es toca. M2 i M3, eliminades. | — | — | — | — |
| ~~D8~~ | **Decidit (29-09-2026):** verificador JS adaptat del de pil-fflonk (`fflonk_verify.js` + `verifyOpenings`) als canvis de pilfflonk, a `pilfflonk/js/`, cridat per `proofman-cli pilfflonk verify` | — | M8, M19 | — | — |
| ~~D4~~ | **Decidit (29-09-2026):** (a), un tipus `Fr` a `fields` i una biblioteca de witness, a la Fase 3 | — | M38 | — | — |
| ~~D5~~ | **Decidit (29-09-2026):** cerca de 2 a 9, amb `--max-constraint-degree` | — | M16 | — | — |
| ~~D6~~ | **Decidit (29-09-2026):** blinding sempre actiu. Als tests i a la CI és fix (llavor fixa injectada al RNG); a producció, aleatori. | — | M18 | — | — |
| ~~D7~~ | **Decidit (29-09-2026):** format equivalent a l'actual. La prova són bytes en *big-endian* (commitments `x‖y` i després avaluacions); la vista JSON és d'estil snarkjs (A.6). | — | M12, M18 | — | — |
| ~~P1~~ | **Decidit (29-09-2026):** proves directes per millorar les agregacions | — | La prioritat de la Fase 4 i l'objectiu de M39 | — | — |
| ~~P2~~ | **Decidit (29-09-2026):** `2^28` és el grau màxim possible (el `ptau` més gran disponible), no el `ptau` que es farà servir; amb `qDeg ≤ 8`, `N ≤ 2^24` | — | M39 | — | — |
| ~~P4~~ | **Decidit (29-09-2026):** pil-fflonk és l'única referència, de funcionalitat i de format de la prova (D7). No cal que cap verificador JS o Solidity existent accepti les proves noves. | — | — | — | — |
| ~~P5~~ | **Decidit (29-09-2026):** com pil-fflonk, cap custom commit; el setup els rebutja | — | M15 | — | — |
| ~~P6~~ | **Decidit (29-09-2026):** exactament el transcript del FFLONK existent (`Keccak256Transcript`) | — | M4 | — | — |
| ~~P8~~ | **Decidit (29-09-2026):** compilador local (`PIL2C_EXEC`); `package.json` només si cal. La branca del compilador no es fa *commit* sense demanar-ho a l'usuari. | — | M28 | — | — |
| ~~P9~~ | **Decidit (29-09-2026):** `compile-pil` rep un paràmetre nou, `-P, --config <json>`, igual que el de `pil2com`, i el passa tal qual. Per defecte no hi és, i el comportament és l'actual (Goldilocks). És un camp `config: Option<String>` a `CompilePilOptions`, amb `None` als 11 llocs on es construeix. | — | M13 | — | — |
| ~~P10~~ | **Decidit (29-09-2026):** d'acord; entra quan es portin els exemples de connexió de pil-fflonk | — | M34 | — | — |
| N1 | A `develop-0.14.0`, les restriccions de l'usuari són sempre `everyRow` (`processor.js:2040`; `when` s'analitza però no s'executa) | Fer servir `L1`/`LLAST` fixes al Fibonacci i provar els altres dominis amb `pilout` sintètics. **L'especificació ja està corregida.** | M13, M24 | Sí | — |
| N2 | Com es desa el *golden* STARK | Un manifest sha256 amb scripts; els fitxers sencers, només en local o com a artefacte de CI | M0 | Sí | Abans de M0 |
| N3 | Arnès de tests C++ | Un target `pilfflonk_test` al Makefile, amb `assert` i sense gtest. Els E2E, a Rust. | M1 | Sí | Abans de M1 |
| N4 | Refinament de l'API C (§5.3):<br>- falten els publics i els proof values per a `Q`;<br>- falta la llavor del RNG (`randombytes_buf_deterministic`);<br>- falta el missatge de l'últim error | L'absorció de les avaluacions ja està decidida: les absorbeix Rust, i `pilfflonk_open` comença fent `squeeze` d'`α_S`. **L'especificació ja està corregida.** La resta, a M1/M18. | M1 (signatures), M18, M19 | Sí | L'esbós abans de M1; tancat abans de M18 |
| ~~N5~~ | **No cal:** amb D3 (d), ffiasm no es toca (M2 eliminada) | — | — | — | — |
| N6 | Codificació de `pilfflonk.srs.bin` | Copiar les seccions del `ptau` (punts afins en Montgomery *little-endian*, com fa la zkey). Està a A.6 com a proposta. | M6, M15 | Sí | Abans de M6 |
| N7 | Moure també a `pil-info` els constructors JSON d'`expressionsinfo`/`verifierinfo` i de la part comuna del `globalInfo` | Sí, per no duplicar-los | M9 | Sí | Abans de M9 |
| N8 | Format del bytecode pilfflonk | El contenidor i els codis d'operació del `"chps"` del STARK, amb dimensió 1, constants de 32 bytes *little-endian* i una versió pròpia | M11, M17 | Sí | Abans de M11 |
| N9 | Format del fitxer de witness i opció de la CLI | Un directori amb:<br>- `instances.json`: `(airgroupId, airId)` en ordre canònic i els air values;<br>- un `.bin` per instància, fila per fila, amb valors de 32 bytes *little-endian*;<br>- `publics.json` i `proof_values.json`.<br>A la CLI, `prove --witness <dir>`. | M13, M18 | Sí | Abans de M13 |
| N10 | Keccak al setup per al *digest* | Per FFI, amb el `keccak_wrapper` de C++, perquè al *workspace* no hi ha cap crate de Keccak. L'alternativa és `sha3` o `tiny-keccak`, que seria una dependència nova. | M15 | Sí | Abans de M15 |
| ~~N11~~ | **Resolt (29-09-2026):** la còpia retallada és a `../pil-stark` (pil-stark `5e20f57` + shplonkjs `7824640`, 1,7 MB). Només es llegeix; no s'hi executa res (D3). | — | M21 | — | — |
| N12 | E2E a la CI abans de P8 | `#[ignore]` i un job que clona el compilador de la branca fixant el commit. Cal que l'usuari hagi pujat la branca. L'alternativa és versionar el `pilout` de la fixture. | M28 | Sí | Abans de M28 |
| N13 | `ptau` dels tests | No es baixa cap `ptau` (l'usuari: són molt grans) ni es fa servir JS. Un ajudant de test en C++ genera un `ptau` petit amb una `τ` fixa (`[τ^i]₁` i `[τ^i]₂` amb ffiasm), en el format binfile de snarkjs i amb només les seccions 2 i 3, que són les que es llegeixen. És determinista, es desa a `target/` i no es versiona. `PILFFLONK_TEST_PTAU` permet fer servir un altre fitxer. Fora dels tests, el `ptau` és una entrada (`--powers-of-tau`) de com a molt `2^28` (P2). | M6, M20, M28 | Sí | Abans de M6 |
| N14 | On van les fixtures, els generadors i els E2E | `pilfflonk/tests/{fixtures,data}/` i `pilfflonk/tests/e2e_*.rs` | M13 | Sí | Abans de M13 |
| N15 | La variant BN254 de `fibonacci-square` | `fibonaccisq.pil:21` declara `commit stage(0) public(rom_root) rom`, un custom commit que és fora d'abast. Cal una variant sense `rom`. | M36 | Sí | Abans de M36 |

---

## 9. Primeres accions

1. **(Usuari) La branca del compilador.** Revisar-la i fer-ne el *commit* quan convingui (P8). No bloqueja res: mentrestant es fa servir el compilador local.
2. **M0.** Escriure `setup/golden/generate.sh` i `setup/golden/check.sh`, i generar el manifest a `ff0ff959` dues vegades per comprovar que el setup és determinista.
3. **M1.** Crear:
   - `pil2-stark/src/pilfflonk/` i `src/api/pilfflonk_api.{hpp,cpp}`;
   - les entrades a les tres llistes del Makefile;
   - `bindings_pilfflonk.rs` i `ffi_pilfflonk.rs`;
   - els tres crates, buits.

   Criteri: un primer `pilfflonk_*` cridat des d'un test Rust.
4. **M13 (la part PIL).** Portar `pil-fflonk/pil/sm_fibonacci/fibonacci.pil` a PIL2 amb `L1`/`LLAST` i compilar-lo amb `proofman-setup compile-pil -P bn254.json` (P9), amb `PIL2C_EXEC` apuntant al compilador local. Comprovar amb `proofman-cli pilout inspect` que té `baseField = r`, 5 restriccions `everyRow` i cap *hint*.
5. **M21, preparació.** Convertir `pil-fflonk/config/pilfflonk.fflonkinfo.json` i `pilfflonk.shkey.json` en el *golden* de l'agrupació, sense executar JS.

---

### Fitxers crítics

- `setup/pil2-stark/src/pil/info.rs`: el punt d'entrada de les passades, amb `StarkStruct` i `max_deg`; és on es fa el tall de M9 i M10.
- `setup/pil2-stark/src/expr/helpers.rs`: `NEG_ONE` i els usos de `FIELD_EXTENSION` per a les dimensions (M10).
- `pil-fflonk/src/shplonk.cpp`: `ShPlonkProver`, la base de M7 i M18.
- `pil2-stark/Makefile`: les llistes de fonts (`:228, 231, 234`) i els targets `fflonkSetup`, `plonkSetup` i `plonkProve` (`:115-118`), per a M1.
- `pil-stark/src/fflonk/helpers/fflonk_verify.js` i `shplonkjs/src/helpers/verifier.js` (a `../pil-stark`): el verificador que s'adapta (M8, M19).
- `setup/pil2-stark/node_modules/snarkjs/src/Keccak256Transcript.js` i `proofman/src/snark_wrapper.rs:580-638`: el transcript JS i com es crida un verificador JS.
