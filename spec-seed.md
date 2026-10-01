# Especificació: backend fflonk per a PIL2 a pil2-proofman

**Estat:** esborrany v4 (29-09-2026). Aquesta versió incorpora una revisió completa feta per cinc agents contra el codi.

**Base de codi de la implementació:**
- `pil2-proofman`: `pre-develop-1.4.0-alpha` (`ff0ff959`), que ja té el setup STARK en Rust.
- `../pil2-compiler`: `develop-0.14.0` (`503862c`), amb els canvis d'aquest projecte a la branca local `develop-0.14.0-pil2-fflonk` (§4.1). El setup fixa la branca `develop-0.14.0` a `setup/pil2-stark/package.json`; el commit concret queda al `package-lock.json`, que no es versiona.

**Nom.** Tots els components nous porten el prefix **`pilfflonk`** (el nom del projecte original), per no confondre'ls amb el *wrap* final fflonk que ja existeix. Aquests noms ja estan agafats: `namespace Fflonk` (`FflonkProver`, `FflonkSetup`), `fflonk_setup_c`, `generate_fflonk_zkey_c`, `SnarkProtocol::Fflonk`, `zkey_fflonk.*`, el target `fflonkSetup` del Makefile, `FflonkVerifier.sol` i el directori `pil2-stark/src/fflonk_setup/`.

**Convenció de rutes.** Cada ruta és relativa a l'arrel del seu repositori. Les de pil2-proofman no porten prefix; les dels altres repositoris porten el nom del repositori (`pil-fflonk/…`, `pil2-compiler/…`, `pil-stark/…`, `shplonkjs/…`).

## Com llegir aquest document

El document va del *què* al *com* i segueix el camí de les dades, del programa PIL2 fins a la prova verificada. Es llegeix de dalt a baix:

1. **Què volem aconseguir** (§1)
2. **El flux complet en una pàgina** (§2)
3. **D'on partim:** el sistema antic, el repositori actual i el compilador (§3)
4. **Com funcionarà cada pas del flux** (§4)
5. **On viu el codi** (§5)
6. **En quin ordre es construeix** (§6)
7. **Què està decidit i què no** (§7)

Els annexos són la referència detallada per a qui implementi:
- **A:** el protocol normatiu;
- **B:** el mapatge PIL1 → PIL2;
- **C:** l'inventari del sistema antic;
- **D:** els fitxers clau;
- **E:** la base analitzada;
- **F:** les troballes col·laterals;
- **G:** el programa PIL1 de la fixture de pil-fflonk;
- **H:** el rendiment de la versió CPU (M39);
- **I:** el gas del verificador Solidity (M42).

### Glossari

| Terme | Significat |
|---|---|
| `r`, `Fr` | L'ordre del grup de BN254 i el seu camp escalar. Tots els valors dels polinomis són elements d'`Fr`. |
| KZG, SRS, `ptau` | KZG és l'esquema de compromisos polinomials sobre corba el·líptica. L'SRS són les potències `[τ^i]₁` i `[τ]₂`, i el fitxer `ptau` de snarkjs les conté. |
| fflonk, `f_i`, `k`, *layout* | fflonk empaqueta `k` polinomis `p_j` en un de sol, `f_i(X) = Σ_j p_j(X^k)·X^j`. El *layout* és la llista de `f_i` d'una AIR. |
| SHPLONK, `W`, `W'` | L'esquema d'obertura que obre molts polinomis en molts punts amb dos commitments, `W` i `W'`, i un sol *pairing*. |
| `xiSeed`, `ξ`, `powerW` | `xiSeed` és el repte que surt del transcript. El punt d'avaluació és `ξ = xiSeed^powerW`, on `powerW` és el mínim comú múltiple de tots els `k`. |
| `std_vc`, `std_xi` | Reptes que afegeix el setup: `std_vc` plega les restriccions, i `std_xi` és el repte del punt d'avaluació (aquí, `xiSeed`). |
| AIR, airgroup, instància | Una AIR és una traça amb les seves restriccions; les AIRs s'agrupen en airgroups, i cada AIR pot tenir diverses instàncies en una mateixa prova. |
| stage | Cada ronda de commitments. Els reptes de l'stage `s+1` depenen de tot el que s'ha compromès fins a l'stage `s`. |
| *hint* | Metadada que la std posa al `pilout` per dir com es calcula una columna (per exemple, `gsum_col`). |
| bus (de suma o de producte) | La manera com la std expressa lookups, permutacions i connexions: una columna acumulada que ha de quadrar globalment. |
| im pols | Polinomis intermedis que el setup introdueix per mantenir baix el grau de les restriccions. No s'han de confondre amb les columnes `im_col` que declara la std. |
| `evMap` | La llista ordenada de parelles `(columna, offset)` que s'avaluen al punt `ξ·ω^offset`. |
| *zerofier* `Z_D` | El polinomi que s'anul·la exactament a les files del domini `D` d'una restricció. |
| *coset*, LDE | Avaluar un polinomi sobre `g·H'`, un subgrup desplaçat més gran que la traça, per poder dividir punt a punt. |
| `provingKey/` | El directori de sortidqa del setup. Té la mateixa estructura que el del STARK (§4.2.6). |
| *digest* | L'empremta Keccak-256 de la vkey (`pilfflonk.vkey.json`), que conté tot el que afecta la verificació (A.6). |
| ordre canònic | Les AIRs s'ordenen per `(airgroupId, airId)` i les instàncies per `(airgroupId, airId, índex d'instància)`, que és l'ordre en què les dona el `WitnessSource`. |
| *golden* | Un test que compara una sortida amb una referència desada, byte a byte. |

---

## 1. Què volem aconseguir

**Objectiu:** poder provar programes PIL2 amb **fflonk sobre BN254** dins de pil2-proofman: setup, prova i verificació.

**Què és fflonk, en poques paraules.** És un sistema de prova basat en compromisos KZG sobre la corba BN254. Té tres trets clau:
- **Empaqueta polinomis:** en compromet `k` amb una sola multiplicació multiescalar (MSM).
- **Obre tots els polinomis alhora amb SHPLONK:** la prova són uns quants punts i escalars, i es verifica amb un sol *pairing*, que és barat a Ethereum.
- **Tot és sobre `Fr` de BN254**, a diferència del camí STARK, que fa servir Goldilocks i FRI.

**Punt de partida.** Ja existeix un prover fflonk per a PIL1, a `../pil-fflonk`. No es pot fer servir tal com està, per dos motius:
- parla PIL1, i pil2-proofman treballa amb PIL2;
- depèn d'un altre repositori, `pil-stark` (JS, branca `pilfflonk`): pil-stark hi posa els programes, la compilació, la generació de `fflonkinfo`, l'agrupació i l'únic verificador que existeix, i pil-fflonk hi posa el prover i un setup C++ parcial. Els fitxers de `pil-fflonk/config/` són la sortida d'un exemple de pil-stark (§3.1 i Annex G).

**Dins d'abast**
- Compilar programes PIL2 sobre BN254 amb `../pil2-compiler`.
- Setup en Rust, al costat del setup STARK.
- Prover. **La v1 reprodueix el que fa pil-fflonk, però amb PIL2 (D2):**
  - una AIR i una instància;
  - stages arbitraris;
  - els busos de la std, en mode `STD_MODE_ONE_INSTANCE`;
  - publics.

  Diverses instàncies (i, per tant, diverses AIRs), els air, airgroup i proof values i les restriccions globals queden **fora d'abast** (usuari, 29-09-2026).
- Verificador JS, com el del FFLONK existent, que es verifica amb snarkjs. S'adapta del verificador de pil-fflonk (D8). En una fase posterior, verificador Solidity.

**Fora d'abast**
- **Canviar el comportament del camí STARK.** Algunes peces compartides sí que es toquen (les passades simbòliques del setup, la std, `libstarks`, el Makefile i els *bindings*), però amb una garantia: les sortides STARK surten idèntiques byte a byte. Aquesta és la porta de la Fase 1.
- Recursió o agregació amb proves STARK. El *wrap* final SNARK actual no es modifica.
- Custom commits, periodic columns i public tables. El setup els rebutja amb un error clar.
- Prova distribuïda (MPI).
- Verificador natiu: el FFLONK existent tampoc no en té (D3).
- Compatibilitat amb els formats de pil-fflonk, que es deixarà de fer servir en favor de pilfflonk (P4).
- Els dominis de restricció nous del `pilout` v2 (branca `feature/domain-constraints` del compilador). Els quatre tipus de restricció del v1 (`everyRow`, `firstRow`, `lastRow`, `everyFrame`) sí que hi entren.

**Criteri d'èxit de la v1.** Els exemples de pil-fflonk portats a PIL2 (Annex G) compleixen tres condicions. Per exemple, `all`, que combina Fibonacci, connection, permutation i plookup en una sola AIR.
- es proven amb `proofman-cli pilfflonk prove`;
- el verificador JS i, a la Fase 4, el contracte Solidity n'accepten la prova;
- qualsevol mutació de la prova, dels publics o del witness es rebutja.

---

## 2. El flux complet

```
   programa.pil
        │
        │  PAS 1 · Compilar ─ proofman-setup compile-pil -P (prime = r de BN254)
        ▼
   programa.pilout
        │
        │  PAS 2 · Setup ─ proofman-setup setup-pilfflonk (Rust) + fitxer ptau
        ▼
   provingKey/   (mateixa estructura que el del setup STARK, §4.2.6)
        │
        │  PAS 3 · Witness ─ columnes i air values de l'stage 1, publics i proof values (en Fr)
        │  PAS 4 · Prova ─ proofman-cli pilfflonk prove (Rust orquestra, C++/ffiasm calcula)
        ▼
   proof.json · publics.json
        │
        │  PAS 5 · Verificació ─ proofman-cli pilfflonk verify (verificador JS), o el contracte Solidity
        ▼
   acceptada / rebutjada
```

| Pas | Què fa | Qui ho fa | Entrada | Sortida |
|---|---|---|---|---|
| 1. Compilar | Converteix el PIL2 en un `pilout` sobre el camp BN254 | `pil2com` de `../pil2-compiler`, amb la std adaptada | `.pil` | `.pilout` |
| 2. Setup | Analitza les restriccions, agrupa els polinomis i genera les claus | `proofman-setup setup-pilfflonk` (Rust, amb ffiasm per FFI) | `.pilout`, `.ptau` | `provingKey/` (§4.2.6) |
| 3. Witness | Aporta els valors de l'stage 1 | Una font de witness: un fitxer, i més endavant una biblioteca | Programa i entrades | Columnes, air values, publics i proof values en `Fr` |
| 4. Prova | Executa els stages, compromet i obre | `proofman-cli pilfflonk prove` | `provingKey/` i witness | `proof.json`, `publics.json` |
| 5. Verificació | Refà el transcript, comprova les restriccions i fa el *pairing* | `proofman-cli pilfflonk verify`, que crida el verificador JS com `verify-snark` crida snarkjs, o Solidity, amb el calldata de `proofman-cli pilfflonk calldata` (§4.5) | `pilfflonk.vkey.json`, prova i publics | Sí o no |

### 2.1 Principis que guien el disseny

1. **Un camí germà, no una generalització.** El runtime STARK no es toca. El camí pilfflonk en comparteix el `pilout`, les passades simbòliques del setup, els binaris `proofman-setup` i `proofman-cli`, i `libstarks`.
2. **Rust orquestra i C++ calcula.** L'aritmètica pesada de BN254 del prover (MSM, NTT, polinomis) és C++ amb **ffiasm**, i la de GPU és `pil2-stark/src/bn128/src/{msm,ntt}`. S'exposa amb FFI escrita a mà, amb el mateix patró que el camí STARK, i no s'hi afegeix cap biblioteca de corbes nova. En Rust només hi ha aritmètica d'enters grans al setup (`num-bigint`) i, per D4(a), el tipus `Bn254` de `proofman-fields` per calcular el witness en `Fr` (§4.3).
3. **El setup és en Rust, al costat del STARK.** Les passades simbòliques (restriccions, im pols, mapes, codegen) són una sola implementació, parametritzada pel camp.
4. **El setup decideix i el prover executa.** Els graus, l'agrupació i el bytecode es fixen al setup. El prover no pren cap decisió.
5. **Errors explícits.** No hi ha estat global nou, ni `panic!`, ni crides a `exit()` en codi de biblioteca. Tot allò que no se suporta falla **al setup**, no en provar.
6. **Verificació independent.** Com al FFLONK existent, on el prover és el C++ de rapidsnark i la verificació la fa snarkjs, el verificador és JS i no comparteix codi amb el prover (D3, D8).
   - Té el seu propi camí per a la seqüència del transcript, `Q(ξ)`, la comprovació SHPLONK i el *pairing* (`ffjavascript`).
   - Calcula `Q(ξ)` amb el `qVerifier` que el setup posa a la vkey, com el verificador STARK, i no amb el bytecode del prover.
   - Només comparteix amb el prover el codegen del setup. Per compensar-ho, els tests contrasten els valors amb un recorregut directe del `pilout`.
7. **A partir de la Fase 1, cada fase acaba amb una prova end-to-end** verificada per codi que no l'ha produïda, i amb tests de rebuig.

---

## 3. D'on partim

### 3.1 El sistema antic: pil-fflonk, pil-stark i shplonkjs

El prover fflonk actual es reparteix en tres repositoris. Al commit `b385c38`, pil-fflonk no conté cap fitxer `.pil`: els programes i els seus generadors eren a pil-stark. Ara n'hi ha una còpia local a `pil-fflonk/pil/` (Annex G).

| Peça | Llenguatge | Què aporta |
|---|---|---|
| `../pil-fflonk` | C++17 sobre ffiasm | **Prover** d'una sola AIR PIL1 sobre `Fr` de BN254, amb transcript Keccak-256. Inclou:<br>- **el SHPLONK complet de la banda del prover** (`src/shplonk.cpp`, classe `ShPlonkProver`): empaquetat dels `f_i`, commits, `R`, `Z_T`, `L`, `W`, `W'`, arrels, reptes i inversa en lot;<br>- una **NTT multicolumna de CPU** (`src/ntt_bn128.hpp`);<br>- un **setup C++** (`pfSetup`) que genera la `zkey` a partir d'un `shkey` ja calculat;<br>- una via de witness amb circom (`.exec`, `.dat` i `zkin`).<br>**No té verificador.** |
| `pil-stark`, branca `pilfflonk` | JS | **Preprocessament:** `fflonkinfo.json`, la **decisió de les classes** d'agrupació (`src/fflonk/helpers/fflonk_shkey.js`), la `zkey`, la `vkey` i el codi C++ específic de cada circuit (*chelpers*).<br>**Verificador:** és l'únic que existeix (`src/fflonk/helpers/fflonk_verify.js`); el mateix pil-fflonk verifica amb `node ../pil-stark/src/fflonk/main_verifier.js` (`tools/test_examples.sh:30`).<br>**Solidity:** el contracte `PilFflonkVerifier` (`src/fflonk/solidity/`).<br>**Un prover JS complet** (`src/fflonk/helpers/fflonk_prover.js`), útil com a font de vectors de referència. |
| `shplonkjs` | JS | La **repartició** dels grups en `f_i` (`getFCustom`, `applyExtraScalarMuls` a `src/helpers/setup.js`), les arrels, la verificació SHPLONK (`verifyOpenings`) i el contracte `ShPlonkVerifier`, que és on es fa el *pairing* (`src/solidity/verifier.sol.ejs`) |

```
── repositori pil-stark (JS, branca pilfflonk) ─────────────────────────────────────────────
 test/state_machines/sm_*/*.pil           (els programes PIL1 d'exemple)
 test/state_machines/{sm,sm_*}/sm_*.js    (els generadors de constants i witness, en JS)
        │  test/cfiles/fflonk_gen_*_files.js   (un test mocha per a cada grup d'exemples)
        │    pilcom compile (camp BN254, en memòria)
        │    fflonkInfoGen → fflonkinfo.json      fflonk_shkey + shplonkjs → shkey.json
        │    fflonkSetup + ptau → .zkey → .vkey    const/commit → .const/.commit
        │    main_buildchelpers.js → *.chelpers.*.cpp
        ▼
 tmp/<exemple>.*
        │  pil-fflonk/tools/copy_generated_files.sh (a config/pilfflonk.* i src/chelpers/)
── repositori pil-fflonk (C++) ─────────────────────────────────────────────────────────────
 config/pilfflonk.*  +  src/chelpers/*.cpp  →  make  →  pfProver  →  runtime/proof.json
        │
── de tornada a pil-stark ──────────────────────────────────────────────────────────────────
 node ../pil-stark/src/fflonk/main_verifier.js  →  OK / FAIL
```

**D'on surt `pil-fflonk/config/`.** Coincideix amb l'exemple `all` de pil-stark:
- **Programa i generador.** El programa és `test/state_machines/sm_all/all_main.pil` (Fibonacci, Connection, Permutation i Plookup). El generador és `test/cfiles/fflonk_gen_all_files.js`, amb `extraMuls: 2`, `maxQDegree: 0` i entrades de Fibonacci `[1, 2]`.
- **Mides:** `N = 2^8`, 9 constants, 15 columnes compromeses, 3 publics (`in1`, `in2`, `out`) i 9 `f_i` amb `powerW = 12`. Els noms per stage del `shkey` també coincideixen.
- **Publics:** `runtime/public.json` és `[1, 2, out]`.
- **Versió de pil-stark.** La fixture és una mica anterior a `5e20f57`, perquè el seu `shkey` no té els camps `primeQ`/`n8q`/`primeR`/`n8r`. Per això `pfSetup` hi falla.

**Com funciona el prover** (`pilfflonk_prover.cpp:287-432`). Els stages són fixos:

| Stage | Què fa |
|---|---|
| 0 | Constants i publics |
| 1 | Witness |
| 2 | `h1` i `h2` del plookup |
| 3 | Grans productes `Z` i polinomis intermedis |
| 4 | El quocient `Q` |

Després ve l'obertura SHPLONK.

**Blinding.** Cada columna compromesa dels stages 1–3 que forma part d'un `f_i` rep `nOpenings+1` termes aleatoris `b·X^j·(X^N−1)`. `nOpenings` és el nombre de punts d'obertura del seu `f_i`, comptat després de les fusions. Els afegeix en forma de coeficients, després de la INTT (`pilfflonk_prover.cpp:735-765`). Les constants no en reben, i `Q` només en rep quan es parteix.

**Què n'aprofitem i què no.** Regla: abans de copiar res de pil-fflonk, es comprova si ja existeix en aquest repositori o en alguna dependència. Si existeix, es fa servir com a dependència.
- **Ja existeix a pil2-proofman, i es fa servir tal com és:**
  - **ffiasm:** `Fr`, G1, G2, FFT i MSM.
  - **rapidsnark:**
    - `Polynomial`: `fromEvaluations` amb espai per al blinding, `blindCoefficients` (que suma `(X^N−1)·b(X)`), `divByMonic`, `divByVanishing`, `lagrangePolynomialInterpolation`, `zerofierPolynomial` i `fastEvaluate`;
    - `Evaluations`, per a l'extensió amb la FFT;
    - `CPolynomial`, l'empaquetat fflonk `f(X) = Σ_j p_j(X^n)·X^j`;
    - `Keccak256Transcript`;
    - la lectura del `ptau` amb `BinFile` (`fflonk_setup.cpp:46-49, 527-531`).

  El `FflonkProver` de rapidsnark conté les mateixes peces de SHPLONK (`computeR*`, `computeZT`, `computeL`, `getMontgomeryBatchedInverse`), però fixades per al fflonk de R1CS: tres polinomis `C0`/`C1`/`C2` amb `k` fix.
- **Només existeix a pil-fflonk, i s'adapta:** l'orquestració **genèrica** de SHPLONK sobre una llista arbitrària de `f_i` (`src/shplonk.cpp`, `ShPlonkProver`): les arrels, `R`, `Z_T`, `L`, `W` i `W'`, la inversa en lot i les avaluacions. S'escriu sobre les peces de rapidsnark que acabem de citar, i se'n corregeixen els defectes de l'Annex C.3.
- **Es descarta de pil-fflonk,** perquè ja existeix aquí o no cal:
  - la NTT multicolumna (`ntt_bn128`): la versió de CPU fa servir la FFT d'ffiasm columna a columna, i la de GPU té NTT pròpia;
  - `extend`: el substitueixen `fromEvaluations` i `blindCoefficients`;
  - `computeFCommitments`: el substitueixen `CPolynomial` i `multiMulByScalar`, com fa rapidsnark;
  - el seu `Polynomial`;
  - el seu transcript.
- **Es porta de JS** el que no existeix en cap altre llenguatge:
  - el verificador (`fflonk_verify.js` i `verifyOpenings`), que es queda en JS però s'adapta als canvis de pilfflonk (§4.5, D8);
  - l'algorisme d'agrupació (`fflonk_shkey.js` i `shplonkjs/src/helpers/setup.js`), a Rust;
  - les plantilles Solidity (EJS) de `PilFflonkVerifier` i `ShPlonkVerifier`, a `tera`.

  L'agrupació i les plantilles es reescriuen. El verificador s'adapta i continua en JS, com el del FFLONK existent. `../pil-stark` és la referència de lectura. A banda del verificador, l'únic JS és el compilador PIL2, que ja forma part de la cadena actual.
- **Descartem el que depèn de PIL1:**
  - els stages fixos;
  - el C++ generat per circuit;
  - la maquinària `h1`/`h2`/`Z`.
- **Correcció de *soundness*.** El verificador JS fa el *pairing* amb els commitments de constants que porta la prova, en lloc dels de la vkey. El disseny nou ho corregeix (§4.5).

L'inventari detallat és a l'**Annex C**.

### 3.2 pil2-proofman avui

**Setup STARK en Rust.** El crate és `pil2-stark-setup` (`setup/pil2-stark`) i el binari, `proofman-setup`. Té els subcomandaments `setup`, `stats`, `setup-snark`, `setup-compressed-final`, `setup-recursive-test`, `rebuild-witness-libs`, `compile-pil` i `gen-exps` (`setup/pil2-stark/src/main.rs:23-43`). El setup pilfflonk n'ha de seguir tres patrons:
- **Orquestració en Rust i càlcul en C++ via FFI.** Per exemple, `compute_const_tree_c` → `build_const_tree_c` (`proving_key/bctree.rs`) i `generate_fflonk_zkey_c` → `fflonk_setup_c` (`proving_key/snark_setup.rs:496`).
  - La funció C captura les excepcions i retorna un codi d'estat (`pil2-stark/src/api/starks_api.cpp:1284-1295`).
  - Rust valida les entrades abans de fer la crida (`bctree.rs:24-42`).
  - Hi ha dos detalls que no s'han de copiar: `fflonk_setup_c` fa servir `new`/`delete` sense RAII, i l'embolcall Rust `compute_const_tree_c` fa `panic!` si rep un estat d'error (`provers/starks-lib-c/src/ffi_starks.rs:70-73`).
- **Compilació amb `pil2com`.** `commands/compile_pil.rs:46-66` el crida sense cap primer. `ensure_pil2com_exec` (`proving_key/recursive.rs:1106`) el localitza, i abans que res mira la variable `PIL2C_EXEC`.
- **Solidity amb `tera`.** Les plantilles `tera` (`setup/stark-recurser/stark2circom/circuit_templates/`) només generen l'embolcall i la interfície. El verificador del *wrap* surt de la plantilla de snarkjs (`snark_setup.rs:512-518`). A més, `render()` és `pub(super)` i `tera` només és dependència de `pil2-stark-recurser`, de manera que el patró es copia però no es crida.

Les passades simbòliques fan `panic!` en lloc de retornar errors (p. ex. `pil/prepare.rs:70-80`). Això **no** s'ha de copiar.

**Les passades simbòliques** són a `setup/pil2-stark/src/pil/` (`prepare`, `constraint_poly`, `im_polynomials`, `map`, `codegen`, `cse`, `gen_code`, `info`, `fri_poly`), `expr/` i `types/pilout_info.rs`. Només depenen de `pil2-pilout`, `indexmap`, `serde` i `tracing`.

Tenen dos lligams amb STARK:
- `StarkStruct`, només a `prepare.rs` i `info.rs`;
- FRI, a `gen_code.rs`:
  - el polinomi FRI (`:344-358`);
  - el `queryVerifier` (`:369-397`);
  - `friExp` (`:439-522`);
  - i `challenges_map`, que fa servir el tipus de `fri_poly`.

També estan lligades a Goldilocks:
- `pub const FIELD_EXTENSION: usize = 3` (`types/pilout_info.rs:11`), usat en uns 45 llocs. Hi ha còpies privades a `io/parser_args.rs:7` i `verifier_hashes.rs:294`, i el `qDim` del `starkinfo` està fixat a 3 (`output/stark_info.rs:278`).
- `NEG_ONE` de Goldilocks (`expr/helpers.rs:5`).
- El grau màxim es deriva del *blowup* FRI (`pil/info.rs:61`).
- **Truncacions silencioses que corromprien BN254:**
  - les constants del `pilout` es descodifiquen amb un `u128` (`types/pilout_info.rs:126-135`);
  - els números es desen com a `u64` amb `unwrap_or(0)` (`types/stark_info.rs:254, 425, 524-533, 582-593`; `io/bin_file.rs:268, 338, 365`; `output/global_constraints.rs:108, 137`);
  - `bytes_to_u64_be` es queda els 8 bytes **alts** en escriure el `.const` a partir dels valors del `pilout` (`io/fixed_cols.rs:297-305`).
- **Els *golden tests* actuals no protegeixen les passades.** Només n'hi ha tres, d'AIRs de ZisK; comparen `.bin` a partir de JSON ja fets, i quan falta `setup/golden_reference/` se salten imprimint un avís que cargo amaga (`output/global_info.rs:717-812`).

**El runtime està lligat a Goldilocks:**
- **Camp.** `F: PrimeField64`, que només implementa `Goldilocks`, a `common`, `proofman`, `witness`, `hints` i la std.
- **`ProofMan`.** Exigeix `GoldilocksQuinticExtension` (`proofman/src/proofman.rs:689-690, 834-836`).
- **`StepsParams`.** Són punters sense tipus (`common/src/air_instance.rs:19-33`), que el C++ interpreta com a `Goldilocks::Element*` (`pil2-stark/src/starkpil/steps.hpp:6-20`). Els *hints* fixen `CubicExtensionField<F>` (`hints/src/hints.rs:177-181`).
- **CLI.** L'enum `Field` només té Goldilocks (`cli/src/commands/field.rs:7-10`).
- **Traces.** S'emmagatzemen fila per fila (`common/src/trace.rs:37-47`).
- **`GlobalInfo`** (`common/src/global_info.rs`). `from_file` valida quatre coses:
  - que `hash` sigui `Poseidon1`, `Poseidon2` o `blake3` (`:162-168`);
  - que hi hagi `curve`, un enum obligatori `None|EcGFp5|EcMasFp5` (`:49-63`);
  - que `transcriptArity` sigui coherent amb el hash;
  - i crida `set_hash_family_c`, que fixa estat global de C++.

  Per tant, un `globalInfo` de pilfflonk **no** es pot llegir amb aquest tipus.

**El prover C++:**
- Una taula de funcions, `StarksBackend` (`pil2-stark/src/api/starks_backend.hpp:11-103`), tria CPU o GPU en temps d'execució. Hi ha les entrades del prover STARK i també les del SNARK final.
- `genProof` té fixats `TranscriptGL`, la NTT de Goldilocks i `ExpressionsPack` (`starkpil/gen_proof.hpp:73-86`).
- El nou backend **no entra** en aquesta taula.

**Els càlculs de l'stage 2 només existeixen per a Goldilocks:**
- `calculateImHints` (`gen_proof.hpp:27`) calcula `im_col` i `im_airval`.
- `calculateWitnessSTD` (`:57`) calcula `gsum_col` i `gprod_col`.
- El bin d'expressions (`"chps"` versió 1) desa els números com a `u64` de Goldilocks (`starkpil/expressions/expressions_bin.hpp:75`).
- El repte global combina les contribucions de les instàncies (`proofman/src/challenge_accumulation.rs`). Amb les claus actuals (`"curve": "None"`, `latticeSize` 368) fa una suma de reticle en lloc de corbes. Tot plegat està lligat a Goldilocks.

**BN254 només existeix en C++**, i tot es compila dins de `libstarks`:
- **ffiasm** (`pil2-stark/src/bn128/src/ffiasm/`). És iden3/ffiasm 0.1.5 (`0830252a`) amb pedaços locals: fitxers `.c.hpp`, constructors afegits i una bifurcació `__USE_ASSEMBLY__`. Ofereix:
  - `Fr`, `Fq`, F2, G1 i G2;
  - la FFT `fft`/`ifft`, que és d'un sol vector, sense API de *coset* i amb el `nqr` privat (`fft.hpp:10`; l'API pública és a `:22-28`);
  - la MSM `multiMulByScalar`, que espera escalars **canònics** *little-endian* (`curve.hpp:120-128`, `multiexp.c.hpp:23-34`).
- **MSM i NTT per GPU** (`pil2-stark/src/bn128/src/{msm,ntt}`, sobre sppark). Ara mateix només les fa servir `rapidsnark/plonk_prover_gpu`.
- **rapidsnark** (`pil2-stark/src/rapidsnark/`):
  - els provers `Fflonk::FflonkProver` i `Plonk::PlonkProver` (CPU i GPU), els seus setups, i un prover Groth16 (sense verificador);
  - `Keccak256Transcript`, amb un error: amb el punt zero, esborra els primers 64 bytes ja escrits del buffer i no n'afegeix cap (`keccak_256_transcript.c.hpp:63-65`);
  - `Polynomial` i `Evaluations`;
  - utilitats de binfile i zkey.
- **No hi ha cap *pairing*,** ni F6 ni F12, enlloc de `pil2-stark/src`.

**El *wrap* final SNARK:**
- `SnarkWrapper` (`proofman/src/snark_wrapper.rs`) prova una R1CS de circom. El C++ tria `FflonkProver` o `PlonkProver` segons l'id de protocol de la zkey (`starks_api.cpp:1043-1068`).
- La verificació es delega a `snarkjs` (`snark_wrapper.rs:580-638`). En Rust no hi ha cap verificador SNARK.
- No hi ha cap job de CI end-to-end per al SNARK.

**Compilació i FFI:**
- `provers/starks-lib-c/build.rs` executa `make starks_lib` o `starks_lib_gpu`.
- El Makefile llista **directoris arrel**, i dins de cadascun `find` compila tots els `*.cpp`, `*.c`, `*.cc` i `*.asm`, i també els `*.cu` a GPU (`pil2-stark/Makefile:228, 231, 234`).
- Els cossos de les plantilles C++ van en fitxers `.c.hpp`.
- Tots els directoris de `src/` són al camí d'*includes* (`:150`). Ja hi ha col·lisions que només es resolen per l'ordre de les opcions `-I`, com `alt_bn128.hpp` entre ffiasm i sppark.
- Les declaracions `extern "C"` s'escriuen a mà a `provers/starks-lib-c/bindings_starks.rs`, i `src/ffi_starks.rs` les inclou amb `include!` i hi defineix els embolcalls `*_c`.
- La GPU es detecta en compilar (`nvcc`) i es tria en temps d'execució amb `--gpu`. Hi ha la *feature* `cpu-only`, i `mpi` és una *feature* per defecte.

### 3.3 El compilador: `../pil2-compiler`

El compilador PIL2 (JS, `pil2com`) genera el `pilout`. El setup Rust l'invoca des de `compile-pil`.

**El que ja funciona, sense tocar res:**
- **El format.** El `pilout.proto` admet altres camps: `baseField` és de tipus `bytes` i les constants tenen longitud variable. El de `develop-0.14.0` és idèntic a `pilout/src/pilout.proto`. **Al `pilout` no cal cap canvi.**
- **El camp.** El nucli rep el camp com a paràmetre (`compile(Fr, …)`, `pil2-compiler/src/compiler.js:215`). El *builtin* `PRIME` surt de `Fr.p` (`src/processor.js:66, 221`), i el `baseField` s'escriu a partir de `Fr.p` (`src/proto_out.js:124`).
- **El descodificador.** `buf2bint` descodifica bé els valors de diversos trossos (`src/proto_out.js:734-740`, que avança de 8 en 8).
- **El codi per a valors fixos grans** existeix, però està desactivat o incomplet (vegeu el bloquejador 3).

**Els bloquejadors reals a `develop-0.14.0`.** S'han verificat amb experiments, i les correccions són a §4.1.
1. **No hi ha manera de triar el primer.** `src/pil.js:129` construeix el camp de Goldilocks sense condicions; no hi ha cap opció de CLI, ni `-O`, ni camp al JSON de `-P`.
   - `compile()` es pot cridar com a biblioteca des de `src/compiler.js`, però no és una entrada oficial: el `package.json` declara `main: index.js`, i aquest fitxer no existeix.
2. **`bint2buf` escriu cada tros de 64 bits a l'*offset* `index` en lloc d'`index*8`** (`src/proto_out.js:720`).
   - Per aquest codificador passen el `baseField`, les constants i els valors fixos (`src/proto_out.js:380`).
   - Resultat mesurat: `r` i `r−1` queden corromputs, i `2^64` es codifica buit, és a dir, com a `0`.
   - L'alternativa `bint2uint8` tampoc no serviria: `ProtoOut` es crea sempre sense opcions (`src/processor.js:174`) i, a més, `bint2uint8` barreja `BigInt` i `Number` i falla.
3. **Els valors fixos `≥ 2^64` no funcionen.** Afecta tot negatiu en una columna fixa (`−k ≡ r−k`) i les potències de `GEN`. Resultat mesurat a BN254, amb 1 i 2 corregits:

   | Cas | Resultat a `develop-0.14.0` |
   |---|---|
   | Seqüència geomètrica, com `ID = [1,GEN[BITS]..*..]` de `std_connection` | Error de compilació (`conversion problem`) |
   | Seqüència amb negatius: aritmètica (`[0, -1..+..]`) o de rang (`[min..max]`, a `std_range_check.pil:291`) | Error de compilació |
   | Llista amb un valor gran o negatiu (`[-1, 7]...`) | **Truncament silenciós** als 64 bits baixos |
   | Assignació fila a fila, quan l'únic valor gran és a la primera escriptura | **Truncament silenciós** |
   | Assignació fila a fila amb més valors grans | Error en llegir (`Out-of-bounds … [0..false]`) |
   | `Tables.copy` des d'una columna gran amb files sense escriure (`std_range_check.pil:334`) | Error: `Row … is not defined` |
   | `Tables.fill` amb un negatiu | **Truncament silenciós** a `2^64 − k` |

   Les causes:
   - **Seqüències.** `src/sequence.js:112` fixa `bytes = 8`. La crida a `getMaxBytes()` (`src/sequence/size_of.js:108`), que retorna `true` a partir de `2^64`, es va desactivar al commit `9fdab4a` (04-08-2025, "force bytes to 8"). A més, `expr()` només posa el sufix `n` quan `bytes === 8` (`src/sequence/fast_code_gen.js:186`).
   - **Columnes fixes** (`src/definition_items/fixed_col.js`):
     - quan una columna passa a enter gran, `size` val `false` i el control `row >= this.size` (`:238`) falla sempre;
     - la primera escriptura no crida `checkIfResize` (`#setRowValue`);
     - `createBuffer` crea arrays amb forats (`undefined`) allà on les versions tipades tenen `0`;
     - `fillRowsFrom` no redueix el valor al cos ni redimensiona la columna.

**Limitacions acceptades.**
- L'escriptor `fixed-to-file` (`src/fixed_file.js:35, 47`) treballa amb `u64` i redueix pel primer de Goldilocks.
- El pragma `#pragma extern_fixed_file` (`src/processor.js:528-531`, `src/extern_fixed_file.js`) també llegeix `u64`.
- Al camí BN254 no es fan servir; `fixed-to-file` hi dona un error explícit (C4).

**Fora del compilador, a la std de pil2-proofman.** Només té constants de camp per a Goldilocks:
- `std_constants.pil:1` fa `require "goldilocks.pil"`.
- El `switch (ACTIVE_FIELD)` que copia `GEN[]` i `k_coset` (`:131-153`) només coneix `FIELD_GOLDILOCKS`.
- `GEN` i `k_coset` només es fan servir a `std_connection.pil` (línies 101, 143, 516, 526 i 540). La resta de la std és genèrica, perquè treballa amb `PRIME`.

**Tests del compilador.** Tots fan servir Goldilocks (`test/**`, `F1Field(0xffffffff00000001n)`), i per això no cobreixen aquestes vies. A més, a `develop-0.14.0` n'hi ha dos que ja fallen per motius aliens: `test/basic.js`, per la paraula clau obsoleta `subproof`, i `test/features/sequence.test.js`, que no forma part de l'execució per defecte.

### 3.4 Què canvia de PIL1 a PIL2

Aquests són els sis canvis que condicionen el disseny. La taula completa és a l'**Annex B**.

1. **Diverses AIRs i instàncies.** PIL1 té una sola traça. PIL2 té `airGroups[].airs[]`, cadascuna amb la seva `N`, i múltiples instàncies. Els busos de la std només quadren si les instàncies comparteixen els reptes de l'stage 2.
2. **Stages i reptes arbitraris.** Els reptes fixos de PIL1 (α, β, γ, δ i `a`) passen a ser `numChallenges[stage]`. El setup hi afegeix `std_vc`, per plegar les restriccions, i `std_xi`, el repte del punt d'avaluació, que aquí és `xiSeed` (`setup/pil2-stark/src/pil/constraint_poly.rs:46, 203`).
3. **Els arguments no són nadius.** Lookup, permutation i connection es converteixen en busos de la std, de dos tipus:
   - **LogUp de suma** (`gsum`), el tipus per defecte (`std_permutation.pil:32`);
   - **de producte** (`gprod`).

   Els busos generen *hints*, de dues menes:
   - **del prover:** `gsum_col`, `gprod_col`, `im_col` i `im_airval`. El setup accepta els tres primers, que donen les columnes de l'stage 2 (M30, M31, §4.2.1 i §4.4): `im_col`, les columnes intermèdies que la std afegeix quan els termes d'un bus superen el seu `MAX_CONSTRAINT_DEGREE`, i `gsum_col` i `gprod_col`, la suma i el producte acumulats, que les llegeixen. `im_airval` calcula un air value, que la v1 no té (D2);
   - **del witness o de depuració:** `gsum/gprod_debug_data(_global)`, `range_def`, `specified_ranges(_data)`, `virtual_table_data(_global)` i `std_{sum,prod,rc}_users`. El prover els ignora, i els consumeix `pil2-components/lib/std/rs/src/`.
4. **Nous valors a la prova:** air values, airgroup values (agregats per SUM o PROD), proof values i restriccions globals.
5. **Offsets de fila arbitraris i amb signe.** PIL1 només té `'`, és a dir, `ξ` i `ξω`. PIL2 admet qualsevol `rowOffset`, i per tant qualsevol punt `ξ·ω^s`.
6. **Restriccions amb domini.** El `pilout` preveu quatre dominis (`everyRow`, `firstRow`, `lastRow` i `everyFrame`), cadascun amb el seu *zerofier*. Però a `develop-0.14.0` el compilador **només emet `everyRow`**: `processor.js:2040` crida `constraints.define(…, false, …)`, i `when` s'analitza però no s'executa. Per tant, les vores es poden expressar amb columnes fixes com `L1`, i els altres tres dominis només s'exerciten amb `pilout` sintètics (Fase 1).

---

## 4. Com funcionarà, pas a pas

### 4.1 Pas 1: compilar sobre BN254

**Què canvia al compilador** (`../pil2-compiler`). L'objectiu és tocar el mínim, **i al `pilout.proto` no cal canviar-hi res**.

**Estat.** Implementat a la branca local **`develop-0.14.0-pil2-fflonk`**, creada des de `develop-0.14.0` (`503862c`).
- Encara no té commit ni s'ha pujat, i està pendent de revisió.
- Tots els canvis són a l'índex (*staged*), i l'arbre de treball hi coincideix (comprovat el 29-09-2026).
- En total, `src/` té +29/−10 línies en 6 fitxers, i hi ha 2 fitxers de test nous.

| # | Canvi (estat actual de l'arbre de treball) | Efecte a Goldilocks |
|---|---|---|
| C1 | **Triar el primer** amb el canal de configuració que ja existeix, `-P <config.json>`. `src/pil.js` fa servir `config.prime` si hi és, i Goldilocks si no. `prime` **ha de ser una cadena**, decimal o hexadecimal; si és un número JSON, el compilador dona un error, perquè perdria precisió. No s'afegeix cap opció de CLI. | Cap |
| C2 | **`bint2buf`:** `writeBigUInt64BE(…, index * 8)` (`src/proto_out.js:720`). Queda simètric amb `buf2bint`. | Cap: els valors `< 2^64` només tenen un tros |
| C3 | **Valors fixos `≥ 2^64`.** S'activa la via d'enters grans que ja existeix i se'n corregeixen els errors:<br>- `src/sequence.js:112`: tornar a fer servir `getMaxBytes()`;<br>- `src/sequence/fast_code_gen.js`: sufix `n` sempre que `useBigInt()`;<br>- `src/definition_items/fixed_col.js`: el control de límits quan `size === false`, `checkIfResize` també a la primera escriptura (`#setRowValue`) i `fill(0n)` a la branca `bytes === true` de `createBuffer`. | Cap als programes reals. Només canvien les columnes amb `#pragma fixed_bytes 1/2/4`, que abans truncaven la primera escriptura i ara es redimensionen. Cap `.pil` de pil2-proofman no fa servir aquest pragma. |
| C4 | **Cap truncament silenciós conegut:**<br>- **Seqüències.** Al codi generat, el llaç parcial comprova l'escriptura, i els literals es comproven en generar el codi (`fast_code_gen.js`).<br>- **`Tables.fill`** (`fixed_col.js`, `fillRowsFrom`). Redueix el valor al cos, redimensiona la columna i manté `maxRow`. Si la columna és una seqüència de 64 bits i el valor no hi cap, dona un error.<br>- **`fixed-to-file`** (`fixed_file.js`). Dona un error si el cos té més de 64 bits, abans d'escriure res. | Només canvia `Tables.fill` amb un negatiu (abans el `pilout` tenia `4294967294` en lloc de `p−1`; Annex F.7), o amb un valor a `[p, 2^64)` (canvia la memòria i la sortida `fixed-to-file`, però no el `pilout`). `Tables.fill` sobre columnes `fixed_bytes 1/2/4` abans fallava i ara funciona. |

**Tests nous al compilador:** `test/bn254_fixed.js` i `test/bn254/big_fixed.pil`.
- **Què fan.** Compilen el mateix PIL sobre Goldilocks i sobre BN254 (C1) i llegeixen el `pilout` (C2).
- **Què comproven:**
  - el `baseField`;
  - vuit columnes fixes: seqüència geomètrica, llista, negatiu, seqüència aritmètica, assignació fila a fila, `Tables.fill`, redimensionament després d'un `fill`, i `Tables.copy`;
  - i que `fixed-to-file` falla a BN254.
- **Resultats.** Són 19 tests, i amb el codi de `develop-0.14.0` en fallen 11. La suite completa dona 24 tests correctes (inclosos els 19 nous) i 1 error, `test/basic.js`, el mateix que abans.
- **Casos sense test encara** (comprovats a mà, i funcionen): l'error per `prime` numèric, l'error de `Tables.fill` sobre una seqüència, les constants d'expressió, les seqüències de rang negatives i el llaç parcial.

**Validació** (compilador de la branca contra el de `develop-0.14.0`):
- **BN254:** compilen els 10 programes de la CI de pil2-proofman (`fibonacci-square` i 9 tests de `pil2-components`), amb la std actual i `baseField == r`.
- **Goldilocks:** els mateixos 10 programes generen `pilout` **idèntics byte a byte**, amb el mateix sha256.
- **`fixed-to-file`:** a `fibonacci-square` també s'ha comparat amb `fixed-to-file`, incloent-hi els tres `.fixed`. Es va mesurar amb una versió anterior de la branca, però els canvis posteriors no afecten Goldilocks.
- **Cost de la via d'enters grans:** en un PIL amb dues columnes grans de `2^20` files (una seqüència geomètrica i una assignació fila a fila), BN254 triga un 6 % més que Goldilocks (30,9 s contra 29,2 s) i fa servir un 16 % més de memòria (1,50 GB contra 1,29 GB). Amb `fibonacci-square` no hi ha diferència.

**Documentació.** El `README.md` del compilador explica l'opció `prime` del `-P` i que `fixed-to-file` falla amb camps de més de 64 bits.

**C2 i C3 no tenen alternativa.** La mateixa std genera valors `≥ 2^64` a BN254 (els rangs negatius de `std_range_check` i les potències de `GEN` de `std_connection`), i la via fila a fila, que hauria pogut servir per esquivar les seqüències, també estava afectada.

**Què canvia a la std** (`pil2-components/lib/std/pil/`, a pil2-proofman). És tot PIL, sense tocar el compilador, i aprofita el `switch (ACTIVE_FIELD)` que ja existeix:
- Afegir `FIELD_BN254` i derivar `ACTIVE_FIELD` del *builtin* `PRIME`, que la std ja consulta (`std_range_check.pil:51`, `std_virtual_table.pil:20`).
- Afegir un `bn254.pil` amb `Bn254_Gen[i] = 5^((r−1)/2^i)` per a `i ≤ 28`.
  - BN254 té 2-adicitat 28, i 5 és el no-residu quadràtic més petit.
  - `Bn254_Gen[28]` és l'arrel estàndard 19103219067921713944291392827692070036145651957329286315305642004821462161904.
  - També cal un `Bn254_k` que generi *cosets* disjunts.
- **Només cal per a les connexions** (Fase 2): és l'única part de la std que fa servir `GEN` i `k_coset`. Sense aquest canvi, les connexions compilen igualment, però amb els generadors de Goldilocks, i es perd la garantia de *soundness* de l'argument de còpia (P10, §7.1).

**Integració amb pil2-proofman** (§7.1):
- **P8.** Es fa servir el compilador local amb `PIL2C_EXEC`. `setup/pil2-stark/package.json` (que apunta a `develop-0.14.0`) només es canvia si cal, i la branca del compilador no es fa *commit* sense demanar-ho a l'usuari.
- **P9.** `proofman-setup compile-pil` rep un paràmetre nou, `-P, --config <json>`, igual que el de `pil2com` (com ja passa amb `-o`, `-I` i `-u`), i el passa tal qual. Per defecte no hi és i el comportament és l'actual (Goldilocks). És un camp `config: Option<String>` a `CompilePilOptions`, amb `None` als 11 llocs on es construeix (la CI, els `build.rs` de `pil2-components` i `proving_key/recursive.rs:1033`).

**Garanties:**
- el `pilout` BN254 té `baseField = r`;
- els `pilout` Goldilocks surten idèntics byte a byte, llevat dels casos límit de C3 i C4.

### 4.2 Pas 2: setup (Rust)

**Comanda:**

```
proofman-setup setup-pilfflonk -a <pilout> -b <build_dir> --powers-of-tau <ptau>
    [--max-constraint-degree D]   (per defecte 9, D5)
    [--extra-muls E]              (per defecte 2, com pil-stark)
    [--max-q-degree M]            (per defecte 0: Q no es parteix)
    [--solidity]                  (Fase 4: també pilfflonk.verifier.sol, §4.5)
    [--no-packing]                (només per a tests: força k = 1)

proofman-setup pilfflonk-solidity -k <pilfflonk.vkey.json> -o <pilfflonk.verifier.sol>
```

Els noms d'arguments són els de `setup` (`-a`, `-b`) i `setup-snark` (`--powers-of-tau`). **No** hi ha `-u <fixed_dir>`: al STARK, aquesta opció copia `.fixed` de 8 bytes, i a BN254 no hi ha cap productor de fitxers així (C4).

**`--solidity` i `pilfflonk-solidity` (M40).** Amb `--solidity`, el setup també escriu el verificador Solidity de la vkey, `pilfflonk.verifier.sol`, al costat de `pilfflonk.vkey.json` (§4.2.6, §4.5). `pilfflonk-solidity` l'escriu només a partir d'una vkey que ja existeix: és el patró del repositori per a una sortida derivada d'una clau feta (l'opció `setup --gen-exps` i la subcomanda `gen-exps -p <provingKey>`), i el de snarkjs (`snarkjs zkey export solidityverifier`), on el contracte surt de la clau de verificació. Tots dos escriuen els mateixos bytes. Sense `--solidity` no canvia res: els altres fitxers i el *digest* són idèntics byte a byte, i el `globalInfo` no ho registra, perquè el contracte no forma part de la clau.

El setup és una seqüència de passos. Tots són funcions pures amb tests propis, llevat de dos: la lectura i escriptura de fitxers (§4.2.1, §4.2.6) i els compromisos fixos (§4.2.5), que criden C++.

#### 4.2.1 Llegir i validar

Es descodifica el `pilout` i el setup s'atura amb un error clar en qualsevol d'aquests casos:
- el `baseField` no és el `r` de BN254;
- hi ha custom commits, periodic columns o public tables;
- hi ha més d'una AIR, air values, airgroup values, proof values o restriccions globals (D2: fora d'abast);
- hi ha un *hint* de prover que no és `im_col`, `gsum_col` ni `gprod_col` (M30, M31): `im_airval` calcula un air value (D2), i un nom desconegut es rebutja. Els *hints* de witness i de depuració de la llista de §3.4 s'ignoren de manera explícita, i no van al `<air>.bin`. Els *hints* es comproven abans que els valors, perquè un `im_airval` es digui pel seu nom i no per l'air value que porta;
- hi ha un *hint* `witness_bits`, el que escriu el compilador per a cada columna declarada amb `bits(n)` (`col witness bits(8) x`, `processor.js`, `execWitnessColDeclaration`): demana files de traça empaquetades, i pilfflonk encara no n'accepta (decisió de l'usuari, 30-09-2026; M38b). Abans ja es rebutjava, com qualsevol nom desconegut; ara l'error diu per què (`SetupError::PackedTrace`). Ni la std ni cap *fixture* de pilfflonk no en tenen (comprovat compilant-les totes a BN254), de manera que no hi ha cap clau pilfflonk d'una AIR empaquetada;
- alguna columna de l'stage 2 o superior no la produeix exactament un `im_col`, `gsum_col` o `gprod_col`. Com que el que diu un *hint* és el que en processa `pil-info`, aquesta comprovació es fa després de les passades (`validate::check_prover_hints`), sobre els *hints* que van al `<air>.bin`:
  - la `reference` ha de ser una columna d'un stage ≥ 2, llegida a la seva fila, que no sigui un im pol;
  - el numerador i el denominador (`numerator` i `denominator` d'un `im_col`, `numerator_air` i `denominator_air` d'un `gsum_col` o `gprod_col`) han de ser una expressió, una columna en un punt d'obertura o un número (els operands que pren l'`addHintField` del STARK, sense els air values, D2);
  - només poden llegir columnes que el prover calcula abans del *hint*: les fixes, les dels stages anteriors al de la `reference` i, del seu, les dels *hints* anteriors en l'ordre del STARK (§4.4, pas 2): els `im_col` en l'ordre del `pilout`, després els `gprod_col` i després els `gsum_col`. Així, un `gsum_col` pot llegir els `im_col` del seu stage (la std ho fa sempre que n'hi ha), i un `im_col` els `im_col` d'abans (el bus de producte de la std els encadena: `std_prod.pil`, `piop_gprod_air`, `prev_im`), però cap no llegeix un `im_col` posterior, la seva pròpia columna ni un im pol de l'stage, que es calcula al final. El STARK no ho comprova: `multiplyHintFields` llegiria el que tingués el buffer;
  - un `im_col` ha de ser d'una AIR amb algun `gsum_col` o `gprod_col`: el `calculateImHints` del STARK no en calcula cap en una AIR sense;
  - `result`, si un `gsum_col` o `gprod_col` el té, ha de ser un número, com l'escriu la std en `STD_MODE_ONE_INSTANCE`; si no, és un airgroup value (D2). Abans, el `pilout` ja s'ha rebutjat pels seus airgroup values, i l'error ho diu. Un `im_col` no té `result` ni camps directes;
- dues columnes tindrien el mateix nom a la prova i al *layout*, un cop el setup ha indexat els im pols i les columnes que comparteixen nom i no tenen `lengths` (A.6, M34b); l'error diu quines són;
- alguna constant és `≥ r`;
- el domini estès no cap en la 2-adicitat, és a dir, `nBitsExt > 28` (A.1);
- el `ptau` té menys punts que el `degree` més gran del *layout* (el nombre de coeficients de l'`f_i` més gran, M12);
- el `[τ]₂` del `ptau` no és un punt de G2: el punt a l'infinit (`τ = 0`), fora del *twist* o fora de la torsió `r` (M26; l'infinit, amb el seu nom des de la revisió de M40). Seria l'`X_2` de la vkey, que el verificador JS refusa (`g2FromObject`), i amb l'infinit el *pairing* del contracte Solidity acceptaria una prova falsa (§4.5).

Del `ptau` només es llegeixen les seccions 2 (`[τ^i]₁`) i 3 (`[τ^i]₂`). No cal que estigui preparat per a la fase 2 (secció 12), a diferència del que demana `fflonk_setup.cpp:51`.

#### 4.2.2 Informació simbòlica (compartida amb el setup STARK)

**Què es reutilitza.** Les passades de `setup/pil2-stark/src/{pil,expr}` i `types/pilout_info.rs`: preparació d'expressions, graus, *offsets*, mapes de columnes, `evMap`/obertures, codegen i informació de les restriccions globals.

**Nou crate compartit, `pil-info`** (`setup/pil-info`, decisió D1). Hi passen:
- les passades simbòliques;
- el contenidor binari `"chps"` (`io/bin_file_writer.rs`);
- l'assignació de temporals (`io/parser_args.rs:85-205`, que ara és privada i té `dim == 3` fixat);
- la generació de `pilout.globalConstraints.json`.

Així el setup pilfflonk no depèn de `pil2-stark-setup` i no hi ha cap cicle (§5.2).

**Paràmetres.** Les passades reben un context, `PilInfoCfg`, amb tres paràmetres:
- **el mòdul del camp:** per reduir constants i representar `neg`, sense el `NEG_ONE` cablejat;
- **la dimensió d'extensió:** 3 per a Goldilocks i 1 per a BN254. Substitueix els ~45 usos de `FIELD_EXTENSION`;
- **una política de grau:** el STARK la deriva del *blowup*, i pilfflonk fa servir la seva (§4.2.3).

**Canvis que el STARK també aprofita:**
- descodificar les constants amb un enter gran (`num-bigint`, que ja és al *workspace*), no amb `u128`;
- no truncar els números a `u64` fins al moment d'escriure'ls;
- que les passades retornin `Result` en lloc de fer `panic!`.

**Parts del STARK que no es fan servir:**
- el polinomi FRI i el `queryVerifier`, que a `gen_code` passen a ser un ganxo d'obertura;
- les validacions de `StarkStruct` de `prepare`;
- la seguretat FRI, els arbres de constants i la recursió.

#### 4.2.3 Polinomi de restriccions i polinomis intermedis

- **Plegat.** Les restriccions d'una AIR es pleguen amb `std_vc` pel mètode de Horner, en l'ordre del `pilout`, i cadascuna es divideix pel *zerofier* del seu domini:

  ```
  Q(X) = Σ_{i=0..n−1} std_vc^(n−1−i) · c_i(X) / Z_{D_i}(X)
  ```

  És la mateixa semàntica que el STARK (`constraint_poly.rs:136-147`). La primera restricció rep la potència més alta. Els detalls, incloent-hi el comptatge de grau dels *zerofiers*, són a l'Annex A.1.
- **Selecció d'im pols.** El setup tria els polinomis intermedis que minimitzen `nImPols + qDeg`, amb un grau màxim configurable (per defecte 9, com pil-stark; D5).
  - Els im pols van a **l'últim stage de l'AIR**, com al STARK (`im_polynomials.rs:539`). A la Fase 1, amb `nStages = 1`, doncs, viuen a l'stage 1.
  - Els calcula el prover amb el bytecode, abans de comprometre aquell stage.
  - No s'han de confondre amb les columnes `im_col` que declara la std, que es calculen a partir de *hints*.
- **Partició de `Q`.** Si `qDeg` supera `--max-q-degree` (i aquest no és 0), `Q` es parteix en trossos. En aquest cas, els trossos reben blinding i les seves avaluacions van a la prova (A.1, A.3).

#### 4.2.4 Agrupació fflonk

És una funció pura: `group(polinomis, paràmetres) → Layout`.
- **Entrada:** cada polinomi compromès, amb el seu stage, la fita de grau i el conjunt d'*offsets* on s'obre.
- **Sortida:** la llista de polinomis empaquetats `f_i`. Cada `f_i` és d'un sol stage i d'un sol conjunt d'obertura, i conté els seus polinomis en un ordre fix, amb un factor `k` vàlid.

**Regles.** Són les del sistema antic, generalitzades a *offsets* amb signe:
1. **Classes i fusió** (com `pil-stark/src/fflonk/helpers/fflonk_shkey.js`). Els polinomis es classifiquen per `(stage, O)`. Una classe petita puja a la unió dels *offsets* del seu stage, i les classes que ja són aquesta unió no es mouen mai.
2. **Q.** Té el seu propi `f`.
3. **Repartició d'`extraMuls`** (com `shplonkjs/src/helpers/setup.js`). Es fa en dos nivells: primer dins de cada grup, i després entre grups.
4. **`k` vàlid:** cada `k` ha de complir `k | r−1` i `kN | r−1`.

**Compatibilitat.** Per a *offsets* dins de `{0, 1}`, el resultat ha de coincidir exactament amb el del sistema antic. El detall normatiu és a l'Annex A.2.

#### 4.2.5 Bytecode, compromisos fixos i claus

- **Bytecode del prover** (`<air>.bin`). El codegen compartit genera per a cada AIR el bytecode `Fr` que executarà el prover: expressions dels *hints*, im pols i `Q`.
  - Les constants ocupen 32 bytes.
  - El format és el del `.bin` STARK amb dimensió 1 (revisió 3, A.6): els mateixos camps, seccions i tipus de buffer d'`io/parser_args.rs`, sense els camps de dimensió, amb args de 32 bits i constants de 32 bytes, i els *hints* del prover a la secció 3 (M30).
  - Es reaprofiten el contenidor `"chps"` i l'assignació de temporals, que viuen a `pil-info`.
  - El fitxer té una versió i una mida d'element pròpies.
- **Codi del verificador** (el `qVerifier` de `<air>.verifierinfo.json`). És el codi que calcula `Q(ξ)` a partir de les avaluacions, sense `queryVerifier`. Surt del mateix codegen, en el format JSON del STARK, i el setup en copia el `qVerifier` a la vkey. No hi ha `.verifier.bin`, perquè no hi ha verificador natiu (D3).
- **Restriccions globals.**
  - `pilout.globalConstraints.json` surt de `pil-info`, en el mateix format que el STARK.
  - No cal cap `.bin`: el verificador JS llegeix el JSON. A la v1 el fitxer no té cap restricció (D2).
- **SRS** (`pilfflonk.srs.bin`). El setup n'extreu del `ptau` les potències G1 que calen, tantes com el `degree` més gran del *layout* (M12), i `[τ]₂`.
- **Compromisos fixos** (`<air>.verkey.json`). El setup crida una funció C nova, `pilfflonk_commit_fixed`. C++ fa la INTT de les columnes fixes amb ffiasm, n'empaqueta els `f_i` i fa la MSM.
  - S'escriu sobre `Polynomial::fromEvaluations`, `CPolynomial` (empaquetat) i `multiMulByScalar`, que ja existeixen a rapidsnark i ffiasm. `PilFflonkSetup::computeFCommitments` (`pil-fflonk/src/pilfflonk_setup.cpp:270`) només serveix de referència del resultat esperat, i no es copia.
  - Del patró de `fflonk_setup_c` es copien dues coses: C++ captura les excepcions i retorna un codi d'estat, i Rust valida les mides abans de la crida.
  - Però, a diferència d'aquell patró, el C++ fa servir RAII i l'embolcall Rust retorna `Result`, no fa `panic!`.
- **Vkey** (`pilfflonk.vkey.json`). És autocontinguda, com la `verification_key.json` de snarkjs: conté tot el que necessita el verificador, i res més (A.6). El setup l'escriu al final, amb el *digest*. El verificador JS i el Solidity (Fase 4) només llegeixen aquest fitxer.

#### 4.2.6 Sortida: un `provingKey/` com el del setup STARK

El setup escriu un directori `provingKey/` amb la mateixa jerarquia que genera `proofman-setup setup` (`commands/setup.rs:150-313`, `common/src/global_info.rs:202-245`). Els noms de fitxer són els mateixos sempre que el contingut té el mateix paper; els que depenen del sistema de prova canvien.

```
<build>/provingKey/
├── pilout.globalInfo.json             la part comuna de l'esquema STARK, més "backend": "pilfflonk" (A.6)
├── pilout.globalConstraints.json      el mateix format que el STARK (el genera pil-info)
└── <name>/                            <name> = nom del pilout
    ├── pilfflonk/                     fitxers globals del backend (convenció de get_setup_path)
    │   ├── pilfflonk.srs.bin          potències G1 del ptau que calen, i [τ]₂
    │   ├── pilfflonk.vkey.json        la vkey autocontinguda del verificador, amb el digest (A.6)
    │   └── pilfflonk.verifier.sol     el contracte verificador, amb --solidity (Fase 4, §4.5)
    └── <airgroup>/airs/<air>/air/
        ├── <air>.const                columnes fixes, Fr canònic de 32 bytes little-endian (al STARK, 8 bytes)
        ├── <air>.pilfflonkinfo.json   fa el paper del starkinfo.json: mapes i layout dels f_i
        ├── <air>.expressionsinfo.json el format del STARK, amb dimensió 1
        ├── <air>.verifierinfo.json    el format del STARK, només qVerifier (el llegeix el verificador JS)
        ├── <air>.bin                  bytecode del prover i hints (contenidor "chps", versió pilfflonk)
        └── <air>.verkey.json          commitments dels f_i fixos (al STARK, l'arrel Merkle de .const); també són a la vkey
```

- **Dades derivades.** Els coeficients i les avaluacions esteses de les columnes fixes, les arrels i `powerW` es recalculen en carregar, igual que el STARK recalcula `.consttree` a partir de `.const`.
- **Fitxers que no hi són.** No hi ha `<air>.verkey.bin`: al STARK és la còpia binària de l'arrel Merkle, i aquí no té equivalent.
- **Lectura del `globalInfo`.**
  - El runtime pilfflonk el llegeix amb un tipus propi (a `proofman-pilfflonk`), no amb `common::GlobalInfo`, perquè aquest valida `hash`, `curve` i `transcriptArity` i fixa estat global de C++ (§3.2).
  - Com que `curve` és obligatori per al STARK, les eines STARK rebutgen aquest fitxer per si soles, i no hi ha perill que el carreguin per error.

**Correspondència amb els fitxers de setup originals** (pil-stark / pil-fflonk):

| Fitxer original | Què contenia | On va a parar |
|---|---|---|
| `NAME.fflonkinfo.json` | Mapes de seccions, `evMap`, contextos d'arguments, codi pas a pas, `qDeg`, publics | **Global:** `pilout.globalInfo.json`.<br>**Per AIR:** `<air>.pilfflonkinfo.json`.<br>**Codi:** `<air>.expressionsinfo.json`, `<air>.verifierinfo.json` i `<air>.bin`. |
| `NAME.shkey.json` | Agrupació en `f_i`, `powerW`, arrels `w*` | L'agrupació passa al camp `layout` de `<air>.pilfflonkinfo.json`; `powerW` i les arrels es deriven |
| `NAME.zkey` (12 seccions) | Capçalera, definicions i commitments de `f`, noms per stage, constants (avaluacions, coeficients i extensió), `x_n`, `x_ext`, omegas, PTau | Es reparteix així:<br>- capçalera i definicions → `<air>.pilfflonkinfo.json`;<br>- commitments fixos → `<air>.verkey.json`;<br>- avaluacions de les constants → `<air>.const`;<br>- coeficients, extensió, `x_n`, `x_ext` i omegas → es deriven;<br>- PTau → `pilfflonk.srs.bin`. |
| `NAME.vkey` | `[τ]₂`, commitments de constants, arrels, `polsMap` | `pilfflonk.vkey.json`, amb el mateix paper; els commitments fixos també van a `<air>.verkey.json` |
| `NAME.const` | Constants en `Fr` | `<air>.const`, un per AIR |
| `NAME.chelpers.*.cpp` | C++ generat per a cada circuit | `<air>.bin`, bytecode. Ja no cal recompilar el prover. |
| `NAME.commit` / `NAME.exec` | Witness | No és setup: és una entrada del prover (§4.3) |

### 4.3 Pas 3: witness

**Què rep el prover de fora:**
- les columnes de l'**stage 1** de cada instància;
- els **air values** de l'stage 1;
- els **publics**;
- els **proof values** de l'stage 1.

Les columnes de l'**stage 2 i posteriors** i els im pols no els aporta ningú de fora: els calcula el prover (§4.4).

**Format.** Els valors són `Fr` canònics de 32 bytes *little-endian*, fila per fila, com a les traces actuals. Les conversions a la forma de Montgomery de ffiasm es fan dins del C++.

**Ordre de les instàncies.** Les instàncies es donen en ordre canònic (glossari), que és el que fixen el transcript i la prova.

**D'on surt el witness:**
- **Fases 1 i 2:** d'un fitxer per instància, llegit per un `WitnessSource`, que generen petits generadors de fixtures.
- **Fase 3:** de programes reals. Els `WitnessLibrary<F: PrimeField64>` actuals no poden calcular en BN254: convertir valors de Goldilocks a `Fr` no és correcte per a negatius ni inverses. Cal una font de witness en `Fr` (D4).

**El tipus `Bn254` (M38a).** És `proofman_fields::Bn254` (`fields/src/bn254.rs`): el camp escalar `Fr` de BN254, d'ordre `r`, i no el camp base `Fq`. Es diu com la corba, igual que `Goldilocks` es diu com el seu primer. És en Rust pur i sense cap biblioteca de corbes. Només serveix per calcular el witness: el prover continua fent l'aritmètica de BN254 amb ffiasm.
- **Representació.** Forma de Montgomery, `a·2^256 mod r`, en quatre *limbs* de 64 bits *little-endian*, sempre reduïda (`< r`). És la mateixa de `RawFr::Element` d'ffiasm. Com que els *limbs* són únics, la igualtat i el *hash* hi treballen directament; l'ordre (`Ord`) és el dels valors canònics.
- **Aritmètica.** Multiplicació de Montgomery CIOS; inversa per Fermat (`a^(r−2)`, amb finestres de 4 bits), i `exp_u256` per a exponents de 256 bits. Dividir per 0 fa `panic!`, com `Field::inverse` a Goldilocks; `try_inverse` torna `None`. En *release*, a la màquina de desenvolupament (AMD EPYC 7773X), una multiplicació triga uns 22 ns i una inversa uns 7,5 µs.
- **Traits.** `Field` i `PrimeField`, però no `PrimeField64`, que és de 64 bits. `QuotientMap` per a tots els enters fins a 128 bits (i `usize`/`isize`): un negatiu `x` és `r − |x|`, i tots són canònics, de manera que `from_canonical_checked` no torna mai `None`. Com que els `from_u64` de `PrimeField64` no hi són, un enter es converteix amb `Bn254::from_int`.
- **Constants.** `GENERATOR = 5`, el no-residu quadràtic més petit: el generador d'ffjavascript i d'ffiasm i el desplaçament del *coset* de §4.4. `TWO_ADICITY = 28`, i `W[i] = 5^((r−1)/2^i)` per a `i ≤ 28`, que són els `Bn254_Gen[i]` de la std (`bn254.pil`, M29) i les arrels de la FFT d'ffiasm.
- **Bytes.** `to_le_bytes` i `from_le_bytes` fan servir els 32 bytes *little-endian* canònics del witness i del `.const` (A.6); `from_le_bytes` torna `None` per a un valor `≥ r`.
- **Serde: cadena decimal canònica.** És la codificació JSON d'A.6, la mateixa que `FrBytes`. Només es llegeix aquesta grafia: sense signe, espais ni zeros a l'esquerra, i `< r`. Un número JSON es rebutja. **Diferència amb Goldilocks:** el trait només demana `Serialize + DeserializeOwned`, i Goldilocks es serialitza com a número JSON, però molts lectors no poden llegir un número JSON de 254 bits. `Display` i `Debug` també escriuen el decimal canònic.
- **Conversions.** `From<Bn254> for FrBytes` i `From<FrBytes> for Bn254`, a `pilfflonk/src/field.rs`. Cap de les dues pot fallar, perquè els dos tipus són sempre `< r`: només canvien la representació. `proofman-pilfflonk` depèn de `proofman-fields`, i no al revés.
- **Tests.** Cada operació es compara amb `num-bigint` en valors aleatoris i de vora. Un vector calculat amb el `RawFr` d'ffiasm i també amb Python, que coincideixen, queda fixat al test, amb la forma de Montgomery inclosa. `pilfflonk/tests/std_bn254.rs` compara `W` i `GENERATOR` amb `bn254.pil`.

**La biblioteca de witness (M38b).** Segueix el patró del STARK (decisions de l'usuari del 30-09-2026, §7.1): és una biblioteca dinàmica que calcula el witness en `Bn254` amb les files que genera `pil-helpers`, i que la CLI carrega (M38c). És un camí germà del runtime STARK (§2.1, principi 1): `proofman-pilfflonk` no depèn de `proofman-common` ni de `proofman-witness`, no fa servir cap `ProofCtx` ni `WitnessManager`, i el runtime STARK continua sent `F: PrimeField64`.
- **Files tipades sobre qualsevol camp.** `trace_row!` (`macros/src/{unpacked_row,trait_row}.rs`) ja no demana `PrimeField64` a la fila sense empaquetar ni al seu trait `<Fila>Ops`: només `Copy + Default + Send`, i `Sync + Debug` per implementar el trait. `PrimeField64` queda als accessors de les columnes tipades (`bit`, `ubit(N)`, `u8`…), que converteixen a 64 bits (`from_u8`, `as_canonical_u64`): tenen un `impl` propi i, al trait, `where F: PrimeField64`. La fila empaquetada continua sent de 64 bits. `GenericTrace` ja no tenia cap límit de camp. Una fila `#[repr(C)]` de `Bn254` fa `32·C` bytes, sense farciment, i el buffer d'una traça (`get_buffer`) és el witness fila per fila.
- **`pil-helpers` per a un `pilout` BN254.** El reconeix pel `baseField`, que ha de ser `r`, com el setup (§4.2.1). Llavors genera només el que BN254 pot tenir:
  - files sobre `F` sense empaquetar: una AIR amb *hints* `witness_bits` es rebutja, perquè pilfflonk encara no accepta traces empaquetades;
  - cap `FieldExtension`: tots els valors tenen dimensió 1 (A.6), i els de l'stage 2 o posterior són `F`;
  - els public inputs (`<Program>Publics`) en `Bn254`, que es llegeixen com a cadenes decimals (A.6) i per defecte són 0;
  - cap `PACKED_INFO` ni `PackedInfoConst`.

  La plantilla és la mateixa, amb condicions que a Goldilocks no escriuen res. La sortida de Goldilocks és idèntica byte a byte: comprovat contra la de `HEAD`, amb el mateix `pilout`, als vuit programes de `pil2-components/test` que la regeneren al `build.rs` (i el `git diff` en queda buit) i a `fibonacci-square`. De passada es corregeix un `panic!` amb un `pilout` sense reptes (només l'stage 1, com el Fibonacci): `num_challenges.len() - 1` desbordava. Ara es fa servir `num_stages()`, que dona el mateix per a qualsevol `pilout` amb reptes.
- **L'API** (`pilfflonk/src/witness_library.rs`):
  - `PilfflonkWitnessLibrary::witness(&mut self, shape, public_inputs) -> Witness`: el witness de la forma de la clau (`ProvingKey::witness_shape`) a partir dels public inputs, el camí del JSON, si n'hi ha. Com al STARK, la biblioteca els llegeix ella mateixa: `read_public_inputs` fa el que fa `proofman_common::load_from_json` (sense fitxer, el valor per defecte), però amb errors en lloc de `panic!`;
  - `pilfflonk_witness_library!(Nom)`: exporta el símbol `pilfflonk_init_library` (`PilfflonkWitnessLibInitFn`, ABI de Rust com al STARK), i no l'`init_library` del STARK, de manera que cap dels dos carregadors no pren les biblioteques de l'altre. Com el `witness_library!` del STARK, inicialitza el *logger* de la biblioteca amb `proofman_common::initialize_logger`: la biblioteca depèn de `proofman-common`, com qualsevol que tingui `pil_helpers`;
  - `load_witness_library(camí, verbose)`, amb `libloading`: refusa un fitxer que no és una biblioteca i una biblioteca sense `pilfflonk_init_library`, i diu quan és una biblioteca STARK. La biblioteca no es descarrega mai, com a `load_packed_info`: el que torna (el witness, i errors que poden tenir-ne les *vtables*) pot sobreviure a qualsevol *handle*;
  - `compute_witness`: crida la biblioteca i comprova el witness contra la forma, com `FileWitnessSource::open` comprova un directori;
  - `Stage1Witness::from_rows(n_rows, n_cols, &[Bn254], air_values)`: el buffer d'una traça tipada, amb la comprovació de mida dels altres constructors. Els `Bn254` sempre són `< r`, i no es tornen a comprovar.
- **Diferències amb el STARK, obligades:**
  - el punt d'entrada rep la verbositat com a `u8` (el nombre de `-v`) i no com a `VerboseMode`, perquè `proofman-pilfflonk` no depèn de `proofman-common`; no rep cap `RankInfo`, perquè no hi ha MPI;
  - el *logger* s'inicialitza una sola vegada (`Once`), amb la verbositat de la primera crida: `initialize_logger` fa `panic!` si dos fils l'inicialitzen alhora, i un `panic!` dins d'una biblioteca dinàmica, que té el seu propi *runtime* de Rust, no es pot capturar i avorta el procés (va passar als tests, que carreguen la biblioteca des de diversos fils);
  - la biblioteca torna un `Witness`, en lloc de registrar components en un `WitnessManager`.
- **La biblioteca de prova** és `pilfflonk/tests/fixtures/fibonacci/rs` (crate `pilfflonk-fibonacci`, `dylib` com les del STARK), amb els `pil_helpers` del Fibonacci en BN254. Com que el `pilout` BN254 necessita `PIL2C_EXEC`, els `pil_helpers` es versionen (com els de `fibonacci-square`) en lloc de regenerar-se al `build.rs` (com els de `pil2-components/test`), i un test comprova que són els que escriu `pil-helpers`. El nom del `pilout` és el del fitxer (`fibonacci.pilout`), i d'aquí surt el de `FibonacciPublics`. Carregada amb `load_witness_library`, el seu witness és el del generador de M13 byte a byte, també quan els valors donen la volta a `r`. La prova que en surt és la del generador amb el mateix blinding, i el verificador JS l'accepta i rebutja els mateixos publics amb un altre `out`. `load_witness_library` refusa la biblioteca STARK de `fibonacci-square`, i la de pilfflonk no exporta `init_library`.

**La CLI (M38c).** `proofman-cli pilfflonk prove` i `pilfflonk check` prenen el witness d'un directori (`--witness <dir>`) o d'una biblioteca (`-w`/`--witness-lib <so>`), exactament d'un dels dos, i els public inputs amb `-i`/`--public-inputs <json>`, que només valen amb `--witness-lib`. Els noms i les formes curtes són els de `prove` del STARK (`cli/src/commands/prove.rs`), i no xoquen amb cap opció de pilfflonk (`-k`, `-o`, `-v`, `--max-rows`, `--insecure-blinding-seed`). `check` també fa servir `-i`, com `prove`, i no el `-p` de `verify-constraints`, que hi dona `-i` a un altre fitxer.
- **El flux** és el de M38b: `ProvingKey::load`, `load_witness_library(camí, verbose)` amb el nombre de `-v`, `compute_witness(biblioteca, &pk.witness_shape()?, public_inputs)` i, després, `prove` o `check`, com amb un directori. Les dues comandes comparteixen els arguments (`PilfflonkWitnessArgs`, `cli/src/commands/pilfflonk/mod.rs`) i un `WitnessSource` que delega en el `FileWitnessSource` del directori o en el `Witness` de la biblioteca. Un directori es continua llegint instància a instància, i la prova no canvia: `--witness-lib` només canvia d'on surt el witness.
- **Les regles** són de clap, com a `prove-air`: `--witness` i `--witness-lib` formen un grup obligatori i exclusiu, i `--public-inputs` demana `--witness-lib` i és incompatible amb `--witness` (sense aquesta incompatibilitat, clap acceptava `--witness <dir> --public-inputs <json>`). clap refusa les altres combinacions amb el codi 2, abans de llegir res. L'ajuda diu que els valors dels public inputs són cadenes decimals (`"5"`, no `5`), perquè `Bn254` rebutja un número JSON (A.6).
- **Errors.** Són els de M38b, amb el camí i el motiu: una biblioteca que no hi és, un fitxer que no és una biblioteca, una biblioteca STARK i uns public inputs que la biblioteca no pot llegir (un número JSON, un valor `≥ r` o un fitxer que no hi és). La comanda surt amb 1 i no escriu cap prova.
- **Les biblioteques de les fixtures** són `pilfflonk/tests/fixtures/{connection,all}/rs` (crates `pilfflonk-connection` i `pilfflonk-all`), com la del Fibonacci: `dylib`, membres del *workspace* que no ho són per defecte, i amb els `pil_helpers` versionats i un test que comprova que són els que escriu `pil-helpers`. Cada crate es diu com el *stem* del seu `pilout` (`connection.pilout` i `all.pilout`), d'on surt el nom d'`AllPublics`. La Connection no té publics, i per tant no hi ha `ConnectionPublics`.
  - **Una biblioteca per als dos busos.** Els `pilout` de suma i de producte de cada fixture tenen les mateixes columnes de l'stage 1 i els mateixos publics, i `pil-helpers` hi escriu el mateix llevat de `PILOUT_HASH` (el test ho comprova). Els `pil_helpers` surten del `pilout` de suma.
  - **Les files són les columnes de l'stage 1 de la clau.** Un test fa el setup dels dos busos i compara cada camp de la fila, per la seva posició (`offset_of!`), amb l'entrada de `cmPolsMap` del mateix nom i el seu `stageId`, i `ROW_SIZE` amb el nombre de columnes, sense farciment. Cap de les dues fixtures no té columnes en *array* ni columnes de l'stage 1 declarades per la std, i el test falla si n'apareix alguna en *array*. Com que un test no pot enllaçar una `dylib` de Rust (en duria un segon `std`), inclou els `pil_helpers` de la biblioteca amb `#[path]`.
  - **El witness** és el del generador (`pilfflonk/tests/data/{connection,all}.rs`) byte a byte, en el cas d'`all` per a diversos inputs (`all::witness_of_inputs`, nou), i amb `[1, 2]` els publics són els de pil-fflonk (`runtime/public.json`). A les claus dels dos busos, `check` l'accepta i rebutja el witness trencat del generador.
  - **Public inputs.** Els d'`all` són els del Fibonacci, `in1` i `in2`. La Connection no en llegeix cap: com les biblioteques STARK dels programes sense publics (`pil2-components/test/connection/rs`), no llegeix el fitxer que se li doni.
  - **La mida.** Abans de calcular res, cada biblioteca comprova que l'AIR de la clau té les files i les columnes de la seva traça, amb `WitnessShape::check_trace` (nou). La del Fibonacci també el fa servir, en lloc de la seva comprovació pròpia, i el missatge no canvia.
- **Tests E2E** (`cli/tests/pilfflonk_prove.rs` i `pilfflonk_check.rs`). Amb `--witness-lib`, el Fibonacci, la Connection i `all`, en bus de suma i de producte, donen la mateixa prova (byte a byte, amb la mateixa llavor) que el directori del generador amb `--witness`, i `pilfflonk verify` l'accepta. `check --witness-lib` hi passa, restricció per restricció igual que amb el directori. Les dues comandes refusen les combinacions d'opcions i les biblioteques de més amunt. Cargo compila les biblioteques al costat dels tests, perquè són dependències de desenvolupament de `proofman-cli` (també la STARK, `fibonacci-square`), i els tests les troben amb `built_library`. Aquesta funció era al test del Fibonacci i ara és a `pilfflonk/tests/data/witness_libraries.rs`, que comparteixen tots els tests de biblioteques.

### 4.4 Pas 4: prova

**Repartiment de feina:**
- **Rust (crate `proofman-pilfflonk`).** Carrega el `provingKey/`, decideix l'ordre de les operacions i quins valors s'absorbeixen, i escriu la prova.
- **C++ (`pil2-stark/src/pilfflonk/`).** Conté els buffers de polinomis i l'objecte transcript, i fa tots els càlculs amb ffiasm:
  - **Escalars.** La MSM de ffiasm vol escalars canònics, i per això la conversió des de Montgomery es fa just abans de cada MSM, com a pil-fflonk.
  - **NTT.** A la versió de CPU, la FFT d'ffiasm columna a columna, a través de `Polynomial::fromEvaluations` i `Evaluations`. Una NTT multicolumna optimitzada, o la de GPU, vindrà després.
  - **Coset.** pil-fflonk no en fa servir: divideix per `Z_H` en forma de coeficients. Els *zerofiers* per domini, en canvi, demanen dividir punt a punt, i per això `Q` s'avalua sobre un *coset* estès. La FFT d'ffiasm no té API de *coset*, de manera que l'LDE multiplica pels poders del desplaçament abans de la FFT. El desplaçament és `g = 5`, el no-residu quadràtic més petit, que és el `nqr` de la FFT d'ffiasm i el generador de les arrels: no és a cap subgrup d'ordre `2^k`, i per tant `g·H'` no talla `H`. És intern del prover: el verificador no el veu.
  - **Transcript.** Es fa servir, sense modificar-lo i com a dependència, el `Keccak256Transcript` de `pil2-stark/src/rapidsnark/`, exactament el mateix que el FFLONK existent (P6, `fflonk_prover.c.hpp:831-851`):
    - `addScalar`: `Fr` en 32 bytes *big-endian* canònics;
    - `addPolCommitment`: G1 afí `x‖y`, en *big-endian*;
    - `getChallenge`: `keccak256` reduït a `Fr`;
    - `reset()` i llavor amb el repte anterior a cada ronda.

    Té tres peculiaritats que aquí no fan mal:
    - **El punt zero.** El codifica malament (§3.2), però totes les G1 que s'absorbeixen són commitments amb blinding, o `W` i `W'`, i per tant no es dona a la pràctica. L'API el rebutja amb un error (M4).
    - **Coordenades petites.** `RawFq::toRprBE` d'ffiasm (`fq.cpp:324-339`) exporta paraules de 8 bytes sense alinear-les a la dreta: una coordenada `< 2^192` s'escriu com un nombre més gran, i no com 32 bytes *big-endian*. `RawFr::toRprBE` sí que ho fa bé (`fr.cpp:312`), i per això els escalars no hi estan afectats. Un punt aleatori té una coordenada així amb probabilitat `≈ 2^-61`. L'API rebutja aquests punts amb `PILFFLONK_ERR_INVALID_POINT` (M4), de manera que tot el que s'absorbeix es codifica exactament com diu A.4, i com ho fa el verificador JS.
    - **El buffer és un VLA a la pila** (`getChallenge`). Només seria un problema amb transcripts de molts megabytes. Si passa, es corregeix a la mateixa classe, i el *wrap* final també se'n beneficia.

**La seqüència.** La normativa exacta del transcript és a l'Annex A.4.

1. **Preparar.**
   - La comanda és `proofman-cli pilfflonk prove -k <provingKey> -o <dir>`, amb els mateixos noms d'arguments que `prove`, i amb el witness d'un directori (`--witness <dir>`) o d'una biblioteca (`--witness-lib <so> [--public-inputs <json>]`, M38c, §4.3).
   - Carregar el `provingKey/`: `globalInfo`, `pilfflonk.vkey.json` (per al *digest*), `pilfflonk.srs.bin` i, per a cada AIR, `pilfflonkinfo`, `.bin` i `.const`.
   - Rebre les instàncies en ordre canònic.
   - Iniciar el transcript, que és únic per a tota la prova, i absorbir-hi el *digest*, el nombre d'instàncies per AIR i els publics.
2. **Per a cada stage `s = 1 … nStages`:**
   1. Per a cada instància, el C++ fa aquests passos:
      - calcula les columnes de l'stage: a l'stage 1, les del witness; a partir del 2, les dels *hints* de la std, amb els reptes de l'stage i en l'ordre del STARK (`gen_proof.hpp:141-143`; M30, M31), que la clau fixa en carregar-se (`AirKey::stdHints`), sigui quin sigui el del `<air>.bin`:
        - primer els `im_col`, en l'ordre del `.bin` (el de `getHintIdsByName`), com `calculateImHints` (`gen_proof.hpp:27`) amb `multiplyHintFields` (`hints.cpp`): la columna `reference` és `numerator/denominator` a cada fila de `H`, amb una sola inversió en lot. Com al STARK, només en una AIR amb algun `gsum_col` o `gprod_col`. El STARK no afegeix un denominador que és el número 1 (`addHintField`), i aquí s'inverteix igualment: el resultat és el mateix;
        - després els `gprod_col` i els `gsum_col`, com `calculateWitnessSTD` (`gen_proof.hpp:57`), cadascun com `accMulHintFields`: `numerator_air/denominator_air` a cada fila, amb una sola inversió en lot, i acumulat fila a fila a la columna `reference`, com un producte (`gprod_col`) o com una suma (`gsum_col`). La std en fa una sola de cada per AIR; si n'hi hagués més, es calculen totes, i el STARK només calcula la primera;
        - de l'stage, cada *hint* només pot llegir les columnes dels *hints* d'abans, que ja són als buffers (un `gsum_col` les dels `im_col`, un `im_col` les dels `im_col` anteriors): la clau ho comprova (§4.2.1);
        - com que la v1 no té airgroup values (D2), `result`, `numerator_direct` i `denominator_direct` no es llegeixen, com fa `calculateWitnessSTD` quan `hintFieldNameAirgroupVal` és buit. La clau rebutja un *hint* amb un `result` que no és un número;
        - un denominador 0 en alguna fila és un error clar (`UnsatisfiedError`, `PILFFLONK_ERR_UNSATISFIED`), que diu el *hint*, la columna i la fila: la columna no hi té valor;
      - a l'últim stage, calcula també els im pols, després de les columnes dels *hints*, que poden llegir (el de `gsum − 'gsum·(1 − L1)`, per exemple);
      - en fa la INTT (`Polynomial::fromEvaluations`, amb espai per al blinding);
      - hi afegeix el blinding **en forma de coeficients** (`blindCoefficients`), perquè `(X^N−1)·b` s'anul·la a `H`;
      - empaqueta els `f_i` (`CPolynomial`);
      - els compromet (MSM).
   2. Rust fa absorbir els commitments i els valors de l'stage i, si `s < nStages`, en treu els reptes de l'stage `s+1`.

   Com que el transcript és únic, **els reptes de l'stage 2 són compartits per totes les instàncies**, i els busos entre instàncies quadren sense cap mecanisme de repte global (D2).
3. **Quocient.**
   1. Treure `std_vc`.
   2. Per a cada instància, el C++ avalua `Q` sobre el *coset* estès amb el bytecode, en torna a obtenir els coeficients, el parteix si cal (amb blinding entre els trossos) i en compromet els `f_i`. L'avalua part per part, per defecte un *coset* de `H` cada vegada, amb les columnes que llegeix esteses només a aquella part: `Q` i la prova són els mateixos bit a bit sigui quina sigui la mida de les parts, i la memòria de les columnes esteses passa a ser la d'una part (M39, Annex H.6).
   3. Absorbir els commitments de `Q` i treure `xiSeed`. El punt d'avaluació és `ξ = xiSeed^powerW`.
4. **Avaluacions.**
   - El C++ avalua cada polinomi obert als seus punts `ξ·ω^s`.
   - Les columnes fixes s'avaluen una vegada per AIR, i la resta, per instància.
   - Si `Q` està partit, també s'avaluen els seus trossos.
   - Rust fa absorbir les avaluacions.
5. **Obertura.**
   - El C++ fa una única obertura SHPLONK sobre tots els `f_i`, en l'ordre global de l'Annex A.5. `pilfflonk_open` comença fent `squeeze` d'`α_S` (les avaluacions ja s'han absorbit al pas 4), absorbeix `W`, treu `y`, i produeix `W` i `W'`.
   - La base del codi és l'orquestració de `ShPlonkProver` (`pil-fflonk/src/shplonk.cpp`), generalitzada a *offsets* amb signe i a diverses instàncies. Això implica tres feines:
     - **`Polynomial`:** es fa servir el de rapidsnark. `divByXSubValue`, que només existeix a pil-fflonk, se substitueix per `divByMonic(1, β)` (`rapidsnark/polynomial/polynomial.c.hpp:423`), i `fromCoefficients` no cal.
     - **Dependències a substituir:** `PilFflonkZkey` passa a ser el context carregat del `provingKey/`, i `PilFflonkTranscript` passa a ser el `Keccak256Transcript` de rapidsnark. També en depenen `zklog`, les macros de temps i `nlohmann::json`.
     - **La MSM** (`multiMulByScalar` amb `nx`/`x`) ja és compatible amb el ffiasm vendoritzat.
6. **Escriure la prova.** Rust escriu `proof.json` i `publics.json` (A.6).

**Depuració.** `proofman-cli pilfflonk check` comprova el witness fila a fila, sense provar res, i diu quina restricció i quina fila fallen. L'equivalent STARK més proper és `verify-constraints`. Pren el witness com `prove`, d'un directori o d'una biblioteca (M38c, §4.3).
- **Stages ≥ 2 (M30).** Les seves columnes depenen dels reptes, i `check` els treu com el `verify-constraints` del STARK (`proofman/src/proofman.rs`, `_verify_proof_constraints`): d'un transcript d'elements fixos, sense cap commitment, cap MSM ni cap blinding. Un transcript d'A.4 (el `Keccak256Transcript`, amb la mateixa regla de `squeeze`) absorbeix el `dummy_element` del STARK, `[0, 1, 2, r − 1]`, com a `Fr`; després, per a cada `s = 1 … nStages − 1`, en treu els `numChallenges[s]` reptes de l'stage `s + 1`, un per crida, i torna a absorbir `[0, 1, 2, r − 1]`, com el STARK el torna a posar després del repte global (`check::check_challenges`).
- **Les columnes.** El C++ calcula les de cada stage ≥ 2 amb aquests reptes com ho fa el prover, primer les dels *hints* (els `im_col`, i després els `gprod_col` i els `gsum_col`) i després els im pols, en buffers propis (`pilfflonk_check` rep els reptes dels stages 2 … `nStages`): no es compromet res, i una instància es pot provar abans o després igual que si el `check` no s'hagués fet. Els reptes són fixos, i per tant el mateix witness dona sempre el mateix informe. Un denominador 0 hi és el mateix error que al prover.

### 4.5 Pas 5: verificació

**Verificador JS** (D3, D8). pilfflonk no té verificador natiu, igual que el FFLONK existent, que es verifica amb `proofman-cli verify-snark`: aquesta comanda crida `snarkjs fflonk verify` (`proofman/src/snark_wrapper.rs:580-638`). De la mateixa manera, `proofman-cli pilfflonk verify` crida el verificador JS de pilfflonk, que rep tres fitxers com `snarkjs fflonk verify`: la vkey (`pilfflonk.vkey.json`), els publics i la prova.
- **On viu:** a `pilfflonk/js/`, dins de pil2-proofman. És el primer codi JS del repositori. pil-fflonk i el seu verificador es deixaran de fer servir (P4).
- **Plantilla: el verificador fflonk de snarkjs** (`snarkjs/src/fflonk_verify.js`), el del FFLONK existent. En copia la interfície (`verify(vkey, publics, proof, logger)` → cert o fals), els passos 1–3 (commitments a G1, avaluacions i publics a `F`), el patró del transcript, el càlcul de `F`, `E` i `J` i el `pairingEq` final. Només en difereix on snarkjs és específic del circuit PLONK de circom:
  - snarkjs té fixats tres polinomis (`C0`, `C1` i `C2`, amb `k` = 8, 4 i 3) i els seus punts d'obertura; pilfflonk llegeix la llista de `f_i`, amb els seus `k` i *offsets*, de la vkey, i generalitza `computeR0/1/2` com fa `verifyOpenings` de shplonkjs;
  - snarkjs té la identitat de PLONK escrita al codi; pilfflonk calcula `Q(ξ)` amb el `qVerifier` del setup;
  - snarkjs té fixats els reptes `β`, `γ`, `α` i `y`; pilfflonk treu els de cada stage de `numChallenges` (A.4).

  snarkjs només exporta la seva API pública, i per tant el transcript i les funcions auxiliars es copien. snarkjs és de l'equip, i no cal cap capçalera de llicència.
- **D'on surt la resta:** de `pil-stark/src/fflonk/helpers/fflonk_verify.js` i de `verifyOpenings` (`shplonkjs/src/helpers/verifier.js`), amb els canvis de pilfflonk:
  - la seqüència del transcript d'A.4, amb el *digest* i els reptes de `numChallenges`;
  - els *offsets* amb signe;
  - la partició de `Q`;
  - el format de la prova de D7;
  - la vkey autocontinguda.
- **Dependències:** `ffjavascript` (BN254 i `pairingEq`) i `@noble/hashes` (Keccak-256), les mateixes que fa servir snarkjs 0.7.6. Es declaren a `pilfflonk/js/package.json`. `proofman_pilfflonk::js_verifier` les busca com ho fa Node (el `node_modules/` del directori del verificador o d'un pare) i, si en falta alguna, fa `npm install` en aquell directori, com el pas 2 de `node_deps::ensure_node_deps`. No reutilitza `node_deps`, perquè està lligat a `setup/pil2-stark` i perquè Node només resol els mòduls ES des del directori del verificador (M19). `PILFFLONK_JS` permet apuntar a una còpia del verificador.
- **El transcript JS** reprodueix el `Keccak256Transcript` de rapidsnark, igual que el de snarkjs ho fa per al FFLONK existent.

**Passos:**
1. Valida la forma de l'entrada: punts sobre la corba (el cofactor de G1 és 1), escalars `< r` i longituds que quadren amb la vkey. Recalcula el *digest* de la vkey (A.6) i el compara amb el que porta.
2. Refà la seqüència del transcript (A.4).
3. Calcula `Q(ξ)` a partir de les avaluacions amb el `qVerifier` de la vkey, el mateix mecanisme que el verificador STARK. Si `Q` està partit, comprova que `Σ ξ^(i·M·N)·Q_i(ξ) = Q(ξ)` (A.1).
4. Fa la comprovació SHPLONK amb un *pairing* (A.5), sempre amb els commitments fixos de la vkey. La prova no en porta cap. Això corregeix el defecte de `fflonk_verify.js`, que feia servir els de la prova (C.3.1).
5. Acaba amb un codi de sortida diferent de 0 si la prova no verifica. El `main_verifier.js` antic surt amb 0 també quan falla (Annex C), i això no es copia.

Diverses instàncies i restriccions globals queden fora d'abast (D2): el verificador no agrega airgroup values ni avalua `pilout.globalConstraints.json`, que a la v1 no té cap restricció.

**Sense *pairing* a C++.** Com que no hi ha verificador natiu, no es porten F6, F12 ni el *pairing* a `pil2-stark`, i ffiasm no es toca.

**Verificador Solidity** (Fase 4, M40). `proofman-setup setup-pilfflonk --solidity` escriu `pilfflonk.verifier.sol`, i `proofman-setup pilfflonk-solidity` el mateix a partir de la vkey (§4.2). És el segon verificador de pilfflonk, i accepta exactament les proves que accepta el JS, que és la referència (D8).
- **Mecanisme.** El patró de `setup/stark-recurser/stark2circom/circuit_templates/templates.rs`: una plantilla `tera` inclosa amb `include_str!`, `setup/pilfflonk/src/tera/verifier_pilfflonk.sol.tera`, i un sol `render`. El context el calcula `setup/pilfflonk/src/solidity.rs`: les constants (`[τ]₂`, els commitments fixos, `digest mod r` i les arrels de la unitat), on és cada valor al calldata i a la memòria, i dos fragments de Yul que surten d'un algorisme, el transcript d'A.4 i el `qVerifier` desplegat. Com que aquell `render()` és `pub(super)`, `setup/pilfflonk` afegeix `tera = "1"` com a dependència pròpia, la mateixa que `pil2-stark-recurser` (cap crate nou al `Cargo.lock`), i `serde` per al context.
- **Plantilla: el verificador fflonk de snarkjs 0.7.6** (`templates/verifier_fflonk.sol.ejs`), el del FFLONK existent. En copia l'estructura: un sol contracte (`PilfflonkVerifier`) amb un sol bloc `assembly` de funcions Yul, les funcions de snarkjs (`checkField`, `checkPointBelongsToBN128Curve`, `inverseArray`, `g1_acc`, `g1_mulAcc`, `g1_mulAccC`, `checkPairing`), la inversió en lot de Montgomery amb l'`inv` de la prova, que comprova abans de fer-la servir, les formes tancades dels denominadors de Lagrange (`computeLiS0/1/2`) i els precompilats `0x06`, `0x07` i `0x08`. Només en difereix on snarkjs és específic del seu circuit, com el verificador JS (més amunt): una llista de `f_i` amb els seus `k` i *offsets*, el `qVerifier` i els reptes de `numChallenges`. Cada pas cita la funció JS que fa, perquè es puguin comparar un al costat de l'altre.
- **Referència estructural:** els dos contractes del sistema antic.
  - **`PilFflonkVerifier`** (`pil-stark/src/fflonk/solidity/verifier_pilfflonk.sol.ejs`) desplega el codi del verificador en Yul i comprova la partició de `Q`; d'aquí surt `computeQ`.
  - **`ShPlonkVerifier`** (`shplonkjs/src/solidity/verifier.sol.ejs`) generalitza el SHPLONK a una llista de `f` (les arrels, compartides entre els `f` de la mateixa forma, i la inversió en lot amb bucles, sense `extendLoops`).
  - **Desviació:** el sistema antic en feia dos contractes, i el primer cridava el segon amb `staticcall`. Aquí n'hi ha un de sol, com snarkjs: els verificadors de les *fixtures* ocupen de 5,2 a 12,6 kB, per sota del límit de l'EIP-170 (24.576 bytes), i es desplega i es crida una sola vegada.
- **Interfície,** la de `FflonkVerifier`: `verifyProof(bytes32[W] calldata proof, uint256[P] calldata pubSignals) public view returns (bool)`, amb `P = nPublic` i `W` (vegeu "Calldata") fixats per la vkey a la plantilla. Si `nPublic = 0` no hi ha `pubSignals`, perquè Solidity no admet `uint256[0]` (snarkjs hi posa `uint256[1]`). Una prova que no verifica, o de valors malformats, retorna `false`, com snarkjs; una crida més curta que els arguments la reverteix el descodificador d'ABI de Solidity.
- **Calldata.** `proof` són els bytes de la prova (A.6, D7; `Proof::to_bytes`) en paraules de 32 bytes: els commitments dels `f` no fixos (`x‖y`), `W`, `W'`, les avaluacions en l'ordre de la prova, els `Q_i(ξ)` si `Q` està partit, `inv` i `invZh`. Després, una **inversa auxiliar** per a cada frontera `firstRow` o `lastRow`, en l'ordre de `boundaries`: `1/(ξ − ω^j)`, amb `j = 0` o `j = N − 1`. El `Zi` d'aquestes fronteres és `Z_H(ξ)/(ξ − ω^j)` (A.1), i és l'única divisió del verificador que ni `inv` ni `invZh` no cobreixen; el contracte comprova `(ξ − ω^j)·aux = 1` i refusa `aux ≥ r`, com fa amb `inv`, de manera que el calldata d'una prova és únic. `invZh` ja fa `Zi(everyRow)`, i `everyFrame` és un producte. El format de la prova no canvia: les inverses auxiliars només són del calldata, i una AIR sense aquestes fronteres, com qualsevol programa PIL2 compilat (§3.4), no en té cap i el seu calldata és la prova. `pubSignals` són els publics, en l'ordre de `publics.json`. L'`ξ` de les inverses surt de refer el transcript sobre la prova, com el verificador; ho fa el codificador de calldata (M41, més avall).
- **Codificador de calldata** (M41). `proofman-cli pilfflonk calldata -k <pilfflonk.vkey.json> -p <proof.json|proof.bin> --publics <publics.json> [--format solidity|hex] [-o <fitxer>]` dona els arguments de `verifyProof` d'una prova, com `snarkjs zkey export soliditycalldata` dona els del seu `FflonkVerifier` (`src/fflonk_export_calldata.js`), la plantilla. Els noms dels arguments són els de `pilfflonk-solidity` (`-k`, `-o`) i els de `verify-snark` (`-p`).
  - **Entrades:** els tres fitxers de `pilfflonk verify`, llegits com els llegeix el verificador JS. La vkey es valida (`Vkey::validate`) i se'n comprova el *digest* (`Vkey::check_digest`, la comprovació del prover, que ara és de la vkey). La prova ha de tenir exactament els valors que la vkey anomena (`ProofNames::of_vkey`, els noms de `vkey.js`, que són els de `ProofNames::new` per a la instància de l'AIR); pot ser la vista JSON o, si el nom del fitxer acaba en `.bin`, els bytes d'A.6 (`Proof::read`). Hi ha d'haver `nPublic` publics, `< r`. **Desviació respecte a snarkjs**, que només llegeix la prova i els publics i no comprova res: aquí cal la vkey, perquè la forma de la prova i les inverses auxiliars en depenen.
  - **Què fa:** refà el transcript d'A.4 sobre la prova, pas a pas com `computeChallenges` (`challenges.js`) i amb el `Keccak256Transcript` del C++, el del prover (`verifier_challenges`). En treu `ξ = xiSeed^powerW` i calcula les inverses auxiliars, `1/(ξ − ω^j)` amb `ω = 5^((r−1)/N)` (`auxiliary_inverses`, en `Bn254`). Refà el transcript sempre, també quan no hi ha inverses, de manera que refusa el mateix per a totes les claus.
  - **Què refusa** (codi de sortida 1, un missatge que diu per què, i cap fitxer escrit): una vkey que no es llegeix o que no té el *digest* del seu contingut; la prova d'una altra clau, o uns bytes d'una altra longitud; publics d'un altre nombre o que no són `< r`; i una prova amb un commitment o una `W` que el transcript no absorbeix (fora de la corba, el punt a l'infinit o amb una coordenada `< 2^192`, A.4), a la qual posa nom. Aquesta prova no té `ξ`, i el verificador JS i el contracte la rebutgen. Com que `W'` no s'absorbeix, el codificador no comprova si és a la corba: comprovar la corba és cosa del verificador (`field.rs`), i el contracte ho fa. Si hi ha inverses i `ξ` és una fila del domini, també es refusa, perquè `Z_H(ξ) = 0` i el verificador la rebutja per `invZh`.
  - **Sortida:** amb `--format solidity`, que és el format per defecte, la llista d'arguments com la imprimeix snarkjs: `[0x…,0x…],[0x…]`. Cada paraula és `0x` i 64 xifres hexadecimals; primer van les de `proof` i després, si n'hi ha, les de `pubSignals`. Sense publics només hi ha `[…]`, perquè el contracte no té `pubSignals`. snarkjs posa un espai després d'algunes comes, i aquí no n'hi ha cap. Amb `--format hex`, la crida codificada en ABI: el selector (els 4 primers bytes de `keccak256("verifyProof(bytes32[W],uint256[P])")`, o de `verifyProof(bytes32[W])` sense publics) i els dos arguments, que com que són de mida fixa es codifiquen al seu lloc, tot com `0x` i xifres hexadecimals. És el `data` d'un `eth_call`, o el de `cast call <adreça> --data <hex>`. Amb `-o`, el fitxer només conté el calldata i un salt de línia. Sense `-o`, el calldata s'imprimeix a la sortida estàndard en una línia pròpia, després de la capçalera que `proofman-cli` imprimeix sempre; per a un script, cal `-o`.
  - **On viu:** a `proofman_pilfflonk::calldata` (`pilfflonk/src/calldata.rs`). `Calldata::read` llegeix els fitxers; `Calldata::encode` fa la feina a partir dels valors; `Calldata::with_auxiliary_inverses` fa un calldata amb les inverses que se li donin, per als tests (un calldata que `encode` no dona); i `to_solidity` i `to_hex` escriuen el resultat. `CalldataLayout` era de `pilfflonk-setup`; ara és d'aquí, i el generador del contracte el fa servir. La comanda és a `cli/src/commands/pilfflonk/pilfflonk_calldata.rs`, i l'arnès de M40 (`setup/pilfflonk/tests/solidity.rs`) ja no té codificador propi: fa servir `Calldata::encode` i `verifier_challenges`.
- **Els passos, els del JS** (`verify.js`):
  1. **Passos 1-3** (`checkInput`; `proof.js`, `elements.js`): els commitments, `W` i `W'` són punts afins de G1 (coordenades `< q`, el mòdul del cos base, i no `< r` com a snarkjs, Annex F.13; no `(0, 0)`; sobre `y² = x³ + 3`); els que absorbeix el transcript (els commitments i `W`) no tenen cap coordenada `< 2^192` (`transcript.js`, A.4); les avaluacions, els trossos de `Q`, `inv`, `invZh`, les inverses auxiliars i els publics són `< r` (snarkjs no comprova els publics, Annex F.13). La vkey són les constants del contracte, i les longituds, els tipus dels arguments.
  2. **Pas 4** (`computeChallenges`; `challenges.js`): el transcript d'A.4 en un buffer de memòria. Cada `squeeze` és `keccak256` del buffer mòdul `r`, i el buffer passa a ser el repte, com el `Keccak256Transcript`. Els commitments d'un stage i les avaluacions (amb els trossos de `Q`) són consecutius al calldata, i s'absorbeixen amb un sol `calldatacopy`, des del primer escalar del calldata: la primera avaluació o, si no n'hi ha, el primer tros de `Q` en l'ordre del *layout*, que no sempre és `Q0` (troballa BAIXA de la revisió: abans es copiava des de `Q0`, i una clau sense avaluacions i amb `Q` partit en l'ordre `Q1, Q0` rebutjava les proves bones).
  3. **Pas 5** (`computeZh`, `computeZi`, `computeQ` i `checkQPieces`): `ξ = xiSeed^powerW`, `Z_H(ξ)·invZh = 1`, els `Zi` de cada frontera (`qverifier.js`, `computeZi`), `Q(ξ)` amb el `qVerifier` desplegat (una instrucció Yul per entrada, amb els temporals a memòria i una còpia d'un valor fix sense codi; `executeCode`) i, si està partit, `Σ ξ^(i·M·N)·Q_i(ξ) = Q(ξ)` (`joinQPieces`).
  4. **Pas 6** (`computeRoots`, `computeInversions`; `shplonk.js`): `xiSeed ≠ 0` (`checkOpening`), les arrels de cada `(k, offsets)` diferent, `Z_{T_i}(y) ≠ 0` (`verifyOpening`), els valors que inverteix `inv` en l'ordre d'A.5 i `inverseArray`, que comprova `inv·Π = 1` (`isValidInverse`).
  5. **Pas 7** (`computeR`, `computeFEJ`, `checkPairing`): `r_i(y)`, `F`, `E` i `J` (`computeQuotients`, `computeF`, `computeE`, `computeJ`), amb els commitments fixos de la vkey com a constants (A.5), i el *pairing* amb `X_2`. Cada crida a un precompilat ha de tornar el que torna el precompilat: 64 bytes `0x06` i `0x07`, i 32 bytes que valen 1 el `0x08` (un enduriment respecte de snarkjs, Annex F.13).
- **Formes tancades** (les de snarkjs, que el JS no fa servir). Per a les arrels `x_j` d'un *offset* `s`, que són les de `X^k − z_s` amb `z_s = ξ·ω^s`: `Z_T(y) = Π_s (y^k − z_s)`, i el denominador de Lagrange `(y − x_j)·Π_{x'≠x_j}(x_j − x')` és `(y − x_j)·k·x_j^(k−1)·Π_{s'≠s}(z_s − z_{s'})`, perquè `k·X^(k−1)` és la derivada de `X^k − z_s` i `x_j^k − z_{s'} = z_s − z_{s'}`. Són els mateixos valors que calcula el JS (identitats de polinomis), i ho confirma `inv·Π = 1` a totes les proves: un sol denominador diferent faria que el contracte refusés una prova bona. Els denominadors es compten per a cada `f_i`, repetits si dos `f` comparteixen arrels, com A.5 els llista.
- **Quina vkey es refusa.** El generador refusa una vkey que el verificador no llegiria (`Vkey::validate`), una que tingui un *digest* que no és el del seu contingut (el JS no n'accepta cap prova) i una amb un commitment fix fora de la corba (el punt a l'infinit, el d'una columna fixa que s'anul·la a `τ`, sí que hi és). Des de la revisió de seguretat de M40, `Vkey::validate` també comprova, com el JS, dues coses que abans no mirava, i per tant ho fan igual el setup (en escriure la vkey), el prover (en llegir-la) i el generador:
  - **`X_2` és un punt de G2 que no és el punt a l'infinit:** sobre el *twist* i a la seva torsió `r` (`elements.js`, `g2FromObject`). Ho comprova el C++, amb les funcions de l'SRS (`checkG2`, `pilfflonk_srs.cpp`), a través de `pilfflonk_g2_check` de l'API C: no hi ha aritmètica de G2 a Rust. **Troballa de la revisió (MITJANA):** abans, una vkey amb `X_2 = (0, 0, 0, 0)` i el *digest* refet generava un contracte, el precompilat `0x08` pren aquest punt com l'infinit (EIP-197), `e(A, [τ]₂) = 1`, i qualsevol podia falsificar una prova de qualsevol enunciat amb `W' = y⁻¹·(E + J − F)`. El JS ja la refusava; ara el Rust també. El setup ja refusava un `ptau` amb `[τ]₂` a l'infinit, com a punt fora del *twist*, i ara ho diu pel seu nom (§4.2.1). Mentre el setup no ha llegit l'SRS, la vkey porta `[1]₂` com a `X_2`.
  - **Els *offsets* de cada `f`,** com `checkLayout` (`shplonk.js`): `|s| < N` i dos *offsets* mai no són la mateixa fila mòdul `N` (`Layout::check`; abans només es comprovava que fossin creixents). **Troballa de la revisió (BAIXA).**
  El setup genera el contracte abans d'escriure la vkey, i un error atura el setup com qualsevol altre.
- **Compilació.** Amb l'optimitzador (200 *runs*, el `foundry.toml` dels tests), sense cap avís de solc. Sense l'optimitzador, el d'`all_sum` arriba a 21,8 kB. El contracte només fa servir `shr` (Constantinople) i els precompilats de Byzantium, amb els costos d'Istanbul (EIP-1108); solc 0.8.37 compila per defecte per a una EVM recent, amb `PUSH0` (Shanghai): per a una cadena que no en té, cal el seu `--evm-version`.
- **Eines, fixades** (usuari, 30-09-2026): Foundry v1.8.3 i solc 0.8.37, instal·lats fora del repositori. Els tests les troben a `PILFFLONK_FORGE` i `PILFFLONK_SOLC`, com el compilador a `PIL2C_EXEC`, i sense aquestes variables són `#[ignore]`, de manera que la CI sense Foundry passa. El projecte Foundry és a `pilfflonk/solidity/` (`foundry.toml` i `test/PilfflonkVerifier.t.sol`, sense `forge-std`: el test declara els *cheatcodes* que fa servir). El test de Rust el copia a un directori propi a `target/`, hi escriu el verificador de la clau i els casos, i hi executa `forge test --offline`, amb `FOUNDRY_SOLC=$PILFFLONK_SOLC`: el `foundry.toml` té `offline = true` i `solc = "0.8.37"`, i l'entorn en dona el camí, de manera que Foundry no baixa cap compilador (res a `~/.svm`). Res no es compila dins del repositori.
- **Validació de M40** (`setup/pilfflonk/tests/solidity.rs`; proves amb llavor fixa i un `ptau` d'una `τ` fixa de mida completa, N13):
  - el contracte de cada clau compila amb solc 0.8.37 i Foundry el desplega;
  - Foundry accepta la prova del Fibonacci empaquetat i amb `--no-packing`, de la *fixture* de l'empaquetat (`k = 3, 4, 4`, `powerW = 12`), de la dels *offsets* amb signe (`{−1, 0, 1, 2}`; també amb `Q` partit en tres), d'`all` en bus de suma (stage 2, `powerW = 72`; també amb `Q` partit en dos), i dels `pilout` de `domains.rs` (`firstRow`, `lastRow` i `everyFrame`, amb inverses auxiliars; i sis `everyFrame`, empaquetat i amb `--no-packing`);
  - refusa les mateixes proves amb una avaluació, un commitment, un públic o `W'` canviats (aquest últim només el veu el *pairing*), amb un commitment fora de la corba o que el transcript no absorbeix (`(1, 2)`), i un calldata amb una coordenada `x + q`, un escalar `e + r` o una inversa auxiliar incorrecta: en tots aquests casos `verifyProof` torna `false`, i només un calldata més curt que els arguments reverteix (el test de Foundry ho distingeix);
  - les mutacions d'una avaluació, un commitment o un públic es repeteixen "arreglades": amb `invZh`, `inv` i les inverses auxiliars refetes per al transcript nou, com l'arnès de la revisió, de manera que arriben al *pairing* (amb `Q` sencer) o a `checkQPieces` (amb `Q` partit); amb `Q` partit, també els trossos canviats sense canviar-ne la suma (arriben al *pairing*) i un tros canviat (`checkQPieces`). El test comprova que el cas que arriba al *pairing* costa gairebé el mateix gas que la prova;
  - una clau que cap `pilout` no dona, amb `Q` partit en l'ordre `Q1, Q0` i cap avaluació, amb una prova feta a mà amb la `τ` del `ptau` de test (`foundry_verifies_a_split_q_without_evaluations`): el JS i Foundry l'accepten, i amb el transcript d'abans Foundry la rebutjava;
  - en cada cas diu el mateix que el verificador JS (el test de Foundry falla si no), i el calldata del codificador del test refà el transcript del prover (els `xiSeed` coincideixen). Des de M41 el codificador és el de la CLI (`Calldata::encode`), i els reptes refets (els dels stages, `std_vc` i `xiSeed`) són els del prover;
  - els tests que no necessiten eines (`setup/pilfflonk/tests/setup/solidity.rs`, `setup/pil2-stark/tests/setup_pilfflonk.rs`) comproven que `--solidity` no canvia cap altre fitxer, que `pilfflonk-solidity` escriu els mateixos bytes, què es refusa i la forma del calldata.
- **Gas** (M40, preliminar; l'informe és a l'Annex I). El de la crida a `verifyProof` d'una prova que verifica, mesurat amb `gasleft()` al test de Foundry (sense els 21.000 de la transacció), i el del seu calldata (EIP-2028):

  | Clau | `f` | Paraules de `proof` | Codi (bytes) | Gas de `verifyProof` | Gas del calldata |
  |---|---|---|---|---|---|
  | Fibonacci, empaquetat | 5 | 22 | 5.290 | 180.790 | 12.108 |
  | Fibonacci, `--no-packing` | 6 | 21 | 5.252 | 185.168 | 11.596 |
  | `packed` (`powerW = 12`) | 6 | 44 | 8.787 | 229.280 | 22.896 |
  | `signed` (*offsets* `{−1, 0, 1, 2}`) | 6 | 51 | 11.360 | 236.264 | 26.872 |
  | `signed`, `Q` en 3 trossos | 6 | 54 | 11.537 | 239.509 | 28.444 |
  | `all_sum` (stage 2, `powerW = 72`) | 9 | 55 | 12.545 | 275.771 | 28.548 |
  | `all_sum`, `Q` en 2 trossos | 9 | 57 | 12.638 | 277.627 | 29.656 |
  | `domains` (`firstRow`, `lastRow`, `everyFrame`) | 6 | 28 | 6.132 | 191.270 | 14.692 |
  | `frames` (sis `everyFrame`) | 6 | 34 | 8.656 | 206.119 | 17.424 |
  | `frames`, `--no-packing` | 9 | 36 | 7.504 | 216.594 | 18.460 |
  | a mà: `Q` partit, cap avaluació | 1 | 10 | 3.037 | 142.176 | 5.324 |

  Comprovar el que tornen els precompilats (pas 7) costa de 587 a 907 de gas per prova (de 14 a 22 crides) i de 6 a 14 bytes de codi. Una prova refusada per `invZh`, `inv`, les inverses auxiliars o la comprovació de l'entrada costa de 3.000 a 60.000 de gas; una que arriba al *pairing* i falla, uns 2.500 més que una bona. M42 en troba la causa, que no és el *pairing*: Foundry aïlla cada crida en una transacció pròpia, i les que van després de la primera, com aquesta, costen 2.500 més perquè l'adreça és freda. Mesurada sense aïllament, costa el mateix que la bona (Annex I.1).
- **Validació de M41** (Fase 4, validació 1):
  - **Foundry sobre totes les *fixtures* de les fases 1 a 3** (`cli/tests/pilfflonk_prove.rs`, `foundry_accepts_the_proof_of_every_fixture`). Hi ha 73 claus: cada *fixture* amb els setups del seu E2E, és a dir, agrupada per defecte (amb l'`extraMuls` de pil-fflonk als exemples), amb `--no-packing`, amb el `--max-constraint-degree` i l'`--extra-muls` dels seus tests i amb `Q` partit on `qDeg ≥ 2` ho permet. Les claus són:
    - el Fibonacci (per defecte, amb `--extra-muls 0` i amb `--no-packing`);
    - `packed`, agrupat i amb `--no-packing`, i `signed`, amb els graus 9, 3 i 2 i amb `Q` en 3 i 2 trossos, també amb `--no-packing`;
    - els quatre `pilout` de `domains.rs`, amb els setups de `domain_setups` i amb `Q` partit;
    - `sum_bus`, `sum_bus_degree4`, `prod_bus` i `prod_bus_im` (també amb `Q` partit);
    - `plookup`, `permutation`, `connection`, `range_check` i `all`, en bus de suma i de producte, i amb `Q` partit en els 8 que tenen `qDeg ≥ 2` amb `--max-constraint-degree 3`, com el seu E2E, més `all_prod` en 3 trossos.

    `permutation_sum` i `range_check_sum` tenen `qDeg = 1` amb qualsevol grau, i `Q` no s'hi pot partir. Per a cada clau, el test fa aquests passos:
    - `pilfflonk-solidity` escriu el verificador, i solc el compila sense cap avís i per sota de l'EIP-170;
    - `pilfflonk prove` en fa la prova, amb la llavor fixa. El Fibonacci, la Connection i `all` la treuen de la biblioteca de witness (Fase 3, M38c), i la resta, del directori del generador;
    - `pilfflonk verify` accepta la prova, i `pilfflonk calldata --format hex` en dona el calldata;
    - Foundry l'accepta. També la rebutja, igual que `pilfflonk verify`, amb una avaluació canviada, amb un commitment canviat per `W` i, si n'hi ha, amb un públic canviat: en total, 73 proves acceptades i 192 rebutjades. El test de Foundry comprova que el selector del calldata és el que solc calcula per a `verifyProof`;
    - el calldata de `proof.bin` és el de `proof.json`, la forma Solidity té les mateixes paraules, i `ProofNames::of_vkey` dona els noms de `ProofNames::new`.

    Els setups es fan d'un en un al fil del test, perquè criden el C++ en el procés (M26), i la resta (la CLI, Node, solc i Foundry són programes) la fan quatre fils. Tarda 217 s, amb la màquina molt carregada (una càrrega de 300 a 470 sobre 256 CPU). Amb totes les claus en sèrie, en tardava 783.
  - **El calldata de la CLI és el de l'arnès de M40.** Abans de canviar l'arnès, se'n van desar els casos que veu el JS (104, de les 11 claus). Sobre aquests casos, `pilfflonk calldata` dona, paraula per paraula, el calldata del codificador de M40 en els 84 que codifica. Els altres 20 (un commitment fora de la corba o `(1, 2)`) els refusa, com diu "Codificador de calldata". Els selectors són els de `solc --hashes`, i `cast calldata` de Foundry codifica la forma Solidity de 5 d'aquestes claus (amb publics i sense) en el mateix hex que `--format hex`. Sobre el codificador nou, l'arnès de M40 (els 135 casos) dona els mateixos resultats i el mateix gas.
  - **Sense eines** (`cli/tests/pilfflonk_calldata.rs`, a la CI):
    - sobre `Domains` (dues inverses) i `Frames` (cap inversa i cap públic), el calldata és el selector de solc, els bytes de la prova, les inverses de l'`ξ` del prover (`(ξ − ω^j)·aux = 1`, comprovat amb `num-bigint`) i els publics, en les dues formes, de `proof.json` i de `proof.bin`, amb `-o` i sense;
    - la comanda refusa les entrades que no van juntes (vegeu "Codificador de calldata");
    - un test unitari comprova els selectors d'ERC-20 (`transfer`, `balanceOf`).
- **Gas** (M41; l'informe és a l'Annex I). El de la crida a `verifyProof`, mesurat com a M40, per a les 73 claus, agrupades per família (el test imprimeix la de cada clau). El gas creix amb el nombre de `f`: uns 170.000 i uns 7.500 per `f`, un model que M42 corregeix (Annex I.4).

  | Família | Claus | `f` | Paraules de `proof` | Codi (bytes) | Gas de `verifyProof` | Gas del calldata |
  |---|---|---|---|---|---|---|
  | Fibonacci | 3 | 3–6 | 18–22 | 5.124–5.290 | 170.869 (`--extra-muls 0`) – 185.168 (`--no-packing`) | 10.036–12.108 |
  | `packed` | 2 | 6–19 | 44–57 | 8.787–10.094 | 229.280 – 301.425 (`--no-packing`) | 22.920–29.528 |
  | `signed` | 9 | 6–20 | 40–61 | 9.820–12.662 | 236.264 – 313.881 (`--no-packing`, grau 2) | 21.264–32.016 |
  | `domains.rs` | 21 | 5–15 | 19–54 | 4.030–9.303 | 174.065 (`firstRow`) – 269.241 (`Frames`, grau 2, `--no-packing`) | 10.060–27.640 |
  | Busos de la std | 9 | 6–11 | 26–38 | 6.064–8.153 | 191.791 (`prod_bus`) – 232.387 (`prod_bus_im`, `--no-packing`) | 13.480–19.600 |
  | Exemples de pil-fflonk | 29 | 5–31 | 24–82 | 5.902–15.474 | 186.580 (`permutation_prod`) – 406.897 (`all_sum`, `--no-packing`) | 12.316–42.300 |

  El contracte més gran, `all_prod` amb `--no-packing`, té 15.474 bytes, per sota de l'EIP-170.
- **Validació de M42** (Fase 4, validació 2): *fuzzing* diferencial entre el JS i el Solidity, a `pilfflonk/tests/data/fuzz.rs` (el test `foundry_and_the_js_verifier_agree_on_mutated_proofs` de `cli/tests/pilfflonk_prove.rs`, que fa servir els *helpers* de M40 i M41: `solidity_keys`, `set_up_for_foundry`, `Calldata::encode`, `verifier_challenges`, `fixup` i l'executor de Foundry de `pilfflonk/tests/data/foundry.rs`).
  - **Claus.** 13 de les 73 de M41 (`FUZZ_KEYS`):
    - el Fibonacci, `packed` i `signed` amb `Q` en tres trossos, cadascuna agrupada i amb `--no-packing`;
    - `Domains`, que té les inverses auxiliars de `firstRow` i `lastRow`: agrupada i, amb `Q` partit, amb `--no-packing`;
    - els busos de l'stage 2: `sum_bus` agrupada i `prod_bus` amb `--no-packing`;
    - `all` amb `Q` partit, en bus de suma (dos trossos) i de producte (tres), i `all_sum` amb `--no-packing`, la clau que gasta més gas.
  - **Casos.** Per a cada clau, unes quantes proves honestes de `pilfflonk prove`: la de la llavor de M41 i les d'altres llavors (dues en total, i una més per cada 400 casos de la clau). Cada mutació en pren una a l'atzar, amb una llavor fixa (`FUZZ_SEED`; `PILFFLONK_FUZZ_SEED` la canvia). Hi ha 25 famílies de mutacions:
    - **una paraula qualsevol** del calldata (de `proof`, les inverses auxiliars incloses, o de `pubSignals`): un bit canviat, un valor aleatori (de 256 bits, per sota de `q` o per sota de `r`), o un valor de les fronteres dels cossos (0, 1, `r − 1`, `r`, `r + 1`, `q − 1`, `q`, `2^256 − 1`);
    - **un punt** (un commitment, `W` o `W'`) canviat per un altre punt de G1 (un múltiple aleatori de `[1]₁`, el mateix punt negat o un altre punt de la prova), per un punt fora de la corba amb coordenades per sota de `q`, pel punt a l'infinit `(0, 0)` o per `(1, ±2)`, de coordenada `< 2^192`;
    - **dos valors intercanviats:** dos escalars de l'stage 4 del transcript (avaluacions o trossos de `Q`), o dos punts;
    - **un públic:** `+ 1`, aleatori, `p + r` (el mateix element d'`Fr`, l'àlies que snarkjs no refusa, Annex F.13), `r`, o 256 bits qualssevol;
    - **cada valor amb una comprovació pròpia:** `W` o `W'` per un altre punt de G1; `inv`, `invZh`, una inversa auxiliar o un tros de `Q` amb `+ 1`, un valor aleatori per sota de `r` o `+ r`;
    - **la longitud del calldata:** més curt que els arguments, que reverteix, o amb bytes de més al final, que el contracte no llegeix (el veredicte és el de la prova);
    - **"arreglades"** (`fixup`, que passa de l'arnès de M40 a `pilfflonk/tests/data/mutations.rs`, i `Calldata::encode`), perquè passin `invZh`, `inv` i les inverses auxiliars:
      - una avaluació, un commitment (per un punt aleatori de G1) o un públic canviats, o dues avaluacions o dos commitments intercanviats: arriben a `checkQPieces` o al *pairing*;
      - `W` per un punt aleatori de G1 (només canvia `y`), i els trossos de `Q` rebalancejats sense canviar-ne la suma: arriben al *pairing*;
      - un tros de `Q` canviat: arriba a `checkQPieces`;
      - **`W' = y⁻¹·(E + J − F)`**, la falsificació de la revisió de M40, que anul·la el costat esquerre del *pairing*. És d'una prova que passa `checkQPieces`: honesta, amb `W` canviat i, segons la clau, amb els trossos de `Q` rebalancejats (`Q` partit) o amb una avaluació, un commitment o un públic canviats (`Q` sencer). `F`, `E` i `J` els calcula el JS amb les seves pròpies funcions (`forgeWp` de `js_batch.mjs`). Amb una `X_2` honesta, `e(0, [1]₂) = e(W', [τ]₂)` no es compleix, i el cas ha de fallar al *pairing*.
  - **Què es compara.** Per a cada cas:
    - **el veredicte del JS** sobre la prova i els publics que té el calldata, escrits com `proof.json` i `publics.json`, sigui quin sigui el valor de cada paraula (en decimal, i el JS en fa les comprovacions). El calcula `verify()` de `verify.js`, per lots, en un sol procés de Node per ronda: `pilfflonk/tests/data/js_batch.mjs`, que fa el mateix que `bin/verify.js` amb cada prova. El primer cas de cada família i de cada clau (llevat de la de la longitud, que no canvia la prova) també passa per `js_verifier::verify`, el de la CLI, i hi ha de coincidir;
    - **les comprovacions que només són del calldata:** cada inversa auxiliar ha de ser `1/(ξ − ω^j)` de l'`ξ` de la prova i `< r`, i un calldata més curt que els arguments reverteix;
    - **el que fa `verifyProof` a Foundry,** que ha de ser el mateix: `true`, `false` o un *revert*, i aquest últim només amb el calldata curt;
    - **la comprovació que el refusa,** que també ha de ser la mateixa: la que diu el missatge del JS i la que diu una sonda del contracte. La sonda (`Probe`, a `fuzz.rs`) és una còpia instrumentada del contracte generat, que el test escriu al seu directori i que el generador no escriu mai. Cada `fail()` hi diu el seu lloc, i el cos desa el gas que queda després de cada pas. El test comprova que la sonda diu el mateix que el verificador en tots els casos;
    - **la cobertura:** cada família diu a quines comprovacions han d'arribar els seus casos. Per exemple, un commitment canviat arriba a `invZh`, una `W` canviada a `inv`, i les "arreglades", al *pairing* si `Q` és sencer i a `checkQPieces` o al *pairing* si està partit. A més, un cas que refusa el *pairing* ha de costar com a mínim el 90 % del gas de la prova honesta (el criteri de M40).

    Quan una família té una discrepància, no s'hi fan més casos en cap clau, i el test informa del cas: les paraules que ha canviat, els missatges del JS i el resultat del contracte (el gas i la comprovació).
  - **Foundry.** El fuzzer copia `pilfflonk/solidity/foundry.toml` i `pilfflonk/solidity/fuzz/PilfflonkFuzz.t.sol` a un projecte propi, amb el verificador, la sonda i un fitxer binari per cas (`cases/<i>.bin`, sense JSON), i l'executa per rondes de 250 casos (`FuzzProject` de `foundry.rs`). L'executa sense aïllament (`FOUNDRY_ISOLATE=false`). Amb aïllament, que és el valor per defecte de Foundry, cada crida és una transacció pròpia, i totes les que van després de la primera costen 2.500 més (EIP-2929, un compte fred). Sense, i amb una primera crida que no es compta i que fa créixer la memòria del test, cada cas es mesura com la primera crida del test de M40: la prova honesta costa exactament el que diu la taula de M41.
  - **Execucions** (01-10-2026, amb una càrrega de 21 a 28 sobre 256 CPU; els casos i els resultats no canvien d'una execució a l'altra):
    - **la de la CI:** 400 casos, que són 403 (31 per clau), en 37 s, setups inclosos;
    - **l'ampliada** (`PILFFLONK_FUZZ_CASES=10400`): 10.400 casos (800 per clau) en 94 s. Cada clau hi passa de 5 a 13 s fent els casos, de 12 a 19 s al JS i de 2 a 5 s a Foundry, en quatre fils.

    **No hi ha cap discrepància.** Tots els veredictes coincideixen, i també la comprovació que refusa cada cas. El contracte només reverteix amb el calldata curt, i no accepta cap prova mutada: les acceptades de la família de la longitud són proves honestes amb bytes de més al final. Cada família arriba a les comprovacions que busca. Els casos de l'execució ampliada són aquests:

    | Família | Casos | D'acord | Refusats per (la comprovació del contracte, que és la del JS) | Gas de `verifyProof` |
    |---|---|---|---|---|
    | Honestes | 52 | 52 | acceptada 52 | 180.790–406.897 |
    | Un bit canviat | 465 | 465 | coordenada `≥ q` 1, fora de la corba 199, escalar `≥ r` 5, `invZh` 42, inversa auxiliar 2, `checkQPieces` 65, `inv` 151 | 1.207–58.204 |
    | Una paraula aleatòria | 467 | 467 | coordenada `≥ q` 63, fora de la corba 142, escalar `≥ r` 60, `invZh` 31, inversa auxiliar 8, `checkQPieces` 56, `inv` 107 | 1.021–58.204 |
    | Una frontera dels cossos | 467 | 467 | coordenada `≥ q` 42, fora de la corba 163, escalar `≥ r` 175, `invZh` 9, inversa auxiliar 2, `checkQPieces` 25, `inv` 51 | 1.021–49.416 |
    | Un altre punt de G1 | 468 | 468 | `invZh` 342, `inv` 59, *pairing* 67 | 7.544–406.897 |
    | Un punt fora de la corba | 468 | 468 | fora de la corba 468 | 1.207–9.794 |
    | El punt a l'infinit | 467 | 467 | punt a l'infinit 467 | 1.041–9.628 |
    | `(1, ±2)` | 467 | 467 | coordenada `< 2^192` 390, *pairing* 77 | 3.293–406.897 |
    | Dos escalars de l'stage 4 intercanviats | 465 | 465 | `checkQPieces` 155, `inv` 310 | 12.543–49.416 |
    | Dos punts intercanviats | 466 | 466 | `invZh` 446, `inv` 20 | 7.544–58.204 |
    | Un públic | 465 | 465 | escalar `≥ r` 277, `invZh` 188 | 5.241–22.470 |
    | `W` | 465 | 465 | `inv` 465 | 13.400–58.204 |
    | `W'` | 464 | 464 | *pairing* 464 | 180.790–406.897 |
    | `inv` | 464 | 464 | escalar `≥ r` 146, `inv` 318 | 4.937–58.204 |
    | `invZh` | 467 | 467 | escalar `≥ r` 158, `invZh` 309 | 5.070–22.470 |
    | Una inversa auxiliar | 69 | 69 | escalar `≥ r` 27, inversa auxiliar 42 | 6.136–9.941 |
    | Un tros de `Q` | 164 | 164 | escalar `≥ r` 57, `checkQPieces` 107 | 6.648–27.550 |
    | La longitud del calldata | 467 | 467 | acceptada 232, *revert* 235 | 515–406.903 |
    | Arreglada: una avaluació | 466 | 466 | `checkQPieces` 145, *pairing* 321 | 12.543–406.897 |
    | Arreglada: un commitment | 465 | 465 | `checkQPieces` 164, *pairing* 301 | 12.543–406.897 |
    | Arreglada: un públic | 465 | 465 | `checkQPieces` 164, *pairing* 301 | 12.543–406.897 |
    | Arreglada: dos valors intercanviats | 465 | 465 | `checkQPieces` 160, *pairing* 305 | 12.543–406.897 |
    | Arreglada: `W` | 466 | 466 | *pairing* 466 | 180.790–406.897 |
    | Arreglada: els trossos de `Q` rebalancejats | 165 | 165 | *pairing* 165 | 204.560–277.627 |
    | Arreglada: un tros de `Q` | 165 | 165 | `checkQPieces` 165 | 12.543–27.550 |
    | Arreglada: `W'` falsificada | 466 | 466 | *pairing* 466 | 180.790–406.897 |

    El fuzzer no pot arribar a tres comprovacions, i no hi arriba: `xiSeed = 0` i `Z_T(y) = 0` demanen que el transcript doni un valor concret (probabilitat `≈ 2^-254`), i la del retorn dels precompilats només falla en una cadena sense els precompilats d'EIP-196 i EIP-197 (Annex F.13).
  - **Sense eines** (la CI): el test és `#[ignore]` sense `PILFFLONK_FORGE`, `PILFFLONK_SOLC` i `PIL2C_EXEC`, com l'E2E de M41. L'arnès de M40 (`setup/pilfflonk/tests/solidity.rs`) fa servir ara el `fixup` de `mutations.rs`, i dona els mateixos resultats i el mateix gas.
- **Informe de gas** (Fase 4, validació 3): l'Annex I. Totes les claus, on va el gas de 13 d'elles, com creix, la comparació amb l'`FflonkVerifier` de snarkjs i les oportunitats, que no s'implementen.
- **Obert després de M42:**
  - **Gas.** Les oportunitats de l'Annex I.6, que no s'implementen: el contracte es queda com el va revisar M40. N'hi ha dues que no toquen el contracte: el nombre de *runs* de l'optimitzador, que tria qui el desplega (fins a un 5,5 % menys de gas, amb més codi), i l'agrupació del setup (`--extra-muls 0` gasta un 5,5 % menys al Fibonacci). De les dues observacions de M41, els `add(pMem, …)` no estalvien res, i els `PUSH32` de `q` que l'optimitzador converteix en `codecopy` són part del que donen els *runs* (Annex I.6).
  - **Foundry a la CI (M28).** Els tests de Foundry (de M40, M41 i M42) són `#[ignore]` sense les eines. Perquè la CI els executi, caldria:
    - instal·lar-hi Foundry v1.8.3 i solc 0.8.37, fixats i comprovats amb el seu sha256;
    - donar-hi `PILFFLONK_FORGE`, `PILFFLONK_SOLC` i `PIL2C_EXEC`;
    - executar-los amb `--test-threads 2` (M26). En aquesta màquina, el *fuzzer* de la mida de la CI tarda uns 40 s, i l'E2E de les 73 claus, uns 80 s.

    Dues precaucions: mesurar les mides amb solc i no amb `forge build --sizes`, que escriu a `~/.foundry`; i fer servir la mateixa versió de Foundry, perquè el gas que mesura depèn del seu mode d'aïllament (Annex I.1).
  - **Mida.** El `qVerifier` desplegat creix amb el nombre de restriccions (uns 50 bytes per entrada): una AIR molt més gran que `all` podria passar de l'EIP-170. Llavors caldria partir-lo en dos contractes, com el sistema antic, o avaluar el `qVerifier` amb un bucle sobre el codi.
  - **Llicència del contracte:** `GPL-3.0`, la de la plantilla de snarkjs (`SPDX-License-Identifier`; sense la capçalera de snarkjs, que és de l'equip). La plantilla Solidity del STARK en fa servir `AGPL-3.0`; si l'usuari en vol una altra, és una línia.

---

## 5. On viu el codi

### 5.1 Mapa de components

| Component | Crate / ubicació | Llenguatge | Responsabilitat |
|---|---|---|---|
| Compilador | `../pil2-compiler` (branca `develop-0.14.0-pil2-fflonk`) | JS | `pilout` sobre BN254 (§4.1) |
| Constants de camp | `pil2-components/lib/std/pil/` (nou `bn254.pil`) | PIL | `GEN` i `k_coset` per a BN254 (només connexions) |
| Informació simbòlica | crate `pil-info`, a `setup/pil-info/` (nou; extret de `setup/pil2-stark`; D1) | Rust | Passades simbòliques parametritzades pel camp, contenidor `"chps"`, assignació de temporals i `globalConstraints.json` |
| Setup pilfflonk | crate `pilfflonk-setup`, a `setup/pilfflonk/` (nou) | Rust | Validar, agrupar, generar el bytecode, les claus, el `provingKey/`, el *digest* i el Solidity |
| Subcomandament | `proofman-setup setup-pilfflonk` i `pilfflonk-solidity`, al crate `pil2-stark-setup` | Rust | Crida `pilfflonk-setup` |
| Tipus i orquestrador | crate `proofman-pilfflonk`, a `pilfflonk/` (nou) | Rust | Tipus dels fitxers propis, càrrega, instàncies, bucle d'stages, `WitnessSource`, la biblioteca de witness (M38b), escriptura de la prova |
| Nucli BN254 | `pil2-stark/src/pilfflonk/` (nou). De pil-fflonk només adapta l'orquestració SHPLONK genèrica, i reutilitza `Polynomial`, `Evaluations`, `CPolynomial`, `Keccak256Transcript` i `BinFile` de rapidsnark i la FFT i la MSM d'ffiasm. | C++ (ffiasm) | Intèrpret `Fr`, LDE sobre *coset*, blinding, empaquetat, MSM, `Q`, avaluacions i SHPLONK (la banda del prover) |
| Verificador | `pilfflonk/js/` (nou) | JS (Node; `ffjavascript`, `@noble/hashes`) | Transcript, `Q(ξ)`, SHPLONK i *pairing*. El crida `proofman-cli pilfflonk verify` (§4.5). |
| API C | `pil2-stark/src/api/pilfflonk_api.{hpp,cpp}` (nou) | C++ | La superfície C que veu Rust. Retorna codis d'estat i mai no crida `exitProcess`. |
| *Bindings* | `provers/starks-lib-c/bindings_pilfflonk.rs` (declaracions `extern "C"`, incloses amb `include!`) i `provers/starks-lib-c/src/ffi_pilfflonk.rs` (embolcalls), tots dos nous; a `src/lib.rs` s'hi afegeixen `mod ffi_pilfflonk; pub use ffi_pilfflonk::*;` | Rust | FFI escrita a mà, amb el mateix patró que `bindings_starks.rs` i `ffi_starks.rs` |
| CLI | `cli/` | Rust | `proofman-cli pilfflonk prove \| verify \| check \| calldata`, com a subcomandament niat, seguint el patró de `Pilout` (`cli/src/commands/pilout/mod.rs`); `calldata`, el codificador del calldata del verificador Solidity (§4.5, M41) |
| GPU (Fase 5) | `pil2-stark/src/bn128/src/{msm,ntt}` (ja existeix) | CUDA | MSM i NTT, amb `--gpu` quan la compilació ha detectat `nvcc` |

### 5.2 Dependències

```
proofman-cli ─────────────► proofman-pilfflonk ──────────► proofman-starks-lib-c ──► libstarks (C++, ffiasm)
                                    ▲                               ▲
pil2-stark-setup ─► pilfflonk-setup ┘───────────────────────────────┘  (compromisos fixos)
 (proofman-setup)     │     │
       │              │     └──► pil2-pilout
       └──────────────┴────────► pil-info ──► pil2-pilout
```

- **No hi ha cicles.** `pilfflonk-setup` depèn de `pil-info`, `proofman-pilfflonk` i `proofman-starks-lib-c`, però no de `pil2-stark-setup`. `pil2-stark-setup` depèn de `pil-info`, per les passades, i de `pilfflonk-setup`, per allotjar el subcomandament.
- **Què guanya dependències noves.** Només el binari de setup STARK (`pil-info` i `pilfflonk-setup`). El runtime STARK no en guanya cap.
- **Propietat dels fitxers.**
  - Els tipus dels fitxers propis de pilfflonk tenen un sol propietari, `proofman-pilfflonk`: el setup els escriu i el prover els llegeix.
  - `pilout.globalConstraints.json` el genera `pil-info` per als dos backends.
- **Verificador JS.** No depèn de cap crate: llegeix la vkey, la prova i els publics. Un test E2E garanteix que llegeix els fitxers com els escriu Rust.
- **Lectura des del C++.** El C++ llegeix `<air>.pilfflonkinfo.json` amb `nlohmann/json`, igual que avui llegeix `starkinfo.json`. Un test d'anada i tornada Rust ↔ C++ garanteix que les dues bandes llegeixen el mateix.

### 5.3 Interfícies (esbós)

**API C** (`pil2-stark/src/api/pilfflonk_api.hpp`). Convencions:
- **Codificació.** Els escalars es passen com a 32 bytes canònics *little-endian*, i els punts G1 com a 64 bytes (`x‖y`, *little-endian*). El transcript fa servir *big-endian* internament (A.4).
- **Handles.** Tots els objectes són opacs i tenen una funció `_free`.
- **Errors.** Les funcions que creen objectes retornen `NULL` si hi ha un error, i la resta retornen un codi d'estat `int`. Mai no es crida `exitProcess`.
- **Transcript.** Rust decideix què s'absorbeix i quan, i també absorbeix les avaluacions. L'objecte és de C++, i `pilfflonk_open` fa `squeeze` d'`α_S`, absorbeix `W` i treu `y` (A.4).
- **Pendent de tancar (N4 del pla).** Falten encara els publics i els proof values que calen per calcular `Q`, la llavor del generador aleatori i el missatge de l'últim error.

```c
void* pilfflonk_ctx_new(const char* proving_key_dir);                     // carrega globalInfo, vkey (digest), srs i les AIRs
void  pilfflonk_ctx_free(void* ctx);
void* pilfflonk_transcript_new(void);
void  pilfflonk_transcript_free(void* t);
int   pilfflonk_transcript_absorb(void* t, const uint8_t* data, uint64_t n, uint32_t kind); // kind: Fr | G1
int   pilfflonk_transcript_squeeze(void* t, uint8_t out[32]);
void* pilfflonk_instance_new(void* ctx, uint64_t airgroup_id, uint64_t air_id,
                             const uint8_t* stage1, const uint8_t* air_values);
void  pilfflonk_instance_free(void* inst);
int   pilfflonk_commit_stage(void* inst, uint32_t stage, const uint8_t* challenges, uint8_t* out_g1);
int   pilfflonk_commit_q(void* inst, const uint8_t* challenges, uint8_t* out_g1);
int   pilfflonk_evaluate(void* ctx, void** insts, uint64_t n, const uint8_t xi_seed[32], uint8_t* out_evals);
int   pilfflonk_open(void* ctx, void** insts, uint64_t n, void* t, uint8_t* out_w_wp);
int   pilfflonk_last_status(void);                                         // estat de l'última crida d'aquest fil
int   pilfflonk_srs_from_ptau(const char* ptau_path, uint64_t n_g1, const char* srs_path); // setup
int   pilfflonk_srs_g2(const void* srs, uint64_t i, uint8_t out_g2[128]);     // [1]₂ o [τ]₂, canònic x.c0‖x.c1‖y.c0‖y.c1 (M15)
int   pilfflonk_keccak256(const uint8_t* data, uint64_t len, uint8_t out[32]); // el keccak_wrapper de rapidsnark (M15, N10)
void* pilfflonk_srs_load(const char* srs_path);                            // NULL si falla; pilfflonk_last_status en diu el motiu
void  pilfflonk_srs_free(void* srs);
int   pilfflonk_commit_fixed(const void* srs, uint64_t n_bits, uint64_t k,
                             const uint8_t* evals, uint8_t out_g1[64]);    // setup; un f fix per crida (M6)
```

**Rust:**

```rust
pub trait WitnessSource {
    fn instances(&self) -> Vec<AirInstanceRef>;                                       // en ordre canònic
    fn stage1(&self, instance: usize) -> Result<Stage1Witness, WitnessError>;        // columnes + air values en Fr (bytes)
    fn publics(&self) -> Result<Vec<FrBytes>, WitnessError>;
    fn proof_values(&self) -> Result<Vec<FrBytes>, WitnessError>;
}
pub trait PilfflonkWitnessLibrary {                                                   // M38b
    fn witness(&mut self, shape: &WitnessShape, public_inputs: Option<&Path>) -> Result<Witness, PilfflonkError>;
}
pub fn load_witness_library(path: &Path, verbose: u8) -> Result<Box<dyn PilfflonkWitnessLibrary>, PilfflonkError>;
pub fn group(pols: &[CommittedPol], params: &GroupingParams) -> Result<Layout, GroupingError>; // funció pura
```

### 5.4 Convencions

- **Errors:**
  - **biblioteques** (`pil-info`, `proofman-pilfflonk` i la part de `pilfflonk-setup` que fa de biblioteca): `thiserror`, amb el patró de `common/src/error_manager.rs`;
  - **comandes de setup:** `anyhow`, com ara;
  - **CLI:** `Box<dyn Error + Send + Sync>`, com les altres comandes.
  - No es fa servir `panic!` ni `exit()` en codi de biblioteca. Això inclou les passades que es mouen a `pil-info` i que avui fan `panic!`.
- **C++:**
  - no s'hi afegeix estat global; la memòria es gestiona amb RAII, i els errors es retornen a través de l'API C;
  - els cossos de les plantilles van en fitxers `.c.hpp`, perquè el Makefile compila tots els `*.cpp` dels directoris que llista;
  - els noms de fitxer porten el prefix `pilfflonk_` i el namespace és `PilFflonk`, perquè tots els directoris són al camí d'*includes*.
- **Compilació:** cal afegir `./src/api/pilfflonk_api.*` i `./src/pilfflonk` a les tres llistes de fonts del Makefile (`pil2-stark/Makefile:228, 231, 234`). Sense aquest pas, les llibreries de CPU i GPU no inclouen els símbols. No cal tocar `build.rs`.
- **Registre i temps:** `tracing` i les macros de temps de `util`.
- **Format i lints:** `rustfmt` (`max_width = 120`) i `clippy -D warnings`.
- **Tests:**
  - unitaris;
  - de propietats (`group`, passades simbòliques);
  - *golden*: l'agrupació contra el sistema antic, i totes les sortides del setup STARK abans i després d'extreure `pil-info`;
  - end-to-end;
  - de rebuig.
- **CI:** un job end-to-end nou amb una fixture BN254 (compilar, `setup-pilfflonk`, `prove`, `verify`). Avui no hi ha cap job end-to-end SNARK.

---

## 6. En quin ordre es construeix

**Prioritat:** primer una versió CPU funcional (fases 0 a 3); després, de manera incremental, la GPU (Fase 5) i les millores. A partir de la Fase 1, cada fase acaba amb una prova end-to-end verificada per codi que no l'ha produïda, amb `fmt`, `clippy`, els tests i la CI en verd. El pla d'execució incremental detallat és a `pla-pilfflonk.md`.

| Fase | Lliurable | Validació clau |
|---|---|---|
| 0 | Compilador BN254 (fet); transcript; LDE; KZG i SHPLONK (prover C++, verificador JS) | El verificador JS accepta les obertures SHPLONK del prover C++ |
| 1 | Una AIR, una instància, sense busos | Prova end-to-end i tests de rebuig; setup STARK idèntic byte a byte |
| 2 | Busos de la std en una instància | Lookup, permutation, range check i connection verifiquen |
| 3 | Witness en `Fr` i rendiment | Els exemples de pil-fflonk, amb witness de biblioteca; informe de rendiment |
| 4 | Verificador Solidity | Foundry accepta totes les proves; el verificador JS i el Solidity coincideixen |
| 5 | GPU (necessària, després de la versió CPU) | Resultats idèntics bit a bit als de CPU |

### Fase 0: fonaments

**Abast:**
- **Compilador BN254** (§4.1). **Fet** a la branca `develop-0.14.0-pil2-fflonk`, sense commit a 29-09-2026. Al `pilout.proto`, cap canvi.
- **A `pil2-stark`:**
  - l'LDE sobre *coset* amb la FFT d'ffiasm (`Polynomial`/`Evaluations` de rapidsnark);
  - l'accés al transcript: s'exposa per l'API C el `Keccak256Transcript` de rapidsnark, sense modificar-lo;
  - el lector de `ptau`;
  - el commit KZG;
  - SHPLONK sobre polinomis aleatoris i conjunts de punts arbitraris: el prover (C++) s'adapta de `ShPlonkProver`.
- **Verificador JS, la base** (`pilfflonk/js/`): el transcript i la verificació SHPLONK, adaptada de `verifyOpenings`.
- **Bastida:** l'API C, els *bindings*, les entrades al Makefile i els esquelets dels crates.

**Fora d'abast:**
- qualsevol semàntica de PIL, de manera que en aquesta fase encara no hi ha prova end-to-end;
- la std BN254 (Fase 2).

**Validació:**
1. **Compilador.** A BN254, `−1` fa el viatge d'anada i tornada com a `r−1` en cinc llocs: una llista amb un negatiu, una seqüència geomètrica, una seqüència aritmètica decreixent, una columna fixa assignada fila a fila i un `Tables.fill`/`Tables.copy`. A més, `baseField = r`. (Fet; les constants d'expressió s'han comprovat a mà.)
2. **Programes de la CI a BN254.** Els 10 programes de la CI de pil2-proofman compilen. (Fet.)
3. **Regressió a Goldilocks.** Els mateixos 10 programes surten idèntics byte a byte. (Fet.)
4. **Transcript C++ ↔ JS.** El `Keccak256Transcript` de rapidsnark i el transcript JS donen els mateixos reptes per a seqüències que barregen `Fr` i G1.
5. **SHPLONK.** El verificador JS accepta les obertures del prover C++ i les rebutja si se'n canvia qualsevol bit, també amb *offsets* amb signe i conjunts d'arrels repetits.
6. **Transcript.** El `Keccak256Transcript` exposat coincideix amb un càlcul independent: la codificació d'A.4, el `keccak_wrapper` de C++ i la reducció mòdul `r`.
7. **LDE.** L'LDE sobre el *coset* coincideix amb l'avaluació de Horner.
8. **Wrap final intacte.** `git diff` buit a `pil2-stark/src/bn128/src/ffiasm/`: com que ffiasm no es toca (D3), el *wrap* final no pot canviar.

### Fase 1: una AIR, una instància, sense busos

**Abast:**
- **Referències *golden* del setup STARK, noves**, generades **abans** d'extreure `pil-info`.
  - Cobreixen totes les sortides (`starkinfo`, `expressionsinfo`, `verifierinfo`, `.bin`, `.verifier.bin`, `globalInfo` i `globalConstraints`) dels programes de la CI.
  - Les actuals no serveixen (§3.2). Com que els `*.pilout` no es versionen, la CI els ha de regenerar.
- **Extracció de `pil-info`:** extreure i parametritzar les passades simbòliques (D1). Abans cal esborrar el fitxer orfe `setup/pil2-stark/src/pilout_info.rs`.
- **`setup-pilfflonk`:** validació, restriccions amb els quatre tipus de domini, im pols a l'stage 1, *offsets* amb signe, agrupació, blinding, bytecode, codi del verificador, SRS, commits, *digest* i `provingKey/`.
- **Prover:** l'stage 1 (amb els im pols), `Q` (sense partir) i l'obertura.
- **Verificador JS** (§4.5).
- **Depuració:** `check`.
- **CLI.**
- **Witness:** `WitnessSource` sobre fitxer.

**Fixtures:**
- **El Fibonacci de l'Annex G portat a PIL2.**
  - És la mateixa aritmètica que la fixture de pil-fflonk: columnes fixes `L1`/`LLAST`, perquè el compilador només emet `everyRow` (§3.4), i els publics lligats per restriccions.
  - La còpia de l'original és a `pil-fflonk/pil/sm_fibonacci/`, i l'original, a pil-stark `test/state_machines/sm_fibonacci/`.
- **`pilout` sintètics** construïts amb `prost`, per exercitar `firstRow`, `lastRow` i `everyFrame`.
- **Una AIR sintètica** amb *offsets* `{−1, 0, 1, 2}` i una restricció de grau ≥ 4, que força im pols.

**Fora d'abast:**
- stages ≥ 2 i *hints*;
- air values i airgroup values;
- diverses instàncies;
- la partició de `Q`;
- Solidity;
- rendiment.

**Validació:**
1. **Prova end-to-end.** La prova verifica, i també verifica amb `--no-packing` (`k = 1`, KZG en lot sense empaquetar).
2. **STARK intacte.** Després de l'extracció, el setup STARK genera **sortides idèntiques byte a byte** a les referències *golden*.
3. **Witness mutat.** Amb un witness mutat, `check` indica la fila que falla i `verify` rebutja la prova.
4. **Manipulació.** Una prova o uns publics manipulats es rebutgen, i també una vkey modificada (el *digest* no quadra).
5. **Tres implementacions, un resultat.**
   - A totes les files, els numeradors de les restriccions calculats pel bytecode del prover coincideixen amb els d'un recorregut directe del `pilout`.
   - En punts aleatoris fora de `H`, el `Q(ξ)` del codi del verificador coincideix amb el d'aquest recorregut.
6. **Agrupació.** `group` coincideix amb el sistema antic en els exemples de pil-stark.

### Fase 2: busos de la std en una instància

**Abast:**
- **Std BN254:** `bn254.pil` i `ACTIVE_FIELD` (§4.1).
- **Reptes de l'stage 2.**
- ***Hints* del prover:** `gsum_col`, `gprod_col`, `im_col` i `im_airval`, amb la semàntica de `calculateImHints` i `calculateWitnessSTD` (`pil2-stark/src/starkpil/gen_proof.hpp:27, 57`).
- **La std en mode `STD_MODE_ONE_INSTANCE`** (D2): els busos es tanquen dins de l'AIR, sense airgroup values ni restriccions globals.
- **Partició de `Q`** (`--max-q-degree`), amb blinding i avaluacions dels trossos.

**Fixtures** (M34, a `pilfflonk/tests/fixtures/`; on viu cadascuna i en què difereix del PIL1, a l'Annex G):
- els exemples Plookup, Permutation i Connection de l'Annex G, portats a PIL2 amb la std: `lookup_assumes`/`lookup_proves`, `permutation_assumes`/`permutation_proves` i `connection`, aquesta amb el `k_coset` i el `GEN[BITS]` de BN254 (M29);
- un range check, que pil-fflonk no té: una consulta a una taula fixa de la mateixa AIR. El `range_check` de la std no hi serveix, perquè posa la taula en una AIR pròpia (Annex F.12);
- cadascun en variant de bus de suma i de bus de producte, dins d'una sola AIR (`<exemple>_sum.pil` i `<exemple>_prod.pil`). El lookup de la std només té el bus de suma, perquè un bus de producte no admet multiplicitats; les variants de producte del Plookup i del range check fan servir la permutació de la std (Annex G);
- l'exemple `all` de l'Annex G, amb Fibonacci, connection, permutation i plookup en una sola AIR i els publics `[1, 2, out]` de pil-fflonk (`runtime/public.json`), també en les dues variants.

**Validació** (`cli/tests/pilfflonk_prove.rs` i `pilfflonk_check.rs`):
1. Totes les fixtures proven i verifiquen, empaquetades i amb `--no-packing`, i també amb `Q` partit les que tenen `qDeg ≥ 2`. La Permutation i el range check en bus de suma tenen `qDeg = 1` sigui quin sigui el `--max-constraint-degree`: la cerca d'A.1 hi tria un im pol i `qDeg = 1` abans que cap im pol i `qDeg = 2` (el desempat, el grau més baix), i un `Q` d'un tros no es parteix (M33).
2. Amb una multiplicitat incorrecta, o una permutació, una connexió o un rang trencats, `check` indica la restricció del bus que falla, la de l'última fila (`__L1__'·(0 − gsum)` o `__L1__'·(1 − gprod)`), i el prover refusa el witness (`UnsatisfiedError`): no en surt cap prova que `verify` hagi de rebutjar. `verify` rebutja qualsevol canvi a una prova bona o als seus publics.
3. Les columnes de l'stage 2 coincideixen amb una referència seqüencial ingènua (l'oracle), tant les del prover com les de `check`.

### Fase 3: witness en `Fr` i rendiment

**Abast:**
- una font de witness en `Fr` per a programes reals (D4);
- un informe de temps i memòria.

**Validació:**
1. Els exemples de pil-fflonk es proven amb el witness que calcula la biblioteca, sense fitxers. **M38c:** el Fibonacci, la Connection i `all` (les quatre màquines d'estats de pil-fflonk en una AIR), en bus de suma i de producte, amb `--witness-lib`, i la prova és la del directori del generador (`cli/tests/pilfflonk_prove.rs`). La Permutation i el Plookup per separat no tenen biblioteca pròpia: les seves columnes les calcula la d'`all`.
2. Es genera un informe de temps i memòria fins al límit de P2 (`N ≤ 2^24`). **M39:** l'Annex H.

### Fase 4: verificador Solidity

**Abast:** generació de `pilfflonk.verifier.sol` amb `tera` a partir de `pilfflonk.vkey.json`, com fa snarkjs amb la seva vkey, i el codificador de calldata.

**Validació:**
1. Foundry, l'entorn de proves de Solidity, accepta totes les proves de les fases 1 a 3. **Coberta a M41.**
2. Amb *fuzzing* diferencial sobre proves mutades, el verificador JS i el Solidity coincideixen sempre a l'hora d'acceptar o rebutjar. **Coberta a M42.**
3. Es genera un informe de gas. **Coberta a M42** (Annex I).

**M40** (§4.5, "Verificador Solidity"): `--solidity`, `pilfflonk-solidity`, la plantilla i la definició del calldata, amb Foundry v1.8.3 i solc 0.8.37, i les quatre correccions de la seva revisió de seguretat (`X_2`, els *offsets*, el transcript sense avaluacions i el que tornen els precompilats). Per a la validació 1, Foundry ja accepta les proves de deu claus de les fases 1 i 2 (el Fibonacci, l'empaquetat, els *offsets* amb signe, `all` en bus de suma, `Q` partit i els tres dominis que no són `everyRow`), les refusa mutades i coincideix amb el JS en tots els casos; la resta de les *fixtures* i el codificador de la CLI són de M41, i les validacions 2 i 3, de M42.

**M41** (§4.5, "Codificador de calldata" i "Validació de M41"): `proofman-cli pilfflonk calldata`, el codificador del calldata, a `proofman_pilfflonk::calldata`. És com el `zkey export soliditycalldata` de snarkjs: refà el transcript i afegeix les inverses auxiliars. Foundry accepta les proves de les 73 claus de totes les *fixtures* de les fases 1 a 3, amb el calldata de la CLI, i coincideix amb el JS en les proves canviades. **La validació 1 queda coberta.** Les validacions 2 i 3 són de M42.

**M42** (§4.5, "Validació de M42", i l'Annex I):
- **El *fuzzing* diferencial entre el JS i el Solidity:** 25 famílies de mutacions sobre 13 claus, a la mida de la CI (403 casos) i ampliat (10.400). No hi ha cap discrepància. El JS i Foundry coincideixen en el veredicte i en la comprovació que refusa cada cas, i el contracte retorna `false`: només reverteix amb el calldata curt.
- **L'informe de gas:** les 73 claus, on va el gas, un model corregit, la comparació amb snarkjs i les oportunitats.

**Les validacions 2 i 3 queden cobertes: ho estan totes les de la Fase 4.**

### Fase 5: GPU

Caldrà, però després d'una versió CPU funcional (§7.1). L'informe de rendiment de la Fase 3 en fixa les prioritats.

**Abast:** fer servir `pil2-stark/src/bn128/src/{msm,ntt}` quan s'executa amb `--gpu` i la compilació ha detectat `nvcc`. La MSM de GPU accepta escalars en forma de Montgomery (`mont=true`). Si cal, també es porta l'intèrpret a GPU.

**Validació:** els resultats són idèntics bit a bit als de CPU, i es genera un informe de l'acceleració.

### Fora d'abast: diverses instàncies (D2)

Cada prova té una sola instància d'una sola AIR. Diverses instàncies, diverses AIRs, els air, airgroup i proof values, l'agregació i les restriccions globals queden fora d'abast (usuari, 29-09-2026). Els formats ja deixen el lloc (el transcript absorbeix el nombre d'instàncies, i els noms de la prova preveuen prefixos), però no s'implementen.

---

## 7. Què està decidit i què no

### 7.1 Decisions preses

| Decisió | Origen |
|---|---|
| El setup és en Rust, al costat del setup STARK, i en reutilitza les passades | Usuari |
| L'aritmètica BN254 es fa amb ffiasm (C++). No es fa servir arkworks. | Usuari |
| La base és `pil2-proofman` `pre-develop-1.4.0-alpha`, i la del compilador, `develop-0.14.0` amb el `pilout` v1. El v2 queda fora d'abast. | Usuari |
| La branca `feat/pil2-fflonk` no es fa servir com a referència | Usuari |
| La sortida del setup té la forma d'un `provingKey/` com el del STARK | Usuari |
| El prefix dels components nous és `pilfflonk` | Especificació (evita col·lisions) |
| **P7:** la llicència és l'actual | Usuari (29-09-2026) |
| **D1:** no és un camí totalment paral·lel. El codi que es pugui compartir és una dependència allà on calgui: les passades simbòliques passen al crate `pil-info`, i ffiasm, rapidsnark (el transcript) i les utilitats es fan servir com a dependència | Usuari (29-09-2026) |
| **P1:** l'objectiu són proves directes que millorin les agregacions | Usuari (29-09-2026) |
| **P6:** el transcript és exactament el del FFLONK existent (`Keccak256Transcript` de rapidsnark) | Usuari (29-09-2026) |
| **D2 / abast de la v1:** de moment es reprodueix el que fa pil-fflonk, però amb PIL2: una AIR, una instància i una sola prova. Com a pil-fflonk, no hi ha air values, airgroup values, proof values ni restriccions globals; la std es fa servir en mode `STD_MODE_ONE_INSTANCE` (`std_constants.pil:13`), que tanca els busos dins de l'AIR. Diverses instàncies, i per tant diverses AIRs, queden **fora d'abast**: cada prova té una sola instància d'una sola AIR. | Usuari (29-09-2026) |
| **D4:** (a), un tipus `Fr` al crate `fields` i una biblioteca de witness en `Fr`, a la Fase 3 | Usuari (29-09-2026) |
| **Biblioteca de witness (M38b):**<br>- una biblioteca de witness de pilfflonk és una biblioteca dinàmica, com les del STARK (la CLI, a M38c);<br>- `trace_row!` només demana `PrimeField64` als accessors tipats, i `pil-helpers` genera files sobre el camp per als `pilout` BN254. De moment, pilfflonk no accepta files de traça empaquetades (les dels *hints* `witness_bits`), i l'empaquetat polinòmic del setup continua sent el per defecte;<br>- els public inputs arriben amb `--public-inputs <json>`, que llegeix la biblioteca, com al STARK (M38c). | Usuari (30-09-2026) |
| **D5:** es busquen els graus de 2 a 9 (com pil-stark) i es tria el que minimitza `nImPols + qDeg`; `--max-constraint-degree` canvia el límit | Usuari (29-09-2026) |
| **P2:** `2^28` és el grau màxim que hi pot haver, el del `ptau` més gran disponible (`powersOfTau28_hez_final.ptau`). No és el `ptau` que es farà servir: només fixa el límit superior. Amb la 2-adicitat de BN254, això vol dir `N·2^extendBits ≤ 2^28` i grau de cada `f_i` `< 2^28`: amb `qDeg` fins a 8, `N ≤ 2^24`.<br>El `ptau` és una entrada del setup (`--powers-of-tau`). Els de la cerimònia Hermez són a la llista del README de snarkjs (`github.com/iden3/snarkjs`). **No se'n baixa cap**, perquè són molt grans. Els tests generen un `ptau` petit amb una `τ` fixa, amb ffiasm i sense JS (N13 del pla). | Usuari (29-09-2026) |
| **P5:** el que tingui pil-fflonk, és a dir, cap custom commit. El setup els rebutja. | Usuari (29-09-2026) |
| **P8:** es fa servir el compilador local (`PIL2C_EXEC`). `package.json` només es fixa si cal. La branca del compilador encara no es fa *commit*: si cal, s'ha de demanar a l'usuari. | Usuari (29-09-2026) |
| **P10:** d'acord amb afegir la std de BN254 (`bn254.pil`, `ACTIVE_FIELD`). Entra a la v1 quan es portin els exemples de connexió de pil-fflonk. | Usuari (29-09-2026) |
| **D6:** el blinding sempre és actiu, també als tests i a la CI, de manera que tots tenen els mateixos graus i el mateix *layout*. Als tests i a la CI el blinding és **fix**: el generador rep una llavor fixa (`randombytes_buf_deterministic`) i la prova surt igual a cada execució. A producció el blinding és aleatori (libsodium). La llavor fixa només s'accepta si es demana explícitament, perquè treu el *zero-knowledge*. | Usuari (29-09-2026) |
| **Prioritat:** primer una versió CPU funcional; després, de manera incremental, GPU i millores. La GPU caldrà (Fase 5), però més endavant. | Usuari (29-09-2026) |
| **Reutilització:** abans de copiar codi de pil-fflonk, cal comprovar que no existeixi ja en aquest repositori o en una dependència | Usuari (29-09-2026) |
| **Referència JS:** `../pil-stark`, una còpia retallada amb només el necessari. Només es llegeix. | Usuari (29-09-2026) |
| **`inv` de la prova (29-09-2026):** la prova porta `inv`, com la de snarkjs (`fflonk_prove.js:245`): la inversa del producte de tots els denominadors que inverteix el verificador a la comprovació SHPLONK, en un ordre fix. A diferència del verificador JS de snarkjs, el de pilfflonk el recalcula i rebutja la prova si no quadra, de manera que la prova no és mal·leable. La definició exacta la fixa M18. | Usuari (29-09-2026) |
| **D3:** no hi ha verificador natiu, igual que el FFLONK existent, que es verifica amb snarkjs. No es porta cap *pairing* a C++, i ffiasm no es toca. | Usuari (29-09-2026) |
| **D8:** el verificador és JS, com el del FFLONK existent. S'adapta del de pil-fflonk (`fflonk_verify.js` + `verifyOpenings`) als canvis de pilfflonk, viu a `pilfflonk/js/` i el crida `proofman-cli pilfflonk verify` (§4.5). | Usuari (29-09-2026) |
| **P4:** pil-fflonk es deixarà de fer servir en favor de pilfflonk. No cal cap compatibilitat amb els seus formats ni amb el seu verificador; del seu sistema es manté la forma de la prova (D7). | Usuari (29-09-2026) |
| **Vkey autocontinguda:** el setup escriu `pilfflonk.vkey.json`, com la `verification_key.json` de snarkjs, i el verificador rep vkey, publics i prova, com `snarkjs fflonk verify`. El *digest* és el de la vkey (A.6). | Usuari (29-09-2026) |
| **P9:** `proofman-setup compile-pil` rep un paràmetre nou, `-P, --config <json>`, amb el mateix nom i el mateix fitxer que el `-P, --config` de `pil2com`, i el passa tal qual. Per defecte no hi és, i el comportament és l'actual (Goldilocks). | Usuari (29-09-2026) |
| **D7:** el format de la prova és equivalent a l'actual:<br>- la prova són bytes: primer els commitments G1 (`x‖y`) i després les avaluacions, tot en *big-endian* i en un ordre fix;<br>- la vista JSON és d'estil snarkjs: `protocol`, `curve`, `polynomials` (`[x, y, "1"]`) i `evaluations`, com a pil-fflonk (`shplonk.cpp:948-976`) i al *wrap* final (`pil2-stark/src/starkpil/final_snark_proof.hpp`, `snark_proof_to_json`);<br>- els publics són un array de cadenes decimals. | Usuari (29-09-2026) |
| Compilador:<br>- C1 amb `config.prime` al `-P`;<br>- els 8 bytes forçats a `9fdab4a` s'expliquen perquè sempre es feia servir Goldilocks;<br>- els canvis van a la branca `develop-0.14.0-pil2-fflonk`;<br>- s'hi afegeixen tests BN254 (els dos que ja fallaven queden fora d'abast);<br>- s'accepta la limitació de `fixed-to-file` i `extern_fixed_file`;<br>- s'hi afegeixen proteccions contra el truncament (C4). | Usuari |

### 7.2 Propostes pendents de confirmar

No n'hi ha cap.

### 7.3 Preguntes obertes

No n'hi ha cap. Totes les preguntes (P1–P10) estan decidides (§7.1).

**Context de P7 (llicències).** La llicència és l'actual. Per a la traçabilitat, es mantenen les capçaleres del codi que es porta.
- El ffiasm 0.1.5 vendoritzat ja és GPL-3.0.
- pil-fflonk, d'on s'adapta codi, té `LICENSE` AGPL-3.0 i, des de `0132359`, també `LICENSE-APACHE` i `LICENSE-MIT`.
- `pil2-stark/LICENSE` és AGPL-3.0, mentre que el *workspace* declara `MIT OR Apache-2.0`.

### 7.4 Riscos

| Risc | Mitigació |
|---|---|
| Extreure i parametritzar les passades simbòliques trenca el setup STARK | Referències *golden* noves de totes les sortides, generades abans de l'extracció, i sortides idèntiques byte a byte com a porta de la Fase 1 |
| L'agrupació amb *offsets* arbitraris és complexa: moltes classes, `k` limitat per la 2-adicitat, grau limitat per l'SRS | Una funció pura amb tests de propietats (cada parella `(polinomi, offset)` coberta, un sol stage per `f`, determinisme); tests *golden* contra el sistema antic; errors explícits al setup |
| Un error de *soundness* compartit entre prover i verificador | Commitments fixos presos només de la vkey; *digest* de la vkey; camí de verificació propi; contrast amb el `pilout`; Solidity com a segon verificador; revisió externa de les equacions |
| El transcript del prover (C++) i el del verificador (JS) divergeixen | És la mateixa parella que el FFLONK existent: el `Keccak256Transcript` de rapidsnark al prover i snarkjs al verificador. Test creuat C++ ↔ JS a la Fase 0 (validació 4). |
| Col·lisions de símbols, capçaleres o noms dins de `libstarks`, o fonts que no s'arriben a compilar | Prefix `pilfflonk` i namespace `PilFflonk` a tot arreu; entrades a les tres llistes de fonts del Makefile; test d'enllaç a CPU i a GPU |
| Adaptar el codi de pil-fflonk n'arrossega els defectes | Llista de defectes a corregir (Annex C.3); tests del SHPLONK aïllat abans d'integrar-lo; revisió sota ASan/UBSan |
| El compilador corromp valors amples, a vegades sense avisar | C2–C4, ja fets; els tests BN254 del compilador; comprovació del `baseField` i de les constants `≥ r` al setup |
| L'esquema dels fitxers divergeix entre Rust i C++ | Un sol propietari (`proofman-pilfflonk`) i tests d'anada i tornada Rust ↔ C++ |
| Rendiment i memòria en CPU | La versió CPU va primer i ha de ser funcional, no òptima. Informe a la Fase 3 (Annex H), avaluació de `Q` per parts (M39), i la GPU (Fase 5) sobre el codi MSM/NTT que ja existeix. |
| Discrepàncies de codificació amb Solidity (*endianness*, punt a l'infinit, reducció mòdul `r`) | Codificació fixada a l'Annex A.4 i vectors de test des de la Fase 0. A M40, Foundry i el JS coincideixen en proves reals, mutades i amb valors fora de rang (§4.5); a M42, el *fuzzing* diferencial (10.400 casos de 25 famílies sobre 13 claus) no hi troba cap discrepància |
| La regla d'agrupació generalitzada no es comporta bé amb conjunts d'*offsets* grans | Tests de propietats i mètriques de grau per a les fixtures de les fases 1 a 3; la regla és una sola funció pura, fàcil de canviar |

---

## Annex A. Protocol (normatiu)

### A.1 Polinomi de restriccions

Per a cada AIR, amb les `n` restriccions en l'ordre del `pilout`:

```
Q(X) = Σ_{i=0..n−1} std_vc^(n−1−i) · c_i(X) / Z_{D_i}(X)
```

És el plegat de Horner del STARK (`setup/pil2-stark/src/pil/constraint_poly.rs:136-147`): `acc = acc·std_vc + c_i·(Z_H/Z_{D_i})`, i al final es divideix per `Z_H`.

| Domini | `Z_D(X)` |
|---|---|
| `everyRow` | `X^N − 1` |
| `firstRow` | `X − 1` |
| `lastRow` | `X − ω^{N−1}` |
| `everyFrame{min,max}` | `(X^N − 1)` dividit pels factors `(X − ω^j)` de les files excloses |

El compilador de `develop-0.14.0` només emet `everyRow` (§3.4). Els altres tres dominis s'implementen perquè el format els preveu, i es proven amb `pilout` sintètics.

**Grau:**
- **`qDeg`.** `qDeg = max_i(deg c_i + δ_i) − 1`, en unitats de `N`, després d'introduir els im pols. Aquí `δ_i = 1` si la restricció no és `everyRow`, i `0` si ho és: el factor `Z_H/Z_{D_i}` suma gairebé `N` al grau (`constraint_poly.rs:245-250`).
- **Política de cerca.** Amb la política de cerca `max = D`, el setup prova els graus de 2 a `D` i es queda amb el que minimitza `nImPols + qDeg`. En cas d'empat es queda el grau més baix, com pil-stark (`cp_prover.js`, comparació estricta), perquè el resultat coincideixi amb el del sistema antic. Per exemple, el Fibonacci de l'Annex G acaba amb 1 im pol i `qDeg = 1` (grau 2), i no amb `qDeg = 2` sense im pols (grau 3), que costa el mateix.
- **Coeficients de `Q`.** Si `|O|_max` és el màxim de `|O|` de les columnes que tenen blinding (després de les fusions), `Q` té com a molt `qDeg·N + (qDeg+1)·|O|_max + 1` coeficients.
- **Domini estès.** És la potència de dos més petita que és `≥` el nombre de coeficients de `Q` i `≥` `N + |O|_max + 1`, perquè les columnes amb blinding també s'hi estenen (troballa de M5: amb `qDeg` petit, la primera condició sola podria donar un domini massa petit). Ha de complir `nBitsExt ≤ 28`. És el de `Q` sencer també quan es parteix: el prover calcula `Q` sencer i després el parteix (M33).

**Partició de `Q`.** Per defecte `maxQDegree = 0`, i `Q` no es parteix. Si `maxQDegree > 0` i `qDeg > maxQDegree`:
- **Trossos.** `Q` es parteix en `m = ⌈qDeg/maxQDegree⌉` trossos `Q_0 … Q_{m−1}` de `M·N` coeficients (`M = maxQDegree`): el tros `i` té els coeficients `i·M·N … (i+1)·M·N − 1` de `Q`, i l'últim, els que queden fins a la fita de `Q`, de manera que `Q(X) = Σ_i X^(i·M·N)·Q_i(X)`. Formen un grup propi de l'agrupació (A.2, regla 2): un `f`, que `extraMuls` pot partir en diversos `f`, com al sistema antic (M33).
- **Blinding.** Els trossos en reben com a PLONK: cada frontera entre els trossos `i` i `i+1` rep dos coeficients aleatoris `b_0, b_1` que es compensen, `b_0·X^(M·N) + b_1·X^(M·N+1)` sumats al tros `i` i `b_0 + b_1·X` restats del tros `i+1`. Per això cada tros llevat de l'últim té `M·N + 2` coeficients, i l'últim, la fita de `Q` menys `(m−1)·M·N`, que són almenys `N + 1` (A.3; M33, `proofman_pilfflonk::degrees::QSplit`).
- **`maxQDegree` a les claus.** El `pilfflonkinfo` i la vkey tenen `maxQDegree = M` si `Q` es parteix, i `0` si no (`M = 0` o `qDeg ≤ M`), com el sistema antic (`fflonk_shkey.js:162-163`): `maxQDegree > 0` vol dir que `Q` està partit, i el `globalInfo` guarda l'opció (M33).
- **Avaluacions.** Els valors `Q_i(ξ)` van a la prova i s'absorbeixen al transcript (A.4).
- **Comprovació.** El verificador comprova que `Σ_i ξ^(i·M·N)·Q_i(ξ) = Q(ξ)`.

**Sense partició:**
- El prover calcula `Q` sobre el *coset* estès.
- El verificador no rep `Q(ξ)`, sinó que el calcula a partir de les avaluacions i de `Z_D(ξ)`.

### A.2 Agrupació

**Entrada.** Els polinomis compromesos, cadascun amb `{nom, stage, fita de grau, conjunt d'offsets O}`.
- **Unitat de grau.** Totes les fites de grau es donen en **nombre de coeficients**:
  - constants: `N`;
  - columnes amb blinding: `N + |O| + 1`, amb l'`O` final del seu `f`;
  - `Q`: A.1.

  El sistema antic feia servir el grau inclusiu per a `Q`, que és un de menys. Això no afecta l'agrupació, perquè `Q` va sempre sol, però sí la mida de l'SRS.
- **Columnes que no s'obren.** Les columnes que no s'obren a cap punt no es comprometen, i el setup n'emet un avís.

**Regles:**
1. **Classes i fusió.** Es generalitza `fixFIndex` (`pil-stark/src/fflonk/helpers/fflonk_shkey.js:244-290`, `minPols = 3`):
   - els polinomis es classifiquen per `(stage, O)`;
   - per a cada stage, sigui `U` la unió dels `O` de les seves classes;
   - tota classe amb menys de `minPols` polinomis i `O ≠ U` passa a `U`, i es fusiona amb la classe `U` si ja existeix;
   - les classes amb `O = U` no es mouen mai.

   Per a `O ⊆ {0, 1}`, la regla coincideix exactament amb el sistema antic. Per exemple, `{0}:4, {0,1}:2` es queda en dues classes, i `{0}:5, {1}:1` dona `{0}×5` i `{0,1}×1`.
2. **Q.** `Q`, o els seus trossos, forma un `f` propi.
3. **Repartició d'`extraMuls`.** Es generalitza `applyExtraScalarMuls` (`shplonkjs/src/helpers/setup.js:212-250`).
   - **Nombre de `f`.** El total és `#grups + extraMuls`. Cada grup `g` es parteix en `c_g + 1` trossos consecutius, amb `Σ c_g = extraMuls`.
   - **Mida vàlida.** Cada tros té una mida `k` que ha de complir `k | r−1` i `v₂(k) + nBits ≤ 28`, és a dir, `kN | r−1`. shplonkjs no ho comprova (`setup.js:341-353`); aquí sí que es comprova.
   - **Particions possibles.** Només s'enumeren les seqüències de mides que no decreixen (`shplonkjs/src/utils.js:39-53`).
   - **Cost.** El d'un tros és `max_j(deg_j·k + j)`, i el d'un grup és el màxim dels seus trossos.
   - **Dins d'un grup.** Per a cada nombre de trossos, es tria la partició de cost mínim; si n'hi ha diverses, la primera que s'enumera (`setup.js:19-45`).
   - **Entre grups.** Cada combinació dona un vector amb el cost de cada grup. Aquest vector s'ordena de més gran a més petit i es compara lexicogràficament. Una combinació només substitueix la millor si és estrictament millor (`setup.js:87-99`).
   - **Errors.** Si cap partició és vàlida, o si `extraMuls > #pols − #grups`, el setup dona un error.
4. **Composició:** `f_i(X) = Σ_j p_j(X^k)·X^j`. Dins de cada grup, els polinomis van en ordre invers d'inserció (`setup.js:183`).
5. **Arrels.** No es desen: es deriven en carregar.
   - `w_k = 5^((r−1)/k)`. El 5 és el no-residu quadràtic més petit, i és el que fan servir ffjavascript i la FFT d'ffiasm.
   - `powerW` és el mínim comú múltiple de tots els `k` de tots els `f_i`, i `ξ = xiSeed^powerW`.
   - Per a l'*offset* `s`, les arrels són els `x` tals que `x^k = ξ·ω_N^s`, és a dir, `x_j = xiSeed^(powerW/k) · ω_{kN}^s · w_k^j`. També val per a `s` negatiu.
6. **Compatibilitat.** Per a `O ⊆ {0, 1}`, el resultat (classes, particions, ordre i arrels) ha de coincidir exactament amb el del sistema antic.

**Com ho implementa M21 (`setup/pilfflonk/src/grouping.rs`), i on el JS antic no coincideix amb aquest annex:**
- **Ordre global dels `f`.** El JS els numera per primera aparició a les llistes d'*offsets*, que no sempre va per stage (per exemple, amb 3 o més columnes fixes obertes només a 1). A.5 demana ordre per stage: `group()` ordena de manera estable per stage, però enumera els desempats en l'ordre antic dels grups, de manera que les particions són les del sistema antic. Per a l'exemple `all` tots dos ordres coincideixen.
- **Els trossos de `Q`.** El JS els posa tots en un mateix grup, que `extraMuls` pot partir en diversos `f`; M21 fa el mateix, i M33 ho manté (regla 6). Dins del grup van en ordre invers (regla 4): `Q_{m−1}` primer.
- **Un sol `f`.** El JS n'exigeix com a mínim dos (`shplonk.js:27`); `group()` n'accepta un.
- **`kN | r−1`.** Les mides invàlides no s'arriben a enumerar; el resultat és el mateix que comprovar-ho després.
- **Generalització a *offsets* amb signe:** les llistes van per *offset* creixent (−1 primer), i els moviments de la regla 1 per stage, per `O` lexicogràfic i per ordre d'entrada; per a `O ⊆ {0, 1}` es redueix al JS.
- **Errors clars** (`NoValidPartition`, cota 0) on el JS falla amb una excepció. Una classe de 5, 7, 10, 11, … columnes no pot ser un sol tros, i amb `extraMuls = 2` tres classes així no tenen partició vàlida: el missatge suggereix més `--extra-muls`.

**Com l'usa el setup (M22, `setup/pilfflonk/src/layout.rs`):**
- **Per defecte s'agrupa** amb `--extra-muls` (per defecte 2). `--no-packing` (només per a tests) no fa cap fusió: cada polinomi va sol en un `f` de `k = 1`, amb els seus *offsets*, i `--extra-muls` no es fa servir.
- **`|O|_max` després de les fusions (C.3.2).** El setup crida `fuse()` abans de fixar la fita de `Q`: `|O|_max`, la fita de `Q` i `nBitsExt` (A.1) surten dels *offsets* fusionats, els mateixos que el prover llegeix del *layout*. Una columna fusionada té la fita dels *offsets* del seu `f` (`N + |O_f| + 1`, A.3).
- **L'`evMap`.** Una fusió obre una columna en *offsets* on les passades no l'obrien. L'`evMap` és el de `pil-info`, tal com és, seguit dels parells `(columna, offset)` que el *layout* obre i que no hi eren, ordenats entre ells com `pil-info` ordena els seus (per punt d'obertura, les fixes primer, per `id`). Així els índexs de les entrades de `pil-info`, que són els operands `eval` del `qVerifier` (A.6), no canvien. La prova, el transcript (A.4, pas 4) i el verificador llisten les avaluacions en l'ordre de l'`evMap`, les fixes primer: un parell afegit d'una columna fixa va després de les altres fixes, i un d'una columna compromesa, després de les altres compromeses, al mateix lloc per a tots tres.
- **Cota de la cerca.** Les dues enumeracions de la regla 3 són exhaustives, com al sistema antic, perquè els desempats siguin els seus, i creixen molt de pressa amb `extraMuls`. Abans d'enumerar res, el setup compta els passos que faria (el recompte mateix, les particions que avalua, cadascuna al cost de la longitud del grup, i els nodes i les combinacions completes del recorregut entre grups) i, si passen de `2^26` (menys d'un segon compilat en *release*), dona l'error `SearchTooLarge`, que diu que cal abaixar `--extra-muls`; no poda res. Amb el valor per defecte, un grup de 500 columnes hi cap de sobres.
- **Errors de `--extra-muls`:** `TooManyExtraMuls` (més de `#pols − #grups`), `SearchTooLarge` i `NoValidPartition`, que diu fins a quant es pot pujar `--extra-muls`. El setup s'atura abans d'escriure cap fitxer.

### A.3 Blinding

Tota columna compromesa no fixa `p` amb conjunt d'obertura `O` es transforma així:

```
p'(X) = p(X) + (X^N − 1)·b(X)
```

- `b(X)` té `|O|+1` coeficients aleatoris, i s'afegeix en forma de coeficients, després de la INTT.
- Les constants no tenen blinding.
- `Q` sense partir no té blinding: la seva fita de grau ja inclou la contribució del blinding de les columnes (A.1).
- Els trossos de `Q` partit reben el blinding de PLONK (A.1), com el sistema antic (`pil-fflonk/src/pilfflonk_prover.cpp:697-720`), sense el defecte C.3.4. Els factors surten del mateix generador, després dels de les columnes: frontera per frontera, `b_0` abans que `b_1` (M33).
- El generador aleatori s'injecta. Als tests i a la CI rep una llavor fixa i el blinding és fix; a producció és aleatori (D6). Implementació (M18, `pilfflonk_rng`): sense llavor, `randombytes_buf` de libsodium; amb llavor, blocs de 4096 bytes de `randombytes_buf_deterministic` amb una subllavor per bloc (BLAKE2b-256 de l'índex, amb la llavor com a clau). Cada element de `Fr` es treu per rebuig (32 bytes amb els dos bits alts a zero, acceptat si és `< r`). A la CLI, `--insecure-blinding-seed <64 hex>`, que avisa que la prova no és *zero-knowledge*.

### A.4 Transcript (Keccak-256)

Es fa servir el `Keccak256Transcript` de `pil2-stark/src/rapidsnark/`, el mateix que el FFLONK existent (P6). Aquest annex en descriu la codificació i fixa la seqüència d'absorcions pròpia de pilfflonk.

**Codificació al transcript:**
- `Fr`: 32 bytes *big-endian* en forma canònica.
- G1: `x‖y`, 64 bytes *big-endian*. El punt a l'infinit i els punts amb alguna coordenada `< 2^192` no s'absorbeixen mai: el prover els rebutja, perquè `Keccak256Transcript` no els codifica així (§4.4). A la pràctica no es donen.
- Enters: es codifiquen com a `Fr`.

**`squeeze`:** `h = keccak256(buf) mod r`, i després `buf := enc(h)`. Això és equivalent al `reset()` + `addScalar(h)` del sistema antic.

**Ordre global.** És l'ordre canònic de les AIRs i de les instàncies (glossari). L'ordre global dels `f_i` és el d'A.5.

**Seqüència, amb un únic transcript per prova:**
1. Absorbir:
   - el *digest* de la clau (A.6), com a `Fr`;
   - el nombre d'instàncies de cada AIR, en ordre canònic d'AIRs;
   - els publics.
2. Per a cada `s = 1 … nStages`:
   1. absorbir els commitments dels `f_i` no fixos de l'stage `s`, en l'ordre global;
   2. absorbir, en ordre canònic, els air values de l'stage `s` de cada instància; després, els airgroup values de l'stage `s`; i després, els proof values de l'stage `s`;
   3. si `s < nStages`, fer `squeeze` dels `numChallenges` reptes de l'stage `s+1`, un per crida.
3. Fer `squeeze` de `std_vc` (l'stage `nStages+1`). Absorbir els commitments de `Q` en l'ordre global i fer `squeeze` de `xiSeed`, que és el `std_xi` de l'stage `nStages+2`.
4. Absorbir les avaluacions:
   1. per a cada AIR en ordre canònic, les de les seves columnes fixes, en l'ordre de l'`evMap`;
   2. per a cada instància en ordre canònic, les de la resta de columnes, en l'ordre de l'`evMap`;
   3. si `Q` està partit, a continuació, els `Q_i(ξ)` de cada instància, en l'ordre del seu *layout*: els `f` de l'stage de `Q` en ordre i, de cada un, els seus trossos en ordre (M33; els trossos es reconeixen pel nom, `Q<i>`, A.6).
5. SHPLONK: fer `squeeze` d'`α_S`, absorbir `W` i fer `squeeze` de `y`.

`α_S` és el repte de SHPLONK, i no té res a veure amb l'α de PIL1.

**Diversos stages (M30).** Amb els busos de la std, `nStages = 2` i `numChallenges = [0, 2]`: després de l'stage 1 es treuen `std_alpha` i `std_gamma`, en l'ordre de `stageId`, i el C++ calcula l'stage 2 amb ells. Els reptes del prover (`ProofChallenges::stages`) són els que el verificador JS refà (`computeChallenges`) sobre la mateixa prova, i un test creuat ho comprova (`cli/tests/pilfflonk_prove.rs`, `the_transcripts_agree`), a més de `std_vc` i `xiSeed`.

### A.5 Verificació SHPLONK

**Ordre global dels `f_i`:**
1. Primer, els `f_i` fixos de cada AIR que té alguna instància, en ordre canònic d'AIRs i en l'ordre del seu *layout*. Com que són comuns a totes les instàncies de l'AIR, s'obren una sola vegada.
2. Després, per a cada instància en ordre canònic, els seus `f_i` no fixos en l'ordre del *layout*: per stage ascendent, i `Q` al final.

`f_0` és el primer d'aquesta llista.

La comprovació final és:

```
e(F − E − J + y·W', [1]₂) = e(W', [τ]₂)
```

on:
- `F = [f_0] + Σ_{i≥1} q_i·[f_i]`;
- `E = (r_0(y) + Σ_{i≥1} q_i·r_i(y))·G1`;
- `J = q_0·[W]`;
- `q_0 = Z_{T_0}(y)` i `q_i = α_S^i·Z_{T_0}(y)/Z_{T_i}(y)`, on `T_i` és el conjunt d'arrels de l'`f_i` (A.2);
- `r_i` és el polinomi que interpola els valors de l'`f_i` a `T_i`. Per a cada arrel `x` de l'*offset* `s`, el valor és `f_i(x) = Σ_j p_j(ξ·ω^s)·x^j` (`shplonkjs/src/helpers/verifier.js:107-138`).

El prover fa servir la mateixa convenció. `Z_T` és el producte de tots els `Z_{T_i}`, repeticions incloses, i `W' = L/(Z_{T∖T_0}(y)·(X − y))` (`pil-fflonk/src/shplonk.cpp:83-101, 263-270`).

**Regla:** els commitments dels `f` fixos surten **sempre** de la vkey, mai de la prova.

**`inv` i `invZh` (M18).** `invZh = 1/Z_H(ξ)`, i el verificador comprova `Z_H(ξ)·invZh = 1`. `inv = 1/Π`, on `Π` és el producte d'aquesta llista, en aquest ordre (els `f_i` en l'ordre global, `n` el seu nombre):
1. `Z_{T_i}(y) = Π_{x∈T_i}(y − x)`, per a `i = 1 … n−1` (els denominadors dels `q_i`);
2. per a cada `f_i` (`i = 0 … n−1`) i cada arrel `x_m` de `T_i` en ordre *offset*-major (`x_{m·k+j} = xiSeed^{powerW/k}·ω_{kN}^{s_m}·w_k^j`): `(y − x_m)·Π_{l≠m}(x_m − x_l)` (els denominadors de Lagrange dels `r_i(y)`).

El verificador JS recalcula `Π` i rebutja la prova si `inv·Π ≠ 1` (decisió de l'usuari). Al Fibonacci són 13 factors.

**Referències:** `pil-stark/src/fflonk/helpers/fflonk_verify.js`, `shplonkjs/src/helpers/verifier.js` i `pil-fflonk/src/shplonk.cpp` (la banda del prover).

### A.6 Fitxers del `provingKey/`, de la prova i el *digest* (versió de format 1)

| Fitxer | Camps |
|---|---|
| `pilout.globalInfo.json` | **La part comuna de l'esquema STARK:** `name`, `airs`, `air_groups`, `aggTypes`, `nPublics`, `numChallenges`, `numProofValues`, `proofValuesMap`, `publicsMap`.<br>**Camps nous:** `"backend": "pilfflonk"`, `"field": "bn254"`, `"modulus"` (decimal), `"transcript": "keccak256"`, `"formatVersion": 1` i els paràmetres del setup.<br>**No hi són** `hash`, `curve`, `transcriptArity`, `aggregationArity`, `latticeSize` ni `hasCompressedFinal`. Per això `common::GlobalInfo` el rebutja. |
| `pilout.globalConstraints.json` | El format del STARK (el genera `pil-info`) |
| `<air>.pilfflonkinfo.json` | **Camps equivalents del `starkinfo`:** `nStages`, `nConstants`, `cmPolsMap`, `constPolsMap`, `challengesMap`, `airValuesMap`, `airgroupValuesMap`, `evMap`, `openingPoints`, `boundaries`, `qDeg`, `cExpId` i `mapSectionsN`, amb `qDim = 1`.<br>**Camps nous:** `nBits`, que al STARK és dins de `starkStruct`; `maxQDegree` (0 si `Q` no es parteix, A.1); i `layout`, una llista de `f_i {stage, pols, k, offsets, degree}`. El `degree` és el cost d'A.2 en nombre de coeficients, i l'SRS n'ha de tenir el màxim (no el màxim més 1).<br>**No hi són** `starkStruct` ni res de FRI.<br>L'ordre exacte dels camps i la forma de les entrades (mapes, `evMap`, *boundaries*) són els de `pilfflonk/src/pilfflonk_info.rs` (M12). L'`evMap` és el de `pil-info` seguit dels parells que afegeixen les fusions (A.2, "Com l'usa el setup"). |
| `<air>.expressionsinfo.json` | El format del STARK, amb dimensió 1 per a tots els operands i les constants en decimal |
| `<air>.verifierinfo.json` | El format del STARK: només `qVerifier`, sense `queryVerifier`. El llegeix el verificador JS. |
| `<air>.bin` | Bytecode `Fr` del prover (im pols, `Q` i les expressions a què es refereixen els *hints*), de depuració de restriccions i els *hints* del prover, amb el format del `.bin` STARK i dimensió 1: contenidor `"chps"`, versió `0x7066_0003`, args de 32 bits i constants de 32 bytes *little-endian*. Vegeu "Format de `<air>.bin`", sota la taula (M11, M30). |
| `<air>.const` | Columnes fixes, fila per fila, en `Fr` canònic de 32 bytes *little-endian* |
| `<air>.verkey.json` | Els commitments G1 dels `f_i` fixos de l'AIR, com a cadenes decimals `[x, y]`. Són els mateixos que a la vkey. |
| `pilfflonk.srs.bin` | Contenidor binfile de rapidsnark, tipus `"pfsr"`, versió 1 (M6):<br>- **secció 1** (capçalera, 88 bytes): `u32 n8q = 32`, `q` (LE), `u32 n8r = 32`, `r` (LE), `u64 nG1` (entre 1 i `2^32−1`, el límit de la MSM), `u64 nG2 = 2`;<br>- **secció 2:** `[τ^i]₁` per a `i < nG1`, 64 bytes cadascun;<br>- **secció 3:** `[1]₂` i `[τ]₂`, 128 bytes cadascun (`Fq2` com a `c0‖c1`).<br>Els punts són afins `x‖y`, amb cada coordenada en Montgomery *little-endian*, copiats byte a byte de les seccions 2 i 3 del `ptau`. |
| `pilfflonk.vkey.json` | **Autocontinguda,** com la `verification_key.json` de snarkjs: tot el que necessita el verificador.<br>`protocol` (`"pilfflonk"`), `curve` (`"bn128"`), `formatVersion`, `nPublic`, `power` (`nBits`), `powerW`, `X_2` (`[τ]₂`), `numChallenges`, `evMap`, `layout` (els `f_i` amb `stage`, `pols: [{id, name}]`, `k`, *offsets* i `degree`), `boundaries` (el `qVerifier` hi fa referència per índex), els commitments fixos (`f<i>`, per índex del *layout*), `qDeg`, `maxQDegree`, el `qVerifier` i `digest` (`0x` i 64 dígits hexadecimals; al transcript entra com a `digest mod r`, llegit *big-endian*). `X_2` és `[[x.c0, x.c1], [y.c0, y.c1]]`, un punt de G2 que no és el punt a l'infinit (§4.5, revisió de M40). L'ordre dels camps és el de `pilfflonk/src/vkey.rs` (M12).<br>Els enters grans i els punts, com a cadenes decimals. |
| La prova (no és del `provingKey/`) | **Format equivalent a l'actual (D7):** bytes, com a `gen_final_snark_proof`, i una vista JSON d'estil snarkjs, com a `snark_proof_to_json`.<br>**Bytes, en ordre:**<br>- els commitments G1 (`x‖y`, 32+32 bytes *big-endian*) dels `f` no fixos, en l'ordre global (A.5);<br>- `W` i `W'`;<br>- les avaluacions (32 bytes *big-endian*): primer les fixes per AIR, després les de cada instància en l'ordre de l'`evMap`, i els `Q_i(ξ)` si `Q` està partit, en l'ordre del *layout* (A.4, pas 4);<br>- els air values, els airgroup values i els proof values;<br>- `inv` i `invZh`, com a pil-fflonk.<br>**JSON:** `{"protocol": "pilfflonk", "curve": "bn128", "polynomials": {nom: [x, y, "1"]}, "evaluations": {nom: valor}}`, amb els noms de pil-fflonk (`f<i>`, `W`, `Wp`, `<pol>`, `<pol>w`) i la mateixa convenció estesa per als altres *offsets* i instàncies.<br>**Noms de les avaluacions (M12, `pilfflonk/src/names.rs`):** el nom de la columna al mapa, amb `[i]` per a cada entrada de `lengths`, i el sufix d'*offset*: res per a 0, `w` per a 1 i `w` seguit del decimal amb signe per a la resta (`w2`, `w-1`). Els trossos de `Q` (`Q0` sense partir; `Q0`, `Q1`, … partit) van al final, en l'ordre del *layout*; el tros `i`, el que el verificador multiplica per `ξ^(i·M·N)`, és el que es diu `Q<i>` (al `cmPolsMap`, a l'`stageId` i l'`stagePos` `i`). pil-info anomena tots els im pols `<air>.ImPol`; el setup dona al `k`-è `lengths: [k]`, de manera que el seu nom és `<air>.ImPol[k]` i no col·lideixen (M16). Amb el mateix mecanisme, les columnes d'un mapa (`cmPolsMap` o `constPolsMap`) que comparteixen nom i no tenen `lengths`, com els `im_cluster` que la std declara dins d'un bucle (Annex F.12), són el vector d'aquest nom: el setup dona a la `k`-èsima, en l'ordre del mapa (el del `pilout`), `lengths: [k]`, i es diuen `im_cluster[0]`, `im_cluster[1]`, … (M34b). Un nom que té alguna columna amb `lengths` no canvia, perquè els elements d'un vector comparteixen nom i es distingeixen pels índexs. El compilador anomena les columnes de witness sense el prefix de l'AIR (`l1`, `l1w`) i les fixes amb el prefix (`Fibonacci.L1`). Amb diverses instàncies (després de la v1), els noms porten prefix: `<ag>.<a>:` per als fixos, `<ag>.<a>.<t>:` per als de cada instància i `<ag>:` per als airgroup values. Si dos noms encara col·lideixen (dues columnes d'un mapa o dels dos, els trossos de `Q` inclosos: dos vectors amb el mateix nom, o una columna que es diu com el setup n'anomena una altra, `im_cluster[0]` al costat de dos `im_cluster`), el setup ho rebutja i diu quines són.<br>**Diferència obligada:** els commitments fixos no hi són, perquè surten de la `verkey` (C.3.1). |
| `publics.json` | Un array de cadenes decimals, en l'ordre de `publicsMap`, com a pil-fflonk i al *wrap* final |
| Directori de witness (entrada del prover, no és del `provingKey/`; M13, `pilfflonk/src/witness.rs`) | Exactament aquests fitxers, i cap més:<br>- `instances.json`: un array no buit d'`{"airgroupId", "airId", "airValues": [...]}` en ordre canònic; els `airValues` són els de l'stage 1, en l'ordre de l'`airValuesMap` (buits a la v1);<br>- `instance_<ag>_<a>_<t>.bin`: les columnes de l'stage 1 d'una instància, sense capçalera, fila per fila, cada valor de 32 bytes *little-endian* canònic; la columna `c` de la fila `i` és al byte `(i·C + c)·32`, i el fitxer té exactament `N·C·32` bytes (`C` = columnes de witness de l'stage 1, sense els im pols);<br>- `publics.json` (com el de la prova) i `proof_values.json` (els de l'stage 1; buit a la v1).<br>Tots els valors JSON són cadenes decimals canòniques `< r`. El lector rebutja fitxers de més, mides incorrectes i valors `≥ r`. |

**Format de `<air>.bin` (revisió 3, M11 i M30; l'implementa `setup/pilfflonk/src/bytecode.rs`).** És el `.bin` del prover STARK camp per camp (decisió de l'usuari, 29-09-2026): l'escriu `setup/pil2-stark/src/io/bin_file.rs` amb els ops i args de `io/parser_args.rs`, el llegeix `expressions_bin.cpp` i l'executa `expressions_pack.hpp`. Tots els valors tenen dimensió 1, i només se n'aparta on BN254 i la dimensió 1 ho obliguen:
- **Camps de dimensió:** no hi ha `destDim`, `nTemp3` ni `maxTmp3`, ni temporals de l'extensió; als *hints*, tampoc el `dim` d'una expressió.
- **Args:** u32, no u16, perquè u16 trunca sense avisar els índexs de més de 65535.
- **Constants:** `Fr` canònics de 32 bytes *little-endian*, no u64, al codi i als *hints*.
- ***Hints*:** només els del prover que el setup accepta (`gsum_col`, `gprod_col` i `im_col`); els de witness i de depuració no hi van, perquè el setup els ignora (§4.2.1). Tampoc hi ha el `commitId` d'una columna *custom* (P5).
- **Revisions:** la revisió 2 (M11) tenia la secció 3 buida (`nHints = 0`); la 3 (M30) hi escriu els *hints*. Els lectors comparen la versió per igualtat, i per tant una clau de la revisió 2 s'ha de tornar a generar. Els `im_col` (M31) no han demanat cap revisió nova: els *hints* van amb el seu nom i els seus camps, i el format no canvia; abans de M31, el setup no escrivia cap `im_col` i el lector C++ en refusava la clau.
- **Prefix:** la secció 1 comença amb `version u32`, `n8 u32 = 32`, `r` (32 bytes LE) i `nStages u32`. La versió es repeteix perquè el `BinFile` de rapidsnark només en comprova un màxim; el lector pilfflonk la compara per igualtat i rebutja un `.bin` STARK. `nStages` hi és perquè els tipus d'operand en depenen (el STARK el pren del `starkinfo`), i així el fitxer es pot descodificar sol.
- **Còpies:** una còpia s'escriu com a `add(a, 0)`; el STARK l'escriu com un `add` sense el segon operand, que el seu intèrpret, de 8 args per operació, no pot executar.

```
"chps" | version u32 = 0x7066_0003 ("pf" a la meitat alta, revisió 3 a la baixa) | nSections u32 = 3
3 × { id u32, size u64, payload }, en l'ordre 1, 2, 3
```

- **Secció 1, expressions:** el prefix; `maxTmp`, `maxArgs` i `maxOps` (els màxims de les seccions 1 i 2), `nOps`, `nArgs`, `nNumbers` i `nExpressions`; per a cada expressió `expId`, `destId`, `stage`, `nTemp`, `nOps`, `opsOffset`, `nArgs`, `argsOffset` (u32) i `line` (UTF-8 acabada en NUL); i al final `ops` (u8), `args` (u32) i `numbers` (32 bytes). Com al STARK, el prover troba el codi d'un im pol per l'`expId` de la seva entrada de `cmPolsMap`, i el de `Q` per `cExpId`.
- **Secció 2, restriccions** (depuració): `nOps`, `nArgs`, `nNumbers` i `nConstraints`; per a cada restricció `stage`, `destId`, `firstRow`, `lastRow`, `nTemp`, `nOps`, `opsOffset`, `nArgs`, `argsOffset`, `imPol` i `line`; i al final `ops`, `args` i `numbers`. La restricció val a les files `firstRow ≤ i < lastRow`.
- **Secció 3, *hints*:** `nHints u32` i, per a cada *hint* (en l'ordre del `pilout`; el prover els calcula en l'ordre del STARK, §4.4), com el `write_hints_section` del STARK: `name`; `nFields u32` i, per a cada camp, `name` i `nValues u32` (un, o els elements d'un camp que és un vector); i per a cada valor, `op` (el nom del STARK: `cm`, `const`, `tmp`, `number`, `string`, `public`, `challenge`, `airvalue`, `airgroupvalue` o `proofvalue`), el valor (`number`: 32 bytes, un `Fr` canònic *little-endian*; `string`: una cadena; la resta: `id u32`), `rowOffsetIndex u32` per a `cm` i `const`, i `nPos u32` seguit de les `pos` (u32), la posició al vector; cap per a un sol valor. L'`id` és el del STARK: l'índex a `cmPolsMap` d'una `cm` (no el `stagePos`), a `constPolsMap` d'una `const`, l'`expId` d'una `tmp` (una expressió de la secció 1, que el lector exigeix), i l'índex al seu mapa de la resta. El prover els busca pel nom, com `getHintIdsByName`, i en comprova els operands contra el `pilfflonkinfo` en carregar la clau (§4.2.1).
- **Restricció que és sencera un im pol (M24):** quan la cerca promou tota l'expressió d'una restricció a im pol (passa amb un domini que no és `everyRow`, perquè el seu `Zi` hi suma 1 de grau), `pil-info` en deixa el codi de depuració buit. El codificador l'escriu com una còpia de la columna de l'im pol a la fila, que és el que `pil_code_gen` emet per a qualsevol altra expressió que no és un temporal.

**Ops i args.** Un op (u8) per operació, que al STARK és la combinació de dimensions i aquí sempre val 0. 8 args per operació: `opType dest aType aArg1 aArg2 bType bArg1 bArg2`, amb els `opType` del STARK (0 `add`, 1 `sub`, 2 `mul`, 3 `sub_swap`) i l'ordre d'operands del STARK (per rang de tipus; un `sub` intercanviat passa a `sub_swap`). Els temporals els assigna `get_id_maps` de `pil-info`.

**Operands** `(type, arg1, arg2)`: el tipus és l'índex de buffer del STARK, amb `bs = nStages + 4` (no hi ha *custom commits*, P5). On el STARK multiplica per 3 (la dimensió), aquí és l'índex:

| type | operand | arg1 | arg2 |
|---|---|---|---|
| 0 | columna fixa | columna de `<air>.const` | índex a `openingPoints` |
| 1 … nStages+1 | columna compromesa d'aquest stage | `stagePos` a `cmPolsMap` | índex a `openingPoints` |
| nStages+2 | `Zi` | 1 + índex a `boundaries` | 0 |
| bs | `tmp` | temporal | 0 |
| bs+2 | `public` | `publicsMap` | 0 |
| bs+3 | `number` | índex a `numbers` | 0 |
| bs+4 | `airvalue` | `airValuesMap` | 0 |
| bs+5 | `proofvalue` | `proofValuesMap` | 0 |
| bs+6 | `airgroupvalue` | `airgroupValuesMap` | 0 |
| bs+7 | `challenge` | `challengesMap` | 0 |
| bs+8 | `eval` | `evMap` (només en codi avaluat a ξ) | 0 |

**Semàntica.** El codi de `Q` (`cExpId`) s'avalua punt a punt sobre el *coset* estès; la resta, i les restriccions, sobre `H`. En un domini de `M = 2^e·N` punts, una columna a `o = openingPoints[arg2]` es llegeix al punt `(i + 2^e·o) mod M`. `Zi` de la frontera 0 (`everyRow`) és `1/Z_H(X)`, i el d'una altra frontera `D` és `Z_H(X)/Z_D(X)` (A.1); el codi de `Q` acaba multiplicant per `Zi(everyRow)`. L'última operació del codi d'un im pol o de `Q` escriu un temporal nou (`tmpUsed`), com al STARK. La codificació és determinista.

**El *digest*, normatiu.** Es calcula sobre la vkey:

```
digest = keccak256( "pilfflonk-v1" ‖ canònic(vkey sense el camp digest) )
```

- **Què és `canònic(·)`.** El JSON sense espais, amb les claus ordenades per unitats de codi UTF-16 (l'ordre de `sort()` de JavaScript, que no és el de `str` de Rust), l'escapament de `JSON.stringify`, els enters grans i els punts com a cadenes decimals, i només enters de valor absolut `≤ 2^53−1`. L'ordena explícitament `proofman_pilfflonk::json::canonical_json` (M12): no es pot confiar en `serde_json::Value`, perquè `setup/stark-recurser` activa `preserve_order` i Cargo ho aplica a tot el *workspace*. El JS ha d'escriure les claus en aquest ordre ell mateix, perquè els motors reordenen les claus que semblen enters.
- **Per què només la vkey.** Conté tot el que intervé en la verificació, i el verificador no rep cap altre fitxer. `<air>.bin`, `<air>.const` i `pilfflonk.srs.bin` no hi intervenen; les constants hi queden lligades a través dels commitments fixos.
- **Ús al transcript.** Hi entra `digest mod r`, com a `Fr`.

---

## Annex B. Mapatge PIL1 → PIL2

| Concepte PIL1 | Ús a pil-fflonk | Equivalent PIL2 (`pilout/src/pilout.proto`) | Canvi |
|---|---|---|---|
| Columnes compromeses | Tres seccions fixes, `cm1`–`cm3` | `Air.stageWidths[]`, `Operand.WitnessCol{stage, colIdx, rowOffset}` | Dimensionar a partir de `stageWidths`; dimensió 1 |
| Constants | Seccions 6–8 de la zkey | `Air.fixedCols[]`, `Operand.FixedCol` | Llegir-les com a `Fr` |
| Polinomis intermedis | `imExps`, a l'stage 3 | No són al `pilout`: els tria el setup (els im pols, a l'últim stage). Són diferents de les columnes `im_col` que declara la std, que es calculen a partir de *hints*. | El setup els tria segons la política de grau |
| Identitats | Plegades amb `a` en C++ generat | `Constraint.{EveryRow, FirstRow, LastRow, EveryFrame}` | Plegar-les amb `std_vc` i dividir pel *zerofier* (A.1) |
| Plookup | `h1`/`h2`, `Z` | Bus de suma de la std (`std_lookup.pil`, `std_sum.pil`) | Eliminar; calcular `gsum` a partir de *hints* |
| Permutation | `Z` | Bus de suma per defecte; bus de producte opcional (`std_prod.pil`) | Igual |
| Connection | `S1..S3`, `k1`/`k2` | Permutació sobre `[col, k^i·ID]` (`std_connection.pil`) | Cal `bn254.pil` |
| Publics | `(polId, fila)` o `publicsCode` | `Operand.PublicValue`; l'usuari els lliga amb restriccions | Són una entrada |
| Una AIR, una `N` | Tot el programa comparteix `N` | `airGroups[].airs[]`, instàncies | Llista d'instàncies dins d'una sola prova |
| Reptes fixos | α, β, γ, δ, `a` | `numChallenges[stage]`, `std_vc`, `std_xi` (`xiSeed`) | Transcript guiat per dades (A.4) |
| Valors i restriccions globals | No existeixen | `airValues`, `airGroupValues{SUM\|PROD}`, `numProofValues`, `PilOut.constraints` | Nous elements a la prova i al transcript |
| *Offsets* | `'` (ξ, ξω) | `rowOffset` (`sint32`) | Punts `ξ·ω^s` qualssevol |
| Grau | `extendBits = ⌈log2(qDeg+1)⌉`, `nBitsZK` | El setup STARK el deriva del *blowup* | Política de grau pròpia (A.1) |
| Avaluació d'expressions | C++ generat per circuit | Bytecode interpretat (`ExpressionsPack`, Goldilocks) | Bytecode i intèrpret `Fr` |

---

## Annex C. Inventari del sistema antic

### C.1 Binaris i scripts

| Nom | Fitxer | Entrades → sortides | Observacions |
|---|---|---|---|
| `pfProver` | `pil-fflonk/src/main.cpp` | zkey, fflonkinfo i, o bé `.commit`, o bé `.exec` + circom `verifier.dat` + `*.zkin.json` → `proof.json`, `public.json` | El `.commit` és `Fr` en forma de Montgomery, fila per fila. A la fixture, `pilfflonk.exec` fa 0 bytes, i per tant el mode `.exec` no s'hi pot fer servir. |
| `pfSetup` | `pil-fflonk/src/main_setup.cpp`, `pilfflonk_setup.cpp` | ptau, shkey, fflonkinfo, `.const` → `.zkey` | No agrupa. Falla amb la fixture perquè al `shkey` li falten claus (`pilfflonk_setup.cpp:171-180`). |
| `pfProverTest` | `pil-fflonk/test/` | — | Només hi ha tests de NTT i de `Polynomial` |
| `copy_generated_files.sh`, `test_examples.sh` | `pil-fflonk/tools/` | `../pil-stark/tmp` → `config/` i `src/chelpers/` | Recorren 14 exemples i verifiquen amb JS |
| `main_fflonkinfo.js` | `pil-stark/src/fflonk/` | `.pil` → `fflonkinfo.json` | Reaprofita `src/pil_info/` amb `stark=false` |
| `main_shkey.js` | ídem | `.pil`, fflonkinfo, ptau, `extraMuls`, `maxQDegree` → `shkey.json` | Les classes es decideixen a `fflonk_shkey.js`, i la repartició es fa a shplonkjs |
| `main_setup.js`, `main_exportVerificationKey.js` | ídem | → `.zkey`, `.vkey` | La zkey té 12 seccions (`pil-fflonk/src/zkey_pilfflonk.hpp:18-33`). Per defecte, `extraMuls = 2` i `maxQDegree = 0`. |
| `main_prover.js` | ídem (`helpers/fflonk_prover.js`) | → prova | Prover JS complet: font de vectors de referència |
| `main_verifier.js` | ídem (`helpers/fflonk_verify.js`) | vkey, fflonkinfo, proof, public → OK/FAIL | Surt amb codi 0 també quan falla |
| `main_exportSolidityVerifier.js`, `main_exportCalldata.js` | ídem (`solidity/`) | → 2 contractes; calldata | `PilFflonkVerifier` i `ShPlonkVerifier`, aquest amb la plantilla de shplonkjs |
| `main_buildchelpers.js` | `pil-stark/src/fflonk/` (la lògica és a `chelpers/fflonk_chelpers.js`) | fflonkinfo → `*.chelpers.*.cpp` | `N` i els *strides* queden fixats al codi |

### C.2 Destí de cada peça

| Peça | Destí |
|---|---|
| L'orquestració SHPLONK genèrica de `ShPlonkProver` (`pil-fflonk/src/shplonk.cpp`) | **Adaptar** a `pil2-stark/src/pilfflonk/`, sobre el `Polynomial` i el `CPolynomial` de rapidsnark. Cal generalitzar-la a *offsets* amb signe i a diverses instàncies, i corregir-ne els defectes de C.3. |
| La NTT multicolumna (`ntt_bn128.*`) | **No cal** a la versió de CPU (s'usa la FFT d'ffiasm); per a GPU ja hi ha `bn128/src/ntt` |
| `extend` i el blinding | **Substituir** per `Polynomial::fromEvaluations` i `blindCoefficients` de rapidsnark |
| `computeFCommitments` i `getCommittedPolynomial` | **Substituir** per `CPolynomial` i `multiMulByScalar`, com fa rapidsnark |
| `Polynomial` de pil-fflonk | **Descartar**: s'usa el de rapidsnark (`divByMonic` en lloc de `divByXSubValue`) |
| Flux per stages (`pilfflonk_prover.cpp`) | **Reescriure**: els stages passen a estar guiats per `pilfflonkinfo` |
| Divisió per `Z_H` en coeficients (`divZh`) | **No cal**: `Q` es calcula sobre un *coset* |
| Agrupació (`fflonk_shkey.js` i shplonkjs) | **Portar** a Rust com a funció pura, generalitzada (A.2) |
| Verificador (`fflonk_verify.js` i `verifyOpenings`) | **Adaptar** en JS a `pilfflonk/js/` (D8), amb el *pairing* d'`ffjavascript` |
| Plantilles Solidity (`verifier_pilfflonk.sol.ejs` i la de shplonkjs) | **Referència** per a les plantilles `tera` |
| Prover JS (`fflonk_prover.js`) | **No s'utilitza:** l'únic JS que es manté és el verificador (D8) |
| Plookup `h1`/`h2`, `Z`, `puCtx`/`peCtx`/`ciCtx` | **Descartar**: ara són busos de la std |
| Chelpers, el codi pas a pas de `fflonkinfo` (`Step`/`StepOperation`, `fflonk_info.hpp:186-251`) i els formats zkey/shkey/fflonkinfo | **Substituir** pel `provingKey/` (§4.2.6) |
| `.commit`/`.exec` i la via de witness amb circom | **Descartar**: el witness és una entrada del prover (§4.3) |
| El transcript de pil-fflonk (`pilfflonk_transcript.*`) | **Descartar**: es fa servir el `Keccak256Transcript` de rapidsnark (P6) |
| `logger`, `zklog`, `exit_process` | **Descartar** |
| `FflonkProver` de rapidsnark (R1CS) | **No tocar**: pertany al *wrap* final |

### C.3 Defectes que no s'han de portar

1. **El *pairing* es fa amb les constants de la prova.** `fflonk_verify.js:121-128` substitueix els commitments de la vkey pels de la prova, i `verifyOpenings` els fa servir en el *pairing* (`shplonkjs/src/shplonk.js:203-206`). Mentrestant, el transcript absorbeix els de la vkey (`:51-54`).
2. **A `Q` li pot faltar un coeficient.** `maxPolsOpenings` es calcula abans de les fusions, i el grau de `Q` és inclusiu mentre que la resta de graus són nombres de coeficients. Si una fusió fa passar `|O|` màxim d'1 a 2, el coeficient més alt de `Q` es pot perdre sense avís.
3. **Els polinomis compromesos que no s'obren a cap punt es descarten en silenci** (`fflonk_shkey.js:239-241`).
4. **Error de precedència** al camí `maxQDegree > 0` (`pilfflonk_prover.cpp:703`).
5. **Publics mal indexats.** `publics_first` rep `polId` com a fila (`pilfflonk_prover.cpp:461`), i el codi generat escriu `publicInputs[i]` indexat per fila (`chelpers/pilfflonk.chelpers.publics.cpp:4-8`).
6. **Lectura equivocada.** `fflonk_info.cpp:183-219` llegeix `publicsCode` quan hauria de llegir `step2prev`.
7. **Tipus de `powerW`.** Es serialitza com a cadena quan només hi ha un valor de `k` (`shplonkjs/src/utils.js:11-22`).
8. **Defectes que cal corregir en adaptar el codi C++:**
   - **Doble alliberament.** La `zkey` s'allibera dues vegades (`pilfflonk_prover.cpp:39` i `shplonk.cpp:21`).
   - ***Leaks*:** els buffers d'escalars de la MSM (`shplonk.cpp:901`, `pilfflonk_setup.cpp:380`), `alphas`, els `fTmp` de cada `f`, els arrays de `rootsMap` (`reset()` només fa `clear()`), `polQ` i `CommitmentAndPolynomial`.
   - **Estat global i sortides abruptes.** Hi ha estat global (`zklog`, el *singleton* `AltBn128::Engine::engine`, `bExitingProcess`), i des de biblioteca es criden `exit()` i `exitProcess()` (`pilfflonk_prover.cpp:254, 430, 895`).
   - **Excepcions mal llançades.** `throw new runtime_error(...)` llança un punter, que `catch (const std::exception&)` no captura (`pilfflonk_setup.cpp:69, 84`).
   - **VLA a la pila:** `u_int8_t data[length]` (`pilfflonk_transcript.cpp:44`, i el mateix a rapidsnark) i `FrElement res[nThreads*4]` (`shplonk.cpp:642-643`). Poden desbordar la pila si hi ha moltes instàncies.
   - **Truncació a 32 bits:** `PTauBytes` (`pilfflonk_prover.cpp:205`), `pos`/`index` (`shplonk.cpp:654-655, 819`) i `polDegree` (`shplonk.cpp:807`, `pilfflonk_setup.cpp:310`).
   - **Arrays no inicialitzats.** `lengths` i `polsIds` no s'inicialitzen (`pilfflonk_setup.cpp:290-291`), i `lengths[j] >= 0` sobre un `u64` és sempre cert.
   - **Codi duplicat** entre el setup i `shplonk.cpp` (`find`, `multiExponentiation`, `polynomialFromMontgomery`).
   - **Paral·lelisme perdut, sense conseqüències.** `omp_get_num_threads()/2` fora d'una regió paral·lela val 0, i `ThreadUtils` el passa a 1, de manera que les còpies es fan en sèrie.

---

## Annex D. Fitxers clau

- **Codi a adaptar (pil-fflonk, C++):** `pil-fflonk/src/shplonk.cpp` (`ShPlonkProver`, només l'orquestració genèrica).
- **Codi que es reutilitza tal com és:**
  - `pil2-stark/src/rapidsnark/polynomial/{polynomial,cpolynomial,evaluations}.{hpp,c.hpp}`
  - `pil2-stark/src/rapidsnark/keccak_256_transcript.{hpp,c.hpp}`
  - `pil2-stark/src/rapidsnark/binfile_utils.hpp`
  - `pil2-stark/src/rapidsnark/fflonk_setup.cpp` (lectura del `ptau`)
  - `pil2-stark/src/bn128/src/ffiasm/`
- **Protocol de referència (JS):**
  - `pil-stark/src/fflonk/helpers/{fflonk_info,fflonk_shkey,fflonk_setup,fflonk_verify,fflonk_prover}.js`
  - `shplonkjs/src/helpers/{setup,verifier,prover}.js`, `shplonkjs/src/utils.js`
  - La còpia de referència és a `../pil-stark` (només el necessari; vegeu-ne el `README.md`)
- **Solidity:**
  - `setup/pil2-stark/node_modules/snarkjs/templates/verifier_fflonk.sol.ejs` (la plantilla, snarkjs 0.7.6)
  - `pil-stark/src/fflonk/solidity/verifier_pilfflonk.sol.ejs`
  - `shplonkjs/src/solidity/verifier.sol.ejs` (*pairing*)
  - `setup/stark-recurser/stark2circom/circuit_templates/templates.rs` (patró `tera`)
  - M40: `setup/pilfflonk/src/{solidity.rs,tera/verifier_pilfflonk.sol.tera}`, `pilfflonk/solidity/` (el projecte Foundry) i `setup/pilfflonk/tests/solidity.rs`
- **Setup Rust:**
  - `setup/pil2-stark/src/{pil,expr}/`
  - `setup/pil2-stark/src/types/pilout_info.rs`
  - `setup/pil2-stark/src/io/{parser_args,bin_file,bin_file_writer,fixed_cols}.rs`
  - `setup/pil2-stark/src/output/{global_info,global_constraints,stark_info}.rs`
  - `setup/pil2-stark/src/proving_key/{bctree,snark_setup,recursive}.rs` (patró FFI, `ensure_pil2com_exec`)
  - `setup/pil2-stark/src/commands/{setup,compile_pil}.rs`
- **Runtime:**
  - `common/src/global_info.rs`
  - `common/src/hash_family.rs`
  - `proofman/src/challenge_accumulation.rs`
- **Format PIL2:** `pilout/src/pilout.proto`
- **Witness de l'stage 2:**
  - `pil2-stark/src/starkpil/gen_proof.hpp` (`calculateImHints` :27, `calculateWitnessSTD` :57)
  - `pil2-components/lib/std/pil/{std_sum,std_prod,std_lookup,std_permutation,std_connection,std_constants,goldilocks}.pil`
- **BN254 en C++:**
  - `pil2-stark/src/bn128/src/ffiasm/{alt_bn128,f2field,curve,fft,multiexp}.hpp`
  - `pil2-stark/src/bn128/src/{msm,ntt}/`
  - `pil2-stark/src/rapidsnark/{polynomial/,binfile_utils.hpp,keccak_256_transcript.{hpp,c.hpp}}` (el transcript es reutilitza)
- **Verificador JS (base):**
  - `setup/pil2-stark/node_modules/snarkjs/src/Keccak256Transcript.js` (el transcript JS del FFLONK existent)
  - `setup/pil2-stark/node_modules/ffjavascript/src/engine_pairing.js` (`pairingEq`)
  - `proofman/src/snark_wrapper.rs:580-638` (com es crida un verificador JS)
- **FFI i compilació:**
  - `provers/starks-lib-c/bindings_starks.rs`, `provers/starks-lib-c/src/{lib,ffi_starks}.rs` (patró)
  - `pil2-stark/src/api/starks_api.{hpp,cpp}` (patró de codis d'estat)
  - `pil2-stark/Makefile:150, 180, 228-234`
- **CLI:** `cli/src/main.rs:33-82`, `cli/src/commands/pilout/mod.rs` (patró de subcomandament niat)
- **Compilador** (branca `develop-0.14.0-pil2-fflonk`):
  - `pil2-compiler/src/{pil.js,compiler.js,processor.js,proto_out.js,sequence.js,fixed_file.js,extern_fixed_file.js}`
  - `pil2-compiler/src/sequence/{fast_code_gen,size_of}.js`
  - `pil2-compiler/src/definition_items/fixed_col.js`
  - `pil2-compiler/src/pilout.proto`
  - `pil2-compiler/test/{bn254_fixed.js,bn254/big_fixed.pil}`

---

## Annex E. Base analitzada

| Repositori | Commit | Paper |
|---|---|---|
| `../pil-fflonk` | `b385c38` (`main`), més `pil/`, nou i sense commit | Prover fflonk PIL1 (C++, amb SHPLONK) |
| `pil-stark` (github.com/0xPolygonHermez/pil-stark) | `5e20f57` (`pilfflonk`) | `fflonkinfo`, classes d'agrupació, setup, verificador JS i Solidity, prover JS |
| `shplonkjs` | `7824640` (la versió fixada per pil-stark) | Repartició, arrels, verificació SHPLONK, contracte `ShPlonkVerifier` |
| `pil2-proofman` | `ff0ff959` (`pre-develop-1.4.0-alpha`) | Repositori destí |
| `../pil2-compiler` | `503862c` (`develop-0.14.0`), més la branca local `develop-0.14.0-pil2-fflonk`, sense commit | Compilador PIL2 |
| `iden3/ffiasm` | `0830252a` (0.1.5, vendoritzat) | Aritmètica BN254 del prover |
| `snarkjs` / `ffjavascript` | 0.7.6 / 0.3.1 (a `setup/pil2-stark/node_modules`) | Transcript i *pairing* del verificador JS |

La branca `feat/pil2-fflonk` no s'ha fet servir com a referència, de manera deliberada.

---

## Annex F. Troballes col·laterals

Aquests problemes no bloquegen el backend nou, però han sortit durant l'anàlisi. Alguns queden resolts dins d'aquest projecte, i s'indica on.

1. **El *wrap* final sempre es desa com a PLONK.** `SnarkWrapper.protocol` està fixat a `SnarkProtocol::Plonk` (`proofman/src/snark_wrapper.rs:269`), mentre que `setup-snark` genera fflonk per defecte i `get_snark_protocol_id_c` no es crida mai. Per tant, les proves fflonk es desen i es verifiquen com si fossin PLONK.
2. **`CPolynomial::multiExponentiation` no té `return`** (`pil2-stark/src/rapidsnark/polynomial/cpolynomial.c.hpp:69-75`).
3. **El target `binfile` del Makefile és erroni.** Fa servir `TARGET_BINFILE`, que no està definit; la variable real és `TARGET_BIN_FILE` (`pil2-stark/Makefile:113`).
4. **Fitxers orfes o sobrers:**
   - `setup/pil2-stark/src/pilout_info.rs` no es compila; s'esborra a la Fase 1 (D1);
   - `pil2-stark/src/bn128/src/ffiasm/{fq.o,fr_asm.o}` estan versionats a git;
   - `pil2-stark/src/config/zkglobals.hpp` declara `fec` i `fnec`, però no es defineixen enlloc.
5. **Els *golden tests* del setup no protegeixen les passades.** Només cobreixen tres AIRs de ZisK i no s'executen si falta `setup/golden_reference/`. A la Fase 1 se'n fan de nous.
6. **`Keccak256Transcript` de rapidsnark corromp el transcript amb el punt zero** (`keccak_256_transcript.c.hpp:63-65`). Es fa servir als provers del *wrap* final (`fflonk_prover.c.hpp:318`, `plonk_prover.c.hpp:387`, `plonk_prover_gpu.c.cuh:545`), de manera que és un risc de compatibilitat amb snarkjs. Hi ha un segon cas que afecta els mateixos provers: `RawFq::toRprBE` d'ffiasm (`fq.cpp:324-339`) codifica malament les coordenades `< 2^192`, amb probabilitat `≈ 2^-61` per punt, i llavors snarkjs rebutjaria la prova. Tots dos es podrien corregir amb un canvi d'una línia (a `fq.cpp`, l'`mpz_export` amb paraules de `bytes` com a `fr.cpp:312`), però toca ffiasm i el *wrap* final, i queda fora d'abast. pilfflonk reutilitza la classe tal com és i rebutja aquests punts a l'API (§4.4).
7. **`Tables.fill` amb un valor negatiu ja era incorrecte a Goldilocks.** Per a `-1`, el `pilout` acabava amb `4294967294` en lloc de `p−1` (`pil2-compiler/src/definition_items/fixed_col.js`, `fillRowsFrom`). Queda corregit a la branca del compilador (C4).
8. **Possible error al *zerofier* `lastRow` del prover STARK, no verificat.** `pil2-stark/src/starkpil/setup_ctx.hpp:104` crida `buildOneRowZerofierInv(..., N)`, que faria servir l'arrel `ω^N = 1`, mentre que `:47-56` fa servir `ω^{N−1}`. A més, `:57` sembla comprovar `everyRow` on hauria de ser `everyFrame`. Com que el compilador de `develop-0.14.0` només emet `everyRow` (§3.4), probablement cap AIR real no hi arriba.
9. **Defectes de rapidsnark i ffiasm trobats a M6.** No es toquen; pilfflonk els esquiva:
   - `CPolynomial::getPolynomial` té comportament indefinit quan el grau és menor que 2 (fa `std::log2(0)`), i quan el grau empaquetat és una potència de dos retorna un polinomi amb un coeficient de menys. A més, esborra un prefix d'una potència de dos del buffer. `PilFflonk::pack` ho té en compte.
   - `binfile_writer.cpp:41` fa `delete[]` d'objectes creats amb `new`.
   - `multiexp.c.hpp:30` fa una lectura de 8 bytes desalineada a cada MSM (UBSan).
   - El mode *direct read* de `BinFile`: `readU32LE` i similars desreferencien un punter nul, i `readSectionToParallel` llança una excepció dins d'un `std::thread`, que acaba el procés. pilfflonk només fa servir `readSectionTo`.
   - `fflonk_setup.cpp` fa `throw new runtime_error(...)`, és a dir, llança un punter.
   - GCC 11 dona un error intern en compilar `starks_api.cpp` amb UBSan (a qualsevol nivell d'optimització); les proves amb sanitizers compilen aquell fitxer sense UBSan (M5–M17).
   - **Fuites de memòria a `Polynomial` (M7): corregides a rapidsnark** per decisió de l'usuari (29-09-2026), sense canviar cap resultat: `divByMonic` (`polResult`, `bArr`), `lagrangePolynomialInterpolation` (els polinomis de base), `byXSubValue` (el temporal), `fastDivByVanishing` (`polTmp`), i la propietat del buffer nou quan el polinomi es va crear sobre un buffer reservat (`add`, `addBlinding`, `byXSubValue`, `divByMonic`, i el doble alliberament de `divBy`/`divByVanishing`). LeakSanitizer passa de 2.484.976 bytes en 36.354 blocs a cap fuita, sense supressions; també se'n beneficia el `FflonkProver` del *wrap* final.
   - **Trobat en corregir-les, no corregit (fora d'abast):** `divByMonic` escriu abans del buffer quan `m ≤ grau < 2m−1` (no només quan `grau < m`; pilfflonk ho evita al seu codi, M18); `byXNSubValue` té un ús després d'alliberar i `byX()` perd el buffer vell (cap dels dos té crides); `fflonk_prover.c.hpp:1390, 1397, 1509, 1515` no allibera els `fTmp` (uns 40 bytes per prova); el target `plonkProve` del Makefile no compila (`src/plonk_setup/main_prover.cpp:95`).
   - **`final_snark_proof.hpp:160-168` (M12):** llegeix les coordenades G1 (que són de `Fq`) amb `AltBn128::Fr.fromRprBE`; una coordenada a `[r, q)` sortiria malament (probabilitat `≈ 2^-127`).
   - **`CodeRef` de `pil-info` (M12):** el seu `Deserialize` espera claus en *snake_case* i un `id` obligatori, però escriu `stageId`, `expId`…, i per tant no pot llegir el que escriu. La vkey guarda el `qVerifier` com a JSON opac.
   - **L'encoder de bytecode STARK (M11, no verificat):** el registre de `copy` té 5 arguments u16, però `expressions_pack.hpp` en llegeix 8 per operació; i `bin_file.rs` converteix tots els arguments a u16 sense avisar, de manera que un índex `≥ 65536` es truncaria.
   - **Compilador PIL2 (M23; un altre repositori, no tocat):** `pil2-compiler/src/packed_expressions.js:139` (`rowOffsetToString`) escriu un *offset* positiu més gran que 1 amb un signe menys (`a'2` surt com a `a'-2`; hauria de ser `${e}'${rowOffset}`). Només afecta la línia de depuració que mostra `pilfflonk check`.
   - **Codi de depuració buit a `pil-info` (M24, no verificat al STARK):** `generate_constraints_debug_code` deixa buit el codi d'una restricció que és sencera un im pol; el `bin_file.rs` STARK escriuria una restricció de 0 operacions. Només passa amb restriccions que no són `everyRow`, que el compilador de `develop-0.14.0` no emet mai.
   - **ffjavascript 0.3.1 (M8):** `G1.sub(a, b)` amb `a` afí i `b` jacobià retorna `b − a` (`src/wasm_curve.js:104`, `op2("_subMixed", b, a)`). snarkjs no hi passa; el `computeF` de shplonkjs sí que hi passaria amb un sol `f`. El verificador JS de pilfflonk treballa en coordenades jacobianes per evitar-ho. A més, `Fr.e(v)` no redueix `v = p`, i per això els descodificadors comproven `< r` i `< q` ells mateixos.
   - **Altres casos límit de `Polynomial` (M7):** `lagrangePolynomialInterpolation` falla amb un sol punt; `divByMonic` escriu abans del buffer si el grau és menor que `m` i no comprova el residu; `add()` creix sense actualitzar la longitud; `sub()` desborda si l'altre polinomi és més llarg; `mulScalar`/`subScalar` no actualitzen el grau; `fixDegree` amb longitud 0 llegeix fora de límits; `divByZerofier` dona resultats incorrectes si hi ha menys fils que `n` i `n` no és potència de dos.

10. **Troballes de M30 (el STARK i la std; no es toquen):**
   - `string2opType` (`pil2-stark/src/starkpil/stark_info.cpp:797`) no coneix `proofvalue`: el lector STARK no pot llegir un *hint* amb un proof value. El format pilfflonk l'admet; la v1 no en té.
   - `calculateWitnessSTD` (`gen_proof.hpp:57`) només calcula el primer `gsum_col` i el primer `gprod_col` d'una AIR (`getHintIdsByName` a `hint[1]`). La std en fa un de cada; pilfflonk els calcula tots.
   - El `write_hints_section` del STARK (`setup/pil2-stark/src/io/bin_file.rs`) escriu `id` i `row_offset_index` amb `as u32`: el `rowOffsetIndex` −1 d'un *offset* que no és punt d'obertura (`gen_code.rs`, `process_single_hint_field`) sortiria com `0xFFFFFFFF`. El codificador pilfflonk el rebutja.
   - Amb el `MAX_CONSTRAINT_DEGREE` per defecte (3), el bus de suma no admet dos termes de denominador de grau 1 en una sola fracció (`std_sum.pil`, `piop_gsum_air`: `2 + grau ≤ MAX`), i un lookup `assumes` + `proves` dins d'una AIR necessita un `im_col`. A M30, la fixture `sum_bus` pujava el grau a 4 amb `set_max_constraint_degree`; des de M31 (que calcula els `im_col`) fa servir el grau per defecte, amb un `im_col`, i `sum_bus_degree4.pil` en conserva la variant de grau 4, sense cap. La del bus de producte hi cap amb 3.
   - Possible, no verificat: amb `PROD_EXPRESSIONS_IM_NON_REDUCED ≠ 0`, el `gprod_col` de `std_prod.pil` (`piop_gprod_air`) passa `numerator`/`denominator` sense els termes no reduïts que la restricció sí que multiplica (`numerator_non_reduced`, `denominator_non_reduced`), i la columna no la satisfaria. Per defecte val 0 i no passa.

11. **Troballes de M31 (el STARK i la std; no es toquen):**
   - **Els termes d'alt grau del bus de producte no compilen.** A `std_prod.pil` (`piop_gprod_air`), un terme de grau més gran que `MAX_CONSTRAINT_DEGREE` es redueix amb un `im_high`, i `gprod_e[term] = im_high[idx]` (`:476`, `:498`) reassigna un element de `const expr gprod_e[ARRAY_SIZE]` (`:200`, ja assignat a `:294`). El compilador de `develop-0.14.0-pil2-fflonk` ho rebutja (`setting gprod_e a const element`), i per tant no hi ha cap `pilout` amb `im_high`. La fixture `prod_bus_im` fa servir el camí dels termes de grau baix (`im_low`), que sí que compila i encadena els `im_col`.
   - **El STARK no comprova què llegeix un `im_col`.** `multiplyHintFields` (`hints.cpp`) calcula els `im_col` en l'ordre de `getHintIdsByName` llegint els buffers tal com són: un `im_col` que llegís un `im_col` posterior, la seva pròpia columna o un im pol de l'stage (que es calcula després, `calculateImPolsExpressions`) llegiria el que hi hagués al buffer, sense cap error. La std no ho fa mai (els seus `im_col` només llegeixen els d'abans); pilfflonk ho rebutja al setup i en carregar la clau (§4.2.1).
   - **Un `im_col` sense bus no es calcula.** `calculateImHints` torna sense fer res si l'AIR no té cap `gsum_col` ni `gprod_col`, i la columna es queda sense valor. pilfflonk rebutja aquesta AIR (§4.2.1).

12. **Troballes de M34 (la std; no es toca):**
   - **El bus de suma dona el mateix nom a totes les seves columnes intermèdies.** `piop_gsum_air` (`std_sum.pil:593, 599`) declara `im_single` i `im_cluster` dins d'un bucle, cadascuna amb aquest nom i sense índex: una AIR amb dos `im_cluster` (o dos `im_single`) té dues columnes que es diuen igual. Al STARK no li importa, però la prova de pilfflonk anomena cada avaluació per la seva columna (A.6), i fins a M34b el setup rebutjava l'AIR (`two polynomials are named im_cluster`). Amb el `MAX_CONSTRAINT_DEGREE` per defecte de la std (3) passa amb la Connection (sis termes: dos `im_cluster` i un `im_single`) i amb `all` (deu termes: quatre i un). El bus de producte declara els seus com un vector (`im_low[k]`) i no hi està afectat. **Resolt a M34b (decisió de l'usuari, 30-09-2026):** el setup dona a les columnes que comparteixen nom un índex en l'ordre del `pilout`, `im_cluster[0]`, `im_cluster[1]`, …, amb el mecanisme dels im pols (`lengths: [k]`, A.6). La std, el STARK i el *golden* no canvien, i les fixtures de la Connection i d'`all` en bus de suma tornen al grau per defecte de la std (M34 el pujava a 4 i 6).
   - **Un grup del bus de producte que omple el grau exacte es compta dues vegades.** A `piop_gprod_air` (`std_prod.pil`, termes de grau baix), quan l'últim grup arriba exactament a `MAX_CONSTRAINT_DEGREE` al numerador i al denominador (la branca *perfect match*), l'agrupació torna enrere un terme, però el bucle que construeix el `im_col` el consumeix igualment (el seu `next_degree` val 0 per a l'últim terme, per la condició `offset < len − 1`). Llavors en surt un segon `im_col` que és l'invers del primer (`1 − im_low[k]·im_low[k−1]`, `std_prod.pil:756`), i `gprod·im_low[k] = ('gprod·(1 − L1) + L1)·im_low[k−1]` (`:848`): `gprod` acumula el quadrat de la raó de cada fila. El bus quadra si `(∏P/∏A)² = 1`. Continua sent sòlid, perquè `∏P + ∏A` no és el polinomi zero en `γ` (tots dos són mònics), però costa una columna i un grau. `all` en bus de producte ho té (`im_low[3] = 1/im_low[2]`), i es prova i es verifica. També afecta el STARK. Un esborrany del Plookup de producte amb `SEL·mul` com a selector també hi arribava; la fixture fa servir `mul` sol, amb `(1 − SEL)·mul = 0`.
   - **El `range_check` de la std no es pot fer servir en una sola AIR.** Declara la taula en una AIR pròpia (`U8Air`, `U16Air` o `SpecifiedRanges`, o la d'una taula virtual; `std_range_check.pil`, `declare_*_air`), de manera que el `pilout` té dues AIRs, i pilfflonk el rebutja (D2; `the pilout has 2 AIRs`). A més, en `STD_MODE_ONE_INSTANCE` cada AIR tanca el seu bus sola (`__L1__'·(0 − gsum)` a totes dues), i la suma de l'AIR que només consulta, `−Σ 1/(v + γ)`, no seria 0. El range check de la Fase 2 és una consulta a una taula fixa de la mateixa AIR (Annex G).
   - **El lookup de la std només té el bus de suma.** `lookup_assumes` i `lookup_proves` no tenen `bus_type`: una multiplicitat seria un exponent en un bus de producte. Les variants de producte del Plookup i del range check fan servir la permutació (Annex G).

13. **Troballes de M40 (el verificador Solidity de snarkjs 0.7.6, `templates/verifier_fflonk.sol.ejs`; no es toca):**
   - **Les coordenades dels commitments es comproven contra `r`, no contra `q`.** `checkProofData` crida `checkField` (`lt(v, q)`, i el `q` de la plantilla és l'ordre del grup, el `r` d'aquesta especificació) amb les coordenades de `C1`, `C2`, `W` i `W'`, que són del cos base, de mòdul `q > r`: un punt vàlid amb una coordenada a `[r, q)` es refusaria (probabilitat `≈ 2^-127` per coordenada). El verificador de pilfflonk les compara amb `q`, com el JS (`elements.js`).
   - **Els publics no es comproven.** `pubSignals` entren al transcript tal com són i als càlculs mòdul `r`, de manera que un públic `p` i `p + r` són el mateix enunciat amb dos transcripts. El verificador de pilfflonk refusa un públic `≥ r`, com el JS (`fromObjectPublics`).
   - Amb `nPublic = 0`, la plantilla declara `uint256[1] calldata pubSignals` igualment (`Math.max(nPublic, 1)`). pilfflonk no declara `pubSignals` si no hi ha publics.
   - **No es mira què tornen els precompilats** (troballa BAIXA de la revisió de M40, que depèn de la cadena). `g1_acc`, `g1_mulAcc` i `g1_mulAccC` només comproven que la crida a `0x06` o `0x07` hagi anat bé, i `checkPairing` pren `and(success, mload(mIn))`: una paraula qualsevol que no sigui 0. En una cadena sense aquests precompilats, o on no tornin el que diu l'EIP-196/197, una crida a una adreça buida va bé, no torna res, i el verificador llegeix la memòria que hi havia. **Enduriment deliberat respecte de snarkjs:** el verificador de pilfflonk exigeix `returndatasize() = 64` a `0x06` i `0x07` (`checkPointResult`), i `returndatasize() = 32` i una resposta igual a 1 a `0x08`; costa de 587 a 907 de gas per prova (§4.5).
   - **Un avís de solc** (trobat a M42, en mesurar-ne el gas, Annex I.5). Amb solc 0.8.37, el contracte compila amb l'avís 5667: el paràmetre `proof` de `verifyProof` no es fa servir, perquè el contracte llegeix la prova a posicions fixes del calldata. El de pilfflonk el fa servir (`checkInput(proof, …)`) i compila sense cap avís (§4.5, "Compilació").

---

## Annex G. El programa PIL1 de la fixture de pil-fflonk

**Origen.** És l'exemple `all` de pil-stark (branca `pilfflonk`), `test/state_machines/sm_all/all_main.pil`, més els fitxers que inclou. El genera `test/cfiles/fflonk_gen_all_files.js`.

**On s'ha desat.** Els PIL dels 14 exemples de `tools/test_examples.sh`, juntament amb els generadors JS de constants i witness (`sm_*.js` i `sm/sm_global.js`), s'han desat a **`pil-fflonk/pil/`**:
- són còpies idèntiques de pil-stark `5e20f57` (`test/state_machines/`);
- es manté l'estructura de directoris, de manera que els `include` continuen funcionant;
- hi ha un `README.md` amb la taula d'exemples i els paràmetres de generació (`extraMuls`, `maxQDegree`, entrades);
- encara no tenen commit.

```
constant %N = 2**8;

namespace Global(%N);
    pol constant L1;

// ../sm_fibonacci/fibonacci.pil
namespace Fibonacci(%N);
    pol constant L1, LLAST;
    pol commit l1, l2;
    pol l2c = l2;
    public in1 = l2c(0);
    public in2 = l1(0);
    public out = l1(%N-1);
    (l2' - l1)*(1-LLAST) = 0;
    pol next = l1*l1 + l2*l2;
    (l1' - next)*(1-LLAST) = 0;
    L1 * (l2 - :in1) = 0;
    L1 * (l1 - :in2) = 0;
    LLAST * (l1 - :out) = 0;

// ../sm_connection/connection.pil
namespace Connection(%N);
    pol constant S1, S2, S3;
    pol commit a, b, c;
    {a, b, c} connect {S1, S2, S3};

// ../sm_permutation/permutation.pil
namespace Permutation(%N);
    pol commit a, b;
    pol commit c, d;
    pol commit selC, selD;
    selC {c, c} is selD {d, d};

// ../sm_plookup/plookup.pil
namespace Plookup(%N);
    pol commit sel, a, b;
    pol commit cc;
    pol constant SEL, A, B;
    sel {a, b', a*b'} in SEL {A, B, cc};
```

**Com surt la fixture d'aquest programa:**
- **Constants i witness.** Els calculen en JS `sm/sm_global.js`, `sm_fibonacci/sm_fibonacci.js` (amb entrades `[1, 2]`), `sm_connection/sm_connection.js`, `sm_permutation/sm_permutation.js` i `sm_plookup/sm_plookup.js`, tots a `test/state_machines/`.
- **Columnes que desapareixen.** `Permutation.a` i `Permutation.b` no apareixen en cap restricció, i per tant no es comprometen (defecte 3 de C.3).
- **Columnes que afegeix pil-stark:**
  - `Plookup.H1_0` i `Plookup.H2_0`, a l'stage 2;
  - `Plookup.Z0`, `Permutation.Z0` i `Connection.Z0`, més l'intermedi `Im28`, a l'stage 3;
  - `Q`, a l'stage 4.

**Ús en aquest projecte.** Portat a PIL2, és la font de les fixtures de les fases 1 i 2:
- el Fibonacci, sense busos, a la Fase 1;
- Plookup, Permutation i Connection, amb la std, a la Fase 2 (M34), i l'exemple `all` sencer.

**Els exemples portats a PIL2 (M34).** Són a `pilfflonk/tests/fixtures/`, i els generadors de witness, portats dels `execute` dels `sm_*.js` a Rust, a `pilfflonk/tests/data/`. Cada exemple té un fitxer amb la màquina d'estats i l'`airtemplate`, i dos programes per compilar, `<exemple>_sum.pil` i `<exemple>_prod.pil`, que en fixen el bus (el `bus_type` de cada PIOP de la std) i fan servir la std en `STD_MODE_ONE_INSTANCE`. Les constants es calculen al PIL, com les calculaven els `buildConstants`. Cada PIOP té el seu `opid` (1 el Plookup, 2 la Permutation, 3 la Connection, 4 el range check), de manera que a `all` comparteixen un sol bus.

| PIL1 | PIL2 (fixtures) | `N` | Què canvia |
|---|---|---|---|
| `sm_fibonacci/fibonacci.pil` | `fibonacci/fibonacci.pil` (Fase 1, M13); `data/fibonacci.rs` | 2^8 | Els publics són entrades, lligades a la traça amb `L1` i `LLAST` (PIL1 els lligava a una cel·la). |
| `sm_plookup/plookup.pil`:<br>`sel {a, b', a*b'} in SEL {A, B, cc}` | `plookup/plookup.pil` (`sm_plookup`), `plookup_{sum,prod}.pil`; `data/plookup.rs` | 2^8 | - **Bus de suma:** `lookup_assumes(1, [a, b', a·b'], sel)` i `lookup_proves(1, [A, B, cc], mul)`, amb `mul`, una columna nova: les vegades que es consulta cada fila de la taula. `(1 − SEL)·mul = 0` limita la taula a les files de `SEL`, com PIL1, que movia les altres a un valor aleatori (`pil_info/step2.js`). La std fa binari `sel`, i PIL1 no el restringia.<br>- **Bus de producte:** el lookup de la std no en té, perquè un producte no admet multiplicitats. Les mateixes tuples van a la permutació de la std, amb `mul` com a selector de la taula (binari), de manera que cada fila de la taula es consulta com a molt una vegada. Les deu consultes del generador són files diferents i hi compleixen. |
| `sm_permutation/permutation.pil`:<br>`selC {c, c} is selD {d, d}` | `permutation/permutation.pil` (`sm_permutation`), `permutation_{sum,prod}.pil`; `data/permutation.rs` | 2^8 | `permutation_assumes(2, [c, c], selC)` i `permutation_proves(2, [d, d], selD)`. La std fa binaris els selectors. `a` i `b` no els llegeix cap restricció, i el setup no els compromet, com PIL1 (C.3, defecte 3), però amb un avís (A.2). |
| `sm_connection/connection.pil`:<br>`{a, b, c} connect {S1, S2, S3}` | `connection/connection.pil` (`sm_connection`), `connection_{sum,prod}.pil`; `data/connection.rs` | 2^10 (2^8 a `all`) | `connection(3, [a, b, c], [S1, S2, S3])`. Les `S_i` surten de la identitat `k^(i−1)·ω^fila`, amb el `k_coset` i el `GEN[BITS]` de BN254 (M29; mai `Goldilocks_k`), i dels intercanvis de `sm_connection.js`, en el mateix ordre. Són els `ks` de PIL1 (el `getKs` de pilcom dona `k, k²` amb `F.k = 5^(2^28) = Bn254_k`) i la seva `ω` (`F.w` d'ffjavascript). En bus de suma, amb el `MAX_CONSTRAINT_DEGREE` per defecte de la std, dos `im_cluster`, que el setup anomena `im_cluster[0]` i `im_cluster[1]` (A.6, Annex F.12), i un `im_single`. |
| — | `range_check/range_check.pil`, `range_check_{sum,prod}.pil`; `data/range_check.rs` | 2^6 | Nou: `v ∈ [0, 16)` contra la taula fixa `T = [0 … 15]`, allargada amb 15. **Suma:** `lookup_assumes(4, [v])` i `lookup_proves(4, [T], mul)`. **Producte:** com plookup (i els `h1`/`h2` de PIL1), `h1` seguit de `h2` són els valors de `v` i `T` ordenats, cosa que comprova la permutació de la std, i unes restriccions els fan anar de 0 a 15 amb passos de 0 o 1. El `range_check` de la std no hi serveix (Annex F.12). |
| `sm_all/all_main.pil` | `all/all.pil`, `all_{sum,prod}.pil`; `data/all.rs` | 2^8 | Una sola AIR, `All`. Les màquines d'estats són funcions PIL2 sobre les columnes de witness de qui les crida, i `all.pil` les crida totes, com `all_main.pil` inclou els quatre fitxers. El Fibonacci s'hi torna a escriure, perquè el de la Fase 1 és un `airtemplate`. PIL1 qualificava cada columna amb el seu *namespace* (`Connection.a`, `Permutation.a`, `Plookup.a`). Una AIR de PIL2 té un sol àmbit i la prova anomena cada avaluació per la seva columna (A.6), i per això les columnes de witness porten el nom de la màquina com a prefix (`connection_a`, `permutation_a`, `plookup_a`). `Global.L1` és el `__L1__` de la std. En bus de suma, amb el `MAX_CONSTRAINT_DEGREE` per defecte de la std, quatre `im_cluster`, que el setup anomena `im_cluster[0]` a `im_cluster[3]` (A.6, Annex F.12), i un `im_single`. |

**Comprovació creuada amb pil-fflonk (M34).** Les 9 columnes fixes de `all`, compilades per `pil2com`, coincideixen valor per valor amb les de `pil-fflonk/config/pilfflonk.const`, també les `S1`–`S3` de la connexió. Les 15 columnes de witness de PIL1 que escriu `data/all.rs` coincideixen amb `pilfflonk.commit`, i els publics amb `runtime/public.json`. El valor d'`out` s'ha recalculat també de manera independent, a partir de les restriccions del Fibonacci. Els scripts són fora del repositori, perquè els fitxers de pil-fflonk no hi són; els tests fixen els publics de `all`.

**Columnes de l'stage 2.** Les afegeix la std, no el setup: `gsum` i els `im_single`/`im_cluster` del bus de suma, o `gprod` i els `im_low` del de producte, cadascuna amb el seu *hint* (§3.4). Per exemple, `all` en té 6 en bus de suma (`gsum`, quatre `im_cluster` i un `im_single`) i 5 en bus de producte (`gprod` i quatre `im_low` encadenats), on PIL1 en tenia 6 (`H1_0`, `H2_0`, tres `Z0` i `Im28`).

---

## Annex H. Rendiment de la versió CPU (M39)

Aquest annex és l'informe de temps i memòria de la Fase 3 (validació 2): el setup, el prover per fases i el verificador JS, fins al límit de P2 (`N ≤ 2^24`), amb el que se n'ha tret (l'avaluació de `Q` per parts, H.6) i el que queda per fer (H.9). Les xifres són del 30-09-2026, amb el codi de M39.

### H.1 La màquina i el mètode

**La màquina.** 2 × AMD EPYC 7773X (64 nuclis per sòcol, 2 fils per nucli: 256 fils; 2 nodes NUMA; 1,5 GB de L3), 1 TB de RAM, Ubuntu 22.04.5 (Linux 5.15). GCC 11.4 amb `-O3` i AVX2 (la CPU no té AVX-512), rustc 1.97.1, Node 22.20 i libomp 14 (el de l'Ubuntu, el que enllacen els binaris Rust).

**Condicions.** La màquina és compartida, i cada execució desa la càrrega (1 minut) de la màquina a l'inici: la dels altres usuaris i la cua de l'execució anterior. Les taules en donen el rang, que va de 6 a 225. Una sola mesura alhora, mai en paral·lel amb una altra, i cap procés d'altres usuaris tocat. Les compilacions d'`all_sum` a `2^23` i `2^24` (H.8), d'un sol fil, van coincidir amb una part de les mesures: 1 fil de 256.

**Els binaris.** `cargo build --release --features proofman-starks-lib-c/cpu-only`, amb el compilador de `PIL2C_EXEC` (branca `develop-0.14.0-pil2-fflonk`). Per defecte, OpenMP fa servir els 256 fils; alguns punts es repeteixen amb `OMP_NUM_THREADS=64` (H.7).

**Què es mesura.** L'eina és `pilfflonk/bench/bench.sh` (H.10). Per a cada programa i mida:
- **`pil2com`**, una vegada: temps, pic de RSS i mida del `pilout`;
- **`setup-pilfflonk`**, amb el *layout* empaquetat per defecte (`--extra-muls 2`) i amb `--no-packing`: temps i pic de RSS, i de la clau, `nBitsExt`, `qDeg`, el nombre de `f`, les potències de l'SRS que necessita (el `degree` màxim del *layout*) i les mides del `.const` i de l'SRS;
- **`pilfflonk prove`** amb `--witness <dir>` i `-vv`: temps de paret i pic de RSS (`/usr/bin/time -v`), el temps de cada fase i el pic de RSS de cada fase (vegeu sota);
- **`pilfflonk verify`** de cada prova mesurada: temps i RSS. L'eina s'atura si una prova no verifica: totes les proves de l'informe verifiquen;
- **la mida de la prova** en bytes (A.6): 64 per punt G1 i 32 per escalar.

**Repeticions.** Tres execucions de cada setup i de cada prova, i dues a `all_sum` `2^22` i `2^23`, a les mides senars, amb 64 fils i a `all_prod`. Les taules en donen la mediana i la dispersió, `(màx − mín)/mediana`, i la memòria en GB de `2^30` bytes.

**Les fases.** El prover C++ té temporitzadors `TimerStart`/`TimerStopAndLog` (`pil2-stark/src/utils/timer.hpp`), els del STARK, amb noms `PILFFLONK_*`: la càrrega de la clau (`LOAD_SRS`, `LOAD_AIRS` amb la INTT de les columnes fixes, `FIXED_COMMITMENTS`), la instància (`INSTANCE`), cada stage (`STAGE_<s>`, amb `HINT_COLUMNS_<s>`, `IM_POLS_<s>` i, per a cada `f`, `INTT_<f>` i `COMMIT_<f>`, la MSM), `Q` (`Q`, amb `Q_EXTEND`, `Q_DOMAIN`, `Q_EVALUATE`, `Q_INTERPOLATE` i `Q_COMMIT`), les avaluacions (`EVALUATIONS`) i l'obertura (`OPEN`, amb `SHPLONK_W`, `SHPLONK_COMMIT_W`, `SHPLONK_WP` i `SHPLONK_COMMIT_WP`). Escriuen al registre del nivell *trace* (`-vv`), com al STARK, i no costen res mesurable: una vintena de crides a `gettimeofday` per prova, i dues per `f`. El temps de llegir el witness surt de les marques de temps de dues línies del registre Rust (`··· Reading the witness` i `··· Committing stage 1`).

**La memòria de cada fase.** L'eina llegeix el `VmRSS` del prover cada 0,2 s, amb l'última marca `PILFFLONK_*` del registre en aquell moment, i en treu el pic de cada fase.

**L'SRS.** Un `ptau` de prova amb la `τ` fixa dels tests (`fixed_tau_ptau` de `setup/pilfflonk/src/test_ptau.rs`, `TEST_TAU`; N13), no cap de baixat: 75.497.536 potències (`9·2^23 + 64`, les que necessita el `layout` empaquetat d'`all` a `2^23`), 4,8 GB, fet en 20,7 min amb 21 GB de RSS. Com que la `τ` és pública, només serveix per mesurar. Un de sol serveix per a totes les mides: el lector de l'SRS només llegeix del `ptau` les potències que cal (M6).

### H.2 Els programes i les mides

Tots són les *fixtures* de les fases 1 i 2 amb `N` com a paràmetre: les variants són a `pilfflonk/bench/` i prenen `N = 2^BENCH_BITS` d'un *define* de `pil2com`, que `bench.sh` passa al `-P` (`"defines": {"BENCH_BITS": <bits>}`) amb el `prime`. Les *fixtures* no canvien. A `2^8`, les claus de les variants són les de les *fixtures* fitxer per fitxer, llevat de les línies de depuració (el fitxer font de cada restricció).

| programa | què és | columnes | `qDeg` | `nBitsExt` |
|---|---|---|---|---|
| `fibonacci` | `bench/fibonacci.pil`: l'`airtemplate` de `tests/fixtures/fibonacci/fibonacci.pil` (el seu `airgroup` fixa `2^8`) | stage 1: `l1`, `l2` i 1 im pol; 2 fixes | 1 | `nBits + 1` |
| `all_sum` | `bench/all_sum.pil`: `tests/fixtures/all/all.pil` (M34) amb el bus de suma | stage 1: 16 (14 compromeses; `permutation_a` i `permutation_b` no les llegeix cap restricció); stage 2: 6 (`gsum`, 4 `im_cluster`, 1 `im_single`); 10 fixes (4 d'amplada completa: `S1`–`S3` i l'`ID` de la connexió) | 2 | `nBits + 2` |

També hi ha `bench/all_prod.pil` (el bus de producte), mesurat en dues mides (H.3).

**El witness.** El dels generadors de les *fixtures* (`pilfflonk/tests/data/{fibonacci,all}.rs`) per a `2^nBits` files, que l'exemple `pilfflonk_bench_inputs` de `proofman-cli` (`pilfflonk/bench/inputs.rs`) escriu com a directori de witness (A.6) a partir de la forma de la clau. Per a `all`, `tests/data/all.rs` té ara `witness_of_size(n_bits, inputs)`, i el witness de sempre n'és el cas de `2^8` (els tests no canvien). Les biblioteques de witness de M38 no hi serveixen: els seus `pil_helpers` fixen `N` al tipus de la traça (`GenericTrace<_, 256>`), i fer-les genèriques en `N` hauria demanat biblioteques noves.

**Els *layouts*.** Amb l'empaquetat, `fibonacci` té 5 `f` (`L1` i `LLAST` en un `f` fix de `k = 2`, tres `f` d'una columna a l'stage 1 i `Q`), i l'SRS n'ha de tenir `2N + 1` potències. `all_sum` en té 9 (els fixos de `k = 9` i `k = 1`; a l'stage 1, de `k` = 3, 3 i 8; a l'stage 2, de `k` = 1, 2 i 3; i `Q`), i l'SRS n'ha de tenir `9N + 8`. Amb `--no-packing`, 6 i 31 `f`, i l'SRS és el de `Q`: `2N + 7` a `all_sum`.

**Les mides.** Totes les parells de `2^10` a `2^24` per al `fibonacci` i de `2^10` a `2^22` per a `all_sum`, més `2^21` i `2^23` al `fibonacci` i `2^23` a `all_sum`, el més gran que el compilador en pot fer (H.8). El `fibonacci` es mesura empaquetat i amb `--no-packing` a totes les mides, i `all_sum` també.

### H.3 Resultats: el setup, la prova i el verificador

Amb el codi final (l'avaluació de `Q` per parts, H.6) i els 256 fils; mediana de les execucions i, entre parèntesis, la dispersió. La càrrega és el rang de la de les execucions.

**`pil2com`**, una compilació per programa i mida (el temps no depèn del *layout*):

| programa | N | s | GB | `pilout` MB |
|---|---|---|---|---|
| `fibonacci` | 2^10 | 0,3 | 0,1 | 0,0 |
| `fibonacci` | 2^12 | 0,3 | 0,1 | 0,0 |
| `fibonacci` | 2^14 | 0,3 | 0,1 | 0,1 |
| `fibonacci` | 2^16 | 0,4 | 0,1 | 0,3 |
| `fibonacci` | 2^18 | 0,8 | 0,3 | 1,0 |
| `fibonacci` | 2^20 | 2,2 | 0,7 | 4,2 |
| `fibonacci` | 2^21 | 4,0 | 1,5 | 8,4 |
| `fibonacci` | 2^22 | 7,0 | 2,7 | 16,8 |
| `fibonacci` | 2^23 | 12,7 | 5,4 | 33,6 |
| `fibonacci` | 2^24 | 30,0 | 10,8 | 67,1 |
| `all_sum` | 2^10 | 1,1 | 0,2 | 0,2 |
| `all_sum` | 2^12 | 1,9 | 0,2 | 0,6 |
| `all_sum` | 2^14 | 5,2 | 0,3 | 2,5 |
| `all_sum` | 2^16 | 18,3 | 0,4 | 9,7 |
| `all_sum` | 2^18 | 69,8 | 1,3 | 38,8 |
| `all_sum` | 2^20 | 286,5 | 5,0 | 155,1 |
| `all_sum` | 2^22 | 1155,6 | 19,5 | 620,4 |
| `all_prod` | 2^16 | 18,1 | 0,4 | 9,7 |
| `all_prod` | 2^20 | 267,7 | 5,0 | 155,1 |

`all_sum` a `2^23` i `2^24`: H.8.

**`setup-pilfflonk`:**

| programa | N | *layout* | s (disp. %) | GB | `nBitsExt` | `qDeg` | `f` | potències de l'SRS | `.const` MB | SRS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^10 | `--no-packing` | 0,39 (51) | 0,0 | 11 | 1 | 6 | 1.029 | 0 | 0 |
| `fibonacci` | 2^10 | empaquetat | 0,34 (50) | 0,0 | 11 | 1 | 5 | 2.049 | 0 | 0 |
| `fibonacci` | 2^12 | `--no-packing` | 0,49 (24) | 0,1 | 13 | 1 | 6 | 4.101 | 0 | 0 |
| `fibonacci` | 2^12 | empaquetat | 0,46 (11) | 0,1 | 13 | 1 | 5 | 8.193 | 0 | 1 |
| `fibonacci` | 2^14 | `--no-packing` | 0,83 (27) | 0,2 | 15 | 1 | 6 | 16.389 | 1 | 1 |
| `fibonacci` | 2^14 | empaquetat | 0,60 (28) | 0,5 | 15 | 1 | 5 | 32.769 | 1 | 2 |
| `fibonacci` | 2^16 | `--no-packing` | 1,26 (23) | 1,0 | 17 | 1 | 6 | 65.541 | 4 | 4 |
| `fibonacci` | 2^16 | empaquetat | 1,10 (5) | 2,0 | 17 | 1 | 5 | 131.073 | 4 | 8 |
| `fibonacci` | 2^18 | `--no-packing` | 1,91 (6) | 2,1 | 19 | 1 | 6 | 262.149 | 17 | 17 |
| `fibonacci` | 2^18 | empaquetat | 1,44 (10) | 2,1 | 19 | 1 | 5 | 524.289 | 17 | 34 |
| `fibonacci` | 2^20 | `--no-packing` | 2,80 (1) | 2,4 | 21 | 1 | 6 | 1.048.581 | 67 | 67 |
| `fibonacci` | 2^20 | empaquetat | 2,30 (0) | 2,6 | 21 | 1 | 5 | 2.097.153 | 67 | 134 |
| `fibonacci` | 2^21 | `--no-packing` | 3,89 (2) | 2,7 | 22 | 1 | 6 | 2.097.157 | 134 | 134 |
| `fibonacci` | 2^21 | empaquetat | 3,62 (4) | 3,2 | 22 | 1 | 5 | 4.194.305 | 134 | 268 |
| `fibonacci` | 2^22 | `--no-packing` | 5,78 (11) | 3,4 | 23 | 1 | 6 | 4.194.309 | 268 | 268 |
| `fibonacci` | 2^22 | empaquetat | 6,24 (7) | 4,3 | 23 | 1 | 5 | 8.388.609 | 268 | 537 |
| `fibonacci` | 2^23 | `--no-packing` | 9,86 (1) | 4,9 | 24 | 1 | 6 | 8.388.613 | 537 | 537 |
| `fibonacci` | 2^23 | empaquetat | 11,37 (3) | 6,6 | 24 | 1 | 5 | 16.777.217 | 537 | 1074 |
| `fibonacci` | 2^24 | `--no-packing` | 16,70 (2) | 7,8 | 25 | 1 | 6 | 16.777.221 | 1074 | 1074 |
| `fibonacci` | 2^24 | empaquetat | 20,18 (4) | 11,2 | 25 | 1 | 5 | 33.554.433 | 1074 | 2147 |
| `all_sum` | 2^10 | `--no-packing` | 1,40 (4) | 0,0 | 12 | 2 | 31 | 2.055 | 0 | 0 |
| `all_sum` | 2^10 | empaquetat | 0,80 (6) | 0,1 | 12 | 2 | 9 | 9.224 | 0 | 1 |
| `all_sum` | 2^12 | `--no-packing` | 1,41 (12) | 0,1 | 14 | 2 | 31 | 8.199 | 1 | 1 |
| `all_sum` | 2^12 | empaquetat | 0,98 (22) | 0,5 | 14 | 2 | 9 | 36.872 | 1 | 2 |
| `all_sum` | 2^14 | `--no-packing` | 1,93 (11) | 0,3 | 16 | 2 | 31 | 32.775 | 5 | 2 |
| `all_sum` | 2^14 | empaquetat | 1,29 (7) | 2,0 | 16 | 2 | 9 | 147.464 | 5 | 9 |
| `all_sum` | 2^16 | `--no-packing` | 3,67 (16) | 1,1 | 18 | 2 | 31 | 131.079 | 21 | 8 |
| `all_sum` | 2^16 | empaquetat | 1,91 (3) | 2,2 | 18 | 2 | 9 | 589.832 | 21 | 38 |
| `all_sum` | 2^18 | `--no-packing` | 8,15 (1) | 2,3 | 20 | 2 | 31 | 524.295 | 84 | 34 |
| `all_sum` | 2^18 | empaquetat | 3,67 (7) | 2,7 | 20 | 2 | 9 | 2.359.304 | 84 | 151 |
| `all_sum` | 2^20 | `--no-packing` | 12,17 (3) | 3,1 | 22 | 2 | 31 | 2.097.159 | 336 | 134 |
| `all_sum` | 2^20 | empaquetat | 8,57 (4) | 5,0 | 22 | 2 | 9 | 9.437.192 | 336 | 604 |
| `all_sum` | 2^22 | `--no-packing` | 26,30 (5) | 6,2 | 24 | 2 | 31 | 8.388.615 | 1342 | 537 |
| `all_sum` | 2^22 | empaquetat | 27,19 (12) | 13,8 | 24 | 2 | 9 | 37.748.744 | 1342 | 2416 |
| `all_sum` | 2^23 | `--no-packing` | 45,73 (19) | 10,4 | 25 | 2 | 31 | 16.777.223 | 2684 | 1074 |
| `all_sum` | 2^23 | empaquetat | 53,88 (6) | 25,6 | 25 | 2 | 9 | 75.497.480 | 2684 | 4832 |
| `all_prod` | 2^16 | empaquetat | 2,40 (3) | 2,1 | 18 | 3 | 9 | 524.311 | 21 | 34 |
| `all_prod` | 2^20 | empaquetat | 14,69 (61) | 4,3 | 22 | 3 | 9 | 8.388.631 | 336 | 537 |

**`pilfflonk prove` i `pilfflonk verify`:**

| programa | N | *layout* | execucions | càrrega | prova s (disp. %) | GB | verificació s | prova (bytes) |
|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^10 | `--no-packing` | 3 | 85–99 | 1,45 (24) | 0,0 | 0,30 | 672 |
| `fibonacci` | 2^10 | empaquetat | 3 | 70–70 | 1,38 (44) | 0,0 | 0,30 | 704 |
| `fibonacci` | 2^12 | `--no-packing` | 3 | 104–117 | 1,31 (15) | 0,1 | 0,31 | 672 |
| `fibonacci` | 2^12 | empaquetat | 3 | 99–113 | 1,29 (8) | 0,1 | 0,31 | 704 |
| `fibonacci` | 2^14 | `--no-packing` | 3 | 140–149 | 1,45 (22) | 0,3 | 0,31 | 672 |
| `fibonacci` | 2^14 | empaquetat | 3 | 117–129 | 1,73 (13) | 0,5 | 0,31 | 704 |
| `fibonacci` | 2^16 | `--no-packing` | 3 | 166–180 | 3,19 (9) | 1,1 | 0,31 | 672 |
| `fibonacci` | 2^16 | empaquetat | 3 | 138–158 | 3,42 (6) | 2,0 | 0,30 | 704 |
| `fibonacci` | 2^18 | `--no-packing` | 3 | 199–212 | 7,08 (6) | 2,2 | 0,30 | 672 |
| `fibonacci` | 2^18 | empaquetat | 3 | 172–193 | 7,17 (11) | 2,2 | 0,30 | 704 |
| `fibonacci` | 2^20 | `--no-packing` | 3 | 169–195 | 9,82 (2) | 2,7 | 0,31 | 672 |
| `fibonacci` | 2^20 | empaquetat | 3 | 162–173 | 9,62 (4) | 2,7 | 0,31 | 704 |
| `fibonacci` | 2^21 | `--no-packing` | 2 | 135–142 | 14,03 (4) | 3,4 | 0,32 | 672 |
| `fibonacci` | 2^21 | empaquetat | 2 | 51–98 | 14,37 (5) | 3,5 | 0,30 | 704 |
| `fibonacci` | 2^22 | `--no-packing` | 3 | 169–180 | 18,63 (16) | 4,8 | 0,30 | 672 |
| `fibonacci` | 2^22 | empaquetat | 3 | 136–184 | 18,77 (14) | 5,0 | 0,30 | 704 |
| `fibonacci` | 2^23 | `--no-packing` | 2 | 168–209 | 32,47 (15) | 7,5 | 0,30 | 672 |
| `fibonacci` | 2^23 | empaquetat | 2 | 83–113 | 39,39 (28) | 8,0 | 0,29 | 704 |
| `fibonacci` | 2^24 | `--no-packing` | 3 | 117–172 | 55,45 (1) | 13,0 | 0,30 | 672 |
| `fibonacci` | 2^24 | empaquetat | 3 | 47–152 | 59,12 (15) | 14,0 | 0,30 | 704 |
| `all_sum` | 2^10 | `--no-packing` | 3 | 175–188 | 3,43 (27) | 0,1 | 0,33 | 2624 |
| `all_sum` | 2^10 | empaquetat | 3 | 160–168 | 1,83 (84) | 0,1 | 0,31 | 1760 |
| `all_sum` | 2^12 | `--no-packing` | 3 | 174–181 | 2,64 (9) | 0,1 | 0,33 | 2624 |
| `all_sum` | 2^12 | empaquetat | 3 | 166–181 | 2,46 (12) | 0,5 | 0,32 | 1760 |
| `all_sum` | 2^14 | `--no-packing` | 3 | 169–183 | 4,76 (3) | 0,6 | 0,33 | 2624 |
| `all_sum` | 2^14 | empaquetat | 3 | 153–180 | 4,57 (9) | 2,1 | 0,32 | 1760 |
| `all_sum` | 2^16 | `--no-packing` | 3 | 202–225 | 13,86 (2) | 2,2 | 0,33 | 2624 |
| `all_sum` | 2^16 | empaquetat | 3 | 147–179 | 9,54 (12) | 2,3 | 0,31 | 1760 |
| `all_sum` | 2^18 | `--no-packing` | 3 | 178–210 | 29,16 (3) | 2,8 | 0,33 | 2624 |
| `all_sum` | 2^18 | empaquetat | 3 | 80–152 | 14,16 (2) | 3,0 | 0,31 | 1760 |
| `all_sum` | 2^20 | `--no-packing` | 3 | 112–205 | 42,94 (4) | 5,1 | 0,32 | 2624 |
| `all_sum` | 2^20 | empaquetat | 3 | 21–132 | 26,13 (6) | 5,8 | 0,32 | 1760 |
| `all_sum` | 2^22 | `--no-packing` | 2 | 161–219 | 91,42 (2) | 15,4 | 0,32 | 2624 |
| `all_sum` | 2^22 | empaquetat | 2 | 125–173 | 76,00 (1) | 18,1 | 0,31 | 1760 |
| `all_sum` | 2^23 | `--no-packing` | 2 | 109–207 | 150,19 (5) | 30,8 | 0,33 | 2624 |
| `all_sum` | 2^23 | empaquetat | 2 | 26–149 | 137,83 (1) | 36,3 | 0,32 | 1760 |
| `all_prod` | 2^16 | empaquetat | 2 | 142–161 | 9,07 (0) | 2,3 | 0,32 | 1664 |
| `all_prod` | 2^20 | empaquetat | 2 | 99–155 | 33,90 (12) | 5,6 | 0,32 | 1664 |

**Què en surt:**
- **La prova i el verificador no depenen de `N`.** La prova fa 704 bytes (el `fibonacci` empaquetat), 672 (sense empaquetar), 1.760 i 2.624 (`all_sum`) i 1.664 (`all_prod`): només en depèn el nombre de `f` i d'avaluacions. El verificador JS triga 0,30–0,33 s a totes les mides, gairebé tot l'arrencada de Node i d'ffjavascript.
- **La prova més gran**, el `fibonacci` a `2^24`, triga 59 s i fa servir 14 GB; `all_sum` a `2^22`, 76 s i 18 GB, i a `2^23`, 138 s i 36 GB. El setup triga 20 i 27 s, i el que fa créixer la seva memòria és l'SRS (de `2N + 1` i `9N + 8` potències).
- **L'empaquetat** no canvia gaire el temps del `fibonacci` (només hi empaqueta les dues columnes fixes), i fa `all_sum` un 17–50 % més ràpid (9 `f` en lloc de 31: menys MSM); a canvi, l'SRS és 4,5 vegades més gran, i el setup necessita el doble de memòria.
- **El bus de producte** (`all_prod`) triga com el de suma a `2^16` i un 30 % més a `2^20`: té `qDeg` 3, i el de suma, 2.

### H.4 Les fases del prover

**Temps (s):**

| programa | N | *layout* | clau | witness | *hints* | im pols | INTT | MSM | `Q`: LDE | `Q`: avaluació | `Q`: INTT | `Q`: MSM | `Q`: resta | avaluacions a ξ | `W`, `W'` | MSM de `W`, `W'` | obertura: resta | resta | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^10 | `--no-packing` | 0,53 | 0,00 | 0,00 | 0,00 | 0,02 | 0,46 | 0,04 | 0,01 | 0,01 | 0,11 | 0,01 | 0,00 | 0,03 | 0,20 | 0,01 | 0,02 | 1,45 |
| `fibonacci` | 2^10 | empaquetat | 0,38 | 0,00 | 0,00 | 0,01 | 0,02 | 0,51 | 0,04 | 0,01 | 0,00 | 0,11 | 0,01 | 0,00 | 0,02 | 0,19 | 0,01 | 0,02 | 1,38 |
| `fibonacci` | 2^12 | `--no-packing` | 0,53 | 0,00 | 0,00 | 0,00 | 0,02 | 0,44 | 0,03 | 0,01 | 0,00 | 0,09 | 0,00 | 0,00 | 0,02 | 0,20 | 0,01 | 0,02 | 1,31 |
| `fibonacci` | 2^12 | empaquetat | 0,40 | 0,00 | 0,00 | 0,00 | 0,02 | 0,46 | 0,04 | 0,01 | 0,00 | 0,12 | 0,01 | 0,00 | 0,01 | 0,12 | 0,00 | 0,02 | 1,29 |
| `fibonacci` | 2^14 | `--no-packing` | 0,56 | 0,01 | 0,00 | 0,00 | 0,01 | 0,47 | 0,03 | 0,01 | 0,00 | 0,12 | 0,01 | 0,00 | 0,02 | 0,21 | 0,00 | 0,01 | 1,45 |
| `fibonacci` | 2^14 | empaquetat | 0,67 | 0,02 | 0,00 | 0,01 | 0,01 | 0,57 | 0,03 | 0,01 | 0,00 | 0,11 | 0,00 | 0,00 | 0,02 | 0,23 | 0,00 | 0,02 | 1,73 |
| `fibonacci` | 2^16 | `--no-packing` | 1,02 | 0,02 | 0,00 | 0,00 | 0,01 | 0,99 | 0,03 | 0,01 | 0,00 | 0,32 | 0,00 | 0,00 | 0,06 | 0,63 | 0,01 | 0,03 | 3,19 |
| `fibonacci` | 2^16 | empaquetat | 1,00 | 0,02 | 0,00 | 0,00 | 0,02 | 1,08 | 0,02 | 0,02 | 0,00 | 0,33 | 0,00 | 0,00 | 0,06 | 0,71 | 0,02 | 0,03 | 3,42 |
| `fibonacci` | 2^18 | `--no-packing` | 1,71 | 0,05 | 0,00 | 0,00 | 0,13 | 2,43 | 0,04 | 0,02 | 0,01 | 0,84 | 0,04 | 0,00 | 0,14 | 1,62 | 0,00 | 0,05 | 7,08 |
| `fibonacci` | 2^18 | empaquetat | 1,17 | 0,07 | 0,00 | 0,00 | 0,14 | 2,39 | 0,06 | 0,02 | 0,01 | 0,81 | 0,05 | 0,01 | 0,20 | 1,71 | 0,01 | 0,06 | 7,17 |
| `fibonacci` | 2^20 | `--no-packing` | 2,25 | 0,24 | 0,00 | 0,00 | 0,48 | 2,83 | 0,12 | 0,15 | 0,03 | 0,92 | 0,26 | 0,01 | 0,64 | 1,81 | 0,02 | 0,11 | 9,82 |
| `fibonacci` | 2^20 | empaquetat | 1,60 | 0,23 | 0,00 | 0,00 | 0,47 | 2,90 | 0,13 | 0,15 | 0,03 | 0,91 | 0,30 | 0,01 | 0,78 | 2,07 | 0,00 | 0,10 | 9,62 |
| `fibonacci` | 2^21 | `--no-packing` | 2,85 | 0,49 | 0,00 | 0,01 | 0,92 | 3,53 | 0,28 | 0,27 | 0,08 | 1,18 | 0,56 | 0,03 | 1,43 | 2,16 | 0,02 | 0,23 | 14,03 |
| `fibonacci` | 2^21 | empaquetat | 2,17 | 0,48 | 0,00 | 0,01 | 0,94 | 3,48 | 0,35 | 0,41 | 0,11 | 1,16 | 0,66 | 0,02 | 1,67 | 2,68 | 0,04 | 0,20 | 14,37 |
| `fibonacci` | 2^22 | `--no-packing` | 3,74 | 0,92 | 0,00 | 0,01 | 0,17 | 4,25 | 0,61 | 0,44 | 0,11 | 1,31 | 1,21 | 0,04 | 2,53 | 2,58 | 0,11 | 0,29 | 18,63 |
| `fibonacci` | 2^22 | empaquetat | 3,09 | 0,86 | 0,00 | 0,01 | 0,16 | 4,09 | 0,68 | 0,43 | 0,14 | 1,32 | 1,12 | 0,02 | 3,04 | 3,26 | 0,00 | 0,32 | 18,77 |
| `fibonacci` | 2^23 | `--no-packing` | 5,86 | 1,77 | 0,00 | 0,01 | 0,45 | 5,49 | 1,84 | 0,89 | 3,31 | 1,96 | 1,89 | 0,04 | 4,73 | 3,54 | 0,08 | 0,62 | 32,47 |
| `fibonacci` | 2^23 | empaquetat | 5,87 | 1,73 | 0,00 | 0,01 | 3,79 | 5,58 | 3,13 | 0,94 | 3,00 | 1,99 | 1,88 | 0,09 | 5,22 | 5,24 | 0,14 | 0,78 | 39,39 |
| `fibonacci` | 2^24 | `--no-packing` | 9,46 | 3,23 | 0,00 | 0,02 | 1,07 | 7,68 | 10,12 | 1,56 | 0,74 | 2,99 | 3,19 | 0,07 | 8,88 | 4,81 | 0,00 | 1,08 | 55,45 |
| `fibonacci` | 2^24 | empaquetat | 9,47 | 3,14 | 0,00 | 0,02 | 7,47 | 8,37 | 3,56 | 1,60 | 0,74 | 2,55 | 3,29 | 0,10 | 10,73 | 6,96 | 0,32 | 1,18 | 59,12 |
| `all_sum` | 2^10 | `--no-packing` | 1,33 | 0,01 | 0,01 | 0,00 | 0,06 | 1,57 | 0,28 | 0,02 | 0,00 | 0,08 | 0,04 | 0,01 | 0,08 | 0,14 | 0,01 | 0,02 | 3,43 |
| `all_sum` | 2^10 | empaquetat | 0,57 | 0,01 | 0,03 | 0,00 | 0,10 | 0,72 | 0,14 | 0,01 | 0,00 | 0,06 | 0,00 | 0,00 | 0,01 | 0,16 | 0,06 | 0,03 | 1,83 |
| `all_sum` | 2^12 | `--no-packing` | 1,28 | 0,01 | 0,02 | 0,00 | 0,02 | 0,83 | 0,11 | 0,02 | 0,00 | 0,06 | 0,00 | 0,00 | 0,04 | 0,13 | 0,01 | 0,02 | 2,64 |
| `all_sum` | 2^12 | empaquetat | 0,87 | 0,01 | 0,02 | 0,00 | 0,04 | 0,69 | 0,16 | 0,02 | 0,00 | 0,06 | 0,00 | 0,00 | 0,02 | 0,39 | 0,05 | 0,03 | 2,46 |
| `all_sum` | 2^14 | `--no-packing` | 1,73 | 0,04 | 0,01 | 0,00 | 0,03 | 1,93 | 0,15 | 0,03 | 0,00 | 0,18 | 0,06 | 0,00 | 0,11 | 0,35 | 0,06 | 0,04 | 4,76 |
| `all_sum` | 2^14 | empaquetat | 1,08 | 0,03 | 0,02 | 0,00 | 0,04 | 1,43 | 0,17 | 0,03 | 0,00 | 0,17 | 0,02 | 0,00 | 0,07 | 1,45 | 0,02 | 0,05 | 4,57 |
| `all_sum` | 2^16 | `--no-packing` | 3,69 | 0,14 | 0,04 | 0,00 | 0,14 | 6,65 | 0,25 | 0,04 | 0,00 | 0,76 | 0,07 | 0,01 | 0,30 | 1,55 | 0,03 | 0,07 | 13,86 |
| `all_sum` | 2^16 | empaquetat | 1,60 | 0,13 | 0,04 | 0,00 | 0,08 | 4,35 | 0,21 | 0,03 | 0,00 | 0,77 | 0,05 | 0,01 | 0,22 | 1,77 | 0,04 | 0,08 | 9,54 |
| `all_sum` | 2^18 | `--no-packing` | 7,47 | 0,52 | 0,04 | 0,00 | 0,38 | 16,13 | 0,45 | 0,10 | 0,01 | 0,85 | 0,30 | 0,02 | 0,91 | 1,61 | 0,01 | 0,14 | 29,16 |
| `all_sum` | 2^18 | empaquetat | 2,59 | 0,49 | 0,04 | 0,00 | 0,21 | 5,33 | 0,49 | 0,11 | 0,01 | 0,87 | 0,19 | 0,03 | 0,95 | 2,46 | 0,00 | 0,14 | 14,16 |
| `all_sum` | 2^20 | `--no-packing` | 10,39 | 1,76 | 0,16 | 0,00 | 1,22 | 19,56 | 1,36 | 0,45 | 0,09 | 1,12 | 1,21 | 0,04 | 2,98 | 2,02 | 0,02 | 0,42 | 42,94 |
| `all_sum` | 2^20 | empaquetat | 4,80 | 1,69 | 0,12 | 0,00 | 0,28 | 7,71 | 1,47 | 0,43 | 0,07 | 1,18 | 1,15 | 0,05 | 3,22 | 3,52 | 0,10 | 0,45 | 26,13 |
| `all_sum` | 2^22 | `--no-packing` | 17,87 | 6,62 | 0,58 | 0,00 | 5,01 | 29,37 | 8,22 | 1,26 | 0,37 | 1,75 | 3,57 | 0,08 | 11,82 | 3,37 | 0,07 | 1,46 | 91,42 |
| `all_sum` | 2^22 | empaquetat | 13,30 | 6,81 | 0,51 | 0,00 | 3,88 | 14,21 | 8,52 | 1,21 | 0,35 | 1,72 | 3,79 | 0,09 | 12,01 | 7,64 | 0,31 | 1,65 | 76,00 |
| `all_sum` | 2^23 | `--no-packing` | 27,87 | 11,47 | 1,16 | 0,00 | 13,76 | 35,48 | 19,90 | 2,21 | 0,71 | 2,40 | 6,30 | 0,12 | 21,31 | 4,34 | 0,12 | 3,04 | 150,19 |
| `all_sum` | 2^23 | empaquetat | 23,46 | 11,55 | 0,96 | 0,00 | 8,70 | 18,29 | 22,93 | 2,24 | 0,72 | 2,48 | 6,35 | 0,15 | 24,74 | 11,15 | 0,59 | 3,52 | 137,83 |
| `all_prod` | 2^16 | empaquetat | 2,36 | 0,13 | 0,03 | 0,00 | 0,06 | 3,44 | 0,19 | 0,03 | 0,00 | 0,77 | 0,06 | 0,02 | 0,21 | 1,66 | 0,04 | 0,07 | 9,07 |
| `all_prod` | 2^20 | empaquetat | 8,22 | 1,75 | 0,15 | 0,00 | 1,01 | 7,87 | 2,24 | 0,45 | 0,07 | 1,49 | 0,92 | 0,10 | 4,14 | 4,69 | 0,29 | 0,49 | 33,90 |

**Pic de RSS de cada fase (GB):**

| programa | N | *layout* | clau | witness | stage 1 | stage 2 | `Q`: LDE | `Q`: avaluació | `Q`: INTT | `Q`: MSM | avaluacions | obertura | pic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^10 | `--no-packing` | 0,0 | – | 0,0 | – | – | – | – | 0,0 | – | 0,0 | 0,0 |
| `fibonacci` | 2^10 | empaquetat | 0,0 | – | 0,0 | – | – | – | – | 0,0 | – | 0,0 | 0,0 |
| `fibonacci` | 2^12 | `--no-packing` | 0,1 | – | 0,1 | – | – | – | – | 0,0 | – | 0,0 | 0,1 |
| `fibonacci` | 2^12 | empaquetat | 0,1 | – | 0,1 | – | 0,0 | – | – | 0,1 | – | 0,0 | 0,1 |
| `fibonacci` | 2^14 | `--no-packing` | 0,2 | – | 0,2 | – | – | – | – | 0,3 | – | 0,1 | 0,3 |
| `fibonacci` | 2^14 | empaquetat | 0,5 | – | 0,2 | – | 0,0 | – | – | – | – | 0,2 | 0,5 |
| `fibonacci` | 2^16 | `--no-packing` | 1,0 | – | 1,0 | – | 0,1 | – | 0,0 | 0,6 | – | 1,0 | 1,1 |
| `fibonacci` | 2^16 | empaquetat | 2,0 | – | 1,0 | – | 0,1 | – | – | 0,7 | – | 1,0 | 2,0 |
| `fibonacci` | 2^18 | `--no-packing` | 2,1 | 0,1 | 2,1 | – | 0,2 | – | – | 2,2 | – | 2,2 | 2,2 |
| `fibonacci` | 2^18 | empaquetat | 2,1 | 0,0 | 2,2 | – | 0,2 | – | – | 2,2 | 0,2 | 2,2 | 2,2 |
| `fibonacci` | 2^20 | `--no-packing` | 2,3 | 0,3 | 2,6 | – | 0,8 | 0,8 | – | 2,7 | – | 2,6 | 2,7 |
| `fibonacci` | 2^20 | empaquetat | 2,4 | 0,4 | 2,6 | – | 0,8 | 0,9 | – | 2,7 | – | 2,7 | 2,7 |
| `fibonacci` | 2^21 | `--no-packing` | 2,6 | 0,7 | 3,2 | – | 1,5 | 1,6 | 1,2 | 3,4 | – | 3,2 | 3,4 |
| `fibonacci` | 2^21 | empaquetat | 2,9 | 0,8 | 3,3 | – | 1,6 | 1,8 | 1,3 | 3,5 | – | 3,4 | 3,5 |
| `fibonacci` | 2^22 | `--no-packing` | 3,2 | 1,4 | 4,4 | – | 3,0 | 3,2 | 2,3 | 4,8 | 2,3 | 4,4 | 4,8 |
| `fibonacci` | 2^22 | empaquetat | 3,7 | 1,7 | 4,6 | – | 3,3 | 3,4 | 2,5 | 5,0 | – | 4,9 | 5,0 |
| `fibonacci` | 2^23 | `--no-packing` | 4,5 | 3,2 | 6,7 | – | 6,0 | 6,4 | 4,5 | 7,5 | 4,4 | 6,8 | 7,5 |
| `fibonacci` | 2^23 | empaquetat | 5,5 | 3,7 | 7,3 | – | 6,5 | 6,9 | 5,0 | 8,0 | 4,8 | 7,8 | 8,0 |
| `fibonacci` | 2^24 | `--no-packing` | 7,0 | 6,3 | 11,5 | – | 12,0 | 12,8 | 9,0 | 13,0 | 9,1 | 11,5 | 13,0 |
| `fibonacci` | 2^24 | empaquetat | 9,0 | 7,2 | 12,5 | – | 13,0 | 13,8 | 10,0 | 14,0 | 9,6 | 13,5 | 14,0 |
| `all_sum` | 2^10 | `--no-packing` | 0,0 | – | 0,0 | 0,0 | 0,0 | – | – | 0,1 | – | 0,0 | 0,1 |
| `all_sum` | 2^10 | empaquetat | 0,1 | – | 0,1 | 0,0 | 0,0 | – | – | 0,0 | – | 0,0 | 0,1 |
| `all_sum` | 2^12 | `--no-packing` | 0,1 | 0,0 | 0,1 | 0,1 | 0,0 | – | – | 0,1 | – | 0,0 | 0,1 |
| `all_sum` | 2^12 | empaquetat | 0,5 | – | 0,3 | 0,1 | 0,0 | 0,0 | – | 0,1 | – | 0,5 | 0,5 |
| `all_sum` | 2^14 | `--no-packing` | 0,3 | 0,0 | 0,3 | 0,3 | 0,1 | – | – | 0,6 | – | 0,2 | 0,6 |
| `all_sum` | 2^14 | empaquetat | 2,0 | – | 2,0 | 0,6 | 0,1 | 0,1 | – | 0,2 | – | 2,1 | 2,1 |
| `all_sum` | 2^16 | `--no-packing` | 1,1 | 0,1 | 1,1 | 1,2 | 0,3 | – | – | 2,2 | – | 2,2 | 2,2 |
| `all_sum` | 2^16 | empaquetat | 2,1 | 0,1 | 2,2 | 2,2 | 0,3 | – | – | 2,3 | – | 2,3 | 2,3 |
| `all_sum` | 2^18 | `--no-packing` | 2,2 | 0,4 | 2,6 | 2,7 | 1,0 | 1,0 | – | 2,8 | – | 2,7 | 2,8 |
| `all_sum` | 2^18 | empaquetat | 2,5 | 0,5 | 2,9 | 2,8 | 1,1 | 1,1 | – | 2,9 | – | 3,0 | 3,0 |
| `all_sum` | 2^20 | `--no-packing` | 2,9 | 1,9 | 4,6 | 4,8 | 3,8 | 3,8 | 2,8 | 5,1 | – | 4,9 | 5,1 |
| `all_sum` | 2^20 | empaquetat | 4,1 | 2,3 | 5,7 | 5,3 | 4,2 | 4,3 | 3,3 | 5,5 | 3,2 | 5,8 | 5,8 |
| `all_sum` | 2^22 | `--no-packing` | 5,8 | 8,2 | 12,4 | 13,1 | 15,2 | 15,4 | 11,3 | 14,3 | 11,4 | 13,5 | 15,4 |
| `all_sum` | 2^22 | empaquetat | 10,4 | 9,8 | 16,8 | 15,4 | 16,9 | 17,1 | 13,0 | 16,0 | 12,7 | 18,1 | 18,1 |
| `all_sum` | 2^23 | `--no-packing` | 9,5 | 16,5 | 22,7 | 24,2 | 30,3 | 30,7 | 22,5 | 26,5 | 22,9 | 25,0 | 30,8 |
| `all_sum` | 2^23 | empaquetat | 18,8 | 19,9 | 31,5 | 28,7 | 33,9 | 34,2 | 26,0 | 30,0 | 25,7 | 36,3 | 36,3 |
| `all_prod` | 2^16 | empaquetat | 2,1 | 0,1 | 2,2 | 2,2 | 0,2 | – | – | 2,3 | – | 2,2 | 2,3 |
| `all_prod` | 2^20 | empaquetat | 3,7 | 2,3 | 5,6 | 5,4 | 4,1 | 4,1 | 3,2 | 5,4 | – | 5,6 | 5,6 |

Les columnes: la clau és la càrrega de l'SRS i del `.const`, amb la INTT de les columnes fixes, i les MSM de la comprovació dels commitments fixos (M26); el witness, llegir-lo del directori i fer-ne la instància C++; els *hints*, els im pols, la INTT i la MSM, les dels stages; de `Q`, l'LDE de les columnes que llegeix (`Q_EXTEND`), el domini i l'avaluació, la INTT, la MSM i la resta (els seus *buffers* i la comprovació de la fita); les avaluacions a `ξ`; de l'obertura, el càlcul de `W` i `W'`, les seves MSM i la resta (els interpolants `r_i`); i la resta de la prova, que cap temporitzador no cobreix (els JSON de la clau i el *digest* de la vkey, el transcript, els fitxers de la prova i la sortida del procés). La resta es calcula execució per execució: les medianes de les columnes no sempre sumen el total.

### H.5 Com creix amb `N`

- **Per sota de `2^18`, el temps quasi no creix:** 1,3–1,7 s del `fibonacci` de `2^10` a `2^14`, i gairebé tot és MSM i la càrrega de la clau. Cada MSM costa uns 100 ms encara que tingui 2.000 punts: la MSM d'ffiasm (`ParallelMultiexp`) obre unes 300 regions OpenMP per MSM (per cada tros de 16 bits, `processChunk`, `packThreads` i les `reduce` recursives), amb els 256 fils, i en una màquina carregada cada barrera espera el fil més lent. Amb 64 fils, la prova de `2^10` passa d'1,4 s a 0,18 s (H.7).
- **De `2^20` a `2^24`, el temps creix menys que `N log N`:** el `fibonacci`, ×2,0 de `2^20` a `2^22` i ×3,1 de `2^22` a `2^24` (`N log N` seria ×4,4); `all_sum`, ×1,9 de `2^18` a `2^20` i ×2,9 de `2^20` a `2^22`. La MSM, que és el cost principal, és lineal en `N` (finestres de 16 bits com a molt), amb un cost fix per MSM que es va diluint; les NTT sí que són `N log N`, però en són una part petita. A dalt de tot, el creixement s'acosta a lineal.
- **La memòria creix linealment a partir de `2^20`:** fins llavors la domina la de les MSM (vegeu H.7), 1,6 GB fixos amb 256 fils. Al capdamunt, uns 900 bytes per fila al `fibonacci` (14 GB a `2^24`) i uns 4,6 KB per fila a `all_sum` (18 GB a `2^22`).

### H.6 L'avaluació de `Q` per parts

**Per què calia.** Fins ara, `commitQ` estenia al *coset* sencer `g·H'` (`N' = 2^nBitsExt` punts) totes les columnes que llegeix el codi de `Q`, i després l'hi avaluava (M17, M18). Són `32·N'` bytes per columna, més `Q` mateix i els punts i els `Zi` del domini. `all_sum` té `qDeg = 2`, de manera que `N' = 4N`, i el codi de `Q` llegeix 30 columnes (14 de l'stage 1, 6 de l'stage 2 i les 10 fixes): a `N = 2^22`, 16 GB de columnes esteses, quatre vegades els coeficients de tots els polinomis compromesos (4 GB). Al `fibonacci` (5 columnes, `N' = 2N`), 5,4 GB a `2^24`, el doble. Les mesures abans del canvi (les mateixes, amb el codi d'abans) ho confirmaven:
- a `all_sum` `2^22`, el pic de la prova era el de l'avaluació de `Q`: 28,5 GB, contra 16,8 GB de l'stage 1;
- al `fibonacci` `2^24`, 16,0 GB contra 12,5;
- i els *buffers* de `Q`, posats a zero en un sol fil, costaven 11 s de 89 a `all_sum` `2^22`.

Això és el cas de M39 (una memòria diverses vegades la dels polinomis compromesos). Cap mida no era inviable en aquesta màquina, però `all_sum` a `2^24` (si el compilador en pogués fer el `pilout`, H.8) hauria necessitat uns 110 GB, dels quals 64 per a les columnes esteses.

**Com es fa ara.** `g·H'` és la unió de `N'/S` parts de `S = 2^partBits` punts, `N ≤ S ≤ N'`: la part `p` són els punts `g·ω_{N'}^(p + (N'/S)·i)`, `i < S`, és a dir `c·ω_S^i` amb `c = g·ω_{N'}^p`, i per defecte cada part és un *coset* de `H` (`S = N`). Per a cada part, `commitQ`:
1. estén cada columna que `Q` llegeix a la part amb `Lde::extendCosetPart`: el coeficient `j` es multiplica per `c^j` i es plega a `j mod S` (el polinomi té com a molt `N + |O| + 1 ≤ N'` coeficients), i després ve la FFT d'ffiasm de `S` punts;
2. en construeix el domini amb `ExpressionsDomain::cosetPart`: els punts i els `Zi` de la part, que són els del *coset* sencer als mateixos punts;
3. hi avalua `Q` amb el mateix intèrpret, per blocs de 128 files com el STARK (`expressions_pack.hpp`, `NROWS_PACK`): una columna a l'*offset* `o` es llegeix `o·S/N` punts més enllà dins de la part, com al *coset* sencer;
4. i en desa els valors a les posicions `p + (N'/S)·i` de `Q` al *coset*.

Després, la INTT de `Q` i tota la resta no canvien. Les columnes esteses ocupen `32·S` bytes cadascuna en lloc de `32·N'`, i el domini, el mateix. Amb `S = N'`, la part única és el *coset* sencer: `extendCoset` i `ExpressionsDomain::coset` en són ara aquest cas.

**És el mateix `Q`, bit a bit.** Cada valor és el mateix element del cos al mateix punt, calculat amb aritmètica exacta, i la forma de Montgomery d'ffiasm és canònica. Ho comproven:
- `pilfflonk_lde_test.cpp` (`testExtendCosetParts`): per a cada mida de part i cada part, els valors són els d'`extendCoset` al punt corresponent, byte a byte, amb polinomis de menys, tants i més coeficients que la part, en lot i en un sol lloc;
- `pilfflonk_expressions_test.cpp`: els `Zi` de cada part, de tots els tipus de domini, són els del *coset* sencer, byte a byte;
- `pilfflonk_prover_test.cpp` (`testQInParts`): amb el Fibonacci sencer, amb `Q` partit i amb `Q` partit i empaquetat, per a cada mida de part, els coeficients dels trossos de `Q`, els commitments, les avaluacions, `W`, `W'`, `inv` i `invZh` són els de l'avaluació sencera, i l'API C refusa una mida fora de rang;
- l'E2E de la CLI (`the_parts_of_q_give_the_same_proof`, a `proves_and_rejects_every_change` i `proves_and_verifies`): per a totes les *fixtures* E2E (el Fibonacci empaquetat i no, els *offsets* amb signe, els quatre dominis, `Q` partit, els busos de suma i de producte, i els exemples de pil-fflonk), la prova amb cada mida de part, de `nBits` a `nBitsExt`, és la de `pilfflonk prove` amb la mateixa llavor, byte a byte, i `nBits − 1` i `nBitsExt + 1` es refusen.

**L'API.** `Instance::setQPartBits(bits)` (C++), `pilfflonk_instance_set_q_part_bits` (C) i `ProveOptions::q_part_bits` (Rust, `None` per defecte, que és `nBits`). No canvia res del format ni del protocol: només com el prover recorre `g·H'`.

**Abans i després** (256 fils):

| programa | N | *layout* | pic de `Q`: avaluació (GB) | pic de la prova (GB) | `Q` (s) | prova (s) |
|---|---|---|---|---|---|---|
| `fibonacci` | 2^20 | `--no-packing` | 0,9 → 0,8 | 2,7 → 2,7 | 1,71 → 1,47 | 10,63 → 9,82 |
| `fibonacci` | 2^20 | empaquetat | 0,9 → 0,9 | 2,8 → 2,7 | 1,68 → 1,52 | 9,61 → 9,62 |
| `fibonacci` | 2^22 | `--no-packing` | 3,7 → 3,2 | 5,0 → 4,8 | 4,36 → 3,69 | 20,05 → 18,63 |
| `fibonacci` | 2^22 | empaquetat | 4,0 → 3,4 | 5,2 → 5,0 | 4,37 → 3,69 | 19,93 → 18,77 |
| `fibonacci` | 2^24 | `--no-packing` | 15,0 → 12,8 | 15,0 → 13,0 | 13,48 → 18,60 | 58,42 → 55,45 |
| `fibonacci` | 2^24 | empaquetat | 16,0 → 13,8 | 16,0 → 14,0 | 14,58 → 11,74 | 63,07 → 59,12 |
| `all_sum` | 2^20 | `--no-packing` | 6,7 → 3,8 | 6,7 → 5,1 | 6,77 → 4,24 | 47,98 → 42,94 |
| `all_sum` | 2^20 | empaquetat | 7,1 → 4,3 | 7,2 → 5,8 | 8,17 → 4,31 | 31,40 → 26,13 |
| `all_sum` | 2^22 | `--no-packing` | 26,8 → 15,4 | 26,8 → 15,4 | 28,83 → 15,18 | 108,06 → 91,42 |
| `all_sum` | 2^22 | empaquetat | 28,5 → 17,1 | 28,5 → 18,1 | 27,23 → 15,59 | 88,77 → 76,00 |

La memòria de l'avaluació de `Q` baixa un 40 % a `all_sum` i un 15 % al `fibonacci`, i el pic de la prova ja no és el de `Q`: a `all_sum` `2^22`, de l'stage 1 fins a l'obertura, les fases queden entre 15 i 18 GB. `Q` també triga menys (15,6 s en lloc de 27,2 a `all_sum` `2^22`): posa a zero 4 vegades menys memòria (3,8 s en lloc de 10,7) i fa l'LDE en FFT de `N` punts (8,5 s en lloc de 13,1). El `fibonacci` `2^24` sense empaquetar és l'excepció: la primera LDE de la primera part hi triga 8,4 s en lloc d'1,7 a les tres execucions (H.7).

### H.7 Els colls d'ampolla

**On va el temps**, a les mides més grans (empaquetat, 256 fils):
- **Les MSM: 36–38 %.** Al `fibonacci` `2^24`, 21 s de 59: 8,4 s les de l'stage 1, 2,6 la de `Q`, 7,0 les de `W` i `W'` (de `2N` coeficients) i 3,2 la comprovació dels commitments fixos a la càrrega de la clau. A `all_sum` `2^22`, 29 s de 76. Sense empaquetar, fins al 50 %: `all_sum` en fa 31 en lloc de 9. La MSM d'ffiasm fa uns 6 milions de punts per segon amb 256 fils.
- **`W` i `W'`: 16–18 %** (11–12 s). Per a cada `f` i cada *offset*, una divisió de `rapidsnark` (`divByMonic`, seqüencial), l'empaquetat i sumes de polinomis de `k·N` coeficients.
- **La càrrega de la clau: 16–18 %** (9,5 s al `fibonacci` `2^24`, 13 s a `all_sum` `2^22`): llegir l'SRS (2,2 s per 2 GB) i el `.const` (1 GB), la INTT de les columnes fixes i, sobretot, tornar a calcular els commitments fixos a cada prova (M26: 3,2 i 5,1 s).
- **`Q`: 20 %** (11,7 i 15,6 s): l'LDE de les columnes (3,6 i 8,5 s), la posada a zero dels seus *buffers* i de `Q` (3,3 i 3,8 s), el domini (1,3 s) i la MSM; l'avaluació amb el *bytecode* en si és petita (0,3–0,5 s).
- **El witness: 5–9 %** (3,1 i 6,8 s): llegir 1–2 GB, comprovar que cada valor és canònic (tres vegades: a Rust, a l'API C i a la instància) i copiar-lo a la instància.
- **Els *hints* i els im pols** no hi pesen: 0,5 s a `all_sum` `2^22`, tot i que l'acumulació de `gsum`/`gprod` és seqüencial.

**La memòria de les MSM.** Cada fil té els seus 2^16 *buckets* (`PaddedPoint`, 96 bytes): 1,6 GB amb 256 fils, 0,4 GB amb 64. Fins a `2^18`, és gairebé tot el pic de la prova. Al capdamunt, la memòria és la dels polinomis: l'SRS (`64` bytes per potència), els coeficients i les avaluacions a `H` de cada columna (que la instància guarda totes fins al final), el witness (que Rust també guarda fins al final) i, a `Q`, les columnes d'una part.

**Els fils.** Amb `OMP_NUM_THREADS=64` (i les càrregues de les execucions entre parèntesis):

| programa | N | *layout* | 256 fils: s (càrrega) | 64 fils: s (càrrega) | totes les MSM (s) | pic (GB) |
|---|---|---|---|---|---|---|
| `fibonacci` | 2^10 | empaquetat | 1,38 (70–70) | 0,18 (194–194) | 1,01 → 0,13 | 0,0 → 0,0 |
| `fibonacci` | 2^16 | empaquetat | 3,42 (138–158) | 0,83 (184–194) | 2,97 → 0,69 | 2,0 → 0,5 |
| `fibonacci` | 2^20 | empaquetat | 9,62 (162–173) | 5,37 (166–175) | 6,89 → 3,07 | 2,7 → 1,3 |
| `fibonacci` | 2^24 | empaquetat | 59,12 (47–152) | 58,31 (47–52) | 21,07 → 18,56 | 14,0 → 14,0 |
| `all_sum` | 2^16 | empaquetat | 9,54 (147–179) | 2,79 (52–52) | 8,11 → 2,14 | 2,3 → 0,8 |
| `all_sum` | 2^20 | empaquetat | 26,13 (21–132) | 19,90 (48–54) | 14,91 → 8,90 | 5,8 → 4,5 |
| `all_sum` | 2^22 | empaquetat | 76,00 (125–173) | 72,66 (22–45) | 28,64 → 23,56 | 18,1 → 18,1 |

Amb 64 fils, les proves petites i mitjanes són de 2 a 7 vegades més ràpides i fan servir 4 vegades menys memòria, perquè la MSM paga menys regions i menys *buckets*; a `2^22` i `2^24`, els dos triguen el mateix. Els 256 fils per defecte d'OpenMP no convenen a la MSM d'ffiasm per sota de `2^22`.

**La variància.** En algunes execucions, una sola FFT gran triga 5–20 vegades més que les altres del mateix tipus (per exemple, una INTT de l'stage 1 de `2^24`, 6,8–7,1 s en lloc de 0,3 s, a dues de tres execucions empaquetades; la primera LDE de `Q` sense empaquetar a totes tres). Sempre és la primera passada amb 256 fils sobre memòria que un sol fil acaba de posar a zero, i la segona part amb els mateixos *buffers* és 5 vegades més ràpida. La causa probable és el balanceig automàtic de NUMA del nucli (`numa_balancing = 1`, 2 nodes), que migra les pàgines; no s'ha pogut confirmar, perquè demanaria `numactl` o permisos de `root`. Afecta la mediana d'alguns punts de `2^23` i `2^24` (la dispersió de la taula).

### H.8 El límit: el compilador, no el prover

El prover arriba a `N = 2^24` (el límit de P2) amb el `fibonacci`, en 59 s i 14 GB, i res no fa pensar que no hi arribi amb programes més grans: `all_sum` a `2^24` necessitaria uns 5 minuts i uns 75 GB, i un SRS de `9·2^24 + 8` potències (`< 2^28`, dins de P2). Qui no hi arriba és `pil2com`:

| programa | N | temps | pic de RSS | `pilout` | resultat |
|---|---|---|---|---|---|
| `all_sum` | 2^22 | 19 min 16 s | 19,5 GB | 620 MB | bé |
| `all_sum` | 2^23 | 47 min 58 s | 40,3 GB | 1.241 MB | bé |
| `all_sum` | 2^24 | 2 h 31 min | 95,1 GB | 2.482 MB | error en escriure el `pilout` |

- **Per què.** El compilador guarda les columnes fixes dins del `pilout` (sobre BN254, `fixed-to-file` no hi funciona: §3.3), i `all` en té 4 d'amplada completa (`S1`–`S3` i l'`ID` de la connexió de la std): uns 148 bytes per fila. A `2^24`, el `pilout` fa 2.481.667.577 bytes, i Node no en pot escriure més de `2^31 − 1` en una crida: `RangeError [ERR_OUT_OF_RANGE]` a `fs.writeFileSync` (`pil2-compiler/src/proto_out.js:148`, des de `processor.js:303`), després de 2 h 14 min d'execució, gairebé tot als dos bucles de `N` iteracions de `sm_connection` (`connection.pil:32` i `:40`: 41 i 61 min). Tampoc no seria un missatge protobuf vàlid (el límit és 2 GiB).
- **El temps.** És el del PIL de les *fixtures*: `sm_connection` calcula les permutacions `S1`–`S3` amb bucles del PIL, a 150–220 µs per iteració a `2^24`, i el temps creix més que linealment amb `N`: ×4,0 de `2^20` a `2^22`, però ×2,5 de `2^22` a `2^23` i ×3,2 de `2^23` a `2^24`. Qualsevol programa amb una connexió de la std (`all`, la Connection sola) té el mateix límit.
- **La decisió** (de l'usuari, 30-09-2026): la compilació de `2^24` no es torna a intentar, i la de `2^23`, amb un límit de temps, va acabar a temps; el programa amb busos queda, doncs, a `N ≤ 2^23`, i `2^24` es mesura només amb el `fibonacci`, que té columnes fixes trivials i cobreix el límit de P2 per al prover. La conclusió: **a aquesta mida, el coll d'ampolla és compilar programes grans de BN254 amb el compilador PIL2 en JS, no provar-los.** Es podria resoldre al compilador (escriure el `pilout` per trossos, o les columnes fixes a part, un `fixed-to-file` per a BN254) o calculant les permutacions fora del PIL.
- **Memòria.** Cap procés de l'informe no ha passat de 250 GB (el límit per procés en aquesta màquina compartida, per decisió de l'usuari): el més gran és la compilació de `2^24`, 95 GB. `bench.sh` limita el *heap* de Node a 250 GB (`BENCH_NODE_HEAP_MB`).

### H.9 Oportunitats (no implementades)

Per ordre del guany estimat a les mides grans. Cap no canvia la prova ni el protocol:

| oportunitat | on | guany estimat |
|---|---|---|
| Una MSM que escali: *buckets* compartits o per grups de fils, finestra segons `n` i els fils, o la MSM de `pil2-stark/src/bn128/src/msm` (Fase 5) | `Srs::commit`, ffiasm `multiexp.c.hpp` | les MSM són el 36–50 %: la meitat del temps, i 1,6 GB de *buckets* |
| Limitar els fils de cada MSM segons el seu nombre de punts (o fer les MSM d'un stage en paral·lel, cadascuna amb menys fils) | `Srs::commit` | 1,3–7× a les proves de `2^10` a `2^20` (H.7, 64 fils) |
| No tornar a calcular els commitments fixos a cada prova: fer-ho una vegada per clau i guardar-ne un resum, o només en mode de depuració | `check_srs_and_fixed` (M26) | 3–5 s (5–7 %) a `2^22`–`2^24`; 11 s sense empaquetar a `all_sum` `2^22` |
| Paral·lelitzar la divisió per `Z_{T_i}` de `W` i `W'` (`X^k − a` es divideix per blocs) i l'empaquetat | `ShplonkProver::quotientW/quotientWp` | la meitat dels 11–12 s de `W` i `W'` (8–10 %) |
| Reservar sense posar a zero els *buffers* de `Q` (les columnes de la part i `qValues`) i tocar-los per primer cop en paral·lel; alliberar les avaluacions a `H` de les columnes abans de `Q` | `Instance::commitQ`, `commitF` | 3–4 s de `Q` i la variància NUMA de H.7; un 10–20 % del pic a `Q` |
| No calcular els punts `x` del domini quan totes les restriccions són `everyRow`, i guardar el `Zi` d'`everyRow` com un sol valor per part | `ExpressionsDomain::cosetPart` | 1,3 s i `64·S` bytes a `2^24` |
| Un sol control de canonicitat del witness (ara el fan Rust, l'API C i la instància), i no guardar la còpia Rust del witness fins al final de la prova | `WitnessInstance`, `pilfflonk_instance_new`, `Instance` | 1–2 s i `32·N·C` bytes (1–2 GB) |
| Inverses en lot i acumulació en paral·lel (per trossos, amb un prefix) als *hints* | `computeHintColumns` | petit: 0,5 s a `all_sum` `2^22` |

### H.10 Com es reprodueix

Les eines són a `pilfflonk/bench/` (M39):
- `bench.sh`: `ptau <potències>` escriu el `ptau` de prova; `run <programa> <bits>…` compila (un cop, i en desa el `pilout` a `$BENCH_DIR/pilouts`), fa el setup, el witness, les proves i les verificacions, i ho escriu a `$BENCH_DIR/{compile,setup,prove}.tsv`; `summary` en fa les taules (`summary.mjs`, medianes i dispersió). La capçalera en descriu les variables: `BENCH_DIR`, `BENCH_PTAU`, `BENCH_PACKING`, `BENCH_REPEATS`, `BENCH_KEEP`, `BENCH_NODE_HEAP_MB` i `OMP_NUM_THREADS`.
- `fibonacci.pil`, `all_sum.pil` i `all_prod.pil`: els programes, amb `N = 2^BENCH_BITS`.
- `inputs.rs`: l'exemple `pilfflonk_bench_inputs` de `proofman-cli` (el `ptau` de prova i els witness).

```sh
cargo build --release --features proofman-starks-lib-c/cpu-only \
    --bin proofman-cli --bin proofman-setup --example pilfflonk_bench_inputs
export BENCH_DIR=/tmp/pilfflonk-bench PIL2C_EXEC=<pil2-compiler>/src/pil.js
pilfflonk/bench/bench.sh ptau 75497536          # 9·2^23 + 64 potències: 4,8 GB, uns 20 min
BENCH_PACKING="packed nopacking" BENCH_REPEATS=3 pilfflonk/bench/bench.sh run fibonacci 10 12 14 16 18 20 22 24
BENCH_PACKING="packed nopacking" BENCH_REPEATS=3 pilfflonk/bench/bench.sh run all_sum 10 12 14 16 18 20
OMP_NUM_THREADS=64 BENCH_PACKING=packed BENCH_REPEATS=2 pilfflonk/bench/bench.sh run fibonacci 10 16 20 24
pilfflonk/bench/bench.sh summary
rm -rf "$BENCH_DIR"                              # el ptau, els pilouts i els resultats
```

Per defecte, cada mida esborra les seves claus, el witness i les proves en acabar (`BENCH_KEEP=1` els guarda). Abans d'una mida gran cal comprovar l'espai (`df -h`): a `2^24`, el `fibonacci` necessita uns 5 GB (el `.const` i l'SRS d'una clau, el witness i el `pilout`), a més dels 4,8 GB del `ptau`, i `all_sum` a `2^23`, uns 13 GB.

## Annex I. Gas del verificador Solidity (M42)

Aquest annex és l'informe de gas de la Fase 4 (validació 3). Recull el gas de `verifyProof` i el del seu calldata per a les 73 claus de M41 (I.2), on va el gas (I.3), com creix (I.4), la comparació amb l'`FflonkVerifier` de snarkjs (I.5) i les oportunitats, que no s'implementen (I.6). Les xifres són de l'01-10-2026, amb el contracte de M40, que no canvia.

### I.1 La mesura

**Les eines.** Foundry v1.8.3 i solc 0.8.37, amb l'optimitzador a 200 *runs* (el `foundry.toml` dels tests) i l'EVM per defecte de solc 0.8.37, que té `PUSH0`. Els precompilats tenen els costos d'Istanbul (EIP-1108): `0x06` 150, `0x07` 6.000, i `0x08` 45.000 més 34.000 per parell.

**El gas de `verifyProof`.** És el de la crida, mesurat amb `gasleft()` abans i després, com fa el test de Foundry de M40 (`test/PilfflonkVerifier.t.sol`). Inclou l'execució del contracte i la crida (100 de gas, perquè l'adreça és calenta), i no inclou els 21.000 de la transacció ni el calldata. Foundry aïlla per defecte cada crida en una transacció pròpia, i així només la primera crida d'un test es mesura d'aquesta manera: les següents costen 2.500 més, perquè l'adreça és freda (EIP-2929). Per això l'E2E de M41 mesura la primera crida de cada clau, i el fuzzer executa Foundry sense aïllament (§4.5, "Validació de M42"). Les dues mesures donen el mateix gas.

**On s'executa.**
- **Una transacció que crida `verifyProof` directament** costa uns 21.000, més el calldata, més el gas de `verifyProof`, menys els 100 de la crida: de 201.805 (`fibonacci_k3`) a 470.097 (`all_sum_unpacked`).
- **Un contracte que el crida** la primera vegada en una transacció hi suma els 2.500 de l'adreça freda.

**El calldata** (EIP-2028): 16 per byte que no és zero i 4 per byte zero, del selector i dels arguments. Depèn dels valors de la prova, que aquí són els de la llavor de M41.

**On va el gas.** Ho mesura la sonda del fuzzer (§4.5, "Validació de M42"), amb la prova honesta de cada clau. La sonda és una còpia del contracte que desa `gas()` després de cada pas del cos. La seva crida costa entre 285 i 322 més que la del verificador: són els punts de mesura, que cada pas inclou (uns pocs de gas cadascun), la seva memòria i les 15 paraules que retorna. Com al verificador, el primer pas que escriu a la memòria, i per tant la fa créixer, és el transcript: la sonda guarda a la pila el gas dels dos primers punts fins després del transcript.

### I.2 Les 73 claus

El gas de `verifyProof` i el del calldata de la prova de cada clau de M41 (`foundry_accepts_the_proof_of_every_fixture`, que l'imprimeix). Es mesuren de nou, i coincideixen amb M41: els extrems de cada família de la taula de M41 (§4.5) i els valors de les claus de la taula de M40. A M42, la taula afegeix tres columnes per al model (I.4):
- **Arrels:** `Σ_i k_i·|O_i|`, les arrels de tots els `f`;
- **Horner:** `Σ_i k_i²·|O_i|`, els productes de la regla de Horner que dona el valor de cada `f_i` a cada arrel;
- **`qVerifier`:** les entrades del seu codi.

La columna *model* és la d'I.4.

| Clau | `f` | Arrels | Horner | `qVerifier` | Paraules | Publics | Codi (bytes) | Gas de `verifyProof` | Model | Gas del calldata |
|---|---|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 5 | 9 | 11 | 26 | 22 | 3 | 5.290 | 180.790 | 182.006 | 12.108 |
| `fibonacci_k3` | 3 | 9 | 23 | 26 | 18 | 3 | 5.124 | 170.869 | 169.786 | 10.036 |
| `fibonacci_unpacked` | 6 | 8 | 8 | 26 | 21 | 3 | 5.252 | 185.168 | 187.116 | 11.596 |
| `packed` | 6 | 31 | 121 | 62 | 44 | 2 | 8.787 | 229.280 | 232.132 | 22.920 |
| `packed_unpacked` | 19 | 28 | 28 | 62 | 57 | 2 | 10.094 | 301.425 | 306.162 | 29.528 |
| `signed` | 6 | 38 | 98 | 57 | 51 | 3 | 11.360 | 236.264 | 238.402 | 26.812 |
| `signed_d3` | 7 | 34 | 82 | 63 | 49 | 3 | 11.984 | 243.184 | 238.948 | 25.896 |
| `signed_d2` | 7 | 39 | 119 | 78 | 54 | 3 | 12.662 | 253.359 | 250.738 | 28.420 |
| `signed_unpacked` | 13 | 19 | 19 | 57 | 40 | 3 | 9.820 | 252.505 | 252.772 | 21.264 |
| `signed_unpacked_d2` | 20 | 26 | 26 | 78 | 61 | 3 | 11.889 | 313.881 | 311.768 | 32.016 |
| `signed_split_m1` | 6 | 40 | 106 | 57 | 54 | 3 | 11.537 | 239.509 | 241.802 | 28.408 |
| `signed_split_m1_unpacked` | 15 | 21 | 21 | 57 | 47 | 3 | 10.397 | 270.324 | 268.992 | 24.788 |
| `signed_split_m2` | 6 | 39 | 101 | 57 | 53 | 3 | 11.597 | 238.864 | 240.002 | 27.932 |
| `signed_split_m2_unpacked` | 14 | 20 | 20 | 57 | 44 | 3 | 10.149 | 261.811 | 260.882 | 23.288 |
| `domain_FirstRow` | 5 | 5 | 5 | 13 | 19 | 2 | 4.030 | 174.065 | 174.828 | 10.084 |
| `domain_FirstRow_unpacked` | 5 | 5 | 5 | 13 | 19 | 2 | 4.030 | 174.065 | 174.828 | 10.060 |
| `domain_FirstRow_d2` | 5 | 7 | 11 | 19 | 21 | 2 | 4.521 | 178.939 | 178.664 | 11.108 |
| `domain_FirstRow_d2_unpacked` | 7 | 7 | 7 | 19 | 25 | 2 | 4.631 | 191.596 | 191.684 | 13.132 |
| `domain_FirstRow_split` | 5 | 6 | 8 | 13 | 21 | 2 | 4.395 | 177.686 | 176.428 | 11.084 |
| `domain_LastRow` | 5 | 5 | 5 | 13 | 19 | 2 | 4.030 | 174.065 | 174.828 | 10.060 |
| `domain_LastRow_unpacked` | 5 | 5 | 5 | 13 | 19 | 2 | 4.030 | 174.065 | 174.828 | 10.096 |
| `domain_LastRow_d2` | 5 | 7 | 11 | 19 | 21 | 2 | 4.540 | 178.910 | 178.664 | 11.120 |
| `domain_LastRow_d2_unpacked` | 7 | 7 | 7 | 19 | 25 | 2 | 4.631 | 191.596 | 191.684 | 13.120 |
| `domain_LastRow_split` | 5 | 6 | 8 | 13 | 21 | 2 | 4.439 | 177.570 | 176.428 | 11.132 |
| `domain_Frames` | 6 | 19 | 33 | 39 | 34 | 0 | 8.656 | 206.119 | 205.294 | 17.388 |
| `domain_Frames_unpacked` | 9 | 15 | 15 | 39 | 36 | 0 | 7.504 | 216.594 | 218.424 | 18.460 |
| `domain_Frames_d2` | 7 | 23 | 55 | 57 | 40 | 0 | 9.270 | 225.208 | 221.312 | 20.436 |
| `domain_Frames_d2_unpacked` | 15 | 21 | 21 | 57 | 54 | 0 | 9.303 | 269.241 | 268.992 | 27.640 |
| `domain_Frames_split` | 6 | 20 | 36 | 39 | 36 | 0 | 8.973 | 208.791 | 206.894 | 18.448 |
| `domain_Domains` | 6 | 11 | 13 | 25 | 28 | 2 | 6.132 | 191.270 | 191.410 | 14.704 |
| `domain_Domains_unpacked` | 7 | 9 | 9 | 25 | 28 | 2 | 5.941 | 195.379 | 195.120 | 14.668 |
| `domain_Domains_d2` | 6 | 15 | 33 | 37 | 32 | 2 | 6.952 | 202.262 | 199.882 | 16.704 |
| `domain_Domains_d2_unpacked` | 11 | 13 | 13 | 37 | 40 | 2 | 7.140 | 230.467 | 228.832 | 20.776 |
| `domain_Domains_split` | 6 | 12 | 20 | 25 | 30 | 2 | 6.465 | 196.050 | 193.410 | 15.680 |
| `domain_Domains_split_unpacked` | 8 | 10 | 10 | 25 | 32 | 2 | 6.270 | 204.560 | 203.230 | 16.704 |
| `sum_bus` | 6 | 16 | 34 | 32 | 31 | 1 | 7.031 | 203.284 | 200.752 | 16.040 |
| `sum_bus_unpacked` | 10 | 12 | 12 | 32 | 31 | 1 | 6.679 | 219.706 | 220.192 | 16.028 |
| `sum_bus_degree4` | 6 | 16 | 34 | 34 | 31 | 1 | 7.110 | 203.494 | 200.964 | 16.040 |
| `sum_bus_degree4_unpacked` | 10 | 12 | 12 | 34 | 31 | 1 | 6.777 | 219.887 | 220.404 | 15.992 |
| `prod_bus` | 6 | 11 | 15 | 32 | 26 | 1 | 6.064 | 191.791 | 192.352 | 13.480 |
| `prod_bus_unpacked` | 8 | 10 | 10 | 32 | 29 | 1 | 6.096 | 203.700 | 203.972 | 14.968 |
| `prod_bus_im` | 6 | 15 | 31 | 67 | 30 | 1 | 7.918 | 203.625 | 202.862 | 15.516 |
| `prod_bus_im_unpacked` | 11 | 13 | 13 | 67 | 38 | 1 | 7.785 | 232.387 | 232.012 | 19.600 |
| `prod_bus_im_split` | 6 | 16 | 34 | 67 | 32 | 1 | 8.153 | 206.102 | 204.462 | 16.492 |
| `plookup_sum` | 9 | 16 | 22 | 39 | 33 | 0 | 7.261 | 218.885 | 220.424 | 16.504 |
| `plookup_sum_unpacked` | 12 | 14 | 14 | 39 | 35 | 0 | 7.461 | 237.062 | 237.154 | 17.576 |
| `plookup_sum_split` | 8 | 17 | 29 | 39 | 33 | 0 | 7.384 | 215.373 | 215.714 | 16.564 |
| `plookup_prod` | 9 | 16 | 22 | 47 | 33 | 0 | 7.518 | 219.564 | 221.272 | 16.516 |
| `plookup_prod_unpacked` | 12 | 14 | 14 | 47 | 35 | 0 | 7.712 | 237.799 | 238.002 | 17.552 |
| `plookup_prod_split` | 8 | 17 | 29 | 47 | 33 | 0 | 7.641 | 216.052 | 216.562 | 16.564 |
| `permutation_sum` | 5 | 13 | 29 | 37 | 26 | 0 | 6.341 | 191.382 | 190.172 | 13.328 |
| `permutation_sum_unpacked` | 9 | 11 | 11 | 37 | 32 | 0 | 6.436 | 212.497 | 212.612 | 16.388 |
| `permutation_prod` | 5 | 11 | 19 | 38 | 24 | 0 | 6.178 | 186.580 | 186.678 | 12.316 |
| `permutation_prod_unpacked` | 8 | 10 | 10 | 38 | 29 | 0 | 6.239 | 204.149 | 204.608 | 14.840 |
| `permutation_prod_split` | 6 | 12 | 20 | 38 | 28 | 0 | 6.508 | 195.762 | 194.788 | 14.340 |
| `connection_sum` | 7 | 15 | 31 | 63 | 28 | 0 | 7.851 | 208.979 | 209.148 | 14.352 |
| `connection_sum_unpacked` | 13 | 15 | 15 | 63 | 36 | 0 | 8.443 | 246.729 | 247.808 | 18.484 |
| `connection_sum_split` | 8 | 16 | 30 | 63 | 32 | 0 | 8.193 | 218.367 | 217.058 | 16.388 |
| `connection_prod` | 6 | 14 | 28 | 62 | 25 | 0 | 7.736 | 201.344 | 200.732 | 12.792 |
| `connection_prod_unpacked` | 11 | 13 | 13 | 62 | 30 | 0 | 7.993 | 229.561 | 231.482 | 15.400 |
| `connection_prod_split` | 7 | 15 | 27 | 62 | 29 | 0 | 7.866 | 208.104 | 208.642 | 14.852 |
| `range_check_sum` | 6 | 13 | 19 | 24 | 28 | 0 | 6.032 | 193.847 | 194.504 | 14.352 |
| `range_check_sum_unpacked` | 8 | 10 | 10 | 24 | 27 | 0 | 5.902 | 202.310 | 203.124 | 13.876 |
| `range_check_prod` | 6 | 17 | 29 | 62 | 30 | 0 | 7.999 | 202.353 | 204.732 | 15.388 |
| `range_check_prod_unpacked` | 9 | 13 | 13 | 62 | 30 | 0 | 7.614 | 214.998 | 218.062 | 15.388 |
| `range_check_prod_split` | 6 | 18 | 32 | 62 | 32 | 0 | 8.234 | 204.830 | 206.332 | 16.364 |
| `all_sum` | 9 | 36 | 190 | 147 | 55 | 3 | 12.545 | 275.771 | 274.672 | 28.572 |
| `all_sum_unpacked` | 31 | 35 | 35 | 147 | 82 | 3 | 15.328 | 406.897 | 405.492 | 42.300 |
| `all_sum_split` | 9 | 37 | 193 | 147 | 57 | 3 | 12.638 | 277.627 | 276.272 | 29.572 |
| `all_prod` | 9 | 35 | 157 | 158 | 52 | 3 | 12.963 | 270.923 | 271.238 | 27.024 |
| `all_prod_unpacked` | 30 | 34 | 34 | 158 | 79 | 3 | 15.474 | 399.271 | 398.548 | 40.764 |
| `all_prod_split` | 9 | 38 | 180 | 164 | 56 | 3 | 13.288 | 276.155 | 278.074 | 29.024 |
| `all_prod_split3` | 9 | 37 | 165 | 158 | 55 | 3 | 13.125 | 274.139 | 274.638 | 28.560 |

El més car és `all_sum` amb `--no-packing` (31 `f`), i el contracte més gran, `all_prod` amb `--no-packing` (15.474 bytes, per sota dels 24.576 de l'EIP-170). El calldata va de 10.036 a 42.300 de gas.

### I.3 On va el gas

Aquesta taula és de les 13 claus del fuzzer, amb el gas de cada pas del cos del contracte (els passos de §4.5, "Els passos"):
- **Entrada:** `checkInput`;
- **Transcript:** `computeChallenges`;
- **`Z_H`, `invZh`:** `computeZh`;
- **`Zi`:** `computeZi`, amb les inverses auxiliars;
- **`Q(ξ)`:** `computeQ`;
- **Trossos:** `checkQPieces`;
- **Arrels:** `computeRoots`;
- **Inverses:** `computeInversions` amb `inverseArray`;
- **`r_i(y)`:** `computeR`;
- **F, E, J:** `computeFEJ`;
- ***Pairing*:** `checkPairing`.

La **resta** és la crida, el *dispatch*, la descodificació ABI dels arguments i el retorn: `verifyProof` menys el cos. Els **precompilats** són el mínim que el protocol gasta en precompilats, `113.000 + 6.150·(f + 2) + 100·(2f + 5)`:
- `f + 2` multiplicacions: `q_i·[f_i]` per a `f − 1` dels `f`, `E`, `J` i `y·[W']`;
- `f + 2` sumes;
- un *pairing* de dos parells;
- 100 per cada crida.

| Clau | `f` | `verifyProof` | Entrada | Transcript | `Z_H`, `invZh` | `Zi` | `Q(ξ)` | Trossos | Arrels | Inverses | `r_i(y)` | F, E, J | *Pairing* | Resta | Precompilats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 5 | 180.790 | 4.885 | 1.205 | 1.022 | 32 | 1.933 | – | 1.421 | 6.749 | 1.533 | 40.182 | 121.206 | 599 | 157.550 (87 %) |
| `fibonacci_unpacked` | 6 | 185.168 | 4.752 | 1.175 | 900 | 32 | 1.933 | – | 324 | 6.058 | 1.125 | 47.044 | 121.206 | 596 | 163.900 (89 %) |
| `packed` | 6 | 229.280 | 7.678 | 2.017 | 1.113 | 32 | 4.600 | – | 6.901 | 23.923 | 14.092 | 47.044 | 121.206 | 651 | 163.900 (71 %) |
| `packed_unpacked` | 19 | 301.425 | 11.407 | 2.072 | 696 | 32 | 4.600 | – | 568 | 20.155 | 3.669 | 136.306 | 121.206 | 691 | 246.450 (82 %) |
| `signed_split_m1` | 6 | 239.509 | 9.141 | 2.132 | 1.071 | 32 | 4.160 | 618 | 6.880 | 33.417 | 13.114 | 47.044 | 121.206 | 671 | 163.900 (68 %) |
| `signed_split_m1_unpacked` | 15 | 270.324 | 9.710 | 1.914 | 747 | 32 | 4.160 | 618 | 1.576 | 18.032 | 2.810 | 108.844 | 121.206 | 652 | 221.050 (82 %) |
| `domain_Domains` | 6 | 191.270 | 5.800 | 1.260 | 847 | 712 | 1.834 | – | 1.579 | 8.574 | 1.771 | 47.051 | 121.206 | 613 | 163.900 (86 %) |
| `domain_Domains_split_unpacked` | 8 | 204.560 | 6.832 | 1.304 | 696 | 654 | 1.834 | 521 | 493 | 8.176 | 1.421 | 60.789 | 121.206 | 611 | 176.600 (86 %) |
| `sum_bus` | 6 | 203.284 | 6.066 | 1.687 | 1.071 | 32 | 2.349 | – | 4.668 | 13.924 | 4.591 | 47.051 | 121.206 | 616 | 163.900 (81 %) |
| `prod_bus_unpacked` | 8 | 203.700 | 6.300 | 1.563 | 747 | 32 | 2.326 | – | 493 | 8.182 | 1.427 | 60.789 | 121.206 | 612 | 176.600 (87 %) |
| `all_sum_unpacked` | 31 | 406.897 | 17.115 | 3.548 | 929 | 32 | 10.659 | – | 734 | 28.116 | 5.066 | 218.713 | 121.206 | 756 | 322.650 (79 %) |
| `all_sum_split` | 9 | 277.627 | 10.290 | 3.353 | 1.712 | 32 | 10.659 | 521 | 10.575 | 29.101 | 21.837 | 67.651 | 121.206 | 667 | 182.950 (66 %) |
| `all_prod_split3` | 9 | 274.139 | 9.774 | 3.360 | 1.468 | 32 | 11.546 | 618 | 9.280 | 29.406 | 19.121 | 67.644 | 121.206 | 661 | 182.950 (67 %) |

**Què se'n treu:**
- **Els precompilats són entre el 66 % i el 89 % del gas.** El *pairing* (121.206 a totes les claus) i les multiplicacions de F, E i J en són gairebé tot. `computeFEJ` és `12.716 + 6.866·(f − 1)`: la multiplicació i la suma de cada `f` llevat de `f_0`, que es copia, i les de `E` i `J`.
- **La resta és Yul, i creix amb l'empaquetat.** Les inverses (de 6.058 a 33.417) i `r_i(y)` (de 1.125 a 21.837) creixen amb les arrels i amb `k`. Per això les claus agrupades amb `k` grans (`packed`, `signed`, `all`) hi gasten un terç del total.
- **La resta de passos:**
  - l'entrada costa de 4.752 a 17.115, unes 160–210 per paraula del calldata;
  - `Q(ξ)` costa uns 75 per entrada del `qVerifier`;
  - el transcript (de 1.175 a 3.548) inclou el creixement de la memòria;
  - `Z_H` i els trossos de `Q` costen menys de 2.000 cadascun;
  - la crida i l'ABI costen de 596 a 756.

### I.4 Com creix

**El model de M41 no n'hi ha prou.** M41 deia que el gas creix "uns 170.000 i uns 7.500 per `f`". Sobre les 73 claus, aquest model s'equivoca entre −33.435 i +40.127 de gas (fins al 19 %, amb una desviació quadràtica mitjana de 22.826). La raó és que el nombre de `f` no ho diu tot: empaquetar redueix els `f`, però afegeix arrels i productes de Horner.

**El model corregit** surt de mínims quadrats sobre les 73 claus:

```
gas de verifyProof ≈ 132.900 + 6.710·f + 1.300·arrels + 100·Horner + 106·entrades del qVerifier
```

S'equivoca entre −4.737 i +4.236 (fins a l'1,7 %, desviació quadràtica mitjana de 1.597). Els termes són els d'I.3:
- **la constant:** el *pairing*, `E` i `J`;
- **cada `f`:** una multiplicació i una suma de G1;
- **cada arrel:** un denominador de Lagrange, la seva part de la inversió en lot i el seu terme de `r_i(y)`;
- **cada producte de Horner** i **cada entrada del `qVerifier`:** les seves instruccions de Yul.

**Per al gas, menys `f` sol ser millor,** encara que hi hagi més arrels:
- el Fibonacci gasta 170.869 amb `--extra-muls 0` (3 `f`), 180.790 per defecte (5 `f`) i 185.168 amb `--no-packing` (6);
- `all_sum` gasta 275.771 agrupada (9 `f`) i 406.897 amb `--no-packing` (31).

El calldata creix amb les paraules de la prova: uns 500 de gas per paraula, gairebé totes de bytes que no són zero.

### I.5 Comparació amb l'`FflonkVerifier` de snarkjs

**Com s'ha mesurat.** Tot és local, sense res baixat:
- snarkjs 0.7.6, el de `setup/pil2-stark/node_modules`, i circom 2.2.0 (`/usr/local/bin/circom`);
- un `ptau` fet aquí: `powersoftau new bn128 8`, una contribució i `prepare phase2`;
- dos circuits: `c = a·b`, amb un públic, i `c = a·b`, `d = a + b`, `e = c·d`, amb tres;
- per a cadascun: `fflonk setup`, `zkey export solidityverifier`, `wtns calculate`, `fflonk prove`, `fflonk verify` i `zkey export soliditycalldata`, codificat en ABI amb `cast calldata`;
- el contracte, amb el nom canviat, al test de M40, que en mesura la primera crida com mesura la de pilfflonk.

| Verificador | Publics | `f` | Paraules de `proof` | Codi (bytes) | Gas de `verifyProof` | Gas del calldata |
|---|---|---|---|---|---|---|
| snarkjs, `c = a·b` | 1 | 3 (`C0`, `C1`, `C2`) | 24 | 13.657 | 181.639 | 11.700 |
| snarkjs, tres senyals | 3 | 3 | 24 | 14.287 | 183.200 | 12.376 |
| pilfflonk, Fibonacci, `--extra-muls 0` | 3 | 3 | 18 | 5.124 | 170.869 | 10.036 |
| pilfflonk, Fibonacci | 3 | 5 | 22 | 5.290 | 180.790 | 12.108 |
| pilfflonk, `all_sum` | 3 | 9 | 55 | 12.545 | 275.771 | 28.572 |

**Què en surt:**
- **El de snarkjs gasta el mateix per a qualsevol circuit.** La seva prova sempre té 24 paraules (`C1`, `C2`, `W`, `W'`, 15 avaluacions i `inv`), i la seva F sempre té dues multiplicacions, perquè `C0` és fix. Només els publics el fan créixer: uns 780 de gas per públic.
- **pilfflonk depèn de l'AIR.** Amb tres `f`, com snarkjs, el Fibonacci gasta un 7 % menys que el de snarkjs amb els mateixos tres publics, perquè té menys avaluacions i un `qVerifier` més curt que la identitat de PLONK de snarkjs. Per defecte (5 `f`) en gasta un 1,3 % menys.
- **Totes dues són a prop del mínim dels precompilats** (I.3): `113.000 + 6.150·5 + 1.100 = 144.850` per a tres `f`, i snarkjs també fa cinc multiplicacions i cinc sumes.
- **Els contractes de pilfflonk són més petits** (5,1 kB contra 13,7 kB). snarkjs escriu tot el seu codi en línia recta, sense cap bucle: les arrels de `C0`, `C1` i `C2`, els seus denominadors i la identitat de PLONK.

### I.6 Oportunitats (no implementades)

El contracte es queda tal com el va revisar M40: cap d'aquestes oportunitats no s'implementa. Les mesures s'han fet amb còpies del contracte del Fibonacci i d'`all_sum` amb `--no-packing` al directori de proves, no al generador. Van per ordre de guany:

| Oportunitat | On | Estalvi |
|---|---|---|
| Compilar amb més *runs* de l'optimitzador: és una decisió de qui desplega el contracte, perquè el generador no la fixa. Amb 200 *runs*, `q` apareix una sola vegada al codi del Fibonacci, i es llegeix amb `codecopy` (el punt obert de M41); amb 1.000.000, hi apareix 105 vegades com a `PUSH32` | el `foundry.toml` o el `solc` de qui el desplega | Mesurat. Amb 1.000 o 10.000 *runs*: de −928 a −942 (Fibonacci) i de −3.683 a −3.697 (`all_sum_unpacked`), amb 101–132 i 227–258 bytes més. Amb 1.000.000: −5.698 (−3,2 %) i −22.315 (−5,5 %), però amb 2.095 i 8.187 bytes més: `all_sum_unpacked` passa a 23.515 bytes, a 1 kB de l'EIP-170 |
| Desplegar els bucles i les potències d'exponent conegut: `powMod` amb cadenes d'addició, i `fillRoots`, `fillDens`, `inverseArray` i el bucle de `computeR` desplegats (l'`extendLoops` de shplonkjs) | la plantilla: `checkInput`, `computeZh`, `computeRoots`, `computeInversions`, `computeR` | Estimat, no mesurat. Cada iteració d'un bucle de Yul costa uns 25 de control (la comparació, el salt i l'increment), i una prova en fa de 60 (Fibonacci) a 300 (`all_sum_unpacked`). Una potència d'exponent `e` amb el bucle costa unes 60 per bit, i amb una cadena d'addició, unes 20. En total, de 2.000 a 8.000 (de l'1 % al 2 %), amb més codi |
| Triar l'agrupació pel gas: una opció del setup que tria els grups amb el model d'I.4 i no amb el cost del prover. Cada `extraMuls` que el prover fa servir per partir un grup costa uns 6.700 de gas a cada verificació | `pilfflonk_setup::grouping`, `setup-pilfflonk` | Mesurat al Fibonacci: −9.921 (−5,5 %) amb `--extra-muls 0`. Ja és possible a mà, amb aquesta opció |
| No comprovar la corba dels punts que un precompilat ja comprova. `0x06` i `0x07` fallen amb un punt fora de la corba o amb una coordenada `≥ q` (EIP-196), i tots els punts de la prova hi entren: els commitments a F, `W` a J i `W'` al *pairing*. La comprovació del punt a l'infinit s'hauria de mantenir, perquè els precompilats prenen `(0, 0)` com el punt a l'infinit | `checkPointBelongsToBN128Curve` | Mesurat: −996 (Fibonacci) i −3.818 (`all_sum_unpacked`). El veredicte seria el mateix, però la comprovació que refusa un punt fora de la corba ja no seria la del JS: el refusaria després del transcript |
| No val la pena: `pMem` constant (`let pMem := 0x80`), el punt obert de M41 sobre els `add(pMem, …)` | `verifyProof` | Mesurat: −9. L'optimitzador ja ho fa |
| No val la pena: comprimir els punts del calldata | el format del calldata | Cada punt estalviaria 32 bytes de calldata (uns 500 de gas), però descomprimir-lo és un `0x05` (EIP-2565, uns 1.350) |

### I.7 Com es reprodueix

Amb les eines fixades (§4.5, "Eines") i el compilador:

```sh
export PILFFLONK_FORGE=<forge> PILFFLONK_SOLC=<solc> PIL2C_EXEC=<pil2-compiler>/src/pil.js
# I.2: les 73 claus (imprimeix la taula i el temps)
cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only --test pilfflonk_prove \
    foundry_accepts_the_proof_of_every_fixture -- --ignored --nocapture
# I.3 i la validació de M42: el fuzzer, a la mida de la CI o ampliat
cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only --test pilfflonk_prove \
    foundry_and_the_js_verifier_agree_on_mutated_proofs -- --ignored --nocapture
PILFFLONK_FUZZ_CASES=10400 cargo test … foundry_and_the_js_verifier_agree_on_mutated_proofs -- --ignored --nocapture
```

**Precaucions:**
- **Mesurar la mida del contracte amb solc** (`--bin-runtime`), com fan els tests (`compile_with_solc`). `forge build --sizes` escriu una memòria cau de selectors a `~/.foundry`.
- **Tenir en compte l'aïllament de Foundry:** amb aïllament, que és el valor per defecte, una segona crida del mateix test costa 2.500 més (I.1).
