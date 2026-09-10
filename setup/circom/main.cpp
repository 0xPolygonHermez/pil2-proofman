#include "main.hpp"
#include "exec_layout.hpp"
#include "goldilocks_base_field.hpp"
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#define handle_error(msg) \
           do { perror(msg); exit(EXIT_FAILURE); } while (0)

Circom_Circuit* loadCircuit(std::string const &datFileName) {
    int fd = open(datFileName.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "loadCircuit: open(\"" << datFileName << "\") failed: "
                  << std::strerror(errno) << std::endl;
        return nullptr;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {          /* To obtain file size */
        std::cerr << "loadCircuit: fstat failed: " << std::strerror(errno) << std::endl;
        close(fd);
        return nullptr;
    }

    Circom_Circuit *circuit = new Circom_Circuit;

    u8* bdata = (u8*)mmap(NULL, sb.st_size, PROT_READ , MAP_PRIVATE, fd, 0);
    close(fd);

    circuit->InputHashMap = new HashSignalInfo[get_size_of_input_hashmap()];
    uint dsize = get_size_of_input_hashmap()*sizeof(HashSignalInfo);
    memcpy((void *)(circuit->InputHashMap), (void *)bdata, dsize);

    circuit->witness2SignalList = new u64[get_size_of_witness()];
    uint inisize = dsize;
    dsize = get_size_of_witness()*sizeof(u64);
    memcpy((void *)(circuit->witness2SignalList), (void *)(bdata+inisize), dsize);

    /* in 64 bit constants are not in a map
    circuit->circuitConstants = new u64[get_size_of_constants()];
    if (get_size_of_constants()>0) {
      inisize += dsize;
      dsize = get_size_of_constants()*sizeof(u64);
      memcpy((void *)(circuit->circuitConstants), (void *)(bdata+inisize), dsize);
    }
    */
    
    std::map<u32,IOFieldDefPair> templateInsId2IOSignalInfo1;
    IOFieldDefPair* busInsId2FieldInfo1;
    if (get_size_of_io_map()>0) {
      u32 index[get_size_of_io_map()];
      inisize += dsize;
      dsize = get_size_of_io_map()*sizeof(u32);
      memcpy((void *)index, (void *)(bdata+inisize), dsize);
      inisize += dsize;
      assert(inisize % sizeof(u32) == 0);    
      assert(sb.st_size % sizeof(u32) == 0);
      u32 dataiomap[(sb.st_size-inisize)/sizeof(u32)];
      memcpy((void *)dataiomap, (void *)(bdata+inisize), sb.st_size-inisize);
      u32* pu32 = dataiomap;
      for (int i = 0; i < get_size_of_io_map(); i++) {
	u32 n = *pu32;
	IOFieldDefPair p;
	p.len = n;
	IOFieldDef defs[n];
	pu32 += 1;
	for (u32 j = 0; j <n; j++){
	  defs[j].offset=*pu32;
	  u32 len = *(pu32+1);
	  defs[j].len = len;
	  defs[j].lengths = new u32[len];
	  memcpy((void *)defs[j].lengths,(void *)(pu32+2),len*sizeof(u32));
	  pu32 += len + 2;
	  defs[j].size=*pu32;
	  defs[j].busId=*(pu32+1);	  
	  pu32 += 2;
	}
	p.defs = (IOFieldDef*)calloc(p.len, sizeof(IOFieldDef));
	for (u32 j = 0; j < p.len; j++){
	  p.defs[j] = defs[j];
	}
	templateInsId2IOSignalInfo1[index[i]] = p;
      }
      busInsId2FieldInfo1 = (IOFieldDefPair*)calloc(get_size_of_bus_field_map(), sizeof(IOFieldDefPair));
      for (int i = 0; i < get_size_of_bus_field_map(); i++) {
	u32 n = *pu32;
	IOFieldDefPair p;
	p.len = n;
	IOFieldDef defs[n];
	pu32 += 1;
	for (u32 j = 0; j <n; j++){
	  defs[j].offset=*pu32;
	  u32 len = *(pu32+1);
	  defs[j].len = len;
	  defs[j].lengths = new u32[len];
	  memcpy((void *)defs[j].lengths,(void *)(pu32+2),len*sizeof(u32));
	  pu32 += len + 2;
	  defs[j].size=*pu32;
	  defs[j].busId=*(pu32+1);	  
	  pu32 += 2;
	}
	p.defs = (IOFieldDef*)calloc(10, sizeof(IOFieldDef));
	for (u32 j = 0; j < p.len; j++){
	  p.defs[j] = defs[j];
	}
	busInsId2FieldInfo1[i] = p;
      }
    }
    circuit->templateInsId2IOSignalInfo = move(templateInsId2IOSignalInfo1);
    circuit->busInsId2FieldInfo = busInsId2FieldInfo1;

    munmap(bdata, sb.st_size);
    
    return circuit;
}

bool check_valid_number(std::string & s, uint base){
  bool is_valid = true;
  if (base == 16){
    for (uint i = 0; i < s.size(); i++){
      is_valid &= (
        ('0' <= s[i] && s[i] <= '9') || 
        ('a' <= s[i] && s[i] <= 'f') ||
        ('A' <= s[i] && s[i] <= 'F')
      );
    }
  } else{
    for (uint i = 0; i < s.size(); i++){
      is_valid &= ('0' <= s[i] && s[i] < char(int('0') + base));
    }
  }
  return is_valid;
}

bool json2FrElements (json val, std::vector<u64> & vval){
  if (!val.is_array()) {
    std::string s_aux, s;
    uint base;
    if (val.is_string()) {
      s_aux = val.get<std::string>();
      std::string possible_prefix = s_aux.substr(0, 2);
      if (possible_prefix == "0b" || possible_prefix == "0B"){
        s = s_aux.substr(2, s_aux.size() - 2);
        base = 2;
      } else if (possible_prefix == "0o" || possible_prefix == "0O"){
        s = s_aux.substr(2, s_aux.size() - 2);
        base = 8;
      } else if (possible_prefix == "0x" || possible_prefix == "0X"){
        s = s_aux.substr(2, s_aux.size() - 2);
        base = 16;
      } else{
        s = s_aux;
        base = 10;
      }
      if (!check_valid_number(s, base)){
        std::cerr << "json2FrElements: invalid number in JSON input: " << s_aux << std::endl;
        return false;
      }
    } else if (val.is_number()) {
        double vd = val.get<double>();
        std::stringstream stream;
        stream << std::fixed << std::setprecision(0) << vd;
        s = stream.str();
        base = 10;
    } else {
        std::cerr << "json2FrElements: invalid JSON type" << std::endl;
        return false;
    }
    vval.push_back(strtoull(s.c_str(), NULL, base));
  } else {
    for (uint i = 0; i < val.size(); i++) {
      if (!json2FrElements (val[i], vval)) {
        return false;
      }
    }
  }
  return true;
}

json::value_t check_type(std::string prefix, json in){
  if (not in.is_array()) {
      return in.type();
    } else {
    if (in.size() == 0) return json::value_t::null;
    json::value_t t = check_type(prefix, in[0]);
    for (uint i = 1; i < in.size(); i++) {
      if (t != check_type(prefix, in[i])) {
	fprintf(stderr, "Types are not the same in the the key %s\n",prefix.c_str());
	assert(false);
      }
    }
    return t;
  }
}

void qualify_input(std::string prefix, json &in, json &in1);

void qualify_input_list(std::string prefix, json &in, json &in1){
    if (in.is_array()) {
      for (uint i = 0; i<in.size(); i++) {
	  std::string new_prefix = prefix + "[" + std::to_string(i) + "]";
	  qualify_input_list(new_prefix,in[i],in1);
	}
    } else {
	qualify_input(prefix,in,in1);
    }
}

void qualify_input(std::string prefix, json &in, json &in1) {
  if (in.is_array()) {
    if (in.size() > 0) {
      json::value_t t = check_type(prefix,in);
      if (t == json::value_t::object) {
	qualify_input_list(prefix,in,in1);
      } else {
	in1[prefix] = in;
      }
    } else {
      in1[prefix] = in;
    }
  } else if (in.is_object()) {
    for (json::iterator it = in.begin(); it != in.end(); ++it) {
      std::string new_prefix = prefix.length() == 0 ? it.key() : prefix + "." + it.key();
      qualify_input(new_prefix,it.value(),in1);
    }
  } else {
    in1[prefix] = in;
  }
}

bool loadJsonImpl(Circom_CalcWit *ctx, json &j) {
  u64 nItems = j.size();
  if (nItems == 0){
    ctx->tryRunCircuit();
  }
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    u64 h = fnv1a(it.key());
    std::vector<u64> v;
    if (!json2FrElements(it.value(), v)) {
      std::cerr << "loadJsonImpl: failed to parse values for signal " << it.key() << std::endl;
      return false;
    }
    uint signalSize = ctx->getInputSignalSize(h);
    if (v.size() < signalSize) {
      std::cerr << "loadJsonImpl: signal " << it.key()
                << ": not enough values (got " << v.size()
                << ", expected " << signalSize << ")" << std::endl;
      return false;
    }
    if (v.size() > signalSize) {
      std::cerr << "loadJsonImpl: signal " << it.key()
                << ": too many values (got " << v.size()
                << ", expected " << signalSize << ")" << std::endl;
      return false;
    }
    for (uint i = 0; i<v.size(); i++){
      // setInputSignal is noexcept (asserts on internal errors); no try/catch needed.
      ctx->setInputSignal(h, i, v[i]);
    }
  }
  return true;
}

bool loadJson(Circom_CalcWit *ctx, std::string filename)
{
  std::ifstream inStream(filename);
  if (!inStream) {
    std::cerr << "loadJson: failed to open \"" << filename << "\": "
              << std::strerror(errno) << std::endl;
    return false;
  }
  // nlohmann::json::parse with allow_exceptions=false returns json::value_t::discarded
  // on parse failure instead of throwing.
  json jin = json::parse(inStream, /*cb=*/nullptr, /*allow_exceptions=*/false);
  inStream.close();
  if (jin.is_discarded()) {
    std::cerr << "loadJson: failed to parse JSON from \"" << filename << "\"" << std::endl;
    return false;
  }
  json j;
  std::string prefix = "";
  qualify_input(prefix, jin, j);
  return loadJsonImpl(ctx, j);
}

void freeCircuit(Circom_Circuit *circuit)
{
  delete[] circuit->InputHashMap;
  delete[] circuit->witness2SignalList;
  delete[] circuit->signal_map.sig;
  delete[] circuit->signal_map.adds;
  // delete[] circuit->circuitConstants;
  
  // Free templateInsId2IOSignalInfo map entries
  for (auto& entry : circuit->templateInsId2IOSignalInfo) {
    IOFieldDefPair& pair = entry.second;
    for (u32 i = 0; i < pair.len; i++) {
      delete[] pair.defs[i].lengths;  // Free the lengths array for each IOFieldDef
    }
    free(pair.defs);  // Free the defs array
  }
  
  // Free busInsId2FieldInfo array
  if (circuit->busInsId2FieldInfo != nullptr) {
    for (int i = 0; i < get_size_of_bus_field_map(); i++) {
      IOFieldDefPair& pair = circuit->busInsId2FieldInfo[i];
      for (u32 j = 0; j < pair.len; j++) {
        delete[] pair.defs[j].lengths;  // Free the lengths array for each IOFieldDef
      }
      free(pair.defs);  // Free the defs array
    }
    free(circuit->busInsId2FieldInfo);  // Free the main array
  }
  
  delete circuit;
}

extern "C" __attribute__((visibility("default"))) uint64_t getSizeWitness()  {
  return get_size_of_witness();
}

// Total number of signals — the size of the signalValues buffer (>= sizeWitness).
// Exposed so the Rust side can pre-allocate pooled signalValues buffers.
extern "C" __attribute__((visibility("default"))) uint64_t getTotalSignalNo()  {
  return get_total_signal_no();
}

extern "C" __attribute__((visibility("default"))) void *initCircuit(char* datFile)  {
    Circom_Circuit *circuit = loadCircuit(string(datFile));
    return (void *)circuit;
}

extern "C" __attribute__((visibility("default"))) void freeCircuit(void* circuit_)  {
    Circom_Circuit *circuit = (Circom_Circuit *)circuit_;
    freeCircuit(circuit);
}

extern "C" __attribute__((visibility("default"))) int64_t getWitnessFinal(void *zkin, char* datFile, void* pWitness, uint64_t nMutexes)  {
    //-------------------------------------------
    // Verifier stark proof
    //-------------------------------------------
    Circom_Circuit *circuit = loadCircuit(string(datFile));
    if (!circuit) {
      return -1;
    }

    Circom_CalcWit *ctx = new Circom_CalcWit(circuit, nMutexes);

    if (!loadJsonImpl(ctx, *(json*) zkin)) {
      delete ctx;
      freeCircuit(circuit);
      return -1;
    }

    if (ctx->getRemaingInputsToBeSet() != 0)
    {
      std::cerr << "getWitnessFinal: not all inputs have been set: only "
                << (get_main_input_signal_no() - ctx->getRemaingInputsToBeSet())
                << " out of " << get_main_input_signal_no() << std::endl;
      delete ctx;
      freeCircuit(circuit);
      return -1;
    }

    for(uint64_t i = 0; i < get_main_input_signal_no(); ++i) {
      cout << i << " " << ctx->signalValues[get_main_input_signal_start() + i] << endl;
    }

    if (ctx->errorOccurred) {
      std::cerr << "getWitnessFinal: witness generation failed (assert failed)" << std::endl;
      delete ctx;
      freeCircuit(circuit);
      return -1;
    }

    //-------------------------------------------
    // Compute witness
    //-------------------------------------------
    uint64_t *witness = (uint64_t *)pWitness;
    uint64_t sizeWitness = get_size_of_witness();
    for (uint64_t i = 0; i < sizeWitness; i++)
    {
      ctx->getWitness(i, witness[i]);
    }

    delete ctx;
    freeCircuit(circuit);
    return 0;
}

// Raw witness, for a circuit with no exec map: stark-recurser's equivalence tests link this.
extern "C" __attribute__((visibility("default"))) int64_t getWitness(uint64_t *proof, void* circuit_, void* pWitness, uint64_t nMutexes) {
    Circom_Circuit *circuit = (Circom_Circuit *)circuit_;
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit, nMutexes);

    memcpy(&ctx->signalValues[get_main_input_signal_start()], proof, get_main_input_signal_no() * sizeof(uint64_t));
    ctx->runCircuit();

    if (ctx->errorOccurred) {
        std::cerr << "getWitness: witness generation failed (assert failed)" << std::endl;
        delete ctx;
        return -1;
    }

    uint64_t *witness = (uint64_t *)pWitness;
    uint64_t sizeWitness = get_size_of_witness();
    for (uint64_t i = 0; i < sizeWitness; i++) {
        ctx->getWitness(i, witness[i]);
    }

    delete ctx;
    return 0; // success
}

// Folds witness2SignalList into the exec map so the scatter does one random load per cell
// instead of two. Idempotent per circuit; on failure getWitnessTrace reads the exec map directly.
extern "C" __attribute__((visibility("default"))) int64_t prepareSignalMap(
    void* circuit_, uint64_t *exec_data, uint64_t execWords)
{
    Circom_Circuit *circuit = (Circom_Circuit *)circuit_;
    SignalMap &map = circuit->signal_map;
    if (map.ok) return 0;

    const exec_layout::Header h = exec_layout::header(exec_data, execWords);
    if (!h.valid || h.mapRows == 0 || h.mapCols == 0) return -1;
    const uint64_t entries = h.mapRows * h.mapCols;
    const uint64_t sizeWitness = get_size_of_witness();
    const char *p_sMap = reinterpret_cast<const char *>(&exec_data[exec_layout::map_at(h)]);

    // Resolves one witness index to a signal index, or to an adds_ext reference flagged in the
    // top bit. Returns false on an index neither can hold.
    bool bad = false;
    auto resolve = [&](uint64_t v, const char *what) -> u32 {
        if (v < sizeWitness) {
            const u64 sg = circuit->witness2SignalList[v];
            if (sg < SIGNAL_MAP_ADD_FLAG) return (u32)sg;
            std::cerr << "prepareSignalMap: signal " << sg << " does not fit 31 bits" << std::endl;
        } else if (v - sizeWitness < h.nAdds) {
            return SIGNAL_MAP_ADD_FLAG | (u32)(v - sizeWitness);
        } else {
            std::cerr << "prepareSignalMap: " << what << " " << v << " is past sizeWitness + nAdds" << std::endl;
        }
        bad = true;
        return 0;
    };

    u32 *sig = new u32[entries];
    for (uint64_t i = 0; i < entries && !bad; i++) {
        uint32_t v;
        memcpy(&v, p_sMap + i * sizeof(uint32_t), sizeof(uint32_t));
        sig[i] = (v == 0) ? 0 : resolve(v, "entry");
    }

    const uint64_t *p_adds = &exec_data[exec_layout::HEADER_WORDS];
    u32 *adds = new u32[h.nAdds * 2];
    for (uint64_t i = 0; i < h.nAdds && !bad; i++) {
        adds[i * 2]     = resolve(p_adds[i * 4],     "addition operand");
        adds[i * 2 + 1] = resolve(p_adds[i * 4 + 1], "addition operand");
    }

    if (bad) {
        delete[] sig;
        delete[] adds;
        return -1;
    }
    map.sig = sig;
    map.adds = adds;
    map.ok = true;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int64_t getWitnessTrace(
    uint64_t *proof, void* circuit_, uint64_t *exec_data, void* pTrace, void* pPublics,
    uint64_t N, uint64_t nPublics, uint64_t nCommitedPols, uint64_t nMutexes, void* pSignalValues)
{
    Circom_Circuit *circuit = (Circom_Circuit *)circuit_;

    // Per-phase timing, off unless PIL2_CIRCOM_TIMERS is set: one stderr line per call. The Rust
    // side can only see the whole call (CIRCOM_WITNESS), and the phases inside it have very
    // different costs and fixes -- the solve is circom's, the scatter is ours.
    const bool phase_timing = std::getenv("PIL2_CIRCOM_TIMERS") != nullptr;
    auto tick = [] { return std::chrono::steady_clock::now(); };
    auto span_ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_enter = tick();

    // pSignalValues: optional caller-owned pool buffer (>= get_total_signal_no() u64s).
    // When null, Circom_CalcWit allocates its own. Reused buffers need no zeroing.
    Circom_CalcWit *ctx = new Circom_CalcWit(circuit, nMutexes, (uint64_t*)pSignalValues);
    const auto t_ctor = tick();
    memcpy(&ctx->signalValues[get_main_input_signal_start()], proof, get_main_input_signal_no() * sizeof(uint64_t));
    ctx->runCircuit();
    const auto t_solve = tick();
    if (ctx->errorOccurred) {
        std::cerr << "getWitnessTrace: witness generation failed (assert failed)" << std::endl;
        delete ctx;
        return -1;
    }

    uint64_t sizeWitness = get_size_of_witness();
    // cw(k): the value old code stored in circomWitness[k], read straight from
    // signalValues instead of via a materialized witness buffer.
    auto cw = [&](uint64_t k) -> uint64_t { return ctx->signalValues[circuit->witness2SignalList[k]]; };

    // Layout owned by exec_layout.hpp: [magic|version, nAdds, mapRows, mapCols], additions,
    // then the map as u32 entries packed two per word. Mirrors getCommitedPols in
    // starkpil/recursion_trace/exec_file.hpp, which this call replaces.
    const exec_layout::Header h = exec_layout::header(exec_data, exec_layout::HEADER_WORDS);
    const uint64_t *p_adds = &exec_data[exec_layout::HEADER_WORDS];
    // Entries are u32 pairs inside the u64 buffer, read through a byte pointer because aliasing
    // a uint64_t array as uint32_t is undefined. Compiles to the same load.
    const char *p_sMap = reinterpret_cast<const char *>(&exec_data[exec_layout::map_at(h)]);

    // The loader rejects a map wider than the trace but cannot check the height, not knowing
    // nBits. Clamp both so a bad key cannot walk off either buffer.
    const uint64_t mapRows = h.mapRows < N ? h.mapRows : N;
    const uint64_t mapCols = h.mapCols < nCommitedPols ? h.mapCols : nCommitedPols;

    Goldilocks::Element *trace   = (Goldilocks::Element *)pTrace;
    Goldilocks::Element *publics = (Goldilocks::Element *)pPublics;

    for (uint64_t i = 0; i < nPublics; ++i) {
        publics[i] = Goldilocks::fromU64(cw(1 + i));
    }

    // nAdds extension: the linear-combination signals the old code wrote at
    // circomWitness[sizeWitness + i]. Kept in a small scratch instead of extending the whole
    // witness. Serial: an addition may read a signal an earlier one produced.
    std::vector<uint64_t> adds_ext(h.nAdds);
    const uint32_t *aop = circuit->signal_map.ok ? circuit->signal_map.adds : nullptr;
    auto cw_ext = [&](uint64_t k) -> uint64_t {
        return (k < sizeWitness) ? cw(k) : adds_ext[k - sizeWitness];
    };
    auto op = [&](uint64_t i) -> uint64_t {
        const uint32_t s = aop[i];
        return (s & SIGNAL_MAP_ADD_FLAG) ? adds_ext[s & ~SIGNAL_MAP_ADD_FLAG] : ctx->signalValues[s];
    };
    const uint64_t nThreadsAdds = (nMutexes > 0) ? nMutexes : 1;
    auto one_add = [&](uint64_t i) {
        const uint64_t a = aop ? op(i * 2) : cw_ext(p_adds[i * 4]);
        const uint64_t b = aop ? op(i * 2 + 1) : cw_ext(p_adds[i * 4 + 1]);
        Goldilocks::Element c = Goldilocks::fromU64(a) * Goldilocks::fromU64(p_adds[i * 4 + 2]);
        Goldilocks::Element d = Goldilocks::fromU64(b) * Goldilocks::fromU64(p_adds[i * 4 + 3]);
        adds_ext[i] = Goldilocks::toU64(c + d);
    };
    // An addition reading another one (top bit set on either operand) has to wait for it, and only
    // ever references a lower index; the rest are independent, ~94% of them. Two passes over the
    // same ascending range, so each thread's writes stay contiguous.
    if (aop != nullptr && nThreadsAdds > 1 && h.nAdds >= (1u << 14)) {
        const uint64_t block = (h.nAdds + nThreadsAdds - 1) / nThreadsAdds;
        auto run_block = [&](uint64_t from, uint64_t to) {
            for (uint64_t i = from; i < to; i++) {
                const uint32_t s0 = aop[i * 2], s1 = aop[i * 2 + 1];
                if ((s0 | s1) & SIGNAL_MAP_ADD_FLAG) continue;
                Goldilocks::Element c =
                    Goldilocks::fromU64(ctx->signalValues[s0]) * Goldilocks::fromU64(p_adds[i * 4 + 2]);
                Goldilocks::Element d =
                    Goldilocks::fromU64(ctx->signalValues[s1]) * Goldilocks::fromU64(p_adds[i * 4 + 3]);
                adds_ext[i] = Goldilocks::toU64(c + d);
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(nThreadsAdds - 1);
        for (uint64_t t = 1; t < nThreadsAdds; t++) {
            const uint64_t from = t * block;
            if (from >= h.nAdds) break;
            workers.emplace_back(run_block, from, std::min(h.nAdds, from + block));
        }
        run_block(0, std::min(h.nAdds, block));
        for (auto &w : workers) w.join();
        for (uint64_t i = 0; i < h.nAdds; i++) {
            if ((aop[i * 2] | aop[i * 2 + 1]) & SIGNAL_MAP_ADD_FLAG) one_add(i);
        }
    } else {
        for (uint64_t i = 0; i < h.nAdds; i++) one_add(i);
    }
    const auto t_adds = tick();

    // Parallelized over contiguous ROW blocks -- never over j, which would put several threads on
    // the same 64-byte trace line. Everything read here is immutable, so the blocks need no
    // synchronization.
    // With a prepared map the entries already hold signal indices, so a cell costs one random
    // load instead of chasing witness2SignalList first. Same values either way.
    const uint32_t *sig = circuit->signal_map.ok ? circuit->signal_map.sig : nullptr;
    const uint64_t *adds = adds_ext.data();
    const u64 *signalValues = ctx->signalValues;

    auto scatter_rows = [&](uint64_t lo, uint64_t hi) {
        for (uint64_t i = lo; i < hi; i++) {
            Goldilocks::Element *row = &trace[i * nCommitedPols];
            // Zeroed here so no past-the-end `sig` row pointer is ever formed.
            if (i >= mapRows) {
                memset(row, 0, nCommitedPols * sizeof(Goldilocks::Element));
                continue;
            }
            const uint64_t mapped = mapCols;
            if (sig != nullptr) {
                const uint32_t *srow = sig + i * h.mapCols;
                for (uint64_t j = 0; j < mapped; j++) {
                    const uint32_t s = srow[j];
                    row[j] = s == 0 ? Goldilocks::zero()
                           : Goldilocks::fromU64((s & SIGNAL_MAP_ADD_FLAG) ? adds[s & ~SIGNAL_MAP_ADD_FLAG]
                                                                           : signalValues[s]);
                }
            } else {
                for (uint64_t j = 0; j < mapped; j++) {
                    uint32_t idx;
                    memcpy(&idx, p_sMap + (i * h.mapCols + j) * sizeof(uint32_t), sizeof(uint32_t));
                    row[j] = idx != 0 ? Goldilocks::fromU64(cw_ext(idx)) : Goldilocks::zero();
                }
            }
            memset(row + mapped, 0, (nCommitedPols - mapped) * sizeof(Goldilocks::Element));
        }
    };

    uint64_t nThreads = (nMutexes > 0) ? nMutexes : 1;
    if (nThreads > N) nThreads = (N > 0) ? N : 1;
    if (nThreads <= 1) {
        scatter_rows(0, N);
    } else {
        // Contiguous row-block per thread (last block absorbs the remainder); run block 0 on the
        // calling thread, then join the spawned workers.
        uint64_t block = (N + nThreads - 1) / nThreads;
        std::vector<std::thread> workers;
        workers.reserve(nThreads - 1);
        for (uint64_t t = 1; t < nThreads; t++) {
            uint64_t lo = t * block;
            if (lo >= N) break;
            uint64_t hi = std::min(lo + block, N);
            workers.emplace_back(scatter_rows, lo, hi);
        }
        scatter_rows(0, std::min(block, N));
        for (auto &w : workers) w.join();
    }
    const auto t_scatter = tick();

    delete ctx;
    const auto t_dtor = tick();

    if (phase_timing) {
        // pooled: whether the caller handed us a signalValues buffer, so a self-allocating circuit
        // is visible as such rather than looking like a slow ctor.
        fprintf(stderr,
                "CIRCOM_PHASES N=%llu cols=%llu nAdds=%llu threads=%llu pooled=%d "
                "ctor=%.2f solve=%.2f adds=%.2f scatter=%.2f dtor=%.2f total=%.2f\n",
                (unsigned long long)N, (unsigned long long)nCommitedPols, (unsigned long long)h.nAdds,
                (unsigned long long)nThreads, pSignalValues != nullptr,
                span_ms(t_enter, t_ctor), span_ms(t_ctor, t_solve), span_ms(t_solve, t_adds),
                span_ms(t_adds, t_scatter), span_ms(t_scatter, t_dtor), span_ms(t_enter, t_dtor));
    }
    return 0;
}
