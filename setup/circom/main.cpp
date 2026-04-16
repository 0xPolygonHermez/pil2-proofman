#include "main.hpp"

#define handle_error(msg) \
           do { perror(msg); exit(EXIT_FAILURE); } while (0)

Circom_Circuit* loadCircuit(std::string const &datFileName) {
    Circom_Circuit *circuit = new Circom_Circuit;

    int fd;
    struct stat sb;

    fd = open(datFileName.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cout << ".dat file not found: " << datFileName << "\n";
        throw std::system_error(errno, std::generic_category(), "open");
    }
    
    if (fstat(fd, &sb) == -1) {          /* To obtain file size */
        throw std::system_error(errno, std::generic_category(), "fstat");
    }

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

void json2FrElements (json val, std::vector<u64> & vval){
  if (!val.is_array()) {
    u64 v;
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
        std::ostringstream errStrStream;
        errStrStream << "Invalid number in JSON input: " << s_aux << "\n";
	      throw std::runtime_error(errStrStream.str() );
      }
    } else if (val.is_number()) {
        double vd = val.get<double>();
        std::stringstream stream;
        stream << std::fixed << std::setprecision(0) << vd;
        s = stream.str();
        base = 10;
    } else {
        std::ostringstream errStrStream;
        errStrStream << "Invalid JSON type\n";
	      throw std::runtime_error(errStrStream.str() );
    }
    vval.push_back(strtoull(s.c_str(), NULL, base));
  } else {
    for (uint i = 0; i < val.size(); i++) {
      json2FrElements (val[i], vval);
    }
  }
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

void loadJsonImpl(Circom_CalcWit *ctx, json &j) {
  u64 nItems = j.size();
  // printf("Items : %llu\n",nItems);
  if (nItems == 0){
    ctx->tryRunCircuit();
  }
  for (json::iterator it = j.begin(); it != j.end(); ++it) {
    // std::cout << it.key() << " => " << it.value() << '\n';
    u64 h = fnv1a(it.key());
    std::vector<u64> v;
    json2FrElements(it.value(),v);
    uint signalSize = ctx->getInputSignalSize(h);
    if (v.size() < signalSize) {
	std::ostringstream errStrStream;
	errStrStream << "Error loading signal " << it.key() << ": Not enough values\n";
	throw std::runtime_error(errStrStream.str() );
    }
    if (v.size() > signalSize) {
	std::ostringstream errStrStream;
	errStrStream << "Error loading signal " << it.key() << ": Too many values\n";
	throw std::runtime_error(errStrStream.str() );
    }
    for (uint i = 0; i<v.size(); i++){
      try {
	// std::cout << it.key() << "," << i << " => " << Fr_element2str(&(v[i])) << '\n';
	ctx->setInputSignal(h,i,v[i]);
      } catch (std::runtime_error e) {
	std::ostringstream errStrStream;
	errStrStream << "Error setting signal: " << it.key() << "\n" << e.what();
	throw std::runtime_error(errStrStream.str() );
      }
    }
  }
}

void loadJson(Circom_CalcWit *ctx, std::string filename)
{
  std::ifstream inStream(filename);
  json jin;
  inStream >> jin;
  json j;
  std::string prefix = "";
  qualify_input(prefix, jin, j);
  inStream.close();
  loadJsonImpl(ctx, j);
}

void freeCircuit(Circom_Circuit *circuit)
{
  delete[] circuit->InputHashMap;
  delete[] circuit->witness2SignalList;
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

extern "C" __attribute__((visibility("default"))) void *initCircuit(char* datFile)  {
    Circom_Circuit *circuit = loadCircuit(string(datFile));
    return (void *)circuit;
}

extern "C" __attribute__((visibility("default"))) void freeCircuit(void* circuit_)  {
    Circom_Circuit *circuit = (Circom_Circuit *)circuit_;
    freeCircuit(circuit);
}

extern "C" __attribute__((visibility("default"))) int getWitnessFinal(void *zkin, char* datFile, void* pWitness, uint64_t nMutexes)  {
    try {
      //-------------------------------------------
      // Verifier stark proof
      //-------------------------------------------
      Circom_Circuit *circuit = loadCircuit(string(datFile));

      Circom_CalcWit *ctx = new Circom_CalcWit(circuit, nMutexes);

      loadJsonImpl(ctx, *(json*) zkin);

      if (ctx->getRemaingInputsToBeSet() != 0)
      {
        cout << "Not all inputs have been set. Only " << to_string(get_main_input_signal_no() - ctx->getRemaingInputsToBeSet()) << " out of " << to_string(get_main_input_signal_no()) << endl;
        exit(-1);
      }

      for(uint64_t i = 0; i < get_main_input_signal_no(); ++i) {
        cout << i << " " << ctx->signalValues[get_main_input_signal_start() + i] << endl;
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
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        exit(-1);
    } catch (...) {
        std::cerr << "Unknown runtime error" << std::endl;
        exit(-2);
    }
}

extern "C" __attribute__((visibility("default"))) int getWitness(uint64_t *proof, void* circuit_, void* pWitness, uint64_t nMutexes) {
    try {
        Circom_Circuit *circuit = (Circom_Circuit *)circuit_;
        Circom_CalcWit *ctx = new Circom_CalcWit(circuit, nMutexes);

        memcpy(&ctx->signalValues[get_main_input_signal_start()], proof, get_main_input_signal_no() * sizeof(uint64_t));
        ctx->runCircuit();

        uint64_t *witness = (uint64_t *)pWitness;
        uint64_t sizeWitness = get_size_of_witness();
        for (uint64_t i = 0; i < sizeWitness; i++) {
            ctx->getWitness(i, witness[i]);
        }

        delete ctx;
        return 0; // success
    } catch (const std::exception &e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown runtime error" << std::endl;
        return -2;
    }
}