#include <iomanip>
#include <sstream>
#include <assert.h>
#include "calcwit.hpp"
#include <mutex>

extern void run(Circom_CalcWit* ctx);

// One componentMemory array is kept per circuit and reused by the next witness of it.
//
// The array is 144 B x get_number_of_components() -- 67 MB for ZisK's recursive2 -- and every
// witness of a circuit builds an identical one, so `new[]` faults in the whole thing and runs a
// constructor per element: 16.5 ms of a 95 ms solve. The destructor nulls every pointer member as
// it frees, which is what leaves the array reusable.
//
// One .so serves one circuit, so a cached array always has the right length. Concurrent solves of
// the same circuit take separate arrays; only one is kept, hence the mutex.
static std::mutex componentCacheMutex;
static Circom_Component *componentCache = nullptr;


std::string int_to_hex( u64 i )
{
  std::stringstream stream;
  stream << "0x"
         << std::setfill ('0') << std::setw(16)
         << std::hex << i;
  return stream.str();
}

u64 fnv1a(std::string s) {
  u64 hash = 0xCBF29CE484222325LL;
  for(char& c : s) {
    hash ^= u64(c);
    hash *= 0x100000001B3LL;
  }
  return hash;
}

Circom_CalcWit::Circom_CalcWit(Circom_Circuit *aCircuit, uint maxTh, u64* signalValuesBuf) {
    circuit = aCircuit;
    inputSignalAssignedCounter = get_main_input_signal_no();

    inputSignalAssigned = new bool[inputSignalAssignedCounter];
    memset(inputSignalAssigned, 0, inputSignalAssignedCounter * sizeof(bool));

    if (signalValuesBuf != nullptr) {
        signalValues = signalValuesBuf;
        ownsSignalValues = false;
    } else {
        signalValues = new u64[get_total_signal_no()];
        ownsSignalValues = true;
    }
    signalValues[0] = 1;

    // `new Circom_Component[N]` default-initializes, and every pointer member carries a `= NULL`
    // default member initializer (see circom.hpp), so the six are already null here. The loop that
    // set them again cost ~6 ns per component -- 2.9 ms per recursive2 witness, on 488,841 of them.
    {
        std::lock_guard<std::mutex> lk(componentCacheMutex);
        componentMemory = componentCache;
        componentCache = nullptr;
    }
    if (componentMemory == nullptr) {
        componentMemory = new Circom_Component[get_number_of_components()];
    }

    // circuitConstants = circuit ->circuitConstants;
    templateInsId2IOSignalInfo = circuit->templateInsId2IOSignalInfo;
    busInsId2FieldInfo = circuit->busInsId2FieldInfo;
    listOfTemplateMessages = nullptr; // Initialize to prevent issues

    maxThread = maxTh;
    numThread = 0;
}
Circom_CalcWit::~Circom_CalcWit() {

  // Clean up any component memory that wasn't released during execution. Nulling as it frees is
  // what makes the array reusable: a cached array is handed to the next witness as-is, and its
  // `_create` calls only set the members of components that have subcomponents -- a stale pointer
  // left in a leaf would be freed twice.
  for (uint i = 0; i < get_number_of_components(); i++) {
    Circom_Component &c = componentMemory[i];
    delete[] c.subcomponents;         c.subcomponents = NULL;
    delete[] c.subcomponentsParallel; c.subcomponentsParallel = NULL;
    delete[] c.outputIsSet;           c.outputIsSet = NULL;
    delete[] c.mutexes;               c.mutexes = NULL;
    delete[] c.cvs;                   c.cvs = NULL;
    delete[] c.sbct;                  c.sbct = NULL;
  }
  
  // Let circom handle all component memory cleanup via release_memory_component()
  // Clean up listOfTemplateMessages if allocated
  if (listOfTemplateMessages) {
    delete[] listOfTemplateMessages;
  }
  
  delete[] inputSignalAssigned;
  if (ownsSignalValues) delete[] signalValues;
  {
    std::lock_guard<std::mutex> lk(componentCacheMutex);
    if (componentCache == nullptr) {
      componentCache = componentMemory;
      componentMemory = NULL;
    }
  }
  // NULL when the cache kept it; `delete[] NULL` is a no-op.
  delete[] componentMemory;
}

uint Circom_CalcWit::getInputSignalHashPosition(u64 h) {
  uint n = get_size_of_input_hashmap();
  uint pos = (uint)(h % (u64)n);
  if (circuit->InputHashMap[pos].hash!=h){
    uint inipos = pos;
    pos = (pos+1)%n; 
    while (pos != inipos) {
      if (circuit->InputHashMap[pos].hash == h) return pos;
      if (circuit->InputHashMap[pos].signalid == 0) {
	fprintf(stderr, "Signal not found\n");
	assert(false);
      }
      pos = (pos+1)%n; 
    }
    fprintf(stderr, "Signals not found\n");
    assert(false);
  }
  return pos;
}

void Circom_CalcWit::tryRunCircuit(){ 
  if (inputSignalAssignedCounter == 0) {
    run(this);
  }
}

void Circom_CalcWit::runCircuit(){ 
  run(this);
}

void Circom_CalcWit::setInputSignal(u64 h, uint i,  u64 & val){
  if (inputSignalAssignedCounter == 0) {
    fprintf(stderr, "No more signals to be assigned\n");
    assert(false);
  }
  uint pos = getInputSignalHashPosition(h);
  if (i >= circuit->InputHashMap[pos].signalsize) {
    fprintf(stderr, "Input signal array access exceeds the size\n");
    assert(false);
  }
  
  uint si = circuit->InputHashMap[pos].signalid+i;
  if (inputSignalAssigned[si-get_main_input_signal_start()]) {
    fprintf(stderr, "Signal assigned twice: %d\n", si);
    assert(false);
  }
  signalValues[si] = val;
  inputSignalAssigned[si-get_main_input_signal_start()] = true;
  inputSignalAssignedCounter--;
  tryRunCircuit();
}

u64 Circom_CalcWit::getInputSignalSize(u64 h) {
  uint pos = getInputSignalHashPosition(h);
  return circuit->InputHashMap[pos].signalsize;
}

std::string Circom_CalcWit::getTrace(u64 id_cmp){
  if (id_cmp == 0) return componentMemory[id_cmp].componentName;
  else{
    u64 id_father = componentMemory[id_cmp].idFather;
    std::string my_name = componentMemory[id_cmp].componentName;

    return Circom_CalcWit::getTrace(id_father) + "." + my_name;
  }
}

std::string Circom_CalcWit::generate_position_array(uint* dimensions, uint size_dimensions, uint index){
  std::string positions = "";

  for (uint i = 0 ; i < size_dimensions; i++){
    uint last_pos = index % dimensions[size_dimensions -1 - i];
    index = index / dimensions[size_dimensions -1 - i];
    std::string new_pos = "[" + std::to_string(last_pos) + "]";
    positions =  new_pos + positions;
  }
  return positions;
}

