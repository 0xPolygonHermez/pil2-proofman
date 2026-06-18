#include <iomanip>
#include <sstream>
#include <assert.h>
#include "calcwit.hpp"

extern void run(Circom_CalcWit* ctx);

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

Circom_CalcWit::Circom_CalcWit(Circom_Circuit *aCircuit, uint maxTh) {
    circuit = aCircuit;
    inputSignalAssignedCounter = get_main_input_signal_no();

    inputSignalAssigned = new bool[inputSignalAssignedCounter];
    memset(inputSignalAssigned, 0, inputSignalAssignedCounter * sizeof(bool));

    signalValues = new u64[get_total_signal_no()];
    signalValues[0] = 1;

    componentMemory = new Circom_Component[get_number_of_components()];
    // Initialize all component pointers to NULL to ensure safe cleanup
    for (uint i = 0; i < get_number_of_components(); i++) {
        componentMemory[i].subcomponents = nullptr;
        componentMemory[i].subcomponentsParallel = nullptr;
        componentMemory[i].outputIsSet = nullptr;
        componentMemory[i].mutexes = nullptr;
        componentMemory[i].cvs = nullptr;
        componentMemory[i].sbct = nullptr;
    }

    // circuitConstants = circuit ->circuitConstants;
    templateInsId2IOSignalInfo = circuit->templateInsId2IOSignalInfo;
    busInsId2FieldInfo = circuit->busInsId2FieldInfo;
    listOfTemplateMessages = nullptr; // Initialize to prevent issues

    maxThread = maxTh;
    numThread = 0;
}
Circom_CalcWit::~Circom_CalcWit() {

  // Clean up any component memory that wasn't released during execution
  for (uint i = 0; i < get_number_of_components(); i++) {
    if (componentMemory[i].subcomponents) {
      delete[] componentMemory[i].subcomponents;
    }
    if (componentMemory[i].subcomponentsParallel) {
      delete[] componentMemory[i].subcomponentsParallel;
    }
    if (componentMemory[i].outputIsSet) {
      delete[] componentMemory[i].outputIsSet;
    }
    if (componentMemory[i].mutexes) {
      delete[] componentMemory[i].mutexes;
    }
    if (componentMemory[i].cvs) {
      delete[] componentMemory[i].cvs;
    }
    if (componentMemory[i].sbct) {
      delete[] componentMemory[i].sbct;
    }
  }
  
  // Let circom handle all component memory cleanup via release_memory_component()
  // Clean up listOfTemplateMessages if allocated
  if (listOfTemplateMessages) {
    delete[] listOfTemplateMessages;
  }
  
  delete[] inputSignalAssigned;
  delete[] signalValues;
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

