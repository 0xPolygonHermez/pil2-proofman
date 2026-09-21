#ifndef __CIRCOM_H
#define __CIRCOM_H

#include <map>
#include <gmp.h>
#include <mutex>
#include <condition_variable>
#include <thread>

//#include "fr.hpp"

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

//only for the main inputs
struct __attribute__((__packed__)) HashSignalInfo {
    u64 hash;
    u64 signalid; 
    u64 signalsize; 
};

struct IOFieldDef { 
    u32 offset;
    u32 len;
    u32 *lengths;
    u32 size;
    u32 busId;
};

struct IOFieldDefPair { 
    u32 len;
    IOFieldDef* defs;
};

// Top bit marks an adds_ext reference; the rest is the index.
#define SIGNAL_MAP_ADD_FLAG 0x80000000u

// The exec map with witness2SignalList applied, so a trace cell costs one random load instead
// of two. Dense, mapRows*mapCols, row-major; 0 means empty. Built by prepareSignalMap.
// `adds` does the same for the nAdds operands, 2 per addition; it has no empty encoding.
struct SignalMap {
  u32* sig = NULL;
  u32* adds = NULL;
  bool ok = false;
};

struct Circom_Circuit {
  //  const char *P;
  HashSignalInfo* InputHashMap;
  u64* witness2SignalList;
  //u64* circuitConstants;  
  std::map<u32,IOFieldDefPair> templateInsId2IOSignalInfo;
  IOFieldDefPair* busInsId2FieldInfo;
  SignalMap signal_map;
};

struct Circom_Component {
  u32 templateId;
  u64 signalStart;
  u32 inputCounter;
  std::string templateName;
  std::string componentName;
  u64 idFather; 
  u32* subcomponents = NULL;
  bool* subcomponentsParallel = NULL;
  bool *outputIsSet = NULL;  //one for each output
  std::mutex *mutexes = NULL;  //one for each output
  std::condition_variable *cvs = NULL;
  std::thread *sbct = NULL;//subcomponent threads
};


uint get_main_input_signal_start();
uint get_main_input_signal_no();
uint get_total_signal_no();
uint get_number_of_components();
uint get_size_of_input_hashmap();
uint get_size_of_witness();
//uint get_size_of_constants();
uint get_size_of_io_map();
uint get_size_of_bus_field_map();

#endif  // __CIRCOM_H
