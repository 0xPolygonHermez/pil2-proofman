#ifndef BUILD_CONST_TREE_HPP
#define BUILD_CONST_TREE_HPP

#include <cstdint>
#include <string>
#include "goldilocks_base_field.hpp"
#include "ntt_goldilocks.hpp"
#ifndef __EXCLUDE_BN128__
#include "merkleTreeBN128.hpp"
#endif
#include "merkleTreeGL.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "utils.hpp"

using namespace std;

void buildConstTree(const string constFile, const string starkInfoFile, const string constTreeFile, const string verKeyFile);

#endif