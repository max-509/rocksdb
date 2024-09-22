//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "rocksdb/rocksdb_namespace.h"



namespace ROCKSDB_NAMESPACE {
namespace lru_cache {

/** Determines the size of the hash table, and when and how much it should be resized.
  * Has very small state (one UInt8) and useful for Set-s allocated in automatic memory (see uniqExact as an example).
  */
template <size_t initial_size_degree = 8>
struct HashTableGrower
{
    /// The state of this structure is enough to get the buffer size of the hash table.

    uint8_t size_degree = initial_size_degree;
    static constexpr auto initial_count = 1ULL << initial_size_degree;

    /// If collision resolution chains are contiguous, we can implement erase operation by moving the elements.
    static constexpr auto performs_linear_probing_with_single_step = true;

    static constexpr size_t max_size_degree = 23;

    /// The size of the hash table in the cells.
    size_t bufSize() const               { return 1ULL << size_degree; }

    size_t maxFill() const               { return 1ULL << (size_degree - 1); }
    size_t mask() const                  { return bufSize() - 1; }

    /// From the hash value, get the cell number in the hash table.
    size_t place(size_t x) const         { return x & mask(); }

    /// The next cell in the collision resolution chain.
    size_t next(size_t pos) const        { ++pos; return pos & mask(); }

    /// Whether the hash table is sufficiently full. You need to increase the size of the hash table, or remove something unnecessary from it.
    bool overflow(size_t elems) const    { return elems > maxFill(); }

    /// Increase the size of the hash table.
    void increaseSize()
    {
        size_degree += size_degree >= max_size_degree ? 1 : 2;
    }

    /// Set the buffer size by the number of elements in the hash table. Used when deserializing a hash table.
    void set(size_t num_elems)
    {
        if (num_elems <= 1)
            size_degree = initial_size_degree;
        else if (initial_size_degree > static_cast<size_t>(std::log2(num_elems - 1)) + 2)
            size_degree = initial_size_degree;
        else
            size_degree = static_cast<size_t>(std::log2(num_elems - 1)) + 2;
    }

    void setBufSize(size_t buf_size_)
    {
        size_degree = static_cast<size_t>(std::log2(buf_size_ - 1) + 1);
    }
};

/** Determines the size of the hash table, and when and how much it should be resized.
  * This structure is aligned to cache line boundary and also occupies it all.
  * Precalculates some values to speed up lookups and insertion into the HashTable (and thus has bigger memory footprint than HashTableGrower).
  * This grower assume 0.5 load factor
  */
template <size_t initial_size_degree = 8>
class alignas(64) HashTableGrowerWithPrecalculation
{
    /// The state of this structure is enough to get the buffer size of the hash table.

    uint8_t size_degree = initial_size_degree;
    size_t precalculated_mask = (1ULL << initial_size_degree) - 1;
    size_t precalculated_max_fill = 1ULL << (initial_size_degree - 1);
    static constexpr size_t max_size_degree = 23;

public:
    uint8_t sizeDegree() const { return size_degree; }

    void increaseSizeDegree(uint8_t delta)
    {
        size_degree += delta;
        precalculated_mask = (1ULL << size_degree) - 1;
        precalculated_max_fill = 1ULL << (size_degree - 1);
    }

    static constexpr auto initial_count = 1ULL << initial_size_degree;

    /// If collision resolution chains are contiguous, we can implement erase operation by moving the elements.
    static constexpr auto performs_linear_probing_with_single_step = true;

    /// The size of the hash table in the cells.
    size_t bufSize() const { return 1ULL << size_degree; }

    /// From the hash value, get the cell number in the hash table.
    size_t place(size_t x) const { return x & precalculated_mask; }

    /// The next cell in the collision resolution chain.
    size_t next(size_t pos) const { return (pos + 1) & precalculated_mask; }

    /// Whether the hash table is sufficiently full. You need to increase the size of the hash table, or remove something unnecessary from it.
    bool overflow(size_t elems) const { return elems > precalculated_max_fill; }

    /// Increase the size of the hash table.
    void increaseSize() { increaseSizeDegree(size_degree >= max_size_degree ? 1 : 2); }

    /// Set the buffer size by the number of elements in the hash table. Used when deserializing a hash table.
    void set(size_t num_elems)
    {
        if (num_elems <= 1)
            size_degree = initial_size_degree;
        else if (initial_size_degree > static_cast<size_t>(log2(num_elems - 1)) + 2)
            size_degree = initial_size_degree;
        else
            size_degree = static_cast<size_t>(log2(num_elems - 1)) + 2;
        increaseSizeDegree(0);
    }

    void setBufSize(size_t buf_size_)
    {
        size_degree = static_cast<size_t>(log2(buf_size_ - 1) + 1);
        increaseSizeDegree(0);
    }
};

static_assert(sizeof(HashTableGrowerWithPrecalculation<>) == 64);

/** When used as a Grower, it turns a hash table into something like a lookup table.
  * It remains non-optimal - the cells store the keys.
  * Also, the compiler can not completely remove the code of passing through the collision resolution chain, although it is not needed.
  * NOTE: Better to use FixedHashTable instead.
  */
template <size_t key_bits>
struct HashTableFixedGrower
{
    static constexpr auto initial_count = 1ULL << key_bits;

    static constexpr auto performs_linear_probing_with_single_step = true;

    size_t bufSize() const               { return 1ULL << key_bits; }
    size_t place(size_t x) const         { return x; }
    /// You could write UNREACHABLE(), but the compiler does not optimize everything, and it turns out less efficiently.
    size_t next(size_t pos) const        { return pos + 1; }
    bool overflow(size_t /*elems*/) const { return false; }

    void increaseSize() { std::abort(); }
    void set(size_t /*num_elems*/) {}
    void setBufSize(size_t /*buf_size_*/) {}
};



template <size_t initial_size_degree = 8>
struct StringHashTableGrower : public HashTableGrowerWithPrecalculation<initial_size_degree>
{
    // Smooth growing for string maps
    void increaseSize() { this->increaseSizeDegree(1); }
};


}
}
