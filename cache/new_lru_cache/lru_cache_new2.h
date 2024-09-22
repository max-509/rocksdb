//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#pragma once

#include <memory>
#include <string>

#include "cache/new_lru_cache/allocator.h"
#include "cache/new_lru_cache/grower.h"
#include "cache/secondary_cache_adapter.h"
#include "cache/sharded_cache.h"
#include "port/lang.h"
#include "port/likely.h"
#include "port/malloc.h"
#include "port/port.h"
#include "util/autovector.h"
#include "util/distributed_mutex.h"

namespace ROCKSDB_NAMESPACE {
namespace lru_cache {

// LRU cache implementation. This class is not thread-safe.

// An entry is a variable length heap-allocated structure.
// Entries are referenced by cache and/or by any external entity.
// The cache keeps all its entries in a hash table. Some elements
// are also stored on LRU list.
//
// LRUHandle can be in these states:
// 1. Referenced externally AND in hash table.
//    In that case the entry is *not* in the LRU list
//    (refs >= 1 && in_cache == true)
// 2. Not referenced externally AND in hash table.
//    In that case the entry is in the LRU list and can be freed.
//    (refs == 0 && in_cache == true)
// 3. Referenced externally AND not in hash table.
//    In that case the entry is not in the LRU list and not in hash table.
//    The entry must be freed if refs becomes 0 in this state.
//    (refs >= 1 && in_cache == false)
// If you call LRUCacheShard::Release enough times on an entry in state 1, it
// will go into state 2. To move from state 1 to state 3, either call
// LRUCacheShard::Erase or LRUCacheShard::Insert with the same key (but
// possibly different value). To move from state 2 to state 1, use
// LRUCacheShard::Lookup.
// While refs > 0, public properties like value and deleter must not change.

struct LRUHandle2 : public Cache::Handle {
  Cache::ObjectPtr value;
  const Cache::CacheItemHelper* helper;
  LRUHandle2* next;
  LRUHandle2* prev;
  size_t total_charge;  // TODO(opt): Only allow uint32_t?
  size_t key_length;
  // The hash of key(). Used for fast sharding and comparisons.
  uint32_t hash;
  // The number of external refs to this entry. The cache itself is not counted.
  uint32_t refs;

  // Mutable flags - access controlled by mutex
  // The m_ and M_ prefixes (and im_ and IM_ later) are to hopefully avoid
  // checking an M_ flag on im_flags or an IM_ flag on m_flags.
  uint8_t m_flags;
  enum MFlags : uint8_t {
    // Whether this entry is referenced by the hash table.
    M_IN_CACHE = (1 << 0),
    // Whether this entry has had any lookups (hits).
    M_HAS_HIT = (1 << 1),
    // Whether this entry is in high-pri pool.
    M_IN_HIGH_PRI_POOL = (1 << 2),
    // Whether this entry is in low-pri pool.
    M_IN_LOW_PRI_POOL = (1 << 3),
  };

  // "Immutable" flags - only set in single-threaded context and then
  // can be accessed without mutex
  uint8_t im_flags;
  enum ImFlags : uint8_t {
    // Whether this entry is high priority entry.
    IM_IS_HIGH_PRI = (1 << 0),
    // Whether this entry is low priority entry.
    IM_IS_LOW_PRI = (1 << 1),
    // Marks result handles that should not be inserted into cache
    IM_IS_STANDALONE = (1 << 2),
  };

  // Beginning of the key (MUST BE THE LAST FIELD IN THIS STRUCT!)
  char key_data[1];

  Slice key() const { return Slice(key_data, key_length); }

  // For HandleImpl concept
  uint32_t GetHash() const { return hash; }

  // Increase the reference count by 1.
  void Ref() { refs++; }

  // Just reduce the reference count by 1. Return true if it was last reference.
  bool Unref() {
    assert(refs > 0);
    refs--;
    return refs == 0;
  }

  // Return true if there are external refs, false otherwise.
  bool HasRefs() const { return refs > 0; }

  bool InCache() const { return m_flags & M_IN_CACHE; }
  bool IsHighPri() const { return im_flags & IM_IS_HIGH_PRI; }
  bool InHighPriPool() const { return m_flags & M_IN_HIGH_PRI_POOL; }
  bool IsLowPri() const { return im_flags & IM_IS_LOW_PRI; }
  bool InLowPriPool() const { return m_flags & M_IN_LOW_PRI_POOL; }
  bool HasHit() const { return m_flags & M_HAS_HIT; }
  bool IsStandalone() const { return im_flags & IM_IS_STANDALONE; }

  void SetInCache(bool in_cache) {
    if (in_cache) {
      m_flags |= M_IN_CACHE;
    } else {
      m_flags &= ~M_IN_CACHE;
    }
  }

  void SetPriority(Cache::Priority priority) {
    if (priority == Cache::Priority::HIGH) {
      im_flags |= IM_IS_HIGH_PRI;
      im_flags &= ~IM_IS_LOW_PRI;
    } else if (priority == Cache::Priority::LOW) {
      im_flags &= ~IM_IS_HIGH_PRI;
      im_flags |= IM_IS_LOW_PRI;
    } else {
      im_flags &= ~IM_IS_HIGH_PRI;
      im_flags &= ~IM_IS_LOW_PRI;
    }
  }

  void SetInHighPriPool(bool in_high_pri_pool) {
    if (in_high_pri_pool) {
      m_flags |= M_IN_HIGH_PRI_POOL;
    } else {
      m_flags &= ~M_IN_HIGH_PRI_POOL;
    }
  }

  void SetInLowPriPool(bool in_low_pri_pool) {
    if (in_low_pri_pool) {
      m_flags |= M_IN_LOW_PRI_POOL;
    } else {
      m_flags &= ~M_IN_LOW_PRI_POOL;
    }
  }

  void SetHit() { m_flags |= M_HAS_HIT; }

  void SetIsStandalone(bool is_standalone) {
    if (is_standalone) {
      im_flags |= IM_IS_STANDALONE;
    } else {
      im_flags &= ~IM_IS_STANDALONE;
    }
  }

  void Free(MemoryAllocator* allocator) {
    assert(refs == 0);
    assert(helper);
    if (helper->del_cb) {
      helper->del_cb(value, allocator);
    }

    free(this);
  }

  inline size_t CalcuMetaCharge(
      CacheMetadataChargePolicy metadata_charge_policy) const {
    if (metadata_charge_policy != kFullChargeCacheMetadata) {
      return 0;
    } else {
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
      return malloc_usable_size(
          const_cast<void*>(static_cast<const void*>(this)));
#else
      // This is the size that is used when a new handle is created.
      return sizeof(LRUHandle) - 1 + key_length;
#endif
    }
  }

  // Calculate the memory usage by metadata.
  inline void CalcTotalCharge(
      size_t charge, CacheMetadataChargePolicy metadata_charge_policy) {
    total_charge = charge + CalcuMetaCharge(metadata_charge_policy);
  }

  inline size_t GetCharge(
      CacheMetadataChargePolicy metadata_charge_policy) const {
    size_t meta_charge = CalcuMetaCharge(metadata_charge_policy);
    assert(total_charge >= meta_charge);
    return total_charge - meta_charge;
  }
};

struct LRUCell2 {
  LRUHandle2* handle;

  bool IsZero() const { return nullptr != handle; }

  void SetHandle(LRUHandle2* new_handle) { handle = new_handle; }

  void SetZero() { SetHandle(nullptr); }
};

// We provide our own simple hash table since it removes a whole bunch
// of porting hacks and is also faster than some of the built-in hash
// table implementations in some of the compiler/runtime combinations
// we have tested.  E.g., readrandom speeds up by ~5% over the g++
// 4.4.3's builtin hashtable.
class LRUHandleTable2 : protected Allocator {
 public:
  explicit LRUHandleTable2(int max_upper_hash_bits, MemoryAllocator* allocator)
      : elems_(0),
        max_length_bits_(max_upper_hash_bits),
        allocator_(allocator) {
    Alloc();
  }

  ~LRUHandleTable2() {
    auto alloc = allocator_;
    ApplyToEntriesRange(
        [alloc](LRUHandle2* h) {
          if (!h->HasRefs()) {
            h->Free(alloc);
          }
        },
        0, GetBufferSize());
    Free();
  }

  LRUHandle2* Lookup(const Slice& key, uint32_t hash) {
    return FindPointer(key, hash)->handle;
  }

  LRUHandle2* Insert(LRUHandle2* h) {
    auto place_value = FindCell(h->key(), h->hash, grower_.place(h->hash));
    LRUCell2* found_cell = buf_ + place_value;
    LRUHandle2* old_handle = found_cell->handle;
    bool is_empty_cell = found_cell->IsZero();
    found_cell->handle = h;
    if (is_empty_cell) {
      ++elems_;
      // TODO:
      /*
#if !defined(likely)
#    define likely(x)   (__builtin_expect(!!(x), 1))
#endif
#if !defined(unlikely)
#    define unlikely(x) (__builtin_expect(!!(x), 0))
#endif
       **/
      if (grower_.overflow(elems_)) {
        Resize();
      }
    }
    return old_handle;
  }

  LRUHandle2* Remove(const Slice& key, uint32_t hash) {
    size_t erased_key_position = FindCell(key, hash, grower_.place(hash));
    if (buf_[erased_key_position].IsZero()) {
      return nullptr;
    }
    auto* erased_handle = buf_[erased_key_position].handle;

    assert(elems_ < grower_.bufSize());

    size_t next_position = erased_key_position;
    while (true) {
      next_position = grower_.next(next_position);

      if (buf_[next_position].IsZero()) {
        break;
      }

      size_t optimal_position =
          grower_.place(buf_[next_position].handle->GetHash());
      if (optimal_position == next_position) {
        continue;
      }

      if (next_position > erased_key_position &&
          ((optimal_position > erased_key_position) ||
           (optimal_position < next_position))) {
        continue;
      }

      if (next_position < erased_key_position &&
          ((optimal_position > erased_key_position) ||
           (optimal_position < next_position))) {
        continue;
      }

      buf_[erased_key_position].handle = buf_[next_position].handle;
      erased_key_position = next_position;
    }

    buf_[erased_key_position].SetZero();
    --elems_;

    return erased_handle;
  }

  template <typename T>
  void ApplyToEntriesRange(T func, size_t index_begin, size_t index_end) {
    for (size_t i = index_begin; i < index_end; i++) {
      if (!buf_[i].IsZero()) {
        auto* handle = buf_[i].handle;
        assert(handle->InCache());
        func(handle);
      }
    }
  }

  uint8_t GetLengthBits() const { return grower_.sizeDegree(); }

  size_t GetOccupancyCount() const { return elems_; }

  MemoryAllocator* GetAllocator() const { return allocator_; }

 private:
  // Return a pointer to slot that points to a cache entry that
  // matches key/hash.  If there is no such cache entry, return a
  // pointer to the trailing slot in the corresponding linked list.
  LRUCell2* FindPointer(const Slice& key, uint32_t hash) {
    size_t place_value = FindCell(key, hash, grower_.place(hash));
    LRUCell2* found = buf_ + place_value;
    return found;
  }

  size_t FindCell(const Slice& key, uint32_t hash, size_t place_value) {
    while (!buf_[place_value].IsZero() &&
           (buf_[place_value].handle->GetHash() != hash ||
            key != buf_[place_value].handle->key())) {
      place_value = grower_.next(place_value);
    }
    
    return place_value;
  }

  void Alloc() {
    buf_ = reinterpret_cast<LRUCell2*>(Allocator::alloc(grower_.bufSize()));
  }

  void Free() {
    if (buf_) {
      Allocator::free(static_cast<void*>(buf_), GetBufferSizeInBytes());
      buf_ = nullptr;
    }
  }

  void Resize() {
    const auto old_size = GetBufferSize();
    const auto old_buf_size = GetBufferSizeInBytes();
    grower_.increaseSize();
    buf_ = reinterpret_cast<LRUCell2*>(
        Allocator::realloc(buf_, old_buf_size, grower_.bufSize()));

    size_t i = 0;
    for (; i < old_size; ++i) {
      if (!buf_[i].IsZero()) {
        size_t update_place_value = Reinsert(buf_[i]);
        // TODO: Move left and right links in LRU cache if link will be in
        // LRUCell struct
      }
    }

    size_t new_size = grower_.bufSize();
    for (; i < new_size && !buf_[i].IsZero(); ++i) {
      size_t updated_place_value = Reinsert(buf_[i]);
      // TODO: Move left and right links in LRU cache if link will be in LRUCell
      // struct
    }
  }

  size_t Reinsert(LRUCell2& cell) {
    assert(!cell.IsZero());
    const auto hash_value = cell.handle->GetHash();
    auto place_value = grower_.place(hash_value);

    if (&cell == buf_ + place_value) {
      return place_value;
    }

    place_value = FindCell(cell.handle->key(), hash_value, place_value);

    if (!buf_[place_value].IsZero()) {
      return place_value;
    }
    buf_[place_value].handle = cell.handle;
    cell.SetZero();

    return place_value;
  }

  size_t GetBufferSizeInBytes() const {
    return sizeof(LRUCell2) * grower_.bufSize();
  }

  size_t GetBufferSize() const { return grower_.bufSize(); }

  // The table consists of an array of buckets where each bucket is
  // a linked list of cache entries that hash into the bucket.
  LRUCell2* buf_;

  StringHashTableGrower<8> grower_;

  // Number of elements currently in the table.
  size_t elems_;

  // Set from max_upper_hash_bits (see constructor).
  // TODO: Check overflowing while resizing
  const uint8_t max_length_bits_;

  // From Cache, needed for delete
  MemoryAllocator* const allocator_;
};

// A single shard of sharded cache.
class ALIGN_AS(CACHE_LINE_SIZE) LRUCacheShard2 final : public CacheShardBase {
 public:
  // NOTE: the eviction_callback ptr is saved, as is it assumed to be kept
  // alive in Cache.
  LRUCacheShard2(size_t capacity, bool strict_capacity_limit,
                 double high_pri_pool_ratio, double low_pri_pool_ratio,
                 bool use_adaptive_mutex,
                 CacheMetadataChargePolicy metadata_charge_policy,
                 int max_upper_hash_bits, MemoryAllocator* allocator,
                 const Cache::EvictionCallback* eviction_callback);

 public:  // Type definitions expected as parameter to ShardedCache
  using HandleImpl = LRUHandle2;
  using HashVal = uint32_t;
  using HashCref = uint32_t;

 public:  // Function definitions expected as parameter to ShardedCache
  static inline HashVal ComputeHash(const Slice& key, uint32_t seed) {
    return Lower32of64(GetSliceNPHash64(key, seed));
  }

  // Separate from constructor so caller can easily make an array of LRUCache
  // if current usage is more than new capacity, the function will attempt to
  // free the needed space.
  void SetCapacity(size_t capacity);

  // Set the flag to reject insertion if cache if full.
  void SetStrictCapacityLimit(bool strict_capacity_limit);

  // Set percentage of capacity reserved for high-pri cache entries.
  void SetHighPriorityPoolRatio(double high_pri_pool_ratio);

  // Set percentage of capacity reserved for low-pri cache entries.
  void SetLowPriorityPoolRatio(double low_pri_pool_ratio);

  // Like Cache methods, but with an extra "hash" parameter.
  Status Insert(const Slice& key, uint32_t hash, Cache::ObjectPtr value,
                const Cache::CacheItemHelper* helper, size_t charge,
                LRUHandle2** handle, Cache::Priority priority);

  LRUHandle2* CreateStandalone(const Slice& key, uint32_t hash,
                               Cache::ObjectPtr obj,
                               const Cache::CacheItemHelper* helper,
                               size_t charge, bool allow_uncharged);

  LRUHandle2* Lookup(const Slice& key, uint32_t hash,
                     const Cache::CacheItemHelper* helper,
                     Cache::CreateContext* create_context,
                     Cache::Priority priority, Statistics* stats);

  bool Release(LRUHandle2* handle, bool useful, bool erase_if_last_ref);
  bool Ref(LRUHandle2* handle);
  void Erase(const Slice& key, uint32_t hash);

  // Although in some platforms the update of size_t is atomic, to make sure
  // GetUsage() and GetPinnedUsage() work correctly under any platform, we'll
  // protect them with mutex_.

  size_t GetUsage() const;
  size_t GetPinnedUsage() const;
  size_t GetOccupancyCount() const;
  size_t GetTableAddressCount() const;

  void ApplyToSomeEntries(
      const std::function<void(const Slice& key, Cache::ObjectPtr value,
                               size_t charge,
                               const Cache::CacheItemHelper* helper)>& callback,
      size_t average_entries_per_lock, size_t* state);

  void EraseUnRefEntries();

 public:  // other function definitions
  void TEST_GetLRUList(LRUHandle2** lru, LRUHandle2** lru_low_pri,
                       LRUHandle2** lru_bottom_pri);

  // Retrieves number of elements in LRU, for unit test purpose only.
  // Not threadsafe.
  size_t TEST_GetLRUSize();

  // Retrieves high pri pool ratio
  double GetHighPriPoolRatio();

  // Retrieves low pri pool ratio
  double GetLowPriPoolRatio();

  void AppendPrintableOptions(std::string& /*str*/) const;

 private:
  friend class LRUCache2;
  // Insert an item into the hash table and, if handle is null, insert into
  // the LRU list. Older items are evicted as necessary. Frees `item` on
  // non-OK status.
  Status InsertItem(LRUHandle2* item, LRUHandle2** handle);

  void LRU_Remove(LRUHandle2* e);
  void LRU_Insert(LRUHandle2* e);

  // Overflow the last entry in high-pri pool to low-pri pool until size of
  // high-pri pool is no larger than the size specify by high_pri_pool_pct.
  void MaintainPoolSize();

  // Free some space following strict LRU policy until enough space
  // to hold (usage_ + charge) is freed or the lru list is empty
  // This function is not thread safe - it needs to be executed while
  // holding the mutex_.
  void EvictFromLRU(size_t charge, autovector<LRUHandle2*>* deleted);

  void NotifyEvicted(const autovector<LRUHandle2*>& evicted_handles);

  LRUHandle2* CreateHandle(const Slice& key, uint32_t hash,
                           Cache::ObjectPtr value,
                           const Cache::CacheItemHelper* helper, size_t charge);

  // Initialized before use.
  size_t capacity_;

  // Memory size for entries in high-pri pool.
  size_t high_pri_pool_usage_;

  // Memory size for entries in low-pri pool.
  size_t low_pri_pool_usage_;

  // Whether to reject insertion if cache reaches its full capacity.
  bool strict_capacity_limit_;

  // Ratio of capacity reserved for high priority cache entries.
  double high_pri_pool_ratio_;

  // High-pri pool size, equals to capacity * high_pri_pool_ratio.
  // Remember the value to avoid recomputing each time.
  double high_pri_pool_capacity_;

  // Ratio of capacity reserved for low priority cache entries.
  double low_pri_pool_ratio_;

  // Low-pri pool size, equals to capacity * low_pri_pool_ratio.
  // Remember the value to avoid recomputing each time.
  double low_pri_pool_capacity_;

  // Dummy head of LRU list.
  // lru.prev is newest entry, lru.next is oldest entry.
  // LRU contains items which can be evicted, ie reference only by cache
  LRUHandle2 lru_;

  // Pointer to head of low-pri pool in LRU list.
  LRUHandle2* lru_low_pri_;

  // Pointer to head of bottom-pri pool in LRU list.
  LRUHandle2* lru_bottom_pri_;

  // ------------^^^^^^^^^^^^^-----------
  // Not frequently modified data members
  // ------------------------------------
  //
  // We separate data members that are updated frequently from the ones that
  // are not frequently updated so that they don't share the same cache line
  // which will lead into false cache sharing
  //
  // ------------------------------------
  // Frequently modified data members
  // ------------vvvvvvvvvvvvv-----------
  LRUHandleTable2 table_;

  // Memory size for entries residing in the cache.
  size_t usage_;

  // Memory size for entries residing only in the LRU list.
  size_t lru_usage_;

  // mutex_ protects the following state.
  // We don't count mutex_ as the cache's internal state so semantically we
  // don't mind mutex_ invoking the non-const actions.
  mutable DMutex mutex_;

  // A reference to Cache::eviction_callback_
  const Cache::EvictionCallback& eviction_callback_;
};

class LRUCache2
#ifdef NDEBUG
    final
#endif
    : public ShardedCache<LRUCacheShard2> {
 public:
  explicit LRUCache2(const LRUCacheOptions2& opts);
  const char* Name() const override { return "LRUCache"; }
  ObjectPtr Value(Handle* handle) override;
  size_t GetCharge(Handle* handle) const override;
  const CacheItemHelper* GetCacheItemHelper(Handle* handle) const override;

  // Retrieves number of elements in LRU, for unit test purpose only.
  size_t TEST_GetLRUSize();
  // Retrieves high pri pool ratio.
  double GetHighPriPoolRatio();
};

}  // namespace lru_cache

using LRUCache = lru_cache::LRUCache2;
using LRUHandle = lru_cache::LRUHandle2;
using LRUCacheShard = lru_cache::LRUCacheShard2;

}  // namespace ROCKSDB_NAMESPACE
