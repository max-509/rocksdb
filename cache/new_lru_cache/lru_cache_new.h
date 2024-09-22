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
#include <memory>
#include <string>

#include "cache/grower.h"
#include "cache/secondary_cache_adapter.h"
#include "cache/sharded_cache.h"
#include "monitoring/perf_context_imp.h"
#include "monitoring/statistics_impl.h"
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
// S/ sEntries are referenced by cache and/or by any external entity.
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

enum ImFlags : uint8_t {
    // Whether this entry is high priority entry.
    IM_IS_HIGH_PRI = (1 << 0),
    // Whether this entry is low priority entry.
    IM_IS_LOW_PRI = (1 << 1),
    // Marks result handles that should not be inserted into cache
    IM_IS_STANDALONE = (1 << 2),
  };


struct LRUCellHandle : public Cache::Handle {
  Cache::ObjectPtr value;
  const Cache::CacheItemHelper* helper;

  size_t key_length;

  // The hash of key(). Used for fast sharding and comparisons.
  uint32_t hash;
  // The number of external refs to this entry. The cache itself is not counted.
  uint32_t refs; 

   // Mutable flags - access controlled by mutex
  // The m_ and M_ prefixes (and im_ and IM_ later) are to hopefully avoid
  // checking an M_ flag on im_flags or an IM_ flag on m_flags.
  uint8_t m_flags; 

  // "Immutable" flags - only set in single-threaded context and then
  // can be accessed without mutex
  uint8_t im_flags; 
  char key_data[1];
};

struct LRUCell {
  LRUCellHandle* handle;

  LRUCell* next;
  LRUCell* prev;
  size_t total_charge;  // TODO(opt): Only allow uint32_t?

  bool IsZero() const { return nullptr == handle; }

  Slice key() const { return Slice(handle->key_data, handle->key_length); }

  // For HandleImpl concept
  uint32_t GetHash() const { return handle->hash; }

  // Increase the reference count by 1.
  void Ref() { handle->refs++; }

  // Just reduce the reference count by 1. Return true if it was last reference.
  bool Unref() {
    assert(handle->refs > 0);
    handle->refs--;
    return handle->refs == 0;
  }

  // Return true if there are external refs, false otherwise.
  bool HasRefs() const { return refs > 0; }

  bool InCache() const { return handle->m_flags & M_IN_CACHE; }
  bool IsHighPri() const {
    return handle->im_flags & MIM_IS_HIGH_PRI;
  }
  bool InHighPriPool() const {
    return handle->m_flags & MM_IN_HIGH_PRI_POOL;
  }
  bool IsLowPri() const {
    return handle->im_flags & MIM_IS_LOW_PRI;
  }
  bool InLowPriPool() const {
    return handle->m_flags & MM_IN_LOW_PRI_POOL;
  }
  bool HasHit() const { return handle->m_flags & MM_HAS_HIT; }
  bool IsStandalone() const {
    return handle->im_flags & MIM_IS_STANDALONE;
  }

  void SetInCache(bool in_cache) {
    if (in_cache) {
      m_flags |= MM_IN_CACHE;
    } else {
      m_flags &= ~MM_IN_CACHE;
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
    // TODO: Compute right memory usage
    if (metadata_charge_policy != kFullChargeCacheMetadata) {
      return 0;
    } else {
      // This is the size that is used when a new handle is created.
      return sizeof(LRUCell) + key_data.capacity();
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
// A single shard of sharded cache.
class ALIGN_AS(CACHE_LINE_SIZE) LRUCacheShard final : public CacheShardBase {
 public:
  // NOTE: the eviction_callback ptr is saved, as is it assumed to be kept
  // alive in Cache.
  LRUCacheShard(size_t capacity, bool strict_capacity_limit,
                double high_pri_pool_ratio, double low_pri_pool_ratio,
                bool use_adaptive_mutex,
                CacheMetadataChargePolicy metadata_charge_policy,
                int max_upper_hash_bits, MemoryAllocator* allocator,
                const Cache::EvictionCallback* eviction_callback);

 public:  // Type definitions expected as parameter to ShardedCache
  using HandleImpl = LRUCell;
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
                LRUCell** handle, Cache::Priority priority);

  LRUCell* CreateStandalone(const Slice& key, uint32_t hash,
                            Cache::ObjectPtr obj,
                            const Cache::CacheItemHelper* helper, size_t charge,
                            bool allow_uncharged);

  LRUCell* Lookup(const Slice& key, uint32_t hash,
                  const Cache::CacheItemHelper* helper,
                  Cache::CreateContext* create_context,
                  Cache::Priority priority, Statistics* stats);

  bool Release(LRUCell* handle, bool useful, bool erase_if_last_ref);
  bool Ref(LRUCell* handle);
  void Erase(const Slice& key, uint32_t hash);

  std::pair<LRUCell *, bool> emplace(LRUCell &cell) {
  LRUCell** ptr = FindPointer(h->key(), h->hash);
  LRUCell* old = *ptr;
  h->next_hash = (old == nullptr ? nullptr : old->next_hash);
  *ptr = h;
  if (old == nullptr) {
    ++elems_;
    if ((elems_ >> length_bits_) > 0) {  // elems_ >= length
      // Since each cache entry is fairly large, we aim for a small
      // average linked list length (<= 1).
      Resize();
    }
  }
  return old;
      
  }

  LRUCell* Find(const Slice& key, uint32_t hash) {
    size_t place_value = FindCell(key, hash, grower_.place(hash));
    LRUCell* found = list_.get() + place_value;
    rturn !found->IsZero() ? found : nullptr;
  }

  size_t FindCell(const Slice& key, uint32_t hash, size_t place_value) {
    while (list_[place_value].IsZero() && (list_[place_value].hash != hash ||
                                           key != list_[place_value].key())) {
      place_value = grower_.next(place_value);
    }
    return place_value;
  }

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
  void TEST_GetLRUList(LRUCell** lru, LRUCell** lru_low_pri,
                       LRUCell** lru_bottom_pri);

  // Retrieves number of elements in LRU, for unit test purpose only.
  // Not threadsafe.
  size_t TEST_GetLRUSize();

  // Retrieves high pri pool ratio
  double GetHighPriPoolRatio();

  // Retrieves low pri pool ratio
  double GetLowPriPoolRatio();

  void AppendPrintableOptions(std::string& /*str*/) const;

 private:
  friend class LRUCache;
  // Insert an item into the hash table and, if handle is null, insert into
  // the LRU list. Older items are evicted as necessary. Frees `item` on
  // non-OK status.
  Status InsertItem(LRUCell&& item, LRUCell** handle);

  void LRU_Remove(LRUCell* e);
  void LRU_Insert(LRUCell* e);

  // Overflow the last entry in high-pri pool to low-pri pool until size of
  // high-pri pool is no larger than the size specify by high_pri_pool_pct.
  void MaintainPoolSize();

  // Free some space following strict LRU policy until enough space
  // to hold (usage_ + charge) is freed or the lru list is empty
  // This function is not thread safe - it needs to be executed while
  // holding the mutex_.
  void EvictFromLRU(size_t charge, autovector<LRUCell*>* deleted);

  void NotifyEvicted(const autovector<LRUCell*>& evicted_handles);

  LRUCell CreateHandle(const Slice& key, uint32_t hash, Cache::ObjectPtr value,
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
  LRUCell lru_;

  // Pointer to head of low-pri pool in LRU list.
  LRUCell* lru_low_pri_;

  // Pointer to head of bottom-pri pool in LRU list.
  LRUCell* lru_bottom_pri_;

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

  LRUCell* Lookup(const Slice& key, uint32_t hash);
  LRUCell* Insert(LRUCell* h);
  LRUCell* Remove(const Slice& key, uint32_t hash);

  template <typename T>
  void ApplyToEntriesRange(T func, size_t index_begin, size_t index_end) {
    for (size_t i = index_begin; i < index_end; i++) {
      LRUCell* h = list_[i];
      while (h != nullptr) {
        auto n = h->next_hash;
        assert(h->InCache());
        func(h);
        h = n;
      }
    }
  }

  int GetLengthBits() const { return length_bits_; }

  size_t GetOccupancyCount() const { return elems_; }

  MemoryAllocator* GetAllocator() const { return allocator_; }

 private:
  // Return a pointer to slot that points to a cache entry that
  // matches key/hash.  If there is no such cache entry, return a
  // pointer to the trailing slot in the corresponding linked list.
  LRUCell** FindPointer(const Slice& key, uint32_t hash);

  void Resize();

  // Number of hash bits (upper because lower bits used for sharding)
  // used for table index. Length == 1 << length_bits_
  int length_bits_;

  // The table consists of an array of buckets where each bucket is
  // a linked list of cache entries that hash into the bucket.
  std::unique_ptr<LRUCell[]> list_;
  StringHashTableGrower<> grower_;

  // Number of elements currently in the table.
  uint32_t elems_;

  // Set from max_upper_hash_bits (see constructor).
  const int max_length_bits_;

  // From Cache, needed for delete
  MemoryAllocator* const allocator_;

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

class LRUCache
#ifdef NDEBUG
    final
#endif
    : public ShardedCache<LRUCacheShard> {
 public:
  explicit LRUCache(const LRUCacheOptions& opts);
  const char* Name() const override { return "LRUCache"; }
  ObjectPtr Value(Handle* handle) override;
  size_t GetCharge(Handle* handle) const override;
  const CacheItemHelper* GetCacheItemHelper(Handle* handle) const override;

  // Retrieves number of elements in LRU, for unit test purpose only.
  size_t TEST_GetLRUSize();
  // Retrieves high pri pool ratio.
  double GetHighPriPoolRatio();
};

LRUHandleTable::LRUHandleTable(int max_upper_hash_bits,
                               MemoryAllocator* allocator)
    : length_bits_(/* historical starting size*/ 4),
      list_(reinterpret_cast<LRUCell*>(
          new uint8_t[sizeof(LRUCell) * size_t{1} << length_bits_])),
      elems_(0),
      max_length_bits_(max_upper_hash_bits),
      allocator_(allocator) {}

LRUHandleTable::~LRUHandleTable() {
  auto alloc = allocator_;
  ApplyToEntriesRange(
      [alloc](LRUCell* h) {
        if (!h->HasRefs()) {
          h->Free(alloc);
        }
      },
      0, size_t{1} << length_bits_);
}

LRUCell* LRUHandleTable::Lookup(const Slice& key, uint32_t hash) {
  return *FindPointer(key, hash);
}

LRUCell* LRUHandleTable::Insert(LRUCell* h) {
  LRUCell** ptr = FindPointer(h->key(), h->hash);
  LRUCell* old = *ptr;
  h->next_hash = (old == nullptr ? nullptr : old->next_hash);
  *ptr = h;
  if (old == nullptr) {
    ++elems_;
    if ((elems_ >> length_bits_) > 0) {  // elems_ >= length
      // Since each cache entry is fairly large, we aim for a small
      // average linked list length (<= 1).
      Resize();
    }
  }
  return old;
}

LRUCell* LRUHandleTable::Remove(const Slice& key, uint32_t hash) {
  LRUCell** ptr = FindPointer(key, hash);
  LRUCell* result = *ptr;
  if (result != nullptr) {
    *ptr = result->next_hash;
    --elems_;
  }
  return result;
}

LRUCell** LRUHandleTable::FindPointer(const Slice& key, uint32_t hash) {
  LRUCell** ptr =
      &list_[hash >> (std::numeric_limits<uint32_t>::digits - length_bits_)];
  while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
    ptr = &(*ptr)->next_hash;
  }
  return ptr;
}

void LRUHandleTable::Resize() {
  if (length_bits_ >= max_length_bits_) {
    // Due to reaching limit of hash information, if we made the table bigger,
    // we would allocate more addresses but only the same number would be used.
    return;
  }
  if (length_bits_ >= 31) {
    // Avoid undefined behavior shifting uint32_t by 32.
    return;
  }

  uint32_t old_length = uint32_t{1} << length_bits_;
  int new_length_bits = length_bits_ + 1;
  std::unique_ptr<LRUCell*[]> new_list{
      new LRUCell* [size_t{1} << new_length_bits] {}};
  [[maybe_unused]] uint32_t count = 0;
  for (uint32_t i = 0; i < old_length; i++) {
    LRUCell* h = list_[i];
    while (h != nullptr) {
      LRUCell* next = h->next_hash;
      uint32_t hash = h->hash;
      LRUCell** ptr = &new_list[hash >> (32 - new_length_bits)];
      h->next_hash = *ptr;
      *ptr = h;
      h = next;
      count++;
    }
  }
  assert(elems_ == count);
  list_ = std::move(new_list);
  length_bits_ = new_length_bits;
}

// LRUHandleTableWithOpenAddress::LRUHandleTableWithOpenAddress(int
// max_upper_hash_bits,
//                                MemoryAllocator* allocator)
//     : length_bits_(/* historical starting size*/ 4),
//       buf_(reinterpret_cast<LRUHandle *>(new uint8_t[sizeof(LRUHandle) *
//       size_t{1} << length_bits_])), elems_(0),
//       max_length_bits_(max_upper_hash_bits),
//       allocator_(allocator) {}
//
// LRUHandleTableWithOpenAddress::~LRUHandleTableWithOpenAddress() {
//   auto alloc = allocator_;
//   ApplyToEntriesRange(
//       [alloc](LRUHandle* h) {
//         if (!h->HasRefs()) {
//           h->Free(alloc);
//         }
//       },
//       0, size_t{1} << length_bits_);
// }
//
// LRUHandle* LRUHandleTableWithOpenAddress::Lookup(const Slice& key, uint32_t
// hash) {
//   return *FindPointer(key, hash);
// }
//
// LRUHandle* LRUHandleTableWithOpenAddress::Insert(LRUHandle* h) {
//   LRUHandle** ptr = FindPointer(h->key(), h->hash);
//   LRUHandle* old = *ptr;
//   h->next_hash = (old == nullptr ? nullptr : old->next_hash);
//   *ptr = h;
//   if (old == nullptr) {
//     ++elems_;
//     if ((elems_ >> length_bits_) > 0) {  // elems_ >= length
//       // Since each cache entry is fairly large, we aim for a small
//       // average linked list length (<= 1).
//       Resize();
//     }
//   }
//   return old;
// }
//
// LRUHandle* LRUHandleTableWithOpenAddress::Remove(const Slice& key, uint32_t
// hash) {
//   LRUHandle** ptr = FindPointer(key, hash);
//   LRUHandle* result = *ptr;
//   if (result != nullptr) {
//     *ptr = result->next_hash;
//     --elems_;
//   }
//   return result;
// }
//
// LRUHandle** LRUHandleTableWithOpenAddress::FindPointer(const Slice& key,
// uint32_t hash) {
//   LRUHandle** ptr = &list_[hash >> (32 - length_bits_)];
//   while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
//     ptr = &(*ptr)->next_hash;
//   }
//   return ptr;
// }
//
// void LRUHandleTableWithOpenAddress::Resize() {
//   if (length_bits_ >= max_length_bits_) {
//     // Due to reaching limit of hash information, if we made the table
//     bigger,
//     // we would allocate more addresses but only the same number would be
//     used. return;
//   }
//   if (length_bits_ >= 31) {
//     // Avoid undefined behavior shifting uint32_t by 32.
//     return;
//   }
//
//   uint32_t old_length = uint32_t{1} << length_bits_;
//   int new_length_bits = length_bits_ + 1;
//   std::unique_ptr<LRUHandle* []> new_list {
//       new LRUHandle* [size_t{1} << new_length_bits] {}
//   };
//   [[maybe_unused]] uint32_t count = 0;
//   for (uint32_t i = 0; i < old_length; i++) {
//     LRUHandle* h = list_[i];
//     while (h != nullptr) {
//       LRUHandle* next = h->next_hash;
//       uint32_t hash = h->hash;
//       LRUHandle** ptr = &new_list[hash >> (32 - new_length_bits)];
//       h->next_hash = *ptr;
//       *ptr = h;
//       h = next;
//       count++;
//     }
//   }
//   assert(elems_ == count);
//   list_ = std::move(new_list);
//   length_bits_ = new_length_bits;
// }

LRUCacheShard::LRUCacheShard(size_t capacity, bool strict_capacity_limit,
                             double high_pri_pool_ratio,
                             double low_pri_pool_ratio, bool use_adaptive_mutex,
                             CacheMetadataChargePolicy metadata_charge_policy,
                             int max_upper_hash_bits,
                             MemoryAllocator* allocator,
                             const Cache::EvictionCallback* eviction_callback)
    : CacheShardBase(metadata_charge_policy),
      capacity_(0),
      high_pri_pool_usage_(0),
      low_pri_pool_usage_(0),
      strict_capacity_limit_(strict_capacity_limit),
      high_pri_pool_ratio_(high_pri_pool_ratio),
      high_pri_pool_capacity_(0),
      low_pri_pool_ratio_(low_pri_pool_ratio),
      low_pri_pool_capacity_(0),
      table_(max_upper_hash_bits, allocator),
      usage_(0),
      lru_usage_(0),
      mutex_(use_adaptive_mutex),
      eviction_callback_(*eviction_callback) {
  // Make empty circular linked list.
  lru_.next = &lru_;
  lru_.prev = &lru_;
  lru_low_pri_ = &lru_;
  lru_bottom_pri_ = &lru_;
  SetCapacity(capacity);
}

void LRUCacheShard::EraseUnRefEntries() {
  autovector<LRUCell*> last_reference_list;
  {
    DMutexLock l(mutex_);
    while (lru_.next != &lru_) {
      LRUCell* old = lru_.next;
      // LRU list contains only elements which can be evicted.
      assert(old->InCache() && !old->HasRefs());
      LRU_Remove(old);
      table_.Remove(old->key(), old->hash);
      old->SetInCache(false);
      assert(usage_ >= old->total_charge);
      usage_ -= old->total_charge;
      last_reference_list.push_back(old);
    }
  }

  for (auto entry : last_reference_list) {
    entry->Free(table_.GetAllocator());
  }
}

void LRUCacheShard::ApplyToSomeEntries(
    const std::function<void(const Slice& key, Cache::ObjectPtr value,
                             size_t charge,
                             const Cache::CacheItemHelper* helper)>& callback,
    size_t average_entries_per_lock, size_t* state) {
  // The state is essentially going to be the starting hash, which works
  // nicely even if we resize between calls because we use upper-most
  // hash bits for table indexes.
  DMutexLock l(mutex_);
  int length_bits = table_.GetLengthBits();
  size_t length = size_t{1} << length_bits;

  assert(average_entries_per_lock > 0);
  // Assuming we are called with same average_entries_per_lock repeatedly,
  // this simplifies some logic (index_end will not overflow).
  assert(average_entries_per_lock < length || *state == 0);

  size_t index_begin = *state >> (sizeof(size_t) * 8u - length_bits);
  size_t index_end = index_begin + average_entries_per_lock;
  if (index_end >= length) {
    // Going to end
    index_end = length;
    *state = SIZE_MAX;
  } else {
    *state = index_end << (sizeof(size_t) * 8u - length_bits);
  }

  table_.ApplyToEntriesRange(
      [callback, metadata_charge_policy = metadata_charge_policy_](LRUCell* h) {
        callback(h->key(), h->value, h->GetCharge(metadata_charge_policy),
                 h->helper);
      },
      index_begin, index_end);
}

void LRUCacheShard::TEST_GetLRUList(LRUCell** lru, LRUCell** lru_low_pri,
                                    LRUCell** lru_bottom_pri) {
  DMutexLock l(mutex_);
  *lru = &lru_;
  *lru_low_pri = lru_low_pri_;
  *lru_bottom_pri = lru_bottom_pri_;
}

size_t LRUCacheShard::TEST_GetLRUSize() {
  DMutexLock l(mutex_);
  LRUCell* lru_handle = lru_.next;
  size_t lru_size = 0;
  while (lru_handle != &lru_) {
    lru_size++;
    lru_handle = lru_handle->next;
  }
  return lru_size;
}

double LRUCacheShard::GetHighPriPoolRatio() {
  DMutexLock l(mutex_);
  return high_pri_pool_ratio_;
}

double LRUCacheShard::GetLowPriPoolRatio() {
  DMutexLock l(mutex_);
  return low_pri_pool_ratio_;
}

void LRUCacheShard::LRU_Remove(LRUCell* e) {
  assert(e->next != nullptr);
  assert(e->prev != nullptr);
  if (lru_low_pri_ == e) {
    lru_low_pri_ = e->prev;
  }
  if (lru_bottom_pri_ == e) {
    lru_bottom_pri_ = e->prev;
  }
  e->next->prev = e->prev;
  e->prev->next = e->next;
  e->prev = e->next = nullptr;
  assert(lru_usage_ >= e->total_charge);
  lru_usage_ -= e->total_charge;
  assert(!e->InHighPriPool() || !e->InLowPriPool());
  if (e->InHighPriPool()) {
    assert(high_pri_pool_usage_ >= e->total_charge);
    high_pri_pool_usage_ -= e->total_charge;
  } else if (e->InLowPriPool()) {
    assert(low_pri_pool_usage_ >= e->total_charge);
    low_pri_pool_usage_ -= e->total_charge;
  }
}

void LRUCacheShard::LRU_Insert(LRUCell* e) {
  assert(e->next == nullptr);
  assert(e->prev == nullptr);
  if (high_pri_pool_ratio_ > 0 && (e->IsHighPri() || e->HasHit())) {
    // Inset "e" to head of LRU list.
    e->next = &lru_;
    e->prev = lru_.prev;
    e->prev->next = e;
    e->next->prev = e;
    e->SetInHighPriPool(true);
    e->SetInLowPriPool(false);
    high_pri_pool_usage_ += e->total_charge;
    MaintainPoolSize();
  } else if (low_pri_pool_ratio_ > 0 &&
             (e->IsHighPri() || e->IsLowPri() || e->HasHit())) {
    // Insert "e" to the head of low-pri pool.
    e->next = lru_low_pri_->next;
    e->prev = lru_low_pri_;
    e->prev->next = e;
    e->next->prev = e;
    e->SetInHighPriPool(false);
    e->SetInLowPriPool(true);
    low_pri_pool_usage_ += e->total_charge;
    lru_low_pri_ = e;
    MaintainPoolSize();
  } else {
    // Insert "e" to the head of bottom-pri pool.
    e->next = lru_bottom_pri_->next;
    e->prev = lru_bottom_pri_;
    e->prev->next = e;
    e->next->prev = e;
    e->SetInHighPriPool(false);
    e->SetInLowPriPool(false);
    // if the low-pri pool is empty, lru_low_pri_ also needs to be updated.
    if (lru_bottom_pri_ == lru_low_pri_) {
      lru_low_pri_ = e;
    }
    lru_bottom_pri_ = e;
  }
  lru_usage_ += e->total_charge;
}

void LRUCacheShard::MaintainPoolSize() {
  while (high_pri_pool_usage_ > high_pri_pool_capacity_) {
    // Overflow last entry in high-pri pool to low-pri pool.
    lru_low_pri_ = lru_low_pri_->next;
    assert(lru_low_pri_ != &lru_);
    assert(lru_low_pri_->InHighPriPool());
    lru_low_pri_->SetInHighPriPool(false);
    lru_low_pri_->SetInLowPriPool(true);
    assert(high_pri_pool_usage_ >= lru_low_pri_->total_charge);
    high_pri_pool_usage_ -= lru_low_pri_->total_charge;
    low_pri_pool_usage_ += lru_low_pri_->total_charge;
  }

  while (low_pri_pool_usage_ > low_pri_pool_capacity_) {
    // Overflow last entry in low-pri pool to bottom-pri pool.
    lru_bottom_pri_ = lru_bottom_pri_->next;
    assert(lru_bottom_pri_ != &lru_);
    assert(lru_bottom_pri_->InLowPriPool());
    lru_bottom_pri_->SetInHighPriPool(false);
    lru_bottom_pri_->SetInLowPriPool(false);
    assert(low_pri_pool_usage_ >= lru_bottom_pri_->total_charge);
    low_pri_pool_usage_ -= lru_bottom_pri_->total_charge;
  }
}

void LRUCacheShard::EvictFromLRU(size_t charge, autovector<LRUCell*>* deleted) {
  while ((usage_ + charge) > capacity_ && lru_.next != &lru_) {
    LRUCell* old = lru_.next;
    // LRU list contains only elements which can be evicted.
    assert(old->InCache() && !old->HasRefs());
    LRU_Remove(old);
    RemoveFromTable(old->key(), old->hash);
    old->SetInCache(false);
    assert(usage_ >= old->total_charge);
    usage_ -= old->total_charge;
    deleted->push_back(old);
  }
}

void LRUCacheShard::NotifyEvicted(const autovector<LRUCell*>& evicted_handles) {
  MemoryAllocator* alloc = table_.GetAllocator();
  for (LRUCell* entry : evicted_handles) {
    if (eviction_callback_ &&
        eviction_callback_(entry->key(), static_cast<Cache::Handle*>(entry),
                           entry->HasHit())) {
      // Callback took ownership of obj; just free handle
      free(entry);
    } else {
      // Free the entries here outside of mutex for performance reasons.
      entry->Free(alloc);
    }
  }
}

void LRUCacheShard::SetCapacity(size_t capacity) {
  autovector<LRUCell*> last_reference_list;
  {
    DMutexLock l(mutex_);
    capacity_ = capacity;
    high_pri_pool_capacity_ = capacity_ * high_pri_pool_ratio_;
    low_pri_pool_capacity_ = capacity_ * low_pri_pool_ratio_;
    EvictFromLRU(0, &last_reference_list);
  }

  NotifyEvicted(last_reference_list);
}

void LRUCacheShard::SetStrictCapacityLimit(bool strict_capacity_limit) {
  DMutexLock l(mutex_);
  strict_capacity_limit_ = strict_capacity_limit;
}

Status LRUCacheShard::InsertItem(LRUCell&& e, LRUCell** handle) {
  Status s = Status::OK();
  autovector<LRUCell*> last_reference_list;

  {
    DMutexLock l(mutex_);

    // Free the space following strict LRU policy until enough space
    // is freed or the lru list is empty.
    EvictFromLRU(e->total_charge, &last_reference_list);

    if ((usage_ + e->total_charge) > capacity_ &&
        (strict_capacity_limit_ || handle == nullptr)) {
      e->SetInCache(false);
      if (handle == nullptr) {
        // Don't insert the entry but still return ok, as if the entry inserted
        // into cache and get evicted immediately.
        last_reference_list.push_back(e);
      } else {
        free(e);
        e = nullptr;
        *handle = nullptr;
        s = Status::MemoryLimit("Insert failed due to LRU cache being full.");
      }
    } else {
      // Insert into the cache. Note that the cache might get larger than its
      // capacity if not enough space was freed up.
      LRUCell* old = table_.Insert(e);
      usage_ += e->total_charge;
      if (old != nullptr) {
        s = Status::OkOverwritten();
        assert(old->InCache());
        old->SetInCache(false);
        if (!old->HasRefs()) {
          // old is on LRU because it's in cache and its reference count is 0.
          LRU_Remove(old);
          assert(usage_ >= old->total_charge);
          usage_ -= old->total_charge;
          last_reference_list.push_back(old);
        }
      }
      if (handle == nullptr) {
        LRU_Insert(e);
      } else {
        // If caller already holds a ref, no need to take one here.
        if (!e->HasRefs()) {
          e->Ref();
        }
        *handle = e;
      }
    }
  }

  NotifyEvicted(last_reference_list);

  return s;
}

LRUCell* LRUCacheShard::Lookup(const Slice& key, uint32_t hash,
                               const Cache::CacheItemHelper* /*helper*/,
                               Cache::CreateContext* /*create_context*/,
                               Cache::Priority /*priority*/,
                               Statistics* /*stats*/) {
  DMutexLock l(mutex_);
  LRUCell* e = table_.Lookup(key, hash);
  if (e != nullptr) {
    assert(e->InCache());
    if (!e->HasRefs()) {
      // The entry is in LRU since it's in hash and has no external
      // references.
      LRU_Remove(e);
    }
    e->Ref();
    e->SetHit();
  }
  return e;
}

bool LRUCacheShard::Ref(LRUCell* e) {
  DMutexLock l(mutex_);
  // To create another reference - entry must be already externally referenced.
  assert(e->HasRefs());
  e->Ref();
  return true;
}

void LRUCacheShard::SetHighPriorityPoolRatio(double high_pri_pool_ratio) {
  DMutexLock l(mutex_);
  high_pri_pool_ratio_ = high_pri_pool_ratio;
  high_pri_pool_capacity_ = capacity_ * high_pri_pool_ratio_;
  MaintainPoolSize();
}

void LRUCacheShard::SetLowPriorityPoolRatio(double low_pri_pool_ratio) {
  DMutexLock l(mutex_);
  low_pri_pool_ratio_ = low_pri_pool_ratio;
  low_pri_pool_capacity_ = capacity_ * low_pri_pool_ratio_;
  MaintainPoolSize();
}

bool LRUCacheShard::Release(LRUCell* e, bool /*useful*/,
                            bool erase_if_last_ref) {
  if (e == nullptr) {
    return false;
  }
  bool must_free;
  bool was_in_cache;
  {
    DMutexLock l(mutex_);
    must_free = e->Unref();
    was_in_cache = e->InCache();
    if (must_free && was_in_cache) {
      // The item is still in cache, and nobody else holds a reference to it.
      if (usage_ > capacity_ || erase_if_last_ref) {
        // The LRU list must be empty since the cache is full.
        assert(lru_.next == &lru_ || erase_if_last_ref);
        // Take this opportunity and remove the item.
        table_.Remove(e->key(), e->hash);
        e->SetInCache(false);
      } else {
        // Put the item back on the LRU list, and don't free it.
        LRU_Insert(e);
        must_free = false;
      }
    }
    // If about to be freed, then decrement the cache usage.
    if (must_free) {
      assert(usage_ >= e->total_charge);
      usage_ -= e->total_charge;
    }
  }

  // Free the entry here outside of mutex for performance reasons.
  if (must_free) {
    // Only call eviction callback if we're sure no one requested erasure
    // FIXME: disabled because of test churn
    if (false && was_in_cache && !erase_if_last_ref && eviction_callback_ &&
        eviction_callback_(e->key_data(), static_cast<Cache::Handle*>(e),
                           e->HasHit())) {
      // Callback took ownership of obj; just free handle
      free(e);
    } else {
      e->Free(table_.GetAllocator());
    }
  }
  return must_free;
}

LRUCell LRUCacheShard::CreateHandle(const Slice& key, uint32_t hash,
                                    Cache::ObjectPtr value,
                                    const Cache::CacheItemHelper* helper,
                                    size_t charge) {
  assert(helper);
  // value == nullptr is reserved for indicating failure in SecondaryCache
  assert(!(helper->IsSecondaryCacheCompatible() && value == nullptr));

  LRUCellHandle* handle = static_cast<LRUCellHandle*>(
      malloc(sizeof(LRUCellHandle) - 1 + key.size()));
  handle->value = value;
  handle->helper = helper;
  handle->m_flags = 0;
  handle->im_flags = 0;
  handle->key_length = key.size();
  memcpy(handle->key_data, key.data(), key.size());

  LRUCell e{std::string{key.data(), key.size()},
            value,
            helper,
            hash,
            0,
            0,
            0,
            nullptr,
            nullptr,
            0};

  e->CalcTotalCharge(charge, metadata_charge_policy_);

  return e;
}

Status LRUCacheShard::Insert(const Slice& key, uint32_t hash,
                             Cache::ObjectPtr value,
                             const Cache::CacheItemHelper* helper,
                             size_t charge, LRUCell** handle,
                             Cache::Priority priority) {
  LRUCell e = CreateHandle(key, hash, value, helper, charge);
  e->SetPriority(priority);
  e->SetInCache(true);

  Status s = Status::OK();
  autovector<LRUCell*> last_reference_list;

  {
    DMutexLock l(mutex_);

    // Free the space following strict LRU policy until enough space
    // is freed or the lru list is empty.
    EvictFromLRU(e.total_charge, &last_reference_list);

    if ((usage_ + e.total_charge) > capacity_ &&
        (strict_capacity_limit_ || handle == nullptr)) {
      e.SetInCache(false);
      if (handle == nullptr) {
        // Don't insert the entry but still return ok, as if the entry inserted
        // into cache and get evicted immediately.
        last_reference_list.push_back(e);
      } else {
        *handle = nullptr;
        s = Status::MemoryLimit("Insert failed due to LRU cache being full.");
      }
    } else {
      // Insert into the cache. Note that the cache might get larger than its
      // capacity if not enough space was freed up.
      LRUCell* old = InsertToTable(e);
      usage_ += e->total_charge;
      if (old != nullptr) {
        s = Status::OkOverwritten();
        assert(old->InCache());
        old->SetInCache(false);
        if (!old->HasRefs()) {
          // old is on LRU because it's in cache and its reference count is 0.
          LRU_Remove(old);
          assert(usage_ >= old->total_charge);
          usage_ -= old->total_charge;
          last_reference_list.push_back(old);
        }
      }
      if (handle == nullptr) {
        LRU_Insert(e);
      } else {
        // If caller already holds a ref, no need to take one here.
        if (!e->HasRefs()) {
          e->Ref();
        }
        *handle = e;
      }
    }
  }

  NotifyEvicted(last_reference_list);

  return s;
}

LRUCell* LRUCacheShard::CreateStandalone(const Slice& key, uint32_t hash,
                                         Cache::ObjectPtr value,
                                         const Cache::CacheItemHelper* helper,
                                         size_t charge, bool allow_uncharged) {
  LRUCell* e = CreateHandle(key, hash, value, helper, charge);
  e->SetIsStandalone(true);
  e->Ref();

  autovector<LRUCell*> last_reference_list;

  {
    DMutexLock l(mutex_);

    EvictFromLRU(e->total_charge, &last_reference_list);

    if (strict_capacity_limit_ && (usage_ + e->total_charge) > capacity_) {
      if (allow_uncharged) {
        e->total_charge = 0;
      } else {
        free(e);
        e = nullptr;
      }
    } else {
      usage_ += e->total_charge;
    }
  }

  NotifyEvicted(last_reference_list);
  return e;
}

void LRUCacheShard::Erase(const Slice& key, uint32_t hash) {
  LRUCell* e;
  bool last_reference = false;
  {
    DMutexLock l(mutex_);
    e = table_.Remove(key, hash);
    if (e != nullptr) {
      assert(e->InCache());
      e->SetInCache(false);
      if (!e->HasRefs()) {
        // The entry is in LRU since it's in hash and has no external references
        LRU_Remove(e);
        assert(usage_ >= e->total_charge);
        usage_ -= e->total_charge;
        last_reference = true;
      }
    }
  }

  // Free the entry here outside of mutex for performance reasons.
  // last_reference will only be true if e != nullptr.
  if (last_reference) {
    e->Free(table_.GetAllocator());
  }
}

size_t LRUCacheShard::GetUsage() const {
  DMutexLock l(mutex_);
  return usage_;
}

size_t LRUCacheShard::GetPinnedUsage() const {
  DMutexLock l(mutex_);
  assert(usage_ >= lru_usage_);
  return usage_ - lru_usage_;
}

size_t LRUCacheShard::GetOccupancyCount() const {
  DMutexLock l(mutex_);
  return table_.GetOccupancyCount();
}

size_t LRUCacheShard::GetTableAddressCount() const {
  DMutexLock l(mutex_);
  return size_t{1} << table_.GetLengthBits();
}

void LRUCacheShard::AppendPrintableOptions(std::string& str) const {
  const int kBufferSize = 200;
  char buffer[kBufferSize];
  {
    DMutexLock l(mutex_);
    snprintf(buffer, kBufferSize, "    high_pri_pool_ratio: %.3lf\n",
             high_pri_pool_ratio_);
    snprintf(buffer + strlen(buffer), kBufferSize - strlen(buffer),
             "    low_pri_pool_ratio: %.3lf\n", low_pri_pool_ratio_);
  }
  str.append(buffer);
}

LRUCache::LRUCache(const LRUCacheOptions& opts) : ShardedCache(opts) {
  size_t per_shard = GetPerShardCapacity();
  MemoryAllocator* alloc = memory_allocator();
  InitShards([&](LRUCacheShard* cs) {
    new (cs) LRUCacheShard(per_shard, opts.strict_capacity_limit,
                           opts.high_pri_pool_ratio, opts.low_pri_pool_ratio,
                           opts.use_adaptive_mutex, opts.metadata_charge_policy,
                           /* max_upper_hash_bits */ 32 - opts.num_shard_bits,
                           alloc, &eviction_callback_);
  });
}

Cache::ObjectPtr LRUCache::Value(Handle* handle) {
  auto h = static_cast<const LRUCell*>(handle);
  return h->value;
}

size_t LRUCache::GetCharge(Handle* handle) const {
  return static_cast<const LRUCell*>(handle)->GetCharge(
      GetShard(0).metadata_charge_policy_);
}

const Cache::CacheItemHelper* LRUCache::GetCacheItemHelper(
    Handle* handle) const {
  auto h = static_cast<const LRUCell*>(handle);
  return h->helper;
}

size_t LRUCache::TEST_GetLRUSize() {
  return SumOverShards([](LRUCacheShard& cs) { return cs.TEST_GetLRUSize(); });
}

double LRUCache::GetHighPriPoolRatio() {
  return GetShard(0).GetHighPriPoolRatio();
}

}  // namespace lru_cache

std::shared_ptr<Cache> LRUCacheOptions::MakeSharedCache() const {
  if (num_shard_bits >= 20) {
    return nullptr;  // The cache cannot be sharded into too many fine pieces.
  }
  if (high_pri_pool_ratio < 0.0 || high_pri_pool_ratio > 1.0) {
    // Invalid high_pri_pool_ratio
    return nullptr;
  }
  if (low_pri_pool_ratio < 0.0 || low_pri_pool_ratio > 1.0) {
    // Invalid low_pri_pool_ratio
    return nullptr;
  }
  if (low_pri_pool_ratio + high_pri_pool_ratio > 1.0) {
    // Invalid high_pri_pool_ratio and low_pri_pool_ratio combination
    return nullptr;
  }
  // For sanitized options
  LRUCacheOptions opts = *this;
  if (opts.num_shard_bits < 0) {
    opts.num_shard_bits = GetDefaultCacheShardBits(capacity);
  }
  std::shared_ptr<Cache> cache = std::make_shared<LRUCache>(opts);
  if (secondary_cache) {
    cache = std::make_shared<CacheWithSecondaryAdapter>(cache, secondary_cache);
  }
  return cache;
}

std::shared_ptr<RowCache> LRUCacheOptions::MakeSharedRowCache() const {
  if (secondary_cache) {
    // Not allowed for a RowCache
    return nullptr;
  }
  // Works while RowCache is an alias for Cache
  return MakeSharedCache();
}

}  // namespace lru_cache

using LRUCache = lru_cache::LRUCache;
using LRUHandle = lru_cache::LRUCell;
using LRUCacheShard = lru_cache::LRUCacheShardNew;

}  // namespace ROCKSDB_NAMESPACE
