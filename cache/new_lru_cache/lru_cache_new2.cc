//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "cache/new_lru_cache/lru_cache_new2.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cache/secondary_cache_adapter.h"
#include "monitoring/perf_context_imp.h"
#include "monitoring/statistics_impl.h"
#include "port/lang.h"
#include "util/distributed_mutex.h"

namespace ROCKSDB_NAMESPACE {
namespace lru_cache {

LRUCacheShard2::LRUCacheShard2(size_t capacity, bool strict_capacity_limit,
                               double high_pri_pool_ratio,
                               double low_pri_pool_ratio,
                               bool use_adaptive_mutex,
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

void LRUCacheShard2::EraseUnRefEntries() {
  autovector<LRUHandle2*> last_reference_list;
  {
    DMutexLock l(mutex_);
    while (lru_.next != &lru_) {
      LRUHandle2* old = lru_.next;
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

void LRUCacheShard2::ApplyToSomeEntries(
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
      [callback,
       metadata_charge_policy = metadata_charge_policy_](LRUHandle2* h) {
        callback(h->key(), h->value, h->GetCharge(metadata_charge_policy),
                 h->helper);
      },
      index_begin, index_end);
}

void LRUCacheShard2::TEST_GetLRUList(LRUHandle2** lru, LRUHandle2** lru_low_pri,
                                     LRUHandle2** lru_bottom_pri) {
  DMutexLock l(mutex_);
  *lru = &lru_;
  *lru_low_pri = lru_low_pri_;
  *lru_bottom_pri = lru_bottom_pri_;
}

size_t LRUCacheShard2::TEST_GetLRUSize() {
  DMutexLock l(mutex_);
  LRUHandle2* lru_handle = lru_.next;
  size_t lru_size = 0;
  while (lru_handle != &lru_) {
    lru_size++;
    lru_handle = lru_handle->next;
  }
  return lru_size;
}

double LRUCacheShard2::GetHighPriPoolRatio() {
  DMutexLock l(mutex_);
  return high_pri_pool_ratio_;
}

double LRUCacheShard2::GetLowPriPoolRatio() {
  DMutexLock l(mutex_);
  return low_pri_pool_ratio_;
}

void LRUCacheShard2::LRU_Remove(LRUHandle2* e) {
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

void LRUCacheShard2::LRU_Insert(LRUHandle2* e) {
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

void LRUCacheShard2::MaintainPoolSize() {
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

void LRUCacheShard2::EvictFromLRU(size_t charge,
                                  autovector<LRUHandle2*>* deleted) {
  while ((usage_ + charge) > capacity_ && lru_.next != &lru_) {
    LRUHandle2* old = lru_.next;
    // LRU list contains only elements which can be evicted.
    assert(old->InCache() && !old->HasRefs());
    LRU_Remove(old);
    table_.Remove(old->key(), old->hash);
    old->SetInCache(false);
    assert(usage_ >= old->total_charge);
    usage_ -= old->total_charge;
    deleted->push_back(old);
  }
}

void LRUCacheShard2::NotifyEvicted(
    const autovector<LRUHandle2*>& evicted_handles) {
  MemoryAllocator* alloc = table_.GetAllocator();
  for (LRUHandle2* entry : evicted_handles) {
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

void LRUCacheShard2::SetCapacity(size_t capacity) {
  autovector<LRUHandle2*> last_reference_list;
  {
    DMutexLock l(mutex_);
    capacity_ = capacity;
    high_pri_pool_capacity_ = capacity_ * high_pri_pool_ratio_;
    low_pri_pool_capacity_ = capacity_ * low_pri_pool_ratio_;
    EvictFromLRU(0, &last_reference_list);
  }

  NotifyEvicted(last_reference_list);
}

void LRUCacheShard2::SetStrictCapacityLimit(bool strict_capacity_limit) {
  DMutexLock l(mutex_);
  strict_capacity_limit_ = strict_capacity_limit;
}

Status LRUCacheShard2::InsertItem(LRUHandle2* e, LRUHandle2** handle) {
  Status s = Status::OK();
  autovector<LRUHandle2*> last_reference_list;

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
      LRUHandle2* old = table_.Insert(e);
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

LRUHandle2* LRUCacheShard2::Lookup(const Slice& key, uint32_t hash,
                                   const Cache::CacheItemHelper* /*helper*/,
                                   Cache::CreateContext* /*create_context*/,
                                   Cache::Priority /*priority*/,
                                   Statistics* /*stats*/) {
  DMutexLock l(mutex_);
  LRUHandle2* e = table_.Lookup(key, hash);
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

bool LRUCacheShard2::Ref(LRUHandle2* e) {
  DMutexLock l(mutex_);
  // To create another reference - entry must be already externally referenced.
  assert(e->HasRefs());
  e->Ref();
  return true;
}

void LRUCacheShard2::SetHighPriorityPoolRatio(double high_pri_pool_ratio) {
  DMutexLock l(mutex_);
  high_pri_pool_ratio_ = high_pri_pool_ratio;
  high_pri_pool_capacity_ = capacity_ * high_pri_pool_ratio_;
  MaintainPoolSize();
}

void LRUCacheShard2::SetLowPriorityPoolRatio(double low_pri_pool_ratio) {
  DMutexLock l(mutex_);
  low_pri_pool_ratio_ = low_pri_pool_ratio;
  low_pri_pool_capacity_ = capacity_ * low_pri_pool_ratio_;
  MaintainPoolSize();
}

bool LRUCacheShard2::Release(LRUHandle2* e, bool /*useful*/,
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
        eviction_callback_(e->key(), static_cast<Cache::Handle*>(e),
                           e->HasHit())) {
      // Callback took ownership of obj; just free handle
      free(e);
    } else {
      e->Free(table_.GetAllocator());
    }
  }
  return must_free;
}

LRUHandle2* LRUCacheShard2::CreateHandle(const Slice& key, uint32_t hash,
                                         Cache::ObjectPtr value,
                                         const Cache::CacheItemHelper* helper,
                                         size_t charge) {
  assert(helper);
  // value == nullptr is reserved for indicating failure in SecondaryCache
  assert(!(helper->IsSecondaryCacheCompatible() && value == nullptr));

  // Allocate the memory here outside of the mutex.
  // If the cache is full, we'll have to release it.
  // It shouldn't happen very often though.
  LRUHandle2* e =
      static_cast<LRUHandle2*>(malloc(sizeof(LRUHandle2) - 1 + key.size()));

  e->value = value;
  e->m_flags = 0;
  e->im_flags = 0;
  e->helper = helper;
  e->key_length = key.size();
  e->hash = hash;
  e->refs = 0;
  e->next = e->prev = nullptr;
  memcpy(e->key_data, key.data(), key.size());
  e->CalcTotalCharge(charge, metadata_charge_policy_);

  return e;
}

Status LRUCacheShard2::Insert(const Slice& key, uint32_t hash,
                              Cache::ObjectPtr value,
                              const Cache::CacheItemHelper* helper,
                              size_t charge, LRUHandle2** handle,
                              Cache::Priority priority) {
  LRUHandle2* e = CreateHandle(key, hash, value, helper, charge);
  e->SetPriority(priority);
  e->SetInCache(true);
  return InsertItem(e, handle);
}

LRUHandle2* LRUCacheShard2::CreateStandalone(
    const Slice& key, uint32_t hash, Cache::ObjectPtr value,
    const Cache::CacheItemHelper* helper, size_t charge, bool allow_uncharged) {
  LRUHandle2* e = CreateHandle(key, hash, value, helper, charge);
  e->SetIsStandalone(true);
  e->Ref();

  autovector<LRUHandle2*> last_reference_list;

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

void LRUCacheShard2::Erase(const Slice& key, uint32_t hash) {
  LRUHandle2* e;
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

size_t LRUCacheShard2::GetUsage() const {
  DMutexLock l(mutex_);
  return usage_;
}

size_t LRUCacheShard2::GetPinnedUsage() const {
  DMutexLock l(mutex_);
  assert(usage_ >= lru_usage_);
  return usage_ - lru_usage_;
}

size_t LRUCacheShard2::GetOccupancyCount() const {
  DMutexLock l(mutex_);
  return table_.GetOccupancyCount();
}

size_t LRUCacheShard2::GetTableAddressCount() const {
  DMutexLock l(mutex_);
  return size_t{1} << table_.GetLengthBits();
}

void LRUCacheShard2::AppendPrintableOptions(std::string& str) const {
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

LRUCache2::LRUCache2(const LRUCacheOptions2& opts) : ShardedCache(opts) {
  size_t per_shard = GetPerShardCapacity();
  MemoryAllocator* alloc = memory_allocator();
  InitShards([&](LRUCacheShard2* cs) {
    new (cs)
        LRUCacheShard2(per_shard, opts.strict_capacity_limit,
                       opts.high_pri_pool_ratio, opts.low_pri_pool_ratio,
                       opts.use_adaptive_mutex, opts.metadata_charge_policy,
                       /* max_upper_hash_bits */ 32 - opts.num_shard_bits,
                       alloc, &eviction_callback_);
  });
}

Cache::ObjectPtr LRUCache2::Value(Handle* handle) {
  auto h = static_cast<const LRUHandle2*>(handle);
  return h->value;
}

size_t LRUCache2::GetCharge(Handle* handle) const {
  return static_cast<const LRUHandle2*>(handle)->GetCharge(
      GetShard(0).metadata_charge_policy_);
}

const Cache::CacheItemHelper* LRUCache2::GetCacheItemHelper(
    Handle* handle) const {
  auto h = static_cast<const LRUHandle2*>(handle);
  return h->helper;
}

size_t LRUCache2::TEST_GetLRUSize() {
  return SumOverShards([](LRUCacheShard2& cs) { return cs.TEST_GetLRUSize(); });
}

double LRUCache2::GetHighPriPoolRatio() {
  return GetShard(0).GetHighPriPoolRatio();
}

}  // namespace lru_cache

std::shared_ptr<Cache> LRUCacheOptions2::MakeSharedCache() const {
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
  LRUCacheOptions2 opts = *this;
  if (opts.num_shard_bits < 0) {
    opts.num_shard_bits = GetDefaultCacheShardBits(capacity);
  }
  std::shared_ptr<Cache> cache = std::make_shared<lru_cache::LRUCache2>(opts);
  if (secondary_cache) {
    cache = std::make_shared<CacheWithSecondaryAdapter>(cache, secondary_cache);
  }
  return cache;
}

std::shared_ptr<RowCache> LRUCacheOptions2::MakeSharedRowCache() const {
  if (secondary_cache) {
    // Not allowed for a RowCache
    return nullptr;
  }
  // Works while RowCache is an alias for Cache
  return MakeSharedCache();
}

}  // namespace ROCKSDB_NAMESPACE
