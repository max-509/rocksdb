//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

#include <atomic>
#include <iostream>

#include "db/memtable.h"
#include "inlineskiplist.h"
#include "memory/arena.h"
#include "memtable/skiplist.h"
#include "port/port.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/slice.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/utilities/options_type.h"
#include "util/murmurhash.h"

namespace ROCKSDB_NAMESPACE {
namespace {

class FlinkMemTableRep : public MemTableRep {
 public:
  FlinkMemTableRep(const MemTableRep::KeyComparator& compare,
                   Allocator* allocator, size_t start_keygroup,
                   size_t num_keygroups, size_t keygroup_bytes,
                   int32_t skiplist_height, int32_t skiplist_branching_factor);

  KeyHandle Allocate(const size_t len, char** buf) override;

  void Insert(KeyHandle handle) override;

  bool InsertKey(KeyHandle handle) override;

  void InsertConcurrently(KeyHandle handle) override;

  bool InsertKeyConcurrently(KeyHandle handle) override;

  bool Contains(const char* key) const override;

  size_t ApproximateMemoryUsage() override;

  void Get(const LookupKey& k, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override;

  ~FlinkMemTableRep() override;

  MemTableRep::Iterator* GetIterator(Arena* arena = nullptr) override;
  uint64_t ApproximateNumEntries(const Slice& slice,
                                 const Slice& slice1) override;
  void UniqueRandomSample(const uint64_t num_entries,
                          const uint64_t target_sample_size,
                          std::unordered_set<const char*>* entries) override;

 private:
  friend class DynamicIterator;
  //  using Bucket = SkipList<const char *, const MemTableRep::KeyComparator&>;
  using NonOptimizedSkipList =
      SkipList<const char*, const MemTableRep::KeyComparator&>;
  using Bucket = InlineSkipList<const MemTableRep::KeyComparator&>;

  size_t start_keygroup_;
  size_t num_keygroups_;
  size_t keygroup_bytes_;

  const int32_t skiplist_height_;
  const int32_t skiplist_branching_factor_;
  const uint32_t kScaledInverseBranching_;

  // Maps slices (which are transformed user keys) to buckets of keys sharing
  // the same transform.
  std::atomic<Bucket*>* buckets_;

  const MemTableRep::KeyComparator& compare_;
  // immutable after construction
  Allocator* const allocator_;

  inline Bucket* GetBucket(size_t i) const {
    return buckets_[i].load(std::memory_order_acquire);
  }

  inline size_t GetBucketIdx(const Slice& user_key) const {
    size_t keygroup;
    // TODO: Optimize
    switch (keygroup_bytes_) {
      case 1:
        keygroup = DecodeFixed8<false>(user_key.data());
        break;
      case 2:
        keygroup = DecodeFixed16<false>(user_key.data());
        break;
      default:
        abort();
    }

    size_t bucket_idx = keygroup - start_keygroup_;
    return bucket_idx;
  }

  // Get a bucket from buckets_. If the bucket hasn't been initialized yet,
  // initialize it before returning.
  template <bool UseCAS>
  Bucket* GetInitializedBucket(size_t bucket_idx);

  class Iterator : public MemTableRep::Iterator {
    using SkipListIterator =
        InlineSkipList<const MemTableRep::KeyComparator&>::Iterator;

   public:
    explicit Iterator(FlinkMemTableRep& rep)
        : rep_(rep), bucket_idx_(0), current_iter_(nullptr) {}

    ~Iterator() override = default;

    // Returns true iff the iterator is positioned at a valid node.
    bool Valid() const override {
      return current_iter_.Valid();
    }

    // Returns the key at the current position.
    // REQUIRES: Valid()
    const char* key() const override {
      assert(Valid());
      return current_iter_.key();
    }

    // Advances to the next position.
    // REQUIRES: Valid()
    void Next() override {
      assert(Valid());
      current_iter_.Next();
      if (!current_iter_.Valid()) {
        SeekToFirstForFirstValidBucket();
      }
    }

    // Advances to the previous position.
    // REQUIRES: Valid()
    void Prev() override {
      assert(Valid());
      current_iter_.Prev();
      if (!current_iter_.Valid()) {
        SeekToLastForFirstValidBucket();
      }
    }

    // Advance to the first entry with a key >= target
    void Seek(const Slice& internal_key, const char* memtable_key) override {
      const char* encoded_key = (memtable_key != nullptr)
                                    ? memtable_key
                                    : EncodeKey(&tmp_, internal_key);
      bucket_idx_ = rep_.GetBucketIdx(rep_.UserKey(encoded_key)) + 1;
      auto bucket = rep_.GetBucket(bucket_idx_ - 1);
      if (bucket == nullptr) {
        SeekToFirstForFirstValidBucket();
      } else {
        current_iter_ = SkipListIterator{bucket};
        current_iter_.Seek(encoded_key);
        if (!current_iter_.Valid()) {
          SeekToFirstForFirstValidBucket();
        }
      }
    }

    // Retreat to the last entry with a key <= target
    void SeekForPrev(const Slice& internal_key, const char* memtable_key) override {
      const char* encoded_key = (memtable_key != nullptr)
                                    ? memtable_key
                                    : EncodeKey(&tmp_, internal_key);
      bucket_idx_ = rep_.GetBucketIdx(rep_.UserKey(encoded_key)) + 1;
      auto bucket = rep_.GetBucket(bucket_idx_ - 1);
      if (bucket == nullptr) {
        SeekToLastForFirstValidBucket();
      } else {
        current_iter_ = SkipListIterator{bucket};
        current_iter_.SeekForPrev(encoded_key);
        if (!current_iter_.Valid()) {
          SeekToLastForFirstValidBucket();
        }
      }
    }

    // Position at the first entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToFirst() override {
      bucket_idx_ = 0;
      SeekToFirstForFirstValidBucket();
    }

    // Position at the last entry in collection.
    // Final state of iterator is Valid() iff collection is not empty.
    void SeekToLast() override {
      bucket_idx_ = rep_.num_keygroups_ + 1;
      SeekToLastForFirstValidBucket();
    }

   private:
    FlinkMemTableRep& rep_;
    // Bucket index starts from 1 to num_keygroups inclusive
    size_t bucket_idx_;
    SkipListIterator current_iter_;

    void SeekToLastForFirstValidBucket() {
      assert(bucket_idx_ > 0 && bucket_idx_ <= rep_.num_keygroups_ + 1);
      bucket_idx_--;
      if (bucket_idx_ > 0) {
        auto bucket = FindValidBucketWhileDecrement();
        current_iter_ = SkipListIterator{bucket};
        if (bucket != nullptr) {
          current_iter_.SeekToLast();
        }
      }
    }

    void SeekToFirstForFirstValidBucket() {
      assert(bucket_idx_ <= rep_.num_keygroups_);
      bucket_idx_++;
      if (bucket_idx_ <= rep_.num_keygroups_) {
        auto bucket = FindValidBucketWhileIncrement();
        current_iter_ = SkipListIterator{bucket};
        if (bucket != nullptr) {
          current_iter_.SeekToFirst();
        }
      }
    }

    Bucket *FindValidBucketWhileIncrement() {
      Bucket *bucket = nullptr;
      while (bucket_idx_ <= rep_.num_keygroups_ && (bucket = rep_.GetBucket(bucket_idx_ - 1)) == nullptr) {
        ++bucket_idx_;
      }
      return bucket;
    }

    Bucket *FindValidBucketWhileDecrement() {
      Bucket *bucket = rep_.GetBucket(bucket_idx_ - 1);
      while (bucket_idx_ > 0 && (bucket = rep_.GetBucket(bucket_idx_ - 1)) == nullptr) {
        --bucket_idx_;
      }
      return bucket;
    }

   protected:
    std::string tmp_;  // For passing to EncodeKey
  };
};

FlinkMemTableRep::FlinkMemTableRep(const MemTableRep::KeyComparator& compare,
                                   Allocator* allocator, size_t start_keygroup,
                                   size_t num_keygroups, size_t keygroup_bytes,
                                   int32_t skiplist_height,
                                   int32_t skiplist_branching_factor)
    : MemTableRep(allocator),
      start_keygroup_(start_keygroup),
      num_keygroups_(num_keygroups),
      keygroup_bytes_(keygroup_bytes),
      skiplist_height_(skiplist_height),
      skiplist_branching_factor_(skiplist_branching_factor),
      kScaledInverseBranching_((Random::kMaxNext + 1) /
                               skiplist_branching_factor_),
      compare_(compare),
      allocator_(allocator) {
  auto mem =
      allocator->AllocateAligned(sizeof(std::atomic<void*>) * num_keygroups);
  buckets_ = new (mem) std::atomic<Bucket*>[num_keygroups];

  for (size_t i = 0; i < num_keygroups; ++i) {
    buckets_[i].store(nullptr, std::memory_order_relaxed);
  }
}

FlinkMemTableRep::~FlinkMemTableRep() = default;

template <bool UseCAS>
FlinkMemTableRep::Bucket* FlinkMemTableRep::GetInitializedBucket(
    size_t bucket_idx) {
  if (UseCAS) {
    while (true) {
      auto bucket = buckets_[bucket_idx].load(std::memory_order_acquire);
      if (bucket == nullptr) {
        auto addr = allocator_->AllocateAligned(sizeof(Bucket));
        auto new_bucket =
            new (addr) Bucket(compare_, allocator_, kScaledInverseBranching_,
                              skiplist_height_, skiplist_branching_factor_);
        if (buckets_[bucket_idx].compare_exchange_strong(bucket, new_bucket)) {
          return new_bucket;
        }
      } else {
        return bucket;
      }
    }
  } else {
    auto bucket = buckets_[bucket_idx].load(std::memory_order_acquire);
    if (bucket == nullptr) {
      auto addr = allocator_->AllocateAligned(sizeof(Bucket));
      bucket = new (addr) Bucket(compare_, allocator_, kScaledInverseBranching_,
                                 skiplist_height_, skiplist_branching_factor_);
      buckets_[bucket_idx].store(bucket, std::memory_order_release);
    }
    return bucket;
  }
}

void FlinkMemTableRep::Insert(KeyHandle handle) { InsertKey(handle); }

bool rocksdb::FlinkMemTableRep::InsertKey(KeyHandle handle) {
  const auto* key =
      Bucket::KeyFromAllocatedKey(static_cast<const char*>(handle));
  size_t bucket_idx = GetBucketIdx(UserKey(key));
  auto bucket = GetInitializedBucket<false>(bucket_idx);
  return bucket->Insert(static_cast<const char*>(handle));
}

void rocksdb::FlinkMemTableRep::InsertConcurrently(KeyHandle handle) {
  InsertKeyConcurrently(handle);
}

bool rocksdb::FlinkMemTableRep::InsertKeyConcurrently(KeyHandle handle) {
  const auto* key =
      Bucket::KeyFromAllocatedKey(static_cast<const char*>(handle));
  size_t bucket_idx = GetBucketIdx(UserKey(key));
  auto bucket = GetInitializedBucket<true>(bucket_idx);
  return bucket->InsertConcurrently(static_cast<const char*>(handle));
}

bool FlinkMemTableRep::Contains(const char* key) const {
  size_t bucket_idx = GetBucketIdx(UserKey(key));
  auto bucket = GetBucket(bucket_idx);
  if (bucket == nullptr) {
    return false;
  }
  return bucket->Contains(key);
}

size_t FlinkMemTableRep::ApproximateMemoryUsage() { return 0; }

void FlinkMemTableRep::Get(const LookupKey& k, void* callback_args,
                           bool (*callback_func)(void* arg,
                                                 const char* entry)) {
  auto bucket_idx = GetBucketIdx(k.user_key());
  auto bucket = GetBucket(bucket_idx);
  if (bucket != nullptr) {
    Bucket::Iterator iter(bucket);
    for (iter.Seek(k.memtable_key().data());
         iter.Valid() && callback_func(callback_args, iter.key());
         iter.Next()) {
    }
  }
}

MemTableRep::Iterator* FlinkMemTableRep::GetIterator(Arena* arena) {
  // allocate a new arena of similar size to the one currently in use
  void* mem = arena ? arena->AllocateAligned(sizeof(FlinkMemTableRep::Iterator))
                    :
                    operator new(sizeof(FlinkMemTableRep::Iterator));
  return new (mem) FlinkMemTableRep::Iterator(*this);
}

KeyHandle rocksdb::FlinkMemTableRep::Allocate(const size_t len, char** buf) {
  *buf = Bucket::AllocateKey(len, skiplist_height_, kScaledInverseBranching_,
                             allocator_);
  return static_cast<KeyHandle>(*buf);
  //  return MemTableRep::Allocate(len, buf);
}
uint64_t rocksdb::FlinkMemTableRep::ApproximateNumEntries(const Slice& slice,
                                                          const Slice& slice1) {
  // TODO:
//  std::string tmp;
//  uint64_t start_count =
//      skip_list_.EstimateCount(EncodeKey(&tmp, start_ikey));
//  uint64_t end_count = skip_list_.EstimateCount(EncodeKey(&tmp, end_ikey));
//  return (end_count >= start_count) ? (end_count - start_count) : 0;
  return MemTableRep::ApproximateNumEntries(slice, slice1);
}
void rocksdb::FlinkMemTableRep::UniqueRandomSample(
    const uint64_t num_entries, const uint64_t target_sample_size,
    std::unordered_set<const char*>* entries) {
  // TODO:
  MemTableRep::UniqueRandomSample(num_entries, target_sample_size, entries);
}

struct FlinkMemTableRepOptions {
  static const char* kName() { return "FlinkMemTableRepFactoryOptions"; }
  size_t start_keygroup;
  size_t num_keygroups;
  size_t keygroup_bytes;
  int32_t skiplist_height;
  int32_t skiplist_branching_factor;
};

static std::unordered_map<std::string, OptionTypeInfo> hash_skiplist_info = {
    {"start_keygroup",
     {offsetof(struct FlinkMemTableRepOptions, start_keygroup),
      OptionType::kSizeT, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
    {"num_keygroups",
     {offsetof(struct FlinkMemTableRepOptions, num_keygroups),
      OptionType::kSizeT, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
    {"keygroup_bytes",
     {offsetof(struct FlinkMemTableRepOptions, keygroup_bytes),
      OptionType::kSizeT, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
    {"skiplist_height",
     {offsetof(struct FlinkMemTableRepOptions, skiplist_height),
      OptionType::kInt32T, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
    {"branching_factor",
     {offsetof(struct FlinkMemTableRepOptions, skiplist_branching_factor),
      OptionType::kInt32T, OptionVerificationType::kNormal,
      OptionTypeFlags::kNone}},
};

class FlinkMemTableRepFactory : public MemTableRepFactory {
 public:
  explicit FlinkMemTableRepFactory(size_t start_keygroup, size_t num_keygroups,
                                   size_t keygroup_bytes,
                                   int32_t skiplist_height,
                                   int32_t skiplist_branching_factor) {
    options_.start_keygroup = start_keygroup;
    options_.num_keygroups = num_keygroups;
    options_.keygroup_bytes = keygroup_bytes;
    options_.skiplist_height = skiplist_height;
    options_.skiplist_branching_factor = skiplist_branching_factor;
    RegisterOptions(&options_, &hash_skiplist_info);
  }

  using MemTableRepFactory::CreateMemTableRep;
  MemTableRep* CreateMemTableRep(const MemTableRep::KeyComparator& compare,
                                 Allocator* allocator,
                                 const SliceTransform* transform,
                                 Logger* logger) override;

  static const char* kClassName() { return "FlinkMemTableRepFactory"; }
  static const char* kNickName() { return "flink_memtable"; }

  const char* Name() const override { return kClassName(); }
  const char* NickName() const override { return kNickName(); }

  bool IsInsertConcurrentlySupported() const override;

 private:
  FlinkMemTableRepOptions options_;
};

}  // namespace

MemTableRep* FlinkMemTableRepFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& compare, Allocator* allocator,
    const SliceTransform* /*transform*/, Logger* /*logger*/) {
  return new FlinkMemTableRep(compare, allocator,
                              options_.start_keygroup, options_.num_keygroups,
                              options_.keygroup_bytes, options_.skiplist_height,
                              options_.skiplist_branching_factor);
}
bool FlinkMemTableRepFactory::IsInsertConcurrentlySupported() const {
  return true;
}

MemTableRepFactory* NewFlinkMemTableRepFactory(
    size_t start_keygroup, size_t num_keygroups, size_t keygroup_bytes,
    int32_t skiplist_height, int32_t skiplist_branching_factor) {
  return new FlinkMemTableRepFactory(start_keygroup, num_keygroups,
                                     keygroup_bytes, skiplist_height,
                                     skiplist_branching_factor);
}

}  // namespace ROCKSDB_NAMESPACE
