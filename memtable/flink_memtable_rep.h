// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once
#ifndef ROCKSDB_LITE
#include "rocksdb/slice_transform.h"
#include "rocksdb/memtablerep.h"

namespace ROCKSDB_NAMESPACE {

class FlinkMemTableRepFactory : public MemTableRepFactory {
 public:
  explicit FlinkMemTableRepFactory(size_t start_keygroup, size_t num_keygroups,
                                   size_t keygroup_bytes,
                                   int32_t skiplist_height,
                                   int32_t skiplist_branching_factor) {
    start_keygroup_ = start_keygroup;
    num_keygroups_ = num_keygroups;
    keygroup_bytes_ = keygroup_bytes;
    skiplist_height_ = skiplist_height;
    skiplist_branching_factor_ = skiplist_branching_factor;
  }

  using MemTableRepFactory::CreateMemTableRep;
  MemTableRep* CreateMemTableRep(const MemTableRep::KeyComparator& compare,
                                 Allocator* allocator,
                                 const SliceTransform* transform,
                                 Logger* logger) override;

  const char* Name() const override { return "FlinkMemTableRepFactory"; }

  bool IsInsertConcurrentlySupported() const override;

 private:
  size_t start_keygroup_;
  size_t num_keygroups_;
  size_t keygroup_bytes_;
  int32_t skiplist_height_;
  int32_t skiplist_branching_factor_;
};

}  // namespace ROCKSDB_NAMESPACE
#endif  // ROCKSDB_LITE