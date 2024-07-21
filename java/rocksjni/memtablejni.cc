// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// This file implements the "bridge" between Java and C++ for MemTables.

#include "include/org_rocksdb_HashLinkedListMemTableConfig.h"
#include "include/org_rocksdb_HashSkipListMemTableConfig.h"
#include "include/org_rocksdb_SkipListMemTableConfig.h"
#include "include/org_rocksdb_VectorMemTableConfig.h"
#include "include/org_rocksdb_FlinkMemTableConfig.h"
#include "rocksdb/memtablerep.h"
#include "rocksjni/portal.h"

/*
 * Class:     org_rocksdb_HashSkipListMemTableConfig
 * Method:    newMemTableFactoryHandle
 * Signature: (JII)J
 */
jlong Java_org_rocksdb_HashSkipListMemTableConfig_newMemTableFactoryHandle(
    JNIEnv* env, jobject /*jobj*/, jlong jbucket_count, jint jheight,
    jint jbranching_factor) {
  ROCKSDB_NAMESPACE::Status s =
      ROCKSDB_NAMESPACE::JniUtil::check_if_jlong_fits_size_t(jbucket_count);
  if (s.ok()) {
    return reinterpret_cast<jlong>(ROCKSDB_NAMESPACE::NewHashSkipListRepFactory(
        static_cast<size_t>(jbucket_count), static_cast<int32_t>(jheight),
        static_cast<int32_t>(jbranching_factor)));
  }
  ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni::ThrowNew(env, s);
  return 0;
}

/*
 * Class:     org_rocksdb_HashLinkedListMemTableConfig
 * Method:    newMemTableFactoryHandle
 * Signature: (JJIZI)J
 */
jlong Java_org_rocksdb_HashLinkedListMemTableConfig_newMemTableFactoryHandle(
    JNIEnv* env, jobject /*jobj*/, jlong jbucket_count,
    jlong jhuge_page_tlb_size, jint jbucket_entries_logging_threshold,
    jboolean jif_log_bucket_dist_when_flash, jint jthreshold_use_skiplist) {
  ROCKSDB_NAMESPACE::Status statusBucketCount =
      ROCKSDB_NAMESPACE::JniUtil::check_if_jlong_fits_size_t(jbucket_count);
  ROCKSDB_NAMESPACE::Status statusHugePageTlb =
      ROCKSDB_NAMESPACE::JniUtil::check_if_jlong_fits_size_t(
          jhuge_page_tlb_size);
  if (statusBucketCount.ok() && statusHugePageTlb.ok()) {
    return reinterpret_cast<jlong>(ROCKSDB_NAMESPACE::NewHashLinkListRepFactory(
        static_cast<size_t>(jbucket_count),
        static_cast<size_t>(jhuge_page_tlb_size),
        static_cast<int32_t>(jbucket_entries_logging_threshold),
        static_cast<bool>(jif_log_bucket_dist_when_flash),
        static_cast<int32_t>(jthreshold_use_skiplist)));
  }
  ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni::ThrowNew(
      env, !statusBucketCount.ok() ? statusBucketCount : statusHugePageTlb);
  return 0;
}

/*
 * Class:     org_rocksdb_VectorMemTableConfig
 * Method:    newMemTableFactoryHandle
 * Signature: (J)J
 */
jlong Java_org_rocksdb_VectorMemTableConfig_newMemTableFactoryHandle(
    JNIEnv* env, jobject /*jobj*/, jlong jreserved_size) {
  ROCKSDB_NAMESPACE::Status s =
      ROCKSDB_NAMESPACE::JniUtil::check_if_jlong_fits_size_t(jreserved_size);
  if (s.ok()) {
    return reinterpret_cast<jlong>(new ROCKSDB_NAMESPACE::VectorRepFactory(
        static_cast<size_t>(jreserved_size)));
  }
  ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni::ThrowNew(env, s);
  return 0;
}

/*
 * Class:     org_rocksdb_FlinkMemTableConfig
 * Method:    newMemTableFactoryHandle
 * Signature: (JJJII)J
 */
jlong Java_org_rocksdb_FlinkMemTableConfig_newMemTableFactoryHandle(
    JNIEnv* env, jobject /*jobj*/, jlong jstart_keygroup, jlong jnum_keygroups,
    jlong jkeygroup_bytes, jint jheight, jint jbranching_factor) {
  if (jstart_keygroup < 0) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "Start keygroup must be greater than 0");
    return 0;
  }
  if (jnum_keygroups <= 0) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "Num keygroups greater must be than 0");
    return 0;
  }
  if (jkeygroup_bytes != 1 && jstart_keygroup != 2) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "Num bytes for keygroup prefix must be 1 or 2");
    return 0;
  }
  if (jkeygroup_bytes == 1 && jstart_keygroup >= 256) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "Start keygroup for 1 byte prefix must be less than 256");
    return 0;
  }
  if (jkeygroup_bytes == 1 && jstart_keygroup + jnum_keygroups >= 256) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "End keygroup for 1 byte prefix must be less than 256");
    return 0;
  }
  if (jkeygroup_bytes == 2 &&
      jstart_keygroup > std::numeric_limits<int16_t>::max()) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "Start keygroup for 2 byte prefix must be less than 32768");
    return 0;
  }
  if (jkeygroup_bytes == 2 &&
      jstart_keygroup + jnum_keygroups > std::numeric_limits<int16_t>::max()) {
    ROCKSDB_NAMESPACE::
        JavaException<ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni>::ThrowNew(
            env, "End keygroup for 2 byte prefix must be less than 32768");
    return 0;
  }
  return reinterpret_cast<jlong>(ROCKSDB_NAMESPACE::NewFlinkMemTableRepFactory(
      static_cast<size_t>(jstart_keygroup), static_cast<size_t>(jnum_keygroups),
      static_cast<size_t>(jkeygroup_bytes), static_cast<int32_t>(jheight),
      static_cast<int32_t>(jbranching_factor)));
}

/*
 * Class:     org_rocksdb_SkipListMemTableConfig
 * Method:    newMemTableFactoryHandle0
 * Signature: (J)J
 */
jlong Java_org_rocksdb_SkipListMemTableConfig_newMemTableFactoryHandle0(
    JNIEnv* env, jobject /*jobj*/, jlong jlookahead) {
  ROCKSDB_NAMESPACE::Status s =
      ROCKSDB_NAMESPACE::JniUtil::check_if_jlong_fits_size_t(jlookahead);
  if (s.ok()) {
    return reinterpret_cast<jlong>(new ROCKSDB_NAMESPACE::SkipListFactory(
        static_cast<size_t>(jlookahead)));
  }
  ROCKSDB_NAMESPACE::IllegalArgumentExceptionJni::ThrowNew(env, s);
  return 0;
}
