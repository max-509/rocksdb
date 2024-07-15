// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
package org.rocksdb;

/**
 * The config for hash skip-list mem-table representation.
 * Such mem-table representation contains a fix-sized array of
 * buckets, where each bucket points to a skiplist (or null if the
 * bucket is empty).
 * <p>
 * Note that since this mem-table representation relies on the
 * key prefix, it is required to invoke one of the usePrefixExtractor
 * functions to specify how to extract key prefix given a key.
 * If proper prefix-extractor is not set, then RocksDB will
 * use the default memtable representation (SkipList) instead
 * and post a warning in the LOG.
 */
public class FlinkMemTableConifg extends MemTableConfig {
  public static final int DEFAULT_START_KEYGROUP = -1;
  public static final int DEFAULT_NUM_KEYGROUPS = -1;
  public static final int DEFAULT_KEYGROUP_BYTES = -1;
  public static final int DEFAULT_BRANCHING_FACTOR = 4;
  public static final int DEFAULT_HEIGHT = 12;

  /**
   * FlinkMemTableConifg constructor
   */
  public FlinkMemTableConifg() {
    branchingFactor_ = DEFAULT_BRANCHING_FACTOR;
    height_ = DEFAULT_HEIGHT;
    startKeyGroup_ = DEFAULT_START_KEYGROUP;
    numKeyGroups_ = DEFAULT_NUM_KEYGROUPS;
    keyGroupBytes_ = DEFAULT_KEYGROUP_BYTES;
  }

  /**
   * Set the start keygroup for Flink memtable
   *
   * @param startKeyGroup start keygroup.
   * @return the reference to the current FlinkMemTableConifg.
   */
  public FlinkMemTableConifg setStartKeyGroup(
      final long startKeyGroup) {
    startKeyGroup_ = startKeyGroup;
    return this;
  }

  /**
   * @return the start keygroup
   */
  public long startKeyGroup() {
    return startKeyGroup_;
  }

  /**
   * Set the num keygroups for Flink memtable
   *
   * @param numKeyGroups num keygroups.
   * @return the reference to the current FlinkMemTableConifg.
   */
  public FlinkMemTableConifg setNumKeyGroups(
      final long numKeyGroups) {
    numKeyGroups_ = numKeyGroups;
    return this;
  }

  /**
   * @return the num keygroups
   */
  public long numKeyGroups() {
    return numKeyGroups_;
  }

  /**
   * Set the keygroup bytes for Flink memtable
   *
   * @param keyGroupBytes keygroup bytes.
   * @return the reference to the current FlinkMemTableConifg.
   */
  public FlinkMemTableConifg setKeyGroupBytes(
      final long keyGroupBytes) {
    keyGroupBytes_ = keyGroupBytes;
    return this;
  }

  /**
   * @return the keygroup bytes
   */
  public long keyGroupBytes() {
    return keyGroupBytes_;
  }

  /**
   * Set the height of the skip list.  Default = 4.
   *
   * @param height height to set.
   *
   * @return the reference to the current FlinkMemTableConifg.
   */
  public FlinkMemTableConifg setHeight(final int height) {
    height_ = height;
    return this;
  }

  /**
   * @return the height of the skip list.
   */
  public int height() {
    return height_;
  }

  /**
   * Set the branching factor used in the Flink memtable.
   * This factor controls the probabilistic size ratio between adjacent
   * links in the skip list.
   *
   * @param bf the probabilistic size ratio between adjacent link
   *     lists in the skip list.
   * @return the reference to the current FlinkMemTableConifg.
   */
  public FlinkMemTableConifg setBranchingFactor(
      final int bf) {
    branchingFactor_ = bf;
    return this;
  }

  /**
   * @return branching factor, the probabilistic size ratio between
   *     adjacent links in the skip list.
   */
  public int branchingFactor() {
    return branchingFactor_;
  }

  @Override protected long newMemTableFactoryHandle() {
    return newMemTableFactoryHandle(
        startKeyGroup_, numKeyGroups_, keyGroupBytes_, height_, branchingFactor_);
  }

  private static native long newMemTableFactoryHandle(
      long startKeyGroup, long numKeyGroups, long keyGroupBytes, int height, int branchingFactor) throws IllegalArgumentException;

  private long startKeyGroup_;
  private long numKeyGroups_;
  private long keyGroupBytes_;
  private int branchingFactor_;
  private int height_;
}
