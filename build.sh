#!/usr/bin/env bash

cmake .. -DCMAKE_TOOLCHAIN_FILE=/mnt/disk1/linux/vcpkg/scripts/buildsystems/vcpkg.cmake -DWITH_JNI=true -DJAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64 -DWITH_GFLAGS=1 -DWITH_SNAPPY=1 -DWITH_LZ4=1 -DWITH_ZLIB=1 -DWITH_ZSTD=1 -DWITH_ASAN=1 -DWITH_UBSAN=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo

