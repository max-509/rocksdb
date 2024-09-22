#pragma once

#include <cstring>

#ifdef NDEBUG
    #define ALLOCATOR_ASLR 0
#else
    #define ALLOCATOR_ASLR 1
#endif

#if !defined(OS_DARWIN) && !defined(OS_FREEBSD)
#include <malloc.h>
#endif

#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <sys/mman.h> /// MADV_POPULATE_WRITE


namespace rocksdb {


/// Constant is chosen almost arbitrarily, what I observed is 128KB is too small, 1MB is almost indistinguishable from 64MB and 1GB is too large.
static constexpr size_t POPULATE_THRESHOLD = 16 * 1024 * 1024;
static constexpr size_t MALLOC_MIN_ALIGNMENT = 8;

/** Previously there was a code which tried to use manual mmap and mremap (clickhouse_mremap.h) for large allocations/reallocations (64MB+).
  * Most modern allocators (including jemalloc) don't use mremap, so the idea was to take advantage from mremap system call for large reallocs.
  * Actually jemalloc had support for mremap, but it was intentionally removed from codebase https://github.com/jemalloc/jemalloc/commit/e2deab7a751c8080c2b2cdcfd7b11887332be1bb.
  * Our performance tests also shows that without manual mmap/mremap/munmap clickhouse is overall faster for about 1-2% and up to 5-7x for some types of queries.
  * That is why we don't do manual mmap/mremap/munmap here and completely rely on jemalloc for allocations of any size.
  */

/** Responsible for allocating / freeing memory. Used, for example, in PODArray, Arena.
  * Also used in hash tables.
  * The interface is different from std::allocator
  * - the presence of the method realloc, which for large chunks of memory uses mremap;
  * - passing the size into the `free` method;
  * - by the presence of the `alignment` argument;
  * - the possibility of zeroing memory (used in hash tables);
  */

class Allocator
{
public:
    /// Allocate memory range.
    void *alloc(size_t size, size_t alignment = 0);

    /// Free memory range.
    void free(void * buf, size_t size);

    /** Enlarge memory range.
      * Data from old range is moved to the beginning of new range.
      * Address of memory range could change.
      */
    void * realloc(void * buf, size_t old_size, size_t new_size, size_t alignment = 0);
private:
};

int64_t getPageSize() {
    int64_t page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) {
        abort();
    }
    return page_size;
}

#if defined(MADV_POPULATE_WRITE)
/// Address passed to madvise is required to be aligned to the page boundary.
auto adjustToPageSize(void * buf, size_t len, size_t page_size)
{
    const uintptr_t address_numeric = reinterpret_cast<uintptr_t>(buf);
    const size_t next_page_start = ((address_numeric + page_size - 1) / page_size) * page_size;
    return std::make_pair(reinterpret_cast<void *>(next_page_start), len - (next_page_start - address_numeric));
}
#endif

void prefaultPages([[maybe_unused]] void * buf_, [[maybe_unused]] size_t len_)
{
#if defined(MADV_POPULATE_WRITE)
    if (len_ < POPULATE_THRESHOLD)
        return;

    static const size_t page_size = getPageSize();
    if (len_ < page_size) /// Rounded address should be still within [buf, buf + len).
        return;

    auto [buf, len] = adjustToPageSize(buf_, len_, page_size);
    ::madvise(buf, len, MADV_POPULATE_WRITE);
#endif
}


template<bool clear_memory, bool populate>
void * allocNoTrack(size_t size, size_t alignment)
{
    void * buf;
    if (alignment <= MALLOC_MIN_ALIGNMENT)
    {
        if (clear_memory) {
            buf = ::calloc(size, 1);
        }
        else {
            buf = ::malloc(size);
        }
    }
    else
    {
        buf = nullptr;
        int res = posix_memalign(&buf, alignment, size);

        if (clear_memory) {
            memset(buf, 0, size);
        }
    }

    if constexpr (populate)
        prefaultPages(buf, size);

    return buf;
}

void freeNoTrack(void * buf)
{
#if USE_GWP_ASAN
    if (unlikely(GWPAsan::GuardedAlloc.pointerIsMine(buf)))
    {
        ProfileEvents::increment(ProfileEvents::GWPAsanFree);
        GWPAsan::GuardedAlloc.deallocate(buf);
        return;
    }
#endif

    ::free(buf);
}

void * Allocator::alloc(size_t size, size_t alignment)
{
    void * ptr = allocNoTrack<true, true>(size, alignment);
    return ptr;
}


void Allocator::free(void * buf, size_t size) {
    freeNoTrack(buf);
}

void * Allocator::realloc(void * buf, size_t old_size, size_t new_size, size_t alignment)
{
    if (old_size == new_size)
    {
        /// nothing to do.
        /// BTW, it's not possible to change alignment while doing realloc.
        return buf;
    }

    if (alignment <= MALLOC_MIN_ALIGNMENT)
    {
        void * new_buf = ::realloc(buf, new_size);
        buf = new_buf;

        if (new_size > old_size) {
            memset(reinterpret_cast<char *>(buf) + old_size, 0, new_size - old_size);
        }
    }
    else
    {

        void * new_buf = alloc(new_size, alignment);
        memcpy(new_buf, buf, std::min(old_size, new_size));
        free(buf, old_size);
        buf = new_buf;
    }

    prefaultPages(buf, new_size);

    return buf;
}
}

