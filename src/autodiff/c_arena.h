#ifndef C_ARENA
#define C_ARENA

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace autodiff {

namespace s = std;

// ---------------------------------------------------------------------------
//  CArena<T>: the sole owner of every numeric T-buffer used by CArray<T> and
//  the graph nodes built on it. allocate() carves a fresh block, records it,
//  and returns a non-owning pointer valid until ~CArena, which frees every
//  block at once. CArray<T> and every node are stack value objects holding
//  non-owning pointers into these blocks; the arena, declared first in a
//  scope, must outlive them.
//
//  The arena is pinned: CArray<T> keeps a CArena<T>* back-pointer, so a move
//  of the arena would dangle every array — copy and move are deleted.
//
//  Counting lives here too: one note_* call per buffer-bearing object built on
//  the arena, plus carray_count() for every allocate(). Single-threaded (one
//  arena per computation), so the counters are plain int64_t — the old
//  graph_stats counters were std::atomic only because they were a single
//  global shared across threads.
// ---------------------------------------------------------------------------
template <typename T>
struct CArena {
  CArena() = default;
  ~CArena() = default;

  CArena(const CArena&) = delete;
  CArena& operator=(const CArena&) = delete;
  CArena(CArena&&) = delete;
  CArena& operator=(CArena&&) = delete;

  // Carve n elements, each set to init. Returns a non-owning pointer valid
  // until ~CArena. Counts as one CArray allocation.
  T* allocate(s::size_t n, T init = T{}) {
    auto block = s::make_unique_for_overwrite<T[]>(n);
    T* raw = block.get();
    s::fill_n(raw, n, init);
    mBuffers.push_back(s::move(block));
    ++mCarrayCount;
    return raw;
  }

  // Creation counters — one call per buffer-bearing object built on this arena.
  void note_vnode()      noexcept { ++mVnodeCount; }
  void note_cnode()      noexcept { ++mCnodeCount; }
  void note_onode_add()  noexcept { ++mOnodeAddCount; }
  void note_onode_sub()  noexcept { ++mOnodeSubCount; }
  void note_onode_mul()  noexcept { ++mOnodeMulCount; }
  void note_onode_div()  noexcept { ++mOnodeDivCount; }
  void note_onode_exp()  noexcept { ++mOnodeExpCount; }
  void note_onode_log()  noexcept { ++mOnodeLogCount; }
  void note_onode_sin()  noexcept { ++mOnodeSinCount; }
  void note_onode_cos()  noexcept { ++mOnodeCosCount; }
  void note_onode_tan()  noexcept { ++mOnodeTanCount; }
  void note_onode_sqrt() noexcept { ++mOnodeSqrtCount; }
  void note_onode_abs()  noexcept { ++mOnodeAbsCount; }
  void note_onode_pow()  noexcept { ++mOnodePowCount; }

  int64_t carray_count()     const noexcept { return mCarrayCount; }
  int64_t vnode_count()      const noexcept { return mVnodeCount; }
  int64_t cnode_count()      const noexcept { return mCnodeCount; }
  int64_t onode_add_count()  const noexcept { return mOnodeAddCount; }
  int64_t onode_sub_count()  const noexcept { return mOnodeSubCount; }
  int64_t onode_mul_count()  const noexcept { return mOnodeMulCount; }
  int64_t onode_div_count()  const noexcept { return mOnodeDivCount; }
  int64_t onode_exp_count()  const noexcept { return mOnodeExpCount; }
  int64_t onode_log_count()  const noexcept { return mOnodeLogCount; }
  int64_t onode_sin_count()  const noexcept { return mOnodeSinCount; }
  int64_t onode_cos_count()  const noexcept { return mOnodeCosCount; }
  int64_t onode_tan_count()  const noexcept { return mOnodeTanCount; }
  int64_t onode_sqrt_count() const noexcept { return mOnodeSqrtCount; }
  int64_t onode_abs_count()  const noexcept { return mOnodeAbsCount; }
  int64_t onode_pow_count()  const noexcept { return mOnodePowCount; }

private:
  s::vector<s::unique_ptr<T[]>> mBuffers;
  int64_t mCarrayCount    = 0;
  int64_t mVnodeCount     = 0;
  int64_t mCnodeCount     = 0;
  int64_t mOnodeAddCount  = 0;
  int64_t mOnodeSubCount  = 0;
  int64_t mOnodeMulCount  = 0;
  int64_t mOnodeDivCount  = 0;
  int64_t mOnodeExpCount  = 0;
  int64_t mOnodeLogCount  = 0;
  int64_t mOnodeSinCount  = 0;
  int64_t mOnodeCosCount  = 0;
  int64_t mOnodeTanCount  = 0;
  int64_t mOnodeSqrtCount = 0;
  int64_t mOnodeAbsCount  = 0;
  int64_t mOnodePowCount  = 0;
};

} // autodiff

#endif//C_ARENA
