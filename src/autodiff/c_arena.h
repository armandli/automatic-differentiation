#ifndef C_ARENA
#define C_ARENA

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autodiff {

namespace s = std;

// ---------------------------------------------------------------------------
//  NodeKind / NodeBase: the non-template identity+lifetime base for every
//  graph node (Node<T> and its subclasses in c_graph.h). Nodes are heap
//  objects owned by a CArena (adopt()), never copied, and destroyed
//  polymorphically through this base when the arena dies. NodeKind lets a
//  later reverse pass discriminate leaves from operations without an RTTI
//  round-trip.
// ---------------------------------------------------------------------------
enum class NodeKind { Variable, Constant, Operation };

struct NodeBase {
  NodeKind mKind;

  explicit NodeBase(NodeKind k) noexcept : mKind(k) {}
  NodeBase(const NodeBase&) = delete;
  NodeBase& operator=(const NodeBase&) = delete;
  virtual ~NodeBase() = default;

  virtual void zero_grad() {}   // Node<T> overrides
};

// ---------------------------------------------------------------------------
//  CArena: the sole owner of every numeric buffer used by CArray<T> and of
//  every graph node built on it. It is NOT templated — one arena can own
//  buffers of many element types at once (float and bool and int …), so a
//  CArray<float> and a CArray<bool> can coexist and feed the same operator.
//  allocate<T>() carves a fresh T-block, type-erases the owning pointer, and
//  returns a non-owning T* valid until ~CArena, which frees every block at
//  once. adopt<N>() constructs a node on the heap, takes ownership, and returns
//  a reference stable for the arena's life. CArray<T> is a stack value handle
//  into these blocks; the arena, declared first in a scope, must outlive
//  everything built on it.
//
//  The arena is pinned: CArray<T> keeps a CArena* back-pointer, so a move of
//  the arena would dangle every array — copy and move are deleted.
//
//  Counting lives here too: one note_* call per buffer-bearing object built on
//  the arena, plus carray_count() for every allocate(). Single-threaded (one
//  arena per computation), so the counters are plain int64_t — the old
//  graph_stats counters were std::atomic only because they were a single
//  global shared across threads.
//
//  The node list (mNodes) grows for the arena's whole life — it is a tape, not
//  a scratchpad. Reusing one arena across many forward passes (a training
//  loop) grows it without bound; use a fresh arena per pass for now.
// ---------------------------------------------------------------------------
struct CArena {
  CArena() = default;
  ~CArena() = default;

  CArena(const CArena&) = delete;
  CArena& operator=(const CArena&) = delete;
  CArena(CArena&&) = delete;
  CArena& operator=(CArena&&) = delete;

  // Carve n elements of any fundamental arithmetic type, each set to init.
  // Returns a non-owning T* valid until ~CArena. The owning pointer is stored
  // type-erased (a per-T deleter runs the right delete[]). Counts as one CArray
  // allocation.
  template <typename T>
  T* allocate(s::size_t n, T init = T{}) {
    static_assert(s::is_arithmetic_v<T>,
                  "CArena buffers hold fundamental arithmetic types only");
    auto block = s::make_unique_for_overwrite<T[]>(n);
    T* raw = block.get();
    s::fill_n(raw, n, init);
    mBuffers.push_back(s::unique_ptr<void, void(*)(void*)>(
      block.release(), [](void* p) { delete[] static_cast<T*>(p); }));
    ++mCarrayCount;
    return raw;
  }

  // Construct a node of type N in place, own it, and return it. Exception
  // safe: the owning handle exists before the vector push, so a failed
  // reallocation frees the node instead of leaking it.
  template <typename N, typename... Args>
  N& adopt(Args&&... args) {
    auto p = s::make_unique<N>(s::forward<Args>(args)...);
    N& ref = *p;
    mNodes.push_back(s::move(p));
    return ref;
  }

  // Canonical leaf node a CArray buffer promoted to, or nullptr. Keyed by the
  // buffer pointer (type-erased), which CArena never frees mid-life, so the key
  // stays valid. Callers pass a `const T*` that decays to `const void*`.
  NodeBase* promoted_leaf(const void* key) const {
    auto it = mPromoted.find(key);
    return it == mPromoted.end() ? nullptr : it->second;
  }
  void register_leaf(const void* key, NodeBase* node) { mPromoted.emplace(key, node); }

  // Default requires_grad for CArray<T>s allocated from this arena; each picks it
  // up at construction unless it passes an explicit flag. Off by default.
  void set_auto_requires_grad(bool b) noexcept { mAutoRequiresGrad = b; }
  bool auto_requires_grad()     const noexcept { return mAutoRequiresGrad; }

  // The tape, in creation order (a valid topological order for a reverse pass).
  int64_t   node_count()       const noexcept { return static_cast<int64_t>(mNodes.size()); }
  NodeBase& node_at(int64_t i)  const { return *mNodes[static_cast<s::size_t>(i)]; }

  // Creation counters — one call per buffer-bearing object built on this arena.
  int64_t note_vnode()   noexcept { return mVnodeCount++; }
  int64_t note_cnode()   noexcept { return mCnodeCount++; }
  void note_onode_add()      noexcept { ++mOnodeAddCount; }
  void note_onode_sub()      noexcept { ++mOnodeSubCount; }
  void note_onode_neg()      noexcept { ++mOnodeNegCount; }
  void note_onode_hadamard() noexcept { ++mOnodeHadamardCount; }
  void note_onode_dot()      noexcept { ++mOnodeDotCount; }
  void note_onode_div()      noexcept { ++mOnodeDivCount; }
  void note_onode_sum()      noexcept { ++mOnodeSumCount; }
  void note_onode_max()      noexcept { ++mOnodeMaxCount; }
  void note_onode_min()      noexcept { ++mOnodeMinCount; }
  void note_onode_mean()     noexcept { ++mOnodeMeanCount; }
  void note_onode_softmax()               noexcept { ++mOnodeSoftmaxCount; }
  void note_onode_cross_entropy()         noexcept { ++mOnodeCrossEntropyCount; }
  void note_onode_softmax_cross_entropy() noexcept { ++mOnodeSoftmaxCrossEntropyCount; }
  void note_onode_where()    noexcept { ++mOnodeWhereCount; }
  void note_onode_exp()      noexcept { ++mOnodeExpCount; }
  void note_onode_log()      noexcept { ++mOnodeLogCount; }
  void note_onode_sin()      noexcept { ++mOnodeSinCount; }
  void note_onode_cos()      noexcept { ++mOnodeCosCount; }
  void note_onode_tan()      noexcept { ++mOnodeTanCount; }
  void note_onode_sqrt()     noexcept { ++mOnodeSqrtCount; }
  void note_onode_abs()      noexcept { ++mOnodeAbsCount; }
  void note_onode_pow()      noexcept { ++mOnodePowCount; }

  int64_t carray_count()         const noexcept { return mCarrayCount; }
  int64_t vnode_count()          const noexcept { return mVnodeCount; }
  int64_t cnode_count()          const noexcept { return mCnodeCount; }
  int64_t onode_add_count()      const noexcept { return mOnodeAddCount; }
  int64_t onode_sub_count()      const noexcept { return mOnodeSubCount; }
  int64_t onode_neg_count()      const noexcept { return mOnodeNegCount; }
  int64_t onode_hadamard_count() const noexcept { return mOnodeHadamardCount; }
  int64_t onode_dot_count()      const noexcept { return mOnodeDotCount; }
  int64_t onode_div_count()      const noexcept { return mOnodeDivCount; }
  int64_t onode_sum_count()      const noexcept { return mOnodeSumCount; }
  int64_t onode_max_count()      const noexcept { return mOnodeMaxCount; }
  int64_t onode_min_count()      const noexcept { return mOnodeMinCount; }
  int64_t onode_mean_count()     const noexcept { return mOnodeMeanCount; }
  int64_t onode_softmax_count()               const noexcept { return mOnodeSoftmaxCount; }
  int64_t onode_cross_entropy_count()         const noexcept { return mOnodeCrossEntropyCount; }
  int64_t onode_softmax_cross_entropy_count() const noexcept { return mOnodeSoftmaxCrossEntropyCount; }
  int64_t onode_where_count()    const noexcept { return mOnodeWhereCount; }
  int64_t onode_exp_count()      const noexcept { return mOnodeExpCount; }
  int64_t onode_log_count()      const noexcept { return mOnodeLogCount; }
  int64_t onode_sin_count()      const noexcept { return mOnodeSinCount; }
  int64_t onode_cos_count()      const noexcept { return mOnodeCosCount; }
  int64_t onode_tan_count()      const noexcept { return mOnodeTanCount; }
  int64_t onode_sqrt_count()     const noexcept { return mOnodeSqrtCount; }
  int64_t onode_abs_count()      const noexcept { return mOnodeAbsCount; }
  int64_t onode_pow_count()      const noexcept { return mOnodePowCount; }

private:
  s::vector<s::unique_ptr<void, void(*)(void*)>> mBuffers;   // type-erased owners
  s::vector<s::unique_ptr<NodeBase>>             mNodes;     // destroyed before mBuffers
  s::unordered_map<const void*, NodeBase*>       mPromoted;  // CArray buffer -> canonical leaf
  bool    mAutoRequiresGrad   = false;
  int64_t mCarrayCount        = 0;
  int64_t mVnodeCount         = 0;
  int64_t mCnodeCount         = 0;
  int64_t mOnodeAddCount      = 0;
  int64_t mOnodeSubCount      = 0;
  int64_t mOnodeNegCount      = 0;
  int64_t mOnodeHadamardCount = 0;
  int64_t mOnodeDotCount      = 0;
  int64_t mOnodeDivCount      = 0;
  int64_t mOnodeSumCount      = 0;
  int64_t mOnodeMaxCount      = 0;
  int64_t mOnodeMinCount      = 0;
  int64_t mOnodeMeanCount     = 0;
  int64_t mOnodeSoftmaxCount             = 0;
  int64_t mOnodeCrossEntropyCount        = 0;
  int64_t mOnodeSoftmaxCrossEntropyCount = 0;
  int64_t mOnodeWhereCount    = 0;
  int64_t mOnodeExpCount      = 0;
  int64_t mOnodeLogCount      = 0;
  int64_t mOnodeSinCount      = 0;
  int64_t mOnodeCosCount      = 0;
  int64_t mOnodeTanCount      = 0;
  int64_t mOnodeSqrtCount     = 0;
  int64_t mOnodeAbsCount      = 0;
  int64_t mOnodePowCount      = 0;
};

} // autodiff

#endif//C_ARENA
