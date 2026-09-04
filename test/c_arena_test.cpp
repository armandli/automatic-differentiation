#include <c_arena.h>

#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

using autodiff::CArena;
using autodiff::NodeBase;
using autodiff::NodeKind;

// A minimal concrete NodeBase — this translation unit has no Node<T> types.
struct Dummy : NodeBase {
  int v;
  explicit Dummy(int v) : NodeBase(NodeKind::Operation), v(v) {}
};

TEST(CArena, AllocateReturnsDistinctNonNull) {
  CArena arena;
  double* a = arena.allocate<double>(4);
  double* b = arena.allocate<double>(4);
  EXPECT_NE(a, nullptr);
  EXPECT_NE(b, nullptr);
  EXPECT_NE(a, b);
}

TEST(CArena, AllocateFillsInitValue) {
  CArena arena;
  double* p = arena.allocate<double>(3, 7.0);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(p[i], 7.0);
}

TEST(CArena, AllocateDefaultZeroInitializes) {
  CArena arena;
  double* p = arena.allocate<double>(3);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(p[i], 0.0);
}

TEST(CArena, AllocateOfZeroIsHarmless) {
  CArena arena;
  double* p = arena.allocate<double>(0);
  (void)p;
  EXPECT_EQ(arena.carray_count(), 1);
}

TEST(CArena, CarrayCountBumpsPerAllocate) {
  CArena arena;
  EXPECT_EQ(arena.carray_count(), 0);
  arena.allocate<double>(2);
  EXPECT_EQ(arena.carray_count(), 1);
  arena.allocate<double>(8);
  EXPECT_EQ(arena.carray_count(), 2);
}

TEST(CArena, BuffersStayValidWhileArenaAlive) {
  CArena arena;
  double* blocks[16];
  for (int64_t k = 0; k < 16; ++k) {
    blocks[k] = arena.allocate<double>(4);
    blocks[k][0] = static_cast<double>(k);
  }
  // Later allocations must not disturb earlier blocks.
  for (int64_t k = 0; k < 16; ++k)
    EXPECT_DOUBLE_EQ(blocks[k][0], static_cast<double>(k));
}

TEST(CArena, NonCopyableNonMovable) {
  static_assert(not std::is_copy_constructible_v<CArena>);
  static_assert(not std::is_copy_assignable_v<CArena>);
  static_assert(not std::is_move_constructible_v<CArena>);
  static_assert(not std::is_move_assignable_v<CArena>);
  SUCCEED();
}

TEST(CArena, NoteCountersStartAtZero) {
  CArena arena;
  EXPECT_EQ(arena.vnode_count(), 0);
  EXPECT_EQ(arena.cnode_count(), 0);
  EXPECT_EQ(arena.onode_add_count(), 0);
  EXPECT_EQ(arena.onode_pow_count(), 0);
}

TEST(CArena, NoteCountersIncrementIndependently) {
  CArena arena;
  arena.note_vnode();
  arena.note_vnode();
  arena.note_cnode();
  arena.note_onode_add();
  arena.note_onode_dot();
  arena.note_onode_pow();
  arena.note_onode_pow();

  EXPECT_EQ(arena.vnode_count(), 2);
  EXPECT_EQ(arena.cnode_count(), 1);
  EXPECT_EQ(arena.onode_add_count(), 1);
  EXPECT_EQ(arena.onode_dot_count(), 1);
  EXPECT_EQ(arena.onode_pow_count(), 2);
  EXPECT_EQ(arena.onode_sub_count(), 0);        // untouched
  EXPECT_EQ(arena.onode_hadamard_count(), 0);
}

TEST(CArena, AllocatesNonDoubleBuffers) {
  CArena arena;
  int* p = arena.allocate<int>(4, -1);
  for (int64_t i = 0; i < 4; ++i)
    EXPECT_EQ(p[i], -1);
  EXPECT_EQ(arena.carray_count(), 1);
}

TEST(CArena, HoldsMultipleElementTypes) {
  CArena arena;
  double*    d = arena.allocate<double>(4, 1.5);
  int*       i = arena.allocate<int>(3, -2);
  bool*      b = arena.allocate<bool>(2, true);
  short*     s = arena.allocate<short>(2, 7);
  long long* l = arena.allocate<long long>(2, 1LL << 40);
  char*      c = arena.allocate<char>(2, 'z');

  EXPECT_NE(static_cast<void*>(d), static_cast<void*>(i));
  EXPECT_NE(static_cast<void*>(i), static_cast<void*>(b));
  EXPECT_DOUBLE_EQ(d[3], 1.5);
  EXPECT_EQ(i[0], -2);
  EXPECT_TRUE(b[1]);
  EXPECT_EQ(s[1], 7);
  EXPECT_EQ(l[0], 1LL << 40);
  EXPECT_EQ(c[1], 'z');
  EXPECT_EQ(arena.carray_count(), 6);
}

TEST(CArena, NegCounterStartsAtZero) {
  CArena arena;
  EXPECT_EQ(arena.onode_neg_count(), 0);
  arena.note_onode_neg();
  EXPECT_EQ(arena.onode_neg_count(), 1);
}

TEST(CArena, ReductionCountersIncrementIndependently) {
  CArena arena;
  EXPECT_EQ(arena.onode_sum_count(), 0);
  EXPECT_EQ(arena.onode_mean_count(), 0);
  EXPECT_EQ(arena.onode_max_count(), 0);
  EXPECT_EQ(arena.onode_min_count(), 0);
  arena.note_onode_sum();
  arena.note_onode_sum();
  arena.note_onode_mean();
  arena.note_onode_max();
  EXPECT_EQ(arena.onode_sum_count(), 2);
  EXPECT_EQ(arena.onode_mean_count(), 1);
  EXPECT_EQ(arena.onode_max_count(), 1);
  EXPECT_EQ(arena.onode_min_count(), 0);
}

TEST(CArena, NeuralOpCountersIncrementIndependently) {
  CArena arena;
  EXPECT_EQ(arena.onode_softmax_count(), 0);
  EXPECT_EQ(arena.onode_cross_entropy_count(), 0);
  EXPECT_EQ(arena.onode_softmax_cross_entropy_count(), 0);
  arena.note_onode_softmax();
  arena.note_onode_softmax();
  arena.note_onode_cross_entropy();
  arena.note_onode_softmax_cross_entropy();
  arena.note_onode_softmax_cross_entropy();
  arena.note_onode_softmax_cross_entropy();
  EXPECT_EQ(arena.onode_softmax_count(), 2);
  EXPECT_EQ(arena.onode_cross_entropy_count(), 1);
  EXPECT_EQ(arena.onode_softmax_cross_entropy_count(), 3);
}

TEST(CArena, AutoRequiresGradPolicy) {
  CArena arena;
  EXPECT_FALSE(arena.auto_requires_grad());
  arena.set_auto_requires_grad(true);
  EXPECT_TRUE(arena.auto_requires_grad());
  arena.set_auto_requires_grad(false);
  EXPECT_FALSE(arena.auto_requires_grad());
}

// ---------------------------------------------------------------------------
//  Node ownership
// ---------------------------------------------------------------------------

TEST(CArena, AdoptOwnsNodeAndBumpsCount) {
  CArena arena;
  EXPECT_EQ(arena.node_count(), 0);
  Dummy& d = arena.adopt<Dummy>(42);
  EXPECT_EQ(d.v, 42);
  EXPECT_EQ(arena.node_count(), 1);
  EXPECT_EQ(&arena.node_at(0), static_cast<NodeBase*>(&d));
  EXPECT_EQ(arena.node_at(0).mKind, NodeKind::Operation);
}

TEST(CArena, AdoptReferencesStayStableAcrossReallocation) {
  CArena arena;
  Dummy& first = arena.adopt<Dummy>(0);
  for (int i = 1; i < 128; ++i) arena.adopt<Dummy>(i);
  EXPECT_EQ(first.v, 0);                       // not dangling after vector growth
  EXPECT_EQ(arena.node_count(), 128);
  EXPECT_EQ(&arena.node_at(0), static_cast<NodeBase*>(&first));
}

TEST(CArena, PromotedLeafRoundTrips) {
  CArena arena;
  double* key = arena.allocate<double>(1);
  EXPECT_EQ(arena.promoted_leaf(key), nullptr);
  Dummy& d = arena.adopt<Dummy>(7);
  arena.register_leaf(key, &d);
  EXPECT_EQ(arena.promoted_leaf(key), static_cast<NodeBase*>(&d));
  EXPECT_EQ(arena.promoted_leaf(nullptr), nullptr);
}

} // namespace
