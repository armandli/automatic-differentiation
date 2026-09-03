#include <c_arena.h>
#include <c_array.h>

#include <cstdint>
#include <ostream>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

// Readable failure messages for EXPECT_EQ(Shape, Shape) / EXPECT_EQ(Index, ...).
namespace autodiff {

void PrintTo(const Shape& shape, std::ostream* os){
  *os << '(';
  for (std::size_t i = 0; i < shape.size(); ++i){
    if (i) *os << ", ";
    *os << shape[static_cast<int64_t>(i)];
  }
  *os << ')';
}

void PrintTo(const Index& index, std::ostream* os){
  *os << '[';
  for (std::size_t i = 0; i < index.size(); ++i){
    if (i) *os << ", ";
    *os << index[static_cast<int64_t>(i)];
  }
  *os << ']';
}

} // namespace autodiff

namespace {

using autodiff::CArena;
using autodiff::CArray;
using autodiff::Index;
using autodiff::Shape;
using autodiff::content_equal;

// Fill a (4,5,6) array so every element encodes its own coordinates as
// i*100 + j*10 + k.
CArray<double> coded_456(CArena<double>& arena){
  CArray<double> a(arena, Shape{4, 5, 6});
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 5; ++j)
      for (int64_t k = 0; k < 6; ++k)
        a[i, j, k] = i * 100.0 + j * 10.0 + k;
  return a;
}

// ---------------------------------------------------------------------------
//  Index
// ---------------------------------------------------------------------------

TEST(Index, InitializerListAndSize) {
  const Index idx{1, -2, 3};
  EXPECT_EQ(idx.size(), 3u);
  EXPECT_EQ(idx[0], 1);
  EXPECT_EQ(idx[1], -2);
  EXPECT_EQ(idx[2], 3);
}

TEST(Index, NegativeAxisReadsFromEnd) {
  const Index idx{7, 8, 9};
  EXPECT_EQ(idx[-1], 9);
  EXPECT_EQ(idx[-3], 7);
}

TEST(Index, StoresNegativeEntriesVerbatim) {
  const Index idx{-1, -5};
  EXPECT_EQ(idx[0], -1);
  EXPECT_EQ(idx[1], -5);
}

TEST(Index, Equality) {
  EXPECT_EQ((Index{1, 2, 3}), (Index{1, 2, 3}));
  EXPECT_FALSE((Index{1, 2, 3}) == (Index{1, 2, 4}));
}

// ---------------------------------------------------------------------------
//  Shape
// ---------------------------------------------------------------------------

TEST(Shape, ProductOfEmptyShapeIsOne) {
  EXPECT_EQ((Shape{}).product(), 1);
}

TEST(Shape, Product) {
  EXPECT_EQ((Shape{4, 5, 6}).product(), 120);
  EXPECT_EQ((Shape{7}).product(), 7);
}

TEST(Shape, NegativeAxisSubscript) {
  const Shape shape{4, 5, 6};
  EXPECT_EQ(shape[-1], 6);
  EXPECT_EQ(shape[-2], 5);
  EXPECT_EQ(shape[0], 4);
}

TEST(Shape, Subshape) {
  EXPECT_EQ((Shape{4, 5, 6}).subshape(), (Shape{5, 6}));
  EXPECT_EQ((Shape{6}).subshape(), (Shape{}));
}

TEST(Shape, ResolveInfersMissingAxis) {
  EXPECT_EQ((Shape{2, 2, -1}).resolve(120), (Shape{2, 2, 30}));
  EXPECT_EQ((Shape{-1, 6}).resolve(120), (Shape{20, 6}));
}

TEST(Shape, ResolveRoundsDown) {
  const Shape r = (Shape{7, -1}).resolve(120);
  EXPECT_EQ(r, (Shape{7, 17}));
  EXPECT_EQ(r.product(), 119);
}

TEST(Shape, ResolveWithoutPlaceholderIsIdentity) {
  EXPECT_EQ((Shape{2, 3, 4}).resolve(999), (Shape{2, 3, 4}));
}

TEST(Shape, FlattenRowMajor) {
  const Shape shape{4, 5, 6};
  EXPECT_EQ(shape.flatten(Index{0, 0, 0}), 0);
  EXPECT_EQ(shape.flatten(Index{0, 0, 1}), 1);
  EXPECT_EQ(shape.flatten(Index{0, 1, 0}), 6);
  EXPECT_EQ(shape.flatten(Index{1, 0, 0}), 30);
  EXPECT_EQ(shape.flatten(Index{1, 2, 3}), 45);
}

TEST(Shape, FlattenResolvesNegativeEntries) {
  const Shape shape{4, 5, 6};
  EXPECT_EQ(shape.flatten(Index{-1, -1, -1}), 119);
  EXPECT_EQ(shape.flatten(Index{-1, 0, 0}), 90);
}

TEST(Shape, CheckIndex) {
  const Shape shape{4, 5, 6};
  EXPECT_TRUE(shape.check_index(Index{3, 4, 5}));
  EXPECT_TRUE(shape.check_index(Index{-1, -1, -1}));
  EXPECT_FALSE(shape.check_index(Index{4, 0, 0}));   // axis 0 out of range
  EXPECT_FALSE(shape.check_index(Index{0, -6, 0}));  // axis 1 out of range
  EXPECT_FALSE(shape.check_index(Index{0, 0}));      // rank mismatch
}

TEST(Shape, Equality) {
  EXPECT_EQ((Shape{2, 3}), (Shape{2, 3}));
  EXPECT_FALSE((Shape{2, 3}) == (Shape{3, 2}));
}

TEST(Shape, UsableInConstantExpressions) {
  static_assert((Shape{2, 3, 4}).product() == 24);
  static_assert((Shape{4, 5, 6}).flatten(Index{1, 2, 3}) == 45);
  static_assert((Shape{2, 2, -1}).resolve(120) == Shape{2, 2, 30});
  static_assert((Shape{4, 5, 6}).check_index(Index{-1, -1, -1}));
  SUCCEED();
}

TEST(Shape, UnsqueezeInsertsAtFront) {
  EXPECT_EQ((Shape{3, 4}).unsqueeze(0), (Shape{1, 3, 4}));
}

TEST(Shape, UnsqueezeInsertsAtMiddle) {
  EXPECT_EQ((Shape{3, 4}).unsqueeze(1), (Shape{3, 1, 4}));
}

TEST(Shape, UnsqueezeInsertsAtBack) {
  EXPECT_EQ((Shape{3, 4}).unsqueeze(-1), (Shape{3, 4, 1}));
  EXPECT_EQ((Shape{3, 4}).unsqueeze(2),  (Shape{3, 4, 1}));
}

TEST(Shape, SqueezeRemovesUpToN) {
  EXPECT_EQ((Shape{1, 3, 1, 4, 1}).squeeze(2), (Shape{3, 4, 1}));
}

TEST(Shape, SqueezeRemovesAllWhenNLarge) {
  EXPECT_EQ((Shape{1, 3, 1, 4, 1}).squeeze(10), (Shape{3, 4}));
}

TEST(Shape, SqueezeZeroRemovesNothing) {
  EXPECT_EQ((Shape{1, 3, 1}).squeeze(0), (Shape{1, 3, 1}));
}

// ---------------------------------------------------------------------------
//  CArray: construction, access, ownership
// ---------------------------------------------------------------------------

TEST(CArray, ConstructorFillsInitialValue) {
  CArena<double> arena;
  const CArray<double> a(arena, Shape{2, 3}, 4.5);
  EXPECT_EQ(a.size(), 6u);
  EXPECT_EQ(a.rank(), 2u);
  EXPECT_EQ(a.shape(), (Shape{2, 3}));
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ((a[i, j]), 4.5);
}

TEST(CArray, DefaultConstructedIsEmpty) {
  const CArray<double> a;
  EXPECT_EQ(a.rank(), 0u);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.arena(), nullptr);
}

TEST(CArray, MultiSubscriptReadWrite) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  EXPECT_DOUBLE_EQ((a[1, 2, 3]), 123.0);
  EXPECT_DOUBLE_EQ((a[3, 4, 5]), 345.0);
  a[1, 2, 3] = -9.0;
  EXPECT_DOUBLE_EQ((a[1, 2, 3]), -9.0);
}

TEST(CArray, AtWithIndexObject) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  EXPECT_DOUBLE_EQ(a.at(Index{2, 1, 4}), 214.0);
  a.at(Index{2, 1, 4}) = 1.0;
  EXPECT_DOUBLE_EQ((a[2, 1, 4]), 1.0);
}

TEST(CArray, AtResolvesNegativeIndices) {
  CArena<double> arena;
  const CArray<double> a = coded_456(arena);
  EXPECT_DOUBLE_EQ(a.at(Index{-1, -1, -1}), 345.0);
}

TEST(CArray, ItemOnSingleElementArray) {
  CArena<double> arena;
  CArray<double> a(arena, Shape{1}, 0.0);
  a.item() = 7.0;
  EXPECT_DOUBLE_EQ(a.item(), 7.0);
}

TEST(CArray, CloneIsADeepCopy) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> b = a.clone();
  EXPECT_TRUE(content_equal(a, b));

  a[0, 0, 0] = 999.0;
  EXPECT_DOUBLE_EQ((b[0, 0, 0]), 0.0);
  EXPECT_DOUBLE_EQ((a[0, 0, 0]), 999.0);

  b[1, 1, 1] = -1.0;
  EXPECT_DOUBLE_EQ((a[1, 1, 1]), 111.0);
}

TEST(CArray, MoveAliasesBuffer) {
  CArena<double> arena;
  CArray<double> a(arena, Shape{2, 3}, 1.5);
  const double* before = a.data();
  CArray<double> moved = std::move(a);
  EXPECT_EQ(moved.data(), before);
  EXPECT_DOUBLE_EQ((moved[1, 2]), 1.5);
}

TEST(CArray, CopyIsShallowAlias) {
  static_assert(std::is_copy_constructible_v<CArray<double>>);
  static_assert(std::is_copy_assignable_v<CArray<double>>);
  static_assert(std::is_nothrow_move_constructible_v<CArray<double>>);
  static_assert(std::is_nothrow_move_assignable_v<CArray<double>>);

  CArena<double> arena;
  CArray<double> a(arena, Shape{2, 3}, 1.5);
  CArray<double> b = a;                       // shallow: same buffer
  EXPECT_EQ(b.data(), a.data());
  b[1, 2] = 9.0;
  EXPECT_DOUBLE_EQ((a[1, 2]), 9.0);           // write-through
  EXPECT_EQ(arena.carray_count(), 1);         // the copy did not allocate
}

// ---------------------------------------------------------------------------
//  CArrayView: reshape / sub / unsqueeze / squeeze return a CArray aliasing
//  the same arena storage
// ---------------------------------------------------------------------------

TEST(CArrayView, ReshapeSharesStorage) {
  CArena<double> arena;
  CArray<double> a(arena, Shape{4, 5, 6}, 0.0);
  CArray<double> r = a.reshape(Shape{2, 2, -1});

  EXPECT_EQ(r.shape(), (Shape{2, 2, 30}));
  EXPECT_EQ(r.data(), a.data());

  r[1, 1, 29] = 42.0;                     // last element of the view
  EXPECT_DOUBLE_EQ((a[3, 4, 5]), 42.0);   // ... is the last element of the array

  a[0, 0, 0] = -7.0;
  EXPECT_DOUBLE_EQ((r[0, 0, 0]), -7.0);
}

TEST(CArrayView, ReshapeWithRoundingDownIsAPrefixView) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> r = a.reshape(Shape{7, -1});

  EXPECT_EQ(r.shape(), (Shape{7, 17}));
  EXPECT_EQ(r.size(), 119u);
  EXPECT_EQ(r.data(), a.data());
  EXPECT_DOUBLE_EQ((r[0, 0]), (a[0, 0, 0]));
}

TEST(CArrayView, ReshapeToFlatVector) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> flat = a.reshape(Shape{-1});
  EXPECT_EQ(flat.shape(), (Shape{120}));
  EXPECT_DOUBLE_EQ((flat[45]), (a[1, 2, 3]));
  EXPECT_DOUBLE_EQ((flat[119]), (a[3, 4, 5]));
}

TEST(CArrayView, SubPeelsLeadingAxis) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> slice = a.sub(1);

  EXPECT_EQ(slice.shape(), (Shape{5, 6}));
  EXPECT_EQ(slice.data(), a.data() + 30);
  EXPECT_DOUBLE_EQ((slice[2, 3]), (a[1, 2, 3]));

  slice[0, 0] = -1.0;
  EXPECT_DOUBLE_EQ((a[1, 0, 0]), -1.0);
}

TEST(CArrayView, SubAcceptsNegativeSlice) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  EXPECT_DOUBLE_EQ((a.sub(-1)[4, 5]), (a[3, 4, 5]));
  EXPECT_DOUBLE_EQ((a.sub(-4)[0, 0]), (a[0, 0, 0]));
}

TEST(CArrayView, SubChainsAndOffsetsCompound) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> cell = a.sub(1).sub(2);
  EXPECT_EQ(cell.shape(), (Shape{6}));
  EXPECT_DOUBLE_EQ((cell[3]), (a[1, 2, 3]));

  cell[5] = 88.0;
  EXPECT_DOUBLE_EQ((a[1, 2, 5]), 88.0);
}

TEST(CArrayView, ReshapeThenSub) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> row = a.reshape(Shape{6, -1}).sub(0);
  EXPECT_EQ(row.shape(), (Shape{20}));
  EXPECT_DOUBLE_EQ((row[0]), (a[0, 0, 0]));
  EXPECT_DOUBLE_EQ((row[19]), (a[0, 3, 1]));  // flat index 19 -> (0,3,1)
}

TEST(CArrayView, CloneMaterializesOnlyTheViewSpan) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  CArray<double> piece = a.sub(1).clone();

  EXPECT_EQ(piece.shape(), (Shape{5, 6}));
  EXPECT_TRUE(content_equal(piece, a.sub(1)));

  piece[0, 0] = 123456.0;                 // independent storage
  EXPECT_DOUBLE_EQ((a[1, 0, 0]), 100.0);
}

TEST(CArrayView, ItemThroughRankZeroView) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);
  EXPECT_DOUBLE_EQ(a.sub(2).sub(3).sub(4).item(), (a[2, 3, 4]));
}

TEST(CArrayView, UnsqueezeSharesStorageAndShapeCorrect) {
  CArena<double> arena;
  CArray<double> a = coded_456(arena);  // shape {4,5,6}
  CArray<double> u = a.unsqueeze(0);

  EXPECT_EQ(u.shape(), (Shape{1, 4, 5, 6}));
  EXPECT_EQ(u.data(), a.data());

  // Element access through the extra leading dimension
  EXPECT_DOUBLE_EQ((u[0, 1, 2, 3]), (a[1, 2, 3]));

  // Write through view, read back through original
  u[0, 3, 4, 5] = 999.0;
  EXPECT_DOUBLE_EQ((a[3, 4, 5]), 999.0);
}

TEST(CArrayView, UnsqueezeAtMiddleAndBack) {
  CArena<double> arena;
  CArray<double> a(arena, Shape{3, 4}, 0.0);
  EXPECT_EQ(a.unsqueeze(1).shape(), (Shape{3, 1, 4}));
  EXPECT_EQ(a.unsqueeze(-1).shape(), (Shape{3, 4, 1}));
}

TEST(CArrayView, SqueezeSharesStorageAndShapeCorrect) {
  CArena<double> arena;
  CArray<double> a(arena, Shape{1, 4, 1, 5}, 0.0);

  CArray<double> s1 = a.squeeze(1);
  EXPECT_EQ(s1.shape(), (Shape{4, 1, 5}));
  EXPECT_EQ(s1.data(), a.data());

  CArray<double> s2 = a.squeeze(2);
  EXPECT_EQ(s2.shape(), (Shape{4, 5}));
  EXPECT_EQ(s2.data(), a.data());
}

// ---------------------------------------------------------------------------
//  Views from a const array (shallow const)
// ---------------------------------------------------------------------------

TEST(CArrayView, ViewFromConstArrayReadsThrough) {
  CArena<double> arena;
  const CArray<double> a = coded_456(arena);
  CArray<double> c = a.reshape(Shape{4, 5, 6});
  EXPECT_EQ(c.shape(), (Shape{4, 5, 6}));
  EXPECT_DOUBLE_EQ((c[1, 2, 3]), 123.0);
  EXPECT_DOUBLE_EQ(c.sub(1).at(Index{2, 3}), (a.at(Index{1, 2, 3})));
}

TEST(CArrayView, ReshapeOnConstArrayYieldsCArray) {
  CArena<double> arena;
  const CArray<double> a = coded_456(arena);
  auto flat = a.reshape(Shape{120});
  static_assert(std::is_same_v<decltype(flat), CArray<double>>);
  EXPECT_DOUBLE_EQ((flat[45]), 123.0);
}

TEST(CArrayView, ConstnessOfArrayControlsElementAccess) {
  static_assert(std::is_same_v<
    decltype(std::declval<const CArray<double>&>().at(std::declval<const Index&>())),
    const double&>);
  static_assert(std::is_same_v<
    decltype(std::declval<const CArray<double>&>()[0, 0]),
    const double&>);
  static_assert(std::is_same_v<
    decltype(std::declval<CArray<double>&>().at(std::declval<const Index&>())),
    double&>);
  SUCCEED();
}

// ---------------------------------------------------------------------------
//  content_equal and an end-to-end scenario
// ---------------------------------------------------------------------------

TEST(ContentEqual, ComparesShapeAndValues) {
  CArena<int> arena;
  CArray<int> m(arena, Shape{2, 3});
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      m[i, j] = static_cast<int>(i * 3 + j);

  CArray<int> flat = m.reshape(Shape{6});
  CArray<int> copy = flat.clone();
  EXPECT_TRUE(content_equal(copy, flat));
  EXPECT_TRUE(content_equal(flat, m.reshape(Shape{6})));

  copy[0] = -1;
  EXPECT_FALSE(content_equal(copy, flat));

  EXPECT_FALSE(content_equal(m, m.reshape(Shape{6})));  // same data, other shape
}

TEST(CArray, EndToEndReshapeAndReduce) {
  CArena<double> arena;
  CArray<double> a(arena, Shape{2, 3});
  double expected = 0.0;
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j){
      a[i, j] = static_cast<double>(i) + static_cast<double>(j) * 0.5;
      expected += a[i, j];
    }

  CArray<double> flat = a.reshape(Shape{6});
  double sum = 0.0;
  for (int64_t k = 0; k < 6; ++k)
    sum += flat.sub(k).item();

  EXPECT_DOUBLE_EQ(sum, expected);
}

} // namespace
