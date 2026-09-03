#ifndef C_ARRAY
#define C_ARRAY

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <vector>

#include <c_arena.h>

namespace autodiff {

namespace s = std;

// Index: a multi-dimensional subscript. An entry may be negative, counting back
// from the end of its axis (-1 is the last element). Shape resolves the
// negatives when the index is used.
struct Index {
  constexpr Index() = default;
  constexpr Index(const s::vector<int64_t>& dims): mDimensions(dims) {}
  constexpr Index(s::vector<int64_t>&& dims): mDimensions(s::move(dims)) {}
  constexpr Index(s::initializer_list<int64_t> dims): mDimensions(dims) {}

  constexpr int64_t operator[](int64_t axis) const {
    const int64_t r = static_cast<int64_t>(mDimensions.size());
    if (axis < 0) axis += r;
    assert(axis >= 0 and axis < r);
    return mDimensions[static_cast<s::size_t>(axis)];
  }

  constexpr s::size_t size() const noexcept { return mDimensions.size(); }

  constexpr bool operator==(const Index& o) const {
    return mDimensions == o.mDimensions;
  }
private:
  s::vector<int64_t> mDimensions;
};

// Shape: the dimensional extent of a CArray or a view. Layout is row-major, so
// the last axis is contiguous. One axis may be given as -1 and filled in later
// by resolve().
struct Shape {
  constexpr Shape() = default;
  constexpr Shape(const s::vector<int64_t>& dims): mDimensions(dims) {}
  constexpr Shape(s::vector<int64_t>&& dims): mDimensions(s::move(dims)) {}
  constexpr Shape(s::initializer_list<int64_t> dims): mDimensions(dims) {}

  constexpr int64_t operator[](int64_t axis) const {
    const int64_t r = static_cast<int64_t>(mDimensions.size());
    if (axis < 0) axis += r;
    assert(axis >= 0 and axis < r);
    return mDimensions[static_cast<s::size_t>(axis)];
  }

  constexpr const s::vector<int64_t>& dims() const noexcept { return mDimensions; }
  constexpr s::size_t size() const noexcept { return mDimensions.size(); }
  constexpr s::size_t rank() const noexcept { return mDimensions.size(); }

  constexpr int64_t product() const {
    return s::accumulate(s::begin(mDimensions), s::end(mDimensions), int64_t{1}, s::multiplies<int64_t>{});
  }

  // Replace the lone -1 (if any) with total / product(other axes), rounded
  // down; a shape with no -1 is returned unchanged.
  constexpr Shape resolve(int64_t total) const {
    s::vector<int64_t> out = mDimensions;
    int negs = 0;
    s::size_t at = 0;
    int64_t known = 1;
    for (s::size_t i = 0; i < out.size(); ++i)
      if (out[i] < 0){
        ++negs;
        at = i;
      } else {
        known *= out[i];
      }
    assert(negs <= 1);
    if (negs == 1){
      assert(known > 0);
      out[at] = total / known;
    }
    return Shape(s::move(out));
  }

  // Row-major flat offset; negative entries count from the end of their axis.
  constexpr int64_t flatten(const Index& index) const {
    assert(index.size() == mDimensions.size());
    int64_t flat = 0;
    int64_t stride = 1;
    for (s::size_t k = mDimensions.size(); k-- > 0; ){
      int64_t i = index[static_cast<int64_t>(k)];
      if (i < 0) i += mDimensions[k];
      assert(i >= 0 and i < mDimensions[k]);
      flat += i * stride;
      stride *= mDimensions[k];
    }
    return flat;
  }

  // Like flatten's bounds checks, but returns a bool instead of asserting.
  constexpr bool check_index(const Index& index) const {
    if (index.size() != mDimensions.size()) return false;
    for (s::size_t k = 0; k < mDimensions.size(); ++k){
      int64_t i = index[static_cast<int64_t>(k)];
      if (i < 0) i += mDimensions[k];
      if (i < 0 or i >= mDimensions[k]) return false;
    }
    return true;
  }

  // The shape with the leading axis dropped.
  constexpr Shape subshape() const {
    assert(mDimensions.size() >= 1);
    return Shape(s::vector<int64_t>(s::begin(mDimensions) + 1, s::end(mDimensions)));
  }

  // Insert a size-1 dimension at `axis`. Negative axis counts from the end of
  // the new rank, so -1 appends.
  constexpr Shape unsqueeze(int64_t axis) const {
    const int64_t r = static_cast<int64_t>(mDimensions.size());
    if (axis < 0) axis += r + 1;
    assert(axis >= 0 and axis <= r);
    s::vector<int64_t> out;
    out.reserve(static_cast<s::size_t>(r + 1));
    out.insert(out.end(), mDimensions.begin(),
               mDimensions.begin() + static_cast<s::size_t>(axis));
    out.push_back(1);
    out.insert(out.end(),
               mDimensions.begin() + static_cast<s::size_t>(axis),
               mDimensions.end());
    return Shape(s::move(out));
  }

  // Remove up to `n` size-1 dimensions, scanning left-to-right.
  constexpr Shape squeeze(int n) const {
    s::vector<int64_t> out;
    out.reserve(mDimensions.size());
    for (int64_t d : mDimensions) {
      if (d == 1 and n > 0) { --n; continue; }
      out.push_back(d);
    }
    return Shape(s::move(out));
  }

  constexpr bool operator==(const Shape& o) const {
    return mDimensions == o.mDimensions;
  }
  constexpr bool operator!=(const Shape& o) const {
    return not operator==(o);
  }
private:
  s::vector<int64_t> mDimensions;
};

// Resolve a reshape request against a span of `total` elements: at most one -1,
// a request without -1 must consume `total` exactly, and the result must fit.
inline Shape resolve_reshape(const Shape& request, int64_t total){
  int negs = 0;
  for (int64_t d : request.dims())
    if (d < 0) ++negs;
  assert(negs <= 1);
  if (negs == 0){
    assert(request.product() == total);
    return request;
  }
  const Shape resolved = request.resolve(total);
  assert(resolved.product() <= total);
  return resolved;
}

// ---------------------------------------------------------------------------
//  CArray<T>: a Shape plus a non-owning pointer into a CArena<T> block. The
//  arena owns the storage; CArray is a stack value handle. Copy and move are
//  shallow — both alias the same buffer (deliberate: clone() is the only deep
//  copy). reshape()/sub()/unsqueeze()/squeeze() are const and return a fresh
//  CArray<T> over the same storage; they are mutable-capable (shallow const),
//  so read-only intent is expressed by passing `const CArray<T>&` and using
//  the const element accessors. A default-constructed CArray has no arena;
//  clone() on one asserts.
// ---------------------------------------------------------------------------
template <typename T>
struct CArray {
  CArray() noexcept: mShape(), mData(nullptr), mArena(nullptr) {}

  explicit CArray(CArena<T>& arena, T val)
    : mShape({1}), mData(arena.allocate(1U, val)), mArena(&arena) {}

  explicit CArray(CArena<T>& arena, const Shape& shape, T init_val = T{})
    : mShape(shape)
    , mData(arena.allocate(static_cast<s::size_t>(shape.product()), init_val))
    , mArena(&arena) {}

  CArray(const CArray&) = default;             // shallow: aliases the same buffer
  CArray& operator=(const CArray&) = default;
  CArray(CArray&&) noexcept = default;
  CArray& operator=(CArray&&) noexcept = default;
  ~CArray() = default;

  const Shape& shape() const noexcept { return mShape; }
  const s::vector<int64_t>& dims() const noexcept { return mShape.dims(); }
  s::size_t rank() const noexcept { return mShape.rank(); }
  s::size_t size() const { return static_cast<s::size_t>(mShape.product()); }

  const T* data() const noexcept { return mData; }
  T* data() noexcept { return mData; }

  CArena<T>* arena() const noexcept { return mArena; }

  // The only deep copy: allocate a fresh block in the same arena.
  CArray clone() const {
    assert(mArena != nullptr);
    CArray out(*mArena, mShape);
    s::copy_n(mData, size(), out.mData);
    return out;
  }

  template <s::integral... Is>
  T& operator[](Is... idxs){
    assert(sizeof...(Is) == mShape.rank());
    return mData[static_cast<s::size_t>(
      mShape.flatten(Index{static_cast<int64_t>(idxs)...}))];
  }
  template <s::integral... Is>
  const T& operator[](Is... idxs) const {
    assert(sizeof...(Is) == mShape.rank());
    return mData[static_cast<s::size_t>(
      mShape.flatten(Index{static_cast<int64_t>(idxs)...}))];
  }

  T& at(const Index& index){
    return mData[static_cast<s::size_t>(mShape.flatten(index))];
  }
  const T& at(const Index& index) const {
    return mData[static_cast<s::size_t>(mShape.flatten(index))];
  }

  T& item(){ assert(size() == 1); return mData[0]; }
  const T& item() const { assert(size() == 1); return mData[0]; }

  // Aliasing views: a fresh CArray<T> over this array's storage, no allocation.
  CArray reshape(const Shape& shape) const {
    return CArray(resolve_reshape(shape, mShape.product()), mData, mArena);
  }
  CArray unsqueeze(int64_t axis) const {
    return CArray(mShape.unsqueeze(axis), mData, mArena);
  }
  CArray squeeze(int n) const {
    return CArray(mShape.squeeze(n), mData, mArena);
  }

  // Select slice `i` of the leading axis (negative counts from the end).
  CArray sub(int64_t i) const {
    assert(mShape.rank() >= 1);
    const int64_t d0 = mShape[0];
    if (i < 0) i += d0;
    assert(i >= 0 and i < d0);
    const Shape child = mShape.subshape();
    return CArray(child,
      mData + static_cast<s::size_t>(i) * static_cast<s::size_t>(child.product()),
      mArena);
  }

private:
  // Aliasing-view constructor: shares base's storage, allocates nothing.
  CArray(const Shape& shape, T* base, CArena<T>* arena) noexcept
    : mShape(shape), mData(base), mArena(arena) {}

  Shape      mShape;
  T*         mData;   // non-owning; into a CArena<T> block
  CArena<T>* mArena;  // non-owning; for clone() / further allocation
};

// Exact element-wise equality: same shape, same values in row-major order.
// There is deliberately no operator== on CArray (float == footgun).
template <typename A, typename B>
bool content_equal(const A& a, const B& b){
  if (not (a.shape() == b.shape())) return false;
  const auto* pa = a.data();
  const auto* pb = b.data();
  for (s::size_t i = 0, n = a.size(); i < n; ++i)
    if (not (pa[i] == pb[i])) return false;
  return true;
}

} // autodiff

#endif//C_ARRAY
