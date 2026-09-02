#ifndef C_ARRAY
#define C_ARRAY

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

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

template <typename T> struct CArray;
template <typename T> struct CArrayCRef;

// CArrayRef: a mutable, non-owning view carrying its own Shape and an element
// offset into the owner's buffer, so reshape() and sub() never copy and can be
// chained. The owner must outlive the view (like std::span); a view survives a
// move of the owning CArray, whose buffer address is stable.
template <typename T>
struct CArrayRef {
  CArrayRef() noexcept: mShape(), mOffset(0), mData(nullptr) {}
  CArrayRef(const Shape& shape, T* base, s::size_t offset) noexcept
    : mShape(shape), mOffset(offset), mData(base) {}

  const Shape& shape() const noexcept { return mShape; }
  const s::vector<int64_t>& dims() const noexcept { return mShape.dims(); }
  s::size_t rank() const noexcept { return mShape.rank(); }
  s::size_t size() const { return static_cast<s::size_t>(mShape.product()); }
  s::size_t offset() const noexcept { return mOffset; }

  // Start of this view's span (base + offset).
  T* data() const noexcept { return mData + mOffset; }

  template <s::integral... Is>
  T& operator[](Is... idxs) const {
    assert(sizeof...(Is) == mShape.rank());
    return mData[mOffset + static_cast<s::size_t>(
      mShape.flatten(Index{static_cast<int64_t>(idxs)...}))];
  }

  T& at(const Index& index) const {
    return mData[mOffset + static_cast<s::size_t>(mShape.flatten(index))];
  }

  T& item() const {
    assert(size() == 1);
    return mData[mOffset];
  }

  CArrayRef reshape(const Shape& shape) const {
    return CArrayRef(resolve_reshape(shape, mShape.product()), mData, mOffset);
  }

  // Select slice `i` of the leading axis (negative counts from the end).
  CArrayRef sub(int64_t i) const {
    assert(mShape.rank() >= 1);
    const int64_t d0 = mShape[0];
    if (i < 0) i += d0;
    assert(i >= 0 and i < d0);
    const Shape child = mShape.subshape();
    return CArrayRef(child, mData,
      mOffset + static_cast<s::size_t>(i) * static_cast<s::size_t>(child.product()));
  }

  // Copy this view's span into a fresh owning array.
  CArray<T> clone() const;
private:
  Shape     mShape;
  s::size_t mOffset;
  T*        mData; // points at the owner's element 0; non-owning
};

// CArrayCRef: the read-only twin of CArrayRef.
template <typename T>
struct CArrayCRef {
  CArrayCRef() noexcept: mShape(), mOffset(0), mData(nullptr) {}
  CArrayCRef(const Shape& shape, const T* base, s::size_t offset) noexcept
    : mShape(shape), mOffset(offset), mData(base) {}
  CArrayCRef(const CArrayRef<T>& r) noexcept
    : mShape(r.shape()), mOffset(r.offset()), mData(r.data() - r.offset()) {}
  CArrayCRef(const CArray<T>& a) noexcept;

  const Shape& shape() const noexcept { return mShape; }
  const s::vector<int64_t>& dims() const noexcept { return mShape.dims(); }
  s::size_t rank() const noexcept { return mShape.rank(); }
  s::size_t size() const { return static_cast<s::size_t>(mShape.product()); }
  s::size_t offset() const noexcept { return mOffset; }

  const T* data() const noexcept { return mData + mOffset; }

  template <s::integral... Is>
  const T& operator[](Is... idxs) const {
    assert(sizeof...(Is) == mShape.rank());
    return mData[mOffset + static_cast<s::size_t>(
      mShape.flatten(Index{static_cast<int64_t>(idxs)...}))];
  }

  const T& at(const Index& index) const {
    return mData[mOffset + static_cast<s::size_t>(mShape.flatten(index))];
  }

  const T& item() const {
    assert(size() == 1);
    return mData[mOffset];
  }

  CArrayCRef reshape(const Shape& shape) const {
    return CArrayCRef(resolve_reshape(shape, mShape.product()), mData, mOffset);
  }

  CArrayCRef sub(int64_t i) const {
    assert(mShape.rank() >= 1);
    const int64_t d0 = mShape[0];
    if (i < 0) i += d0;
    assert(i >= 0 and i < d0);
    const Shape child = mShape.subshape();
    return CArrayCRef(child, mData,
      mOffset + static_cast<s::size_t>(i) * static_cast<s::size_t>(child.product()));
  }

  CArray<T> clone() const;
private:
  Shape     mShape;
  s::size_t mOffset;
  const T*  mData;
};

// CArray: owns a contiguous, row-major buffer (last axis contiguous). Copy is
// deleted; duplicate explicitly with clone().
template <typename T>
struct CArray {
  CArray() noexcept: mShape(), mBuffer(nullptr) {}
  explicit CArray(T val): mShape({1}), mBuffer(s::make_unique_for_overwrite<T[]>(1U)) {
    s::fill_n(mBuffer.get(), 1U, val);
  }
  explicit CArray(const Shape& shape, T init_val = T{})
    : mShape(shape),
      mBuffer(s::make_unique_for_overwrite<T[]>(static_cast<s::size_t>(shape.product()))) {
    s::fill_n(mBuffer.get(), static_cast<s::size_t>(shape.product()), init_val);
  }

  CArray(const CArray&) = delete;
  CArray& operator=(const CArray&) = delete;
  CArray(CArray&&) noexcept = default;
  CArray& operator=(CArray&&) noexcept = default;
  ~CArray() = default;

  const Shape& shape() const noexcept { return mShape; }
  const s::vector<int64_t>& dims() const noexcept { return mShape.dims(); }
  s::size_t rank() const noexcept { return mShape.rank(); }
  s::size_t size() const { return static_cast<s::size_t>(mShape.product()); }

  const T* data() const noexcept { return mBuffer.get(); }
  T* data() noexcept { return mBuffer.get(); }

  CArray clone() const {
    CArray out(mShape);
    s::copy_n(mBuffer.get(), size(), out.mBuffer.get());
    return out;
  }

  template <s::integral... Is>
  T& operator[](Is... idxs){
    assert(sizeof...(Is) == mShape.rank());
    return mBuffer[static_cast<s::size_t>(
      mShape.flatten(Index{static_cast<int64_t>(idxs)...}))];
  }
  template <s::integral... Is>
  const T& operator[](Is... idxs) const {
    assert(sizeof...(Is) == mShape.rank());
    return mBuffer[static_cast<s::size_t>(
      mShape.flatten(Index{static_cast<int64_t>(idxs)...}))];
  }

  T& at(const Index& index){
    return mBuffer[static_cast<s::size_t>(mShape.flatten(index))];
  }
  const T& at(const Index& index) const {
    return mBuffer[static_cast<s::size_t>(mShape.flatten(index))];
  }

  T& item(){ assert(size() == 1); return mBuffer[0]; }
  const T& item() const { assert(size() == 1); return mBuffer[0]; }

  CArrayRef<T> ref() noexcept {
    return CArrayRef<T>(mShape, mBuffer.get(), 0);
  }
  CArrayCRef<T> cref() const noexcept {
    return CArrayCRef<T>(mShape, mBuffer.get(), 0);
  }

  CArrayRef<T> reshape(const Shape& shape){
    return CArrayRef<T>(resolve_reshape(shape, mShape.product()), mBuffer.get(), 0);
  }
  CArrayCRef<T> reshape(const Shape& shape) const {
    return CArrayCRef<T>(resolve_reshape(shape, mShape.product()), mBuffer.get(), 0);
  }

  CArrayRef<T> sub(int64_t i){ return ref().sub(i); }
  CArrayCRef<T> sub(int64_t i) const { return cref().sub(i); }
private:
  Shape              mShape;
  s::unique_ptr<T[]> mBuffer;
};

template <typename T>
CArrayCRef<T>::CArrayCRef(const CArray<T>& a) noexcept
  : mShape(a.shape()), mOffset(0), mData(a.data()) {}

template <typename T>
CArray<T> CArrayRef<T>::clone() const {
  CArray<T> out(mShape);
  s::copy_n(mData + mOffset, size(), out.data());
  return out;
}

template <typename T>
CArray<T> CArrayCRef<T>::clone() const {
  CArray<T> out(mShape);
  s::copy_n(mData + mOffset, size(), out.data());
  return out;
}

// Exact element-wise equality for any mix of CArray / CArrayRef / CArrayCRef:
// same shape, same values in row-major order.
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
