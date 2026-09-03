#ifndef C_GRAPH_H
#define C_GRAPH_H

#include <c_array.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>

namespace autodiff {

namespace s = std;

// ---------------------------------------------------------------------------
//  Global per-type node counters (thread-safe, header-only inline statics)
// ---------------------------------------------------------------------------
namespace graph_stats {
  inline s::atomic<int64_t> vnode_count{0};
  inline s::atomic<int64_t> cnode_count{0};
  inline s::atomic<int64_t> onode_add_count{0};
  inline s::atomic<int64_t> onode_sub_count{0};
  inline s::atomic<int64_t> onode_mul_count{0};
  inline s::atomic<int64_t> onode_div_count{0};
  inline s::atomic<int64_t> onode_exp_count{0};
  inline s::atomic<int64_t> onode_log_count{0};
}

enum class Op { Add, Sub, Mul, Div, Exp, Log };

// ---------------------------------------------------------------------------
//  Node<T>: base for all graph nodes.
//  Inherits CArray<T> (the value buffer) and adds a gradient buffer of the
//  same shape. Constructors are protected; use VNode, CNode, or ONode.
// ---------------------------------------------------------------------------
template <typename T>
struct Node : CArray<T> {
  CArray<T> mGrad;

  const CArray<T>& grad() const noexcept { return mGrad; }
  CArray<T>&       grad()       noexcept { return mGrad; }

  void zero_grad() {
    T* p = mGrad.data();
    if (p) s::fill_n(p, mGrad.size(), T{});
  }

  virtual ~Node() = default;

protected:
  Node() : CArray<T>(), mGrad() {}

  explicit Node(const Shape& shape, T val = T{})
    : CArray<T>(shape, val), mGrad(shape, T{}) {}

  // Takes ownership of a pre-computed result (used by ONode).
  // The base CArray<T> is initialized first, so shape() is valid when
  // mGrad is constructed.
  explicit Node(CArray<T>&& data)
    : CArray<T>(s::move(data)), mGrad(CArray<T>::shape(), T{}) {}
};

// ---------------------------------------------------------------------------
//  VNode<T>: variable node — participates in gradient computation.
// ---------------------------------------------------------------------------
template <typename T>
struct VNode : Node<T> {
  explicit VNode(const Shape& shape, T val = T{})
    : Node<T>(shape, val) {
    ++graph_stats::vnode_count;
  }
};

// ---------------------------------------------------------------------------
//  CNode<T>: constant node — treated as a fixed leaf with no gradient.
// ---------------------------------------------------------------------------
template <typename T>
struct CNode : Node<T> {
  explicit CNode(const Shape& shape, T val = T{})
    : Node<T>(shape, val) {
    ++graph_stats::cnode_count;
  }
};

// ---------------------------------------------------------------------------
//  ONode<T>: operation node — owns the result, links to its input nodes.
//  mRight is null for unary operations (Exp, Log).
//  Lifetime is caller-managed: operations return ONode<T>* (heap-allocated).
// ---------------------------------------------------------------------------
template <typename T>
struct ONode : Node<T> {
  Op       mOp;
  Node<T>* mLeft;
  Node<T>* mRight;

  ONode(Op op, CArray<T>&& result, Node<T>* left, Node<T>* right = nullptr)
    : Node<T>(s::move(result))
    , mOp(op), mLeft(left), mRight(right)
  {
    switch (op) {
      case Op::Add: ++graph_stats::onode_add_count; break;
      case Op::Sub: ++graph_stats::onode_sub_count; break;
      case Op::Mul: ++graph_stats::onode_mul_count; break;
      case Op::Div: ++graph_stats::onode_div_count; break;
      case Op::Exp: ++graph_stats::onode_exp_count; break;
      case Op::Log: ++graph_stats::onode_log_count; break;
    }
  }

  Op       op()    const noexcept { return mOp; }
  Node<T>* left()  const noexcept { return mLeft; }
  Node<T>* right() const noexcept { return mRight; }
};

// ---------------------------------------------------------------------------
//  Internal element-wise loop helpers
// ---------------------------------------------------------------------------
namespace detail {

template <typename T, typename BinOp>
ONode<T>* make_binary(Op op, Node<T>* a, Node<T>* b, BinOp f) {
  assert(a->shape() == b->shape());
  CArray<T> result(a->shape());
  const T* pa = a->data();
  const T* pb = b->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = a->size(); i < n; ++i)
    pr[i] = f(pa[i], pb[i]);
  return new ONode<T>(op, s::move(result), a, b);
}

template <typename T, typename UnOp>
ONode<T>* make_unary(Op op, Node<T>* a, UnOp f) {
  CArray<T> result(a->shape());
  const T* pa = a->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = a->size(); i < n; ++i)
    pr[i] = f(pa[i]);
  return new ONode<T>(op, s::move(result), a, nullptr);
}

} // detail

// ---------------------------------------------------------------------------
//  Operations — heap-allocate an ONode; caller is responsible for deletion.
// ---------------------------------------------------------------------------
template <typename T>
ONode<T>* graph_add(Node<T>* a, Node<T>* b) {
  return detail::make_binary(Op::Add, a, b, s::plus<T>{});
}

template <typename T>
ONode<T>* graph_sub(Node<T>* a, Node<T>* b) {
  return detail::make_binary(Op::Sub, a, b, s::minus<T>{});
}

template <typename T>
ONode<T>* graph_mul(Node<T>* a, Node<T>* b) {
  return detail::make_binary(Op::Mul, a, b, s::multiplies<T>{});
}

template <typename T>
ONode<T>* graph_div(Node<T>* a, Node<T>* b) {
  return detail::make_binary(Op::Div, a, b, s::divides<T>{});
}

template <typename T>
ONode<T>* graph_exp(Node<T>* a) {
  return detail::make_unary(Op::Exp, a, [](T x){ return s::exp(x); });
}

template <typename T>
ONode<T>* graph_log(Node<T>* a) {
  return detail::make_unary(Op::Log, a, [](T x){ return s::log(x); });
}

// ---------------------------------------------------------------------------
//  Operator overloads on Node<T>& (binary only; exp/log remain free fns)
// ---------------------------------------------------------------------------
template <typename T>
ONode<T>* operator+(Node<T>& a, Node<T>& b) { return graph_add(&a, &b); }

template <typename T>
ONode<T>* operator-(Node<T>& a, Node<T>& b) { return graph_sub(&a, &b); }

template <typename T>
ONode<T>* operator*(Node<T>& a, Node<T>& b) { return graph_mul(&a, &b); }

template <typename T>
ONode<T>* operator/(Node<T>& a, Node<T>& b) { return graph_div(&a, &b); }

} // autodiff

#endif // C_GRAPH_H
