#ifndef C_GRAPH_H
#define C_GRAPH_H

#include <c_arena.h>
#include <c_array.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace autodiff {

namespace s = std;

enum class Op { Add, Sub, Hadamard, Dot, Div, Exp, Log, Sin, Cos, Tan, Sqrt, Abs, Pow };

// ---------------------------------------------------------------------------
//  Node<T>: base for all graph nodes.
//  Inherits CArray<T> (the value buffer) and adds a gradient buffer of the
//  same shape. Constructors are protected; use VNode, CNode, or ONode. Every
//  node is a stack value object; its buffers belong to the CArena<T> passed
//  in, which must outlive it.
// ---------------------------------------------------------------------------
template <typename T>
struct Node : CArray<T> {
  CArray<T> mGrad;
  s::string mName;

  const CArray<T>& grad() const noexcept { return mGrad; }
  CArray<T>&       grad()       noexcept { return mGrad; }
  const s::string& name() const noexcept { return mName; }

  void zero_grad() {
    T* p = mGrad.data();
    if (p) s::fill_n(p, mGrad.size(), T{});
  }

  virtual ~Node() = default;

protected:
  Node() : CArray<T>(), mGrad() {}

  // The user-declared destructor suppresses the implicit move; restore the
  // full set so ONode can be returned by value without degrading to a copy.
  Node(const Node&) = default;
  Node& operator=(const Node&) = default;
  Node(Node&&) noexcept = default;
  Node& operator=(Node&&) noexcept = default;

  Node(CArena<T>& arena, const Shape& shape, T val = T{})
    : CArray<T>(arena, shape, val), mGrad(arena, shape, T{}) {}

  // Adopts a pre-computed result (used by ONode). The CArray<T> base is
  // move-constructed first, so shape() is valid when mGrad is built.
  Node(CArena<T>& arena, CArray<T>&& data)
    : CArray<T>(s::move(data)), mGrad(arena, CArray<T>::shape(), T{}) {}
};

// ---------------------------------------------------------------------------
//  VNode<T>: variable node — participates in gradient computation.
// ---------------------------------------------------------------------------
template <typename T>
struct VNode : Node<T> {
  VNode(CArena<T>& arena, const Shape& shape, T val = T{})
    : Node<T>(arena, shape, val) {
    this->mName = "var_" + s::to_string(arena.note_vnode());
  }

  VNode(CArena<T>& arena, const Shape& shape, s::string_view name, T val = T{})
    : Node<T>(arena, shape, val) {
    arena.note_vnode();
    this->mName = name;
  }
};

// ---------------------------------------------------------------------------
//  CNode<T>: constant node — treated as a fixed leaf with no gradient.
// ---------------------------------------------------------------------------
template <typename T>
struct CNode : Node<T> {
  CNode(CArena<T>& arena, const Shape& shape, T val = T{})
    : Node<T>(arena, shape, val) {
    this->mName = "const_" + s::to_string(arena.note_cnode());
  }

  CNode(CArena<T>& arena, const Shape& shape, s::string_view name, T val = T{})
    : Node<T>(arena, shape, val) {
    arena.note_cnode();
    this->mName = name;
  }
};

// ---------------------------------------------------------------------------
//  ONode<T>: operation node — holds the result, links to its input nodes.
//  mRight is null for unary operations (Exp, Log, Sin, ...). Returned by value
//  from the graph_* factories; the input nodes it points at must outlive it.
// ---------------------------------------------------------------------------
template <typename T>
struct ONode : Node<T> {
  Op       mOp;
  Node<T>* mLeft;
  Node<T>* mRight;

  ONode(const ONode&) = default;
  ONode& operator=(const ONode&) = default;
  ONode(ONode&&) noexcept = default;
  ONode& operator=(ONode&&) noexcept = default;

  ONode(CArena<T>& arena, Op op, CArray<T>&& result,
        Node<T>* left, Node<T>* right = nullptr)
    : Node<T>(arena, s::move(result))
    , mOp(op), mLeft(left), mRight(right)
  {
    switch (op) {
      case Op::Add:      arena.note_onode_add();      break;
      case Op::Sub:      arena.note_onode_sub();      break;
      case Op::Hadamard: arena.note_onode_hadamard(); break;
      case Op::Dot:      arena.note_onode_dot();      break;
      case Op::Div:      arena.note_onode_div();      break;
      case Op::Exp:      arena.note_onode_exp();      break;
      case Op::Log:      arena.note_onode_log();      break;
      case Op::Sin:      arena.note_onode_sin();      break;
      case Op::Cos:      arena.note_onode_cos();      break;
      case Op::Tan:      arena.note_onode_tan();      break;
      case Op::Sqrt:     arena.note_onode_sqrt();     break;
      case Op::Abs:      arena.note_onode_abs();      break;
      case Op::Pow:      arena.note_onode_pow();      break;
    }
  }

  Op       op()    const noexcept { return mOp; }
  Node<T>* left()  const noexcept { return mLeft; }
  Node<T>* right() const noexcept { return mRight; }
};

// ---------------------------------------------------------------------------
//  Internal loop helpers
// ---------------------------------------------------------------------------
namespace detail {

template <typename T, typename BinOp>
ONode<T> make_binary(CArena<T>& arena, Op op, Node<T>* a, Node<T>* b, BinOp f) {
  assert(a->arena() == b->arena());
  assert(a->shape() == b->shape());
  CArray<T> result(arena, a->shape());
  const T* pa = a->data();
  const T* pb = b->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = a->size(); i < n; ++i)
    pr[i] = f(pa[i], pb[i]);
  return ONode<T>(arena, op, s::move(result), a, b);
}

template <typename T, typename UnOp>
ONode<T> make_unary(CArena<T>& arena, Op op, Node<T>* a, UnOp f) {
  CArray<T> result(arena, a->shape());
  const T* pa = a->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = a->size(); i < n; ++i)
    pr[i] = f(pa[i]);
  return ONode<T>(arena, op, s::move(result), a, nullptr);
}

// Matrix multiply / dot for rank-1 and rank-2 operands, numpy-matmul shape
// rules: a rank-1 left operand is treated as a row (1,k) and its leading axis
// dropped from the result; a rank-1 right operand is a column (k,1) and its
// trailing axis dropped; a 1-D * 1-D contraction yields a single value {1}.
// The product is always computed as the (m,k)*(k,n) triple loop in row-major
// flat indices, valid for every case because out.product() == m*n.
template <typename T>
ONode<T> make_dot(CArena<T>& arena, Node<T>* a, Node<T>* b) {
  assert(a->arena() == b->arena());
  const s::size_t ra = a->rank(), rb = b->rank();
  assert((ra == 1 or ra == 2) and (rb == 1 or rb == 2));

  const int64_t m = (ra == 2) ? a->shape()[0] : 1;
  const int64_t k = (ra == 2) ? a->shape()[1] : a->shape()[0];
  const int64_t n = (rb == 2) ? b->shape()[1] : 1;
  assert(k == b->shape()[0]);   // rb == 1 -> shape()[0]; rb == 2 -> rows

  Shape out{m, n};                     // the rank-2 * rank-2 case
  if (ra == 1 and rb == 1)
    out = Shape{1};
  else if (ra == 1)
    out = Shape{n};
  else if (rb == 1)
    out = Shape{m};

  CArray<T> result(arena, out);
  const T* pa = a->data();
  const T* pb = b->data();
  T* pr = result.data();
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j){
      T acc{};
      for (int64_t p = 0; p < k; ++p)
        acc += pa[i * k + p] * pb[p * n + j];
      pr[i * n + j] = acc;
    }
  return ONode<T>(arena, Op::Dot, s::move(result), a, b);
}

} // detail

// ---------------------------------------------------------------------------
//  Operations — build an ONode over the operands' arena and return it by
//  value. Nothing is heap-allocated; the arena owns every buffer.
// ---------------------------------------------------------------------------
template <typename T>
ONode<T> graph_add(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Add, a, b, s::plus<T>{});
}

template <typename T>
ONode<T> graph_sub(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Sub, a, b, s::minus<T>{});
}

// Element-wise (Hadamard) product; both operands must share a shape.
template <typename T>
ONode<T> graph_hadamard(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Hadamard, a, b, s::multiplies<T>{});
}

// Matrix multiply / dot; see detail::make_dot for the rank-1/rank-2 shape rules.
template <typename T>
ONode<T> graph_dot(Node<T>* a, Node<T>* b) {
  return detail::make_dot(*a->arena(), a, b);
}

template <typename T>
ONode<T> graph_div(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Div, a, b, s::divides<T>{});
}

template <typename T>
ONode<T> graph_exp(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Exp, a, [](T x){ return s::exp(x); });
}

template <typename T>
ONode<T> graph_log(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Log, a, [](T x){ return s::log(x); });
}

template <typename T>
ONode<T> graph_sin(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Sin, a, [](T x){ return s::sin(x); });
}

template <typename T>
ONode<T> graph_cos(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Cos, a, [](T x){ return s::cos(x); });
}

template <typename T>
ONode<T> graph_tan(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Tan, a, [](T x){ return s::tan(x); });
}

template <typename T>
ONode<T> graph_sqrt(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Sqrt, a, [](T x){ return s::sqrt(x); });
}

template <typename T>
ONode<T> graph_abs(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Abs, a, [](T x){ return s::abs(x); });
}

template <typename T>
ONode<T> graph_pow(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Pow, a, b, [](T x, T y){ return s::pow(x, y); });
}

// ---------------------------------------------------------------------------
//  Operator overloads on Node<T>& (binary only; unary ops remain free fns).
//  Lvalue operands only, so intermediates in a chain must be named.
//
//  `*` is element-wise, `&` is matrix multiply / dot, `^` is pow. Note that C++
//  gives `&` and `^` lower precedence than `+ - * /`, so mixed expressions need
//  parentheses: write `(a & b) + c` and `(x ^ n) * c`.
// ---------------------------------------------------------------------------
template <typename T>
ONode<T> operator+(Node<T>& a, Node<T>& b) { return graph_add(&a, &b); }

template <typename T>
ONode<T> operator-(Node<T>& a, Node<T>& b) { return graph_sub(&a, &b); }

template <typename T>
ONode<T> operator*(Node<T>& a, Node<T>& b) { return graph_hadamard(&a, &b); }

template <typename T>
ONode<T> operator/(Node<T>& a, Node<T>& b) { return graph_div(&a, &b); }

template <typename T>
ONode<T> operator^(Node<T>& a, Node<T>& b) { return graph_pow(&a, &b); }

template <typename T>
ONode<T> operator&(Node<T>& a, Node<T>& b) { return graph_dot(&a, &b); }

} // autodiff

#endif // C_GRAPH_H
