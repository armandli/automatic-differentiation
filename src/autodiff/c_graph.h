#ifndef C_GRAPH_H
#define C_GRAPH_H

#include <c_arena.h>
#include <c_array.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace autodiff {

namespace s = std;

enum class Op { Add, Sub, Neg, Hadamard, Dot, Div, Sum, Max, Min, Mean,
                Softmax, CrossEntropy, SoftmaxCrossEntropy, Where,
                Reshape, Squeeze, Unsqueeze,
                Exp, Log, Sin, Cos, Tan, Sqrt, Abs, Pow };

// ---------------------------------------------------------------------------
//  Node<T>: base for all graph nodes.
//  Inherits NodeBase (identity + NodeKind tag + polymorphic lifetime) and
//  CArray<T> (the value buffer), and adds a gradient buffer of the same shape.
//  Constructors are protected; use VNode, CNode, or ONode. Every node is a
//  heap object owned by the CArena passed in (via arena.adopt); it is never
//  copied or moved, and the arena — declared first in a scope — must outlive
//  every CArray<T> handle that still points at it.
// ---------------------------------------------------------------------------
template <typename T>
struct Node : NodeBase, CArray<T> {
  CArray<T> mGrad;
  s::string mName;

  const CArray<T>& grad() const noexcept { return mGrad; }
  CArray<T>&       grad()       noexcept { return mGrad; }
  const s::string& name() const noexcept { return mName; }

  void zero_grad() override {
    T* p = mGrad.data();
    if (p) s::fill_n(p, mGrad.size(), T{});
  }

protected:
  Node(NodeKind k, CArena& arena, const Shape& shape, T val = T{})
    : NodeBase(k), CArray<T>(arena, shape, val), mGrad(arena, shape, T{}) {}

  // Adopts a pre-computed result (used by ONode and the aliasing leaf ctors).
  // The CArray<T> base is move-constructed first, so shape() is valid when
  // mGrad is built.
  Node(NodeKind k, CArena& arena, CArray<T>&& data)
    : NodeBase(k), CArray<T>(s::move(data)), mGrad(arena, CArray<T>::shape(), T{}) {}
};

// One input-gradient contribution produced by ONode<T>::backward: the input node
// to route it to, plus a freshly-allocated gradient buffer the caller owns.
// Defined out of line in dreverse_ad.h along with the backward() members below.
template <typename T>
using GradList = s::vector<s::pair<Node<T>*, CArray<T>>>;

// ---------------------------------------------------------------------------
//  VNode<T>: variable node — participates in gradient computation.
// ---------------------------------------------------------------------------
template <typename T>
struct VNode : Node<T> {
  VNode(CArena& arena, const Shape& shape, T val = T{})
    : Node<T>(NodeKind::Variable, arena, shape, val) {
    this->set_requires_grad(true);
    this->mName = "var_" + s::to_string(arena.note_vnode());
  }

  VNode(CArena& arena, const Shape& shape, s::string_view name, T val = T{})
    : Node<T>(NodeKind::Variable, arena, shape, val) {
    this->set_requires_grad(true);
    arena.note_vnode();
    this->mName = name;
  }

  // Promote an existing array to a variable leaf, aliasing its value buffer.
  VNode(CArena& arena, const CArray<T>& src)
    : Node<T>(NodeKind::Variable, arena, CArray<T>(src)) {
    assert(src.arena() == &arena);
    this->set_requires_grad(true);
    this->mName = "var_" + s::to_string(arena.note_vnode());
  }

  // Reverse pass: store the incoming adjoint as this variable's gradient.
  // Defined in dreverse_ad.h.
  void backward(const CArray<T>& adjoint);
};

// ---------------------------------------------------------------------------
//  CNode<T>: constant node — treated as a fixed leaf with no gradient.
// ---------------------------------------------------------------------------
template <typename T>
struct CNode : Node<T> {
  CNode(CArena& arena, const Shape& shape, T val = T{})
    : Node<T>(NodeKind::Constant, arena, shape, val) {
    this->mName = "const_" + s::to_string(arena.note_cnode());
  }

  CNode(CArena& arena, const Shape& shape, s::string_view name, T val = T{})
    : Node<T>(NodeKind::Constant, arena, shape, val) {
    arena.note_cnode();
    this->mName = name;
  }

  // Promote an existing array to a constant leaf, aliasing its value buffer.
  CNode(CArena& arena, const CArray<T>& src)
    : Node<T>(NodeKind::Constant, arena, CArray<T>(src)) {
    assert(src.arena() == &arena);
    this->set_requires_grad(false);
    this->mName = "const_" + s::to_string(arena.note_cnode());
  }

  // Reverse pass: a constant has no gradient, so the adjoint is discarded.
  // Defined in dreverse_ad.h.
  void backward(const CArray<T>& adjoint);
};

// ---------------------------------------------------------------------------
//  ONode<T>: operation node — holds the result, links to its input nodes.
//  mRight is null for unary operations (Neg, Exp, Log, Sin, Softmax, most
//  reductions, ...); for CrossEntropy/SoftmaxCrossEntropy it is the (constant,
//  non-differentiable) integer class-label operand. mCond is null except for
//  Where, where mLeft/mRight are the two value operands a/b and mCond is the
//  (constant, non-differentiable) 0/1 selector.
//  mAxis is the resolved non-negative axis for the axis-wise ops (Sum/Max/Min/
//  Mean/Softmax/CrossEntropy/SoftmaxCrossEntropy), or -1 for a whole-array
//  reduction and every non-axis op. Built by the graph_* factories into the
//  operands' arena; the input nodes it points at are arena-owned too, so they
//  outlive it.
// ---------------------------------------------------------------------------
template <typename T>
struct ONode : Node<T> {
  Op       mOp;
  Node<T>* mLeft;
  Node<T>* mRight;
  int64_t  mAxis;
  Node<T>* mCond;

  ONode(CArena& arena, Op op, CArray<T>&& result,
        Node<T>* left, Node<T>* right = nullptr, int64_t axis = -1,
        Node<T>* cond = nullptr)
    : Node<T>(NodeKind::Operation, arena, s::move(result))
    , mOp(op), mLeft(left), mRight(right), mAxis(axis), mCond(cond)
  {
    switch (op) {
      case Op::Add:      arena.note_onode_add();      break;
      case Op::Sub:      arena.note_onode_sub();      break;
      case Op::Neg:      arena.note_onode_neg();      break;
      case Op::Hadamard: arena.note_onode_hadamard(); break;
      case Op::Dot:      arena.note_onode_dot();      break;
      case Op::Div:      arena.note_onode_div();      break;
      case Op::Sum:      arena.note_onode_sum();      break;
      case Op::Max:      arena.note_onode_max();      break;
      case Op::Min:      arena.note_onode_min();      break;
      case Op::Mean:     arena.note_onode_mean();     break;
      case Op::Softmax:             arena.note_onode_softmax();               break;
      case Op::CrossEntropy:        arena.note_onode_cross_entropy();         break;
      case Op::SoftmaxCrossEntropy: arena.note_onode_softmax_cross_entropy(); break;
      case Op::Where:               arena.note_onode_where();                 break;
      case Op::Reshape:   arena.note_onode_reshape();   break;
      case Op::Squeeze:   arena.note_onode_squeeze();   break;
      case Op::Unsqueeze: arena.note_onode_unsqueeze(); break;
      case Op::Exp:      arena.note_onode_exp();      break;
      case Op::Log:      arena.note_onode_log();      break;
      case Op::Sin:      arena.note_onode_sin();      break;
      case Op::Cos:      arena.note_onode_cos();      break;
      case Op::Tan:      arena.note_onode_tan();      break;
      case Op::Sqrt:     arena.note_onode_sqrt();     break;
      case Op::Abs:      arena.note_onode_abs();      break;
      case Op::Pow:      arena.note_onode_pow();      break;
    }
    this->set_requires_grad((mLeft  && mLeft->requires_grad()) ||
                            (mRight && mRight->requires_grad()));
  }

  Op       op()    const noexcept { return mOp; }
  Node<T>* left()  const noexcept { return mLeft; }
  Node<T>* right() const noexcept { return mRight; }
  int64_t  axis()  const noexcept { return mAxis; }
  Node<T>* cond()  const noexcept { return mCond; }

  // Reverse pass: given the adjoint flowing in from downstream, return one
  // (input node, gradient) pair per differentiable input. No entry is produced
  // for mCond, nor for mRight of CrossEntropy/SoftmaxCrossEntropy (class
  // labels). Every returned buffer is freshly allocated. Defined in
  // dreverse_ad.h — one gradient rule per Op.
  GradList<T> backward(const CArray<T>& adjoint);
};

// ---------------------------------------------------------------------------
//  Internal loop helpers
// ---------------------------------------------------------------------------
namespace detail {

// Element-wise binary op. One operand may be a single value (size 1); it is
// broadcast against the other's shape. Any other shape mismatch asserts.
template <typename T, typename BinOp>
ONode<T>& make_binary(CArena& arena, Op op, Node<T>* a, Node<T>* b, BinOp f) {
  assert(a->arena() == b->arena());
  const bool as = a->size() == 1, bs = b->size() == 1;
  assert(as or bs or a->shape() == b->shape());
  CArray<T> result(arena, as ? b->shape() : a->shape());
  const T* pa = a->data();
  const T* pb = b->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = result.size(); i < n; ++i)
    pr[i] = f(pa[as ? 0 : i], pb[bs ? 0 : i]);
  return arena.template adopt<ONode<T>>(arena, op, s::move(result), a, b);
}

template <typename T, typename UnOp>
ONode<T>& make_unary(CArena& arena, Op op, Node<T>* a, UnOp f) {
  CArray<T> result(arena, a->shape());
  const T* pa = a->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = a->size(); i < n; ++i)
    pr[i] = f(pa[i]);
  return arena.template adopt<ONode<T>>(arena, op, s::move(result), a, nullptr);
}

// Matrix multiply / dot for rank-1 and rank-2 operands, numpy-matmul shape
// rules: a rank-1 left operand is treated as a row (1,k) and its leading axis
// dropped from the result; a rank-1 right operand is a column (k,1) and its
// trailing axis dropped; a 1-D * 1-D contraction yields a single value {1}.
// The product is always computed as the (m,k)*(k,n) triple loop in row-major
// flat indices, valid for every case because out.product() == m*n.
template <typename T>
ONode<T>& make_dot(CArena& arena, Node<T>* a, Node<T>* b) {
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
  return arena.template adopt<ONode<T>>(arena, Op::Dot, s::move(result), a, b);
}

// Split a row-major layout for an axis-wise op. When `axis` is absent the whole
// array is one group; otherwise `axis` (negative allowed) resolves to `ax`. A
// group has `axlen` elements, there are `outer * inner` of them, and element k of
// group (o, j) lives at flat index (o * axlen + k) * inner + j.
struct AxisExtents { int64_t outer, axlen, inner, ax; };  // ax = resolved axis, or -1

inline AxisExtents axis_extents(const Shape& in, s::optional<int64_t> axis) {
  const int64_t rank = static_cast<int64_t>(in.rank());
  if (not axis) return {1, in.product(), 1, -1};
  int64_t ax = *axis;
  if (ax < 0) ax += rank;
  assert(ax >= 0 and ax < rank);
  int64_t outer = 1, inner = 1;
  for (int64_t d = 0;      d < ax;   ++d) outer *= in[d];
  for (int64_t d = ax + 1; d < rank; ++d) inner *= in[d];
  return {outer, in[ax], inner, ax};
}

// Result shape of an axis-wise reduction: `in` with axis `ax` removed, or
// Shape{1} for a whole-array reduction (ax < 0) or when removal empties it.
inline Shape reduced_shape(const Shape& in, int64_t ax) {
  if (ax < 0) return Shape{1};
  s::vector<int64_t> od;
  for (int64_t d = 0, rank = static_cast<int64_t>(in.rank()); d < rank; ++d)
    if (d != ax) od.push_back(in[d]);
  if (od.empty()) od.push_back(1);
  return Shape{s::move(od)};
}

// Reduce `a` along one axis, or over every element when `axis` is absent. Each
// group's accumulator is seeded with its first element and folded with f(acc, x)
// over the rest, so one fold serves sum (plus), max and min. When `average` is
// set the fold result is divided by the group size — a plus-fold plus `average`
// is the mean. Output shape = reduced_shape(shape, ax). The resolved axis is
// stored on the ONode: a reverse pass broadcasts the Sum grad back along it (the
// Mean grad likewise, scaled by 1/count), and routes the Max/Min grad to each
// group's arg-extreme element (recomputed from mLeft).
template <typename T, typename Fold>
ONode<T>& make_reduce(CArena& arena, Op op, Node<T>* a,
                      s::optional<int64_t> axis, Fold f, bool average = false) {
  assert(a->size() >= 1);
  const auto [outer, axlen, inner, ax] = axis_extents(a->shape(), axis);
  const T* pin = a->data();
  CArray<T> result(arena, reduced_shape(a->shape(), ax));
  T* pr = result.data();
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j){
      const int64_t base = o * axlen * inner + j;
      T acc = pin[base];
      for (int64_t k = 1; k < axlen; ++k)
        acc = f(acc, pin[base + k * inner]);
      pr[o * inner + j] = average ? acc / static_cast<T>(axlen) : acc;
    }
  return arena.template adopt<ONode<T>>(arena, op, s::move(result), a, nullptr, ax);
}

// softmax along `axis` (whole array when absent): exp(x - m) / sum(exp(x - m))
// per group, with m the group max for numerical stability. Shape-preserving. The
// resolved axis is stored on the ONode (the reverse-pass softmax Jacobian needs
// it).
template <typename T>
ONode<T>& make_softmax(CArena& arena, Node<T>* a, s::optional<int64_t> axis) {
  assert(a->size() >= 1);
  const auto [outer, axlen, inner, ax] = axis_extents(a->shape(), axis);
  const T* pin = a->data();
  CArray<T> result(arena, a->shape());
  T* pr = result.data();
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j){
      const int64_t base = o * axlen * inner + j;
      T m = pin[base];
      for (int64_t k = 1; k < axlen; ++k) m = s::max(m, pin[base + k * inner]);
      T sum{};
      for (int64_t k = 0; k < axlen; ++k){
        const T e = s::exp(pin[base + k * inner] - m);
        pr[base + k * inner] = e;
        sum += e;
      }
      for (int64_t k = 0; k < axlen; ++k) pr[base + k * inner] /= sum;
    }
  return arena.template adopt<ONode<T>>(arena, Op::Softmax, s::move(result), a, nullptr, ax);
}

// Categorical cross-entropy of `pred` against integer class labels `target`,
// reduced along `axis` (whole array when absent). `target` carries one class
// index per output group (target->shape() == reduced_shape(pred->shape(), ax)),
// read as T and rounded to the nearest integer. from_logits == false: `pred` is
// a probability distribution and each looked-up probability is clamped to
// [kEps, 1] before the log, so an exact zero gives a large finite loss.
// from_logits == true: `pred` is raw logits and the numerically stable
// logsumexp(logits) - logits[label] form is used (no clamp needed). mLeft =
// pred, mRight = target, mAxis = ax.
template <typename T>
ONode<T>& make_cross_entropy(CArena& arena, Op op, Node<T>* pred,
                             Node<T>* target, s::optional<int64_t> axis,
                             bool from_logits) {
  assert(pred->arena() == target->arena());
  assert(pred->size() >= 1);
  const auto [outer, axlen, inner, ax] = axis_extents(pred->shape(), axis);
  const Shape out = reduced_shape(pred->shape(), ax);
  assert(target->shape() == out);
  const T* pp = pred->data();
  const T* pt = target->data();
  CArray<T> result(arena, out);
  T* pr = result.data();
  constexpr T kEps = static_cast<T>(1e-12);
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j){
      const int64_t base = o * axlen * inner + j;
      const int64_t c = static_cast<int64_t>(s::llround(pt[o * inner + j]));
      assert(c >= 0 and c < axlen);
      if (from_logits){
        T m = pp[base];
        for (int64_t k = 1; k < axlen; ++k) m = s::max(m, pp[base + k * inner]);
        T se{};
        for (int64_t k = 0; k < axlen; ++k) se += s::exp(pp[base + k * inner] - m);
        pr[o * inner + j] = (m + s::log(se)) - pp[base + c * inner];
      } else {
        pr[o * inner + j] = -s::log(s::clamp(pp[base + c * inner], kEps, T{1}));
      }
    }
  return arena.template adopt<ONode<T>>(arena, op, s::move(result), pred, target, ax);
}

// Element-wise select: result[i] = cond[i] != 0 ? a[i] : b[i]. cond, a and b
// share one shape and arena; the result matches. The condition is kept as mCond
// (a constant 0/1 leaf) so a reverse pass can route grad_out to a where the
// condition holds and to b elsewhere; no gradient flows to the condition.
template <typename T>
ONode<T>& make_where(CArena& arena, Node<T>* cond, Node<T>* a, Node<T>* b) {
  assert(cond->arena() == a->arena() and a->arena() == b->arena());
  assert(cond->shape() == a->shape() and a->shape() == b->shape());
  CArray<T> result(arena, a->shape());
  const T* pc = cond->data();
  const T* pa = a->data();
  const T* pb = b->data();
  T* pr = result.data();
  for (s::size_t i = 0, n = result.size(); i < n; ++i)
    pr[i] = (pc[i] != T{}) ? pa[i] : pb[i];
  return arena.template adopt<ONode<T>>(arena, Op::Where, s::move(result),
                                        a, b, -1, cond);
}

// Wrap an aliasing shape-view of `a` (produced by CArray<T>::reshape/squeeze/
// unsqueeze, which Node<T> inherits) in an ONode. No allocation, no elementwise
// loop: the view already shares `a`'s buffer, just under a different Shape.
template <typename T>
ONode<T>& make_reshape(CArena& arena, Op op, Node<T>* a, CArray<T>&& view) {
  return arena.template adopt<ONode<T>>(arena, op, s::move(view), a, nullptr);
}

} // detail

// ---------------------------------------------------------------------------
//  Operations — build an ONode into the operands' arena and return a reference
//  to it. The arena owns every node; nothing here is stack-allocated.
// ---------------------------------------------------------------------------
template <typename T>
ONode<T>& graph_add(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Add, a, b, s::plus<T>{});
}

template <typename T>
ONode<T>& graph_sub(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Sub, a, b, s::minus<T>{});
}

template <typename T>
ONode<T>& graph_neg(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Neg, a, [](T x){ return -x; });
}

// Reductions: axis absent -> reduce every element to Shape{1}; axis given (may be
// negative) -> that axis is removed from the shape. See detail::make_reduce.
template <typename T>
ONode<T>& graph_sum(Node<T>* a, s::optional<int64_t> axis = s::nullopt) {
  return detail::make_reduce(*a->arena(), Op::Sum, a, axis, s::plus<T>{});
}

template <typename T>
ONode<T>& graph_mean(Node<T>* a, s::optional<int64_t> axis = s::nullopt) {
  return detail::make_reduce(*a->arena(), Op::Mean, a, axis, s::plus<T>{}, true);
}

template <typename T>
ONode<T>& graph_max(Node<T>* a, s::optional<int64_t> axis = s::nullopt) {
  return detail::make_reduce(*a->arena(), Op::Max, a, axis,
                             [](T acc, T x){ return s::max(acc, x); });
}

template <typename T>
ONode<T>& graph_min(Node<T>* a, s::optional<int64_t> axis = s::nullopt) {
  return detail::make_reduce(*a->arena(), Op::Min, a, axis,
                             [](T acc, T x){ return s::min(acc, x); });
}

// softmax(x) normalizes every element; softmax(x, k) normalizes along axis k.
// Shape-preserving (unlike the reductions). See detail::make_softmax.
template <typename T>
ONode<T>& graph_softmax(Node<T>* a, s::optional<int64_t> axis = s::nullopt) {
  return detail::make_softmax(*a->arena(), a, axis);
}

// Cross-entropy against integer class labels, reduced along `axis` (whole array
// when absent). `pred` is a probability distribution (clamped away from 0 before
// the log); `logits` for graph_softmax_cross_entropy is raw and handled with a
// stable fused logsumexp. `target` holds one class index per output group. See
// detail::make_cross_entropy.
template <typename T>
ONode<T>& graph_cross_entropy(Node<T>* pred, Node<T>* target,
                              s::optional<int64_t> axis = s::nullopt) {
  return detail::make_cross_entropy(*pred->arena(), Op::CrossEntropy,
                                    pred, target, axis, /*from_logits=*/false);
}

template <typename T>
ONode<T>& graph_softmax_cross_entropy(Node<T>* logits, Node<T>* target,
                                      s::optional<int64_t> axis = s::nullopt) {
  return detail::make_cross_entropy(*logits->arena(), Op::SoftmaxCrossEntropy,
                                    logits, target, axis, /*from_logits=*/true);
}

// Element-wise select: a[i] where cond[i] is nonzero, else b[i]. cond, a and b
// share the exact shape, which the result matches. See detail::make_where.
template <typename T>
ONode<T>& graph_where(Node<T>* cond, Node<T>* a, Node<T>* b) {
  return detail::make_where(*a->arena(), cond, a, b);
}

// Shape-changing views, promoted to graph operators so a variable used both in
// its original shape and through a reshaped/squeezed/unsqueezed view elsewhere
// in the same graph still accumulates gradient correctly (see detail::make_reshape).
template <typename T>
ONode<T>& graph_reshape(Node<T>* a, const Shape& shape) {
  return detail::make_reshape(*a->arena(), Op::Reshape, a, a->reshape(shape));
}

template <typename T>
ONode<T>& graph_squeeze(Node<T>* a, int n) {
  return detail::make_reshape(*a->arena(), Op::Squeeze, a, a->squeeze(n));
}

template <typename T>
ONode<T>& graph_unsqueeze(Node<T>* a, int64_t axis) {
  return detail::make_reshape(*a->arena(), Op::Unsqueeze, a, a->unsqueeze(axis));
}

// Element-wise (Hadamard) product; shapes must match (or one be a scalar).
template <typename T>
ONode<T>& graph_hadamard(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Hadamard, a, b, s::multiplies<T>{});
}

// Matrix multiply / dot; see detail::make_dot for the rank-1/rank-2 shape rules.
template <typename T>
ONode<T>& graph_dot(Node<T>* a, Node<T>* b) {
  return detail::make_dot(*a->arena(), a, b);
}

template <typename T>
ONode<T>& graph_div(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Div, a, b, s::divides<T>{});
}

template <typename T>
ONode<T>& graph_exp(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Exp, a, [](T x){ return s::exp(x); });
}

template <typename T>
ONode<T>& graph_log(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Log, a, [](T x){ return s::log(x); });
}

template <typename T>
ONode<T>& graph_sin(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Sin, a, [](T x){ return s::sin(x); });
}

template <typename T>
ONode<T>& graph_cos(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Cos, a, [](T x){ return s::cos(x); });
}

template <typename T>
ONode<T>& graph_tan(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Tan, a, [](T x){ return s::tan(x); });
}

template <typename T>
ONode<T>& graph_sqrt(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Sqrt, a, [](T x){ return s::sqrt(x); });
}

template <typename T>
ONode<T>& graph_abs(Node<T>* a) {
  return detail::make_unary(*a->arena(), Op::Abs, a, [](T x){ return s::abs(x); });
}

template <typename T>
ONode<T>& graph_pow(Node<T>* a, Node<T>* b) {
  return detail::make_binary(*a->arena(), Op::Pow, a, b, [](T x, T y){ return s::pow(x, y); });
}

// ---------------------------------------------------------------------------
//  Promotion layer: let CArray<T> and numeric literals flow into the operators.
//  A CArray promotes to its canonical leaf — VNode if requires_grad(), else
//  CNode — memoized by buffer pointer so the same array used twice maps to one
//  node (correct gradient accumulation). A literal promotes to a fresh scalar
//  CNode in the other operand's arena. An existing Node passes straight through.
// ---------------------------------------------------------------------------
namespace detail {

template <typename X>
concept is_graph_operand =
  requires { typename s::remove_cvref_t<X>::value_type; } and
  s::is_base_of_v<CArray<typename s::remove_cvref_t<X>::value_type>,
                  s::remove_cvref_t<X>>;

// Result element type of a binary operator. With a scalar on one side the tensor
// operand's type wins (the scalar is cast to it — unchanged behaviour). With a
// tensor on both sides the C++ usual-arithmetic-conversion type wins, so
// float & bool -> float, int + double -> double. common_type_t<T,T> == T, so a
// same-type expression is byte-identical to before.
template <typename A, typename B, bool BothOperands>
struct graph_value_impl {
  using type = typename s::remove_cvref_t<
    s::conditional_t<is_graph_operand<A>, A, B>>::value_type;
};
template <typename A, typename B>
struct graph_value_impl<A, B, true> {
  using type = s::common_type_t<typename s::remove_cvref_t<A>::value_type,
                                typename s::remove_cvref_t<B>::value_type>;
};
template <typename A, typename B>
using graph_value_t = typename graph_value_impl<
  A, B, is_graph_operand<A> and is_graph_operand<B>>::type;

template <typename A, typename B>
CArena& pick_arena(const A& a, const B& b) {
  if constexpr (is_graph_operand<A>) {
    if constexpr (is_graph_operand<B>) assert(a.arena() == b.arena());
    assert(a.arena());
    return *a.arena();
  } else {
    assert(b.arena());
    return *b.arena();
  }
}

template <typename T>
Node<T>& promote_scalar(CArena& arena, T v) {
  // A numeric literal is always a constant, even when the arena's
  // auto_requires_grad policy is on — pass the flag explicitly.
  return arena.template adopt<CNode<T>>(arena, CArray<T>(arena, v, false));
}

template <typename T>
Node<T>& promote_leaf(CArena& arena, const CArray<T>& src) {
  assert(src.arena() == &arena);
  const T* key = src.data();
  if (NodeBase* hit = arena.promoted_leaf(key)) {
    Node<T>& ex = static_cast<Node<T>&>(*hit);
    assert(ex.shape() == src.shape() and
           "promoting a differently-shaped view of an already-promoted array");
    return ex;
  }
  Node<T>& n = src.requires_grad()
    ? static_cast<Node<T>&>(arena.template adopt<VNode<T>>(arena, src))
    : static_cast<Node<T>&>(arena.template adopt<CNode<T>>(arena, src));
  arena.register_leaf(key, static_cast<NodeBase*>(&n));
  return n;
}

// Element type differs from the operator's result type: snapshot `src` into a
// fresh CArray<TR> (element-wise static_cast) and promote that as a constant
// leaf. An implicit cast is not a differentiable edge in this model — a
// gradient does not flow back to `src` (a real Op::Cast node is a follow-up).
template <typename TR, typename TS>
Node<TR>& convert_leaf(CArena& arena, const CArray<TS>& src) {
  assert(src.arena() == &arena);
  CArray<TR> conv(arena, src.shape());
  const TS* ps = src.data();
  TR* pc = conv.data();
  for (s::size_t i = 0, n = src.size(); i < n; ++i)
    pc[i] = static_cast<TR>(ps[i]);
  return static_cast<Node<TR>&>(arena.adopt<CNode<TR>>(arena, conv));
}

template <typename T, typename X>
Node<T>& to_node(CArena& arena, X&& x) {
  using U = s::remove_cvref_t<X>;
  if constexpr (s::is_base_of_v<Node<T>, U>) {
    static_assert(s::is_lvalue_reference_v<X>,
                  "a temporary graph node operand would dangle — name it or pass a CArray");
    return x;
  } else if constexpr (is_graph_operand<X>) {
    using TS = typename U::value_type;
    if constexpr (s::is_same_v<TS, T>)
      return promote_leaf<T>(arena, x);
    else
      return convert_leaf<T>(arena, static_cast<const CArray<TS>&>(x));
  } else {
    return promote_scalar<T>(arena, static_cast<T>(x));
  }
}

} // detail

// ---------------------------------------------------------------------------
//  Operator overloads. Each operand may be a Node, a CArray, or a numeric
//  literal (in either position), and is promoted as described above. Results
//  are arena-owned references, so chains need no named intermediates:
//      auto& y = w & x + 1.0;
//  `*` is element-wise, `&` is matrix multiply / dot, `^` is pow. C++ gives
//  `&` and `^` lower precedence than `+ - * /`, so mixed expressions still need
//  parentheses: write `(a & b) + c` and `(x ^ n) * c`.
// ---------------------------------------------------------------------------
#define AUTODIFF_GRAPH_BINOP(SYM, FN)                                             \
  template <typename A, typename B>                                               \
    requires (detail::is_graph_operand<A> or detail::is_graph_operand<B>)         \
  auto& operator SYM(A&& a, B&& b) {                                              \
    using T = detail::graph_value_t<A, B>;                                        \
    CArena& ar = detail::pick_arena(a, b);                                        \
    return FN(&detail::to_node<T>(ar, s::forward<A>(a)),                          \
              &detail::to_node<T>(ar, s::forward<B>(b)));                         \
  }

AUTODIFF_GRAPH_BINOP(+, graph_add)
AUTODIFF_GRAPH_BINOP(-, graph_sub)
AUTODIFF_GRAPH_BINOP(*, graph_hadamard)
AUTODIFF_GRAPH_BINOP(/, graph_div)
AUTODIFF_GRAPH_BINOP(^, graph_pow)
AUTODIFF_GRAPH_BINOP(&, graph_dot)

#undef AUTODIFF_GRAPH_BINOP

template <typename A>
  requires detail::is_graph_operand<A>
auto& operator-(A&& a) {
  using T = typename s::remove_cvref_t<A>::value_type;
  return graph_neg(&detail::to_node<T>(*a.arena(), s::forward<A>(a)));
}

// ---------------------------------------------------------------------------
//  Unary math on a Node or CArray operand (ADL picks these over std::exp etc.
//  for graph operands; scalar arguments fall through to <cmath>).
// ---------------------------------------------------------------------------
#define AUTODIFF_GRAPH_UNARY(NAME, FN)                                            \
  template <typename A>                                                           \
    requires detail::is_graph_operand<A>                                          \
  auto& NAME(A&& a) {                                                             \
    using T = typename s::remove_cvref_t<A>::value_type;                          \
    return FN(&detail::to_node<T>(*a.arena(), s::forward<A>(a)));                  \
  }

AUTODIFF_GRAPH_UNARY(exp,  graph_exp)
AUTODIFF_GRAPH_UNARY(log,  graph_log)
AUTODIFF_GRAPH_UNARY(sin,  graph_sin)
AUTODIFF_GRAPH_UNARY(cos,  graph_cos)
AUTODIFF_GRAPH_UNARY(tan,  graph_tan)
AUTODIFF_GRAPH_UNARY(sqrt, graph_sqrt)
AUTODIFF_GRAPH_UNARY(abs,  graph_abs)

#undef AUTODIFF_GRAPH_UNARY

// pow as a free function, avoiding operator^'s loose precedence.
template <typename A, typename B>
  requires (detail::is_graph_operand<A> or detail::is_graph_operand<B>)
auto& pow(A&& a, B&& b) {
  using T = detail::graph_value_t<A, B>;
  CArena& ar = detail::pick_arena(a, b);
  return graph_pow(&detail::to_node<T>(ar, s::forward<A>(a)),
                   &detail::to_node<T>(ar, s::forward<B>(b)));
}

// ---------------------------------------------------------------------------
//  Reductions on a Node or CArray operand: sum(x) / mean(x) / max(x) / min(x)
//  fold every element; sum(x, k) / mean(x, k) / max(x, k) / min(x, k) fold axis k
//  (negative allowed). Resolved by ADL for graph operands — no clash with
//  std::max/std::min (the 1-arg call and the optional<int64_t> 2nd arg match no
//  std overload); use graph_max / graph_min if `using namespace std;` is also in
//  effect.
// ---------------------------------------------------------------------------
#define AUTODIFF_GRAPH_REDUCE(NAME, FN)                                           \
  template <typename A>                                                           \
    requires detail::is_graph_operand<A>                                          \
  auto& NAME(A&& a, s::optional<int64_t> axis = s::nullopt) {                     \
    using T = typename s::remove_cvref_t<A>::value_type;                          \
    return FN(&detail::to_node<T>(*a.arena(), s::forward<A>(a)), axis);           \
  }

AUTODIFF_GRAPH_REDUCE(sum,  graph_sum)
AUTODIFF_GRAPH_REDUCE(mean, graph_mean)
AUTODIFF_GRAPH_REDUCE(max,  graph_max)
AUTODIFF_GRAPH_REDUCE(min,  graph_min)

#undef AUTODIFF_GRAPH_REDUCE

// ---------------------------------------------------------------------------
//  Neural-net ops on a Node or CArray operand, ADL-resolved like the reductions:
//    softmax(x)       / softmax(x, k)                — normalize, shape-preserving
//    cross_entropy(p, y)         / cross_entropy(p, y, k)
//    softmax_cross_entropy(z, y) / softmax_cross_entropy(z, y, k)
//  `y` is an integer class-label tensor shaped like the axis-reduced result.
// ---------------------------------------------------------------------------
template <typename A>
  requires detail::is_graph_operand<A>
auto& softmax(A&& a, s::optional<int64_t> axis = s::nullopt) {
  using T = typename s::remove_cvref_t<A>::value_type;
  return graph_softmax(&detail::to_node<T>(*a.arena(), s::forward<A>(a)), axis);
}

#define AUTODIFF_GRAPH_BINOP_AXIS(NAME, FN)                                       \
  template <typename A, typename B>                                               \
    requires (detail::is_graph_operand<A> or detail::is_graph_operand<B>)         \
  auto& NAME(A&& a, B&& b, s::optional<int64_t> axis = s::nullopt) {              \
    using T = detail::graph_value_t<A, B>;                                        \
    CArena& ar = detail::pick_arena(a, b);                                        \
    return FN(&detail::to_node<T>(ar, s::forward<A>(a)),                          \
             &detail::to_node<T>(ar, s::forward<B>(b)), axis);                    \
  }

AUTODIFF_GRAPH_BINOP_AXIS(cross_entropy,         graph_cross_entropy)
AUTODIFF_GRAPH_BINOP_AXIS(softmax_cross_entropy, graph_softmax_cross_entropy)

#undef AUTODIFF_GRAPH_BINOP_AXIS

// where(cond, a, b): element-wise select — a[i] where cond[i] is nonzero, else
// b[i]. cond is typically a CArray<bool> (converted to a 0/1 constant leaf that
// receives no gradient); a and b must share the exact shape, which the result
// matches. Named function only (no ternary operator); ADL-resolved for graph
// operands, no std::where clash. Bare scalars for a/b do not match.
template <typename C, typename A, typename B>
  requires (detail::is_graph_operand<C> and
            detail::is_graph_operand<A> and detail::is_graph_operand<B>)
auto& where(C&& cond, A&& a, B&& b) {
  using T = detail::graph_value_t<A, B>;
  CArena& ar = detail::pick_arena(a, b);
  return graph_where(&detail::to_node<T>(ar, s::forward<C>(cond)),
                     &detail::to_node<T>(ar, s::forward<A>(a)),
                     &detail::to_node<T>(ar, s::forward<B>(b)));
}

// reshape(x, shape) / squeeze(x, n) / unsqueeze(x, axis): shape-changing views
// of a Node or CArray operand, ADL-resolved like the reductions. See
// graph_reshape/graph_squeeze/graph_unsqueeze and detail::make_reshape.
template <typename A>
  requires detail::is_graph_operand<A>
auto& reshape(A&& a, const Shape& shape) {
  using T = typename s::remove_cvref_t<A>::value_type;
  return graph_reshape(&detail::to_node<T>(*a.arena(), s::forward<A>(a)), shape);
}

template <typename A>
  requires detail::is_graph_operand<A>
auto& squeeze(A&& a, int n) {
  using T = typename s::remove_cvref_t<A>::value_type;
  return graph_squeeze(&detail::to_node<T>(*a.arena(), s::forward<A>(a)), n);
}

template <typename A>
  requires detail::is_graph_operand<A>
auto& unsqueeze(A&& a, int64_t axis) {
  using T = typename s::remove_cvref_t<A>::value_type;
  return graph_unsqueeze(&detail::to_node<T>(*a.arena(), s::forward<A>(a)), axis);
}

// ---------------------------------------------------------------------------
//  Explicit leaf factories and gradient access.
// ---------------------------------------------------------------------------

// Promote x to a variable leaf now (aliasing its buffer) and return the node,
// so its .grad() / .name() are reachable before the expression is built. x must
// not have been used in the graph yet.
template <typename T>
VNode<T>& graph_variable(CArray<T>& x) {
  assert(x.arena());
  assert(x.arena()->promoted_leaf(x.data()) == nullptr and
         "graph_variable: array already promoted");
  x.set_requires_grad(true);
  return static_cast<VNode<T>&>(detail::promote_leaf(*x.arena(), x));
}

template <typename T>
CNode<T>& graph_constant(CArray<T>& x) {
  assert(x.arena());
  assert(x.arena()->promoted_leaf(x.data()) == nullptr and
         "graph_constant: array already promoted");
  x.set_requires_grad(false);
  return static_cast<CNode<T>&>(detail::promote_leaf(*x.arena(), x));
}

// Standalone leaf with its own fresh buffer (not tied to a CArray handle).
template <typename T>
VNode<T>& graph_variable(CArena& arena, const Shape& shape, T val = T{}) {
  return arena.template adopt<VNode<T>>(arena, shape, val);
}

template <typename T>
CNode<T>& graph_constant(CArena& arena, const Shape& shape, T val = T{}) {
  return arena.template adopt<CNode<T>>(arena, shape, val);
}

// Gradient buffer of the leaf x was promoted to. Only valid after x has flowed
// through an operator (or graph_variable/graph_constant); asserts otherwise.
template <typename T>
const CArray<T>& grad_of(const CArray<T>& x) {
  assert(x.arena());
  NodeBase* nb = x.arena()->promoted_leaf(x.data());
  assert(nb and "grad_of: array was never used in a graph");
  return static_cast<const Node<T>&>(*nb).grad();
}

// Zero every gradient buffer in the arena's tape (all element types).
inline void zero_grad(CArena& arena) {
  for (int64_t i = 0, n = arena.node_count(); i < n; ++i)
    arena.node_at(i).zero_grad();
}

} // autodiff

#endif // C_GRAPH_H
