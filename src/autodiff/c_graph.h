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
                Exp, Log, Sin, Cos, Tan, Sqrt, Abs, Pow };

// ---------------------------------------------------------------------------
//  Node<T>: base for all graph nodes.
//  Inherits NodeBase (identity + NodeKind tag + polymorphic lifetime) and
//  CArray<T> (the value buffer), and adds a gradient buffer of the same shape.
//  Constructors are protected; use VNode, CNode, or ONode. Every node is a
//  heap object owned by the CArena<T> passed in (via arena.adopt); it is never
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
  Node(NodeKind k, CArena<T>& arena, const Shape& shape, T val = T{})
    : NodeBase(k), CArray<T>(arena, shape, val), mGrad(arena, shape, T{}) {}

  // Adopts a pre-computed result (used by ONode and the aliasing leaf ctors).
  // The CArray<T> base is move-constructed first, so shape() is valid when
  // mGrad is built.
  Node(NodeKind k, CArena<T>& arena, CArray<T>&& data)
    : NodeBase(k), CArray<T>(s::move(data)), mGrad(arena, CArray<T>::shape(), T{}) {}
};

// ---------------------------------------------------------------------------
//  VNode<T>: variable node — participates in gradient computation.
// ---------------------------------------------------------------------------
template <typename T>
struct VNode : Node<T> {
  VNode(CArena<T>& arena, const Shape& shape, T val = T{})
    : Node<T>(NodeKind::Variable, arena, shape, val) {
    this->set_requires_grad(true);
    this->mName = "var_" + s::to_string(arena.note_vnode());
  }

  VNode(CArena<T>& arena, const Shape& shape, s::string_view name, T val = T{})
    : Node<T>(NodeKind::Variable, arena, shape, val) {
    this->set_requires_grad(true);
    arena.note_vnode();
    this->mName = name;
  }

  // Promote an existing array to a variable leaf, aliasing its value buffer.
  VNode(CArena<T>& arena, const CArray<T>& src)
    : Node<T>(NodeKind::Variable, arena, CArray<T>(src)) {
    assert(src.arena() == &arena);
    this->set_requires_grad(true);
    this->mName = "var_" + s::to_string(arena.note_vnode());
  }
};

// ---------------------------------------------------------------------------
//  CNode<T>: constant node — treated as a fixed leaf with no gradient.
// ---------------------------------------------------------------------------
template <typename T>
struct CNode : Node<T> {
  CNode(CArena<T>& arena, const Shape& shape, T val = T{})
    : Node<T>(NodeKind::Constant, arena, shape, val) {
    this->mName = "const_" + s::to_string(arena.note_cnode());
  }

  CNode(CArena<T>& arena, const Shape& shape, s::string_view name, T val = T{})
    : Node<T>(NodeKind::Constant, arena, shape, val) {
    arena.note_cnode();
    this->mName = name;
  }

  // Promote an existing array to a constant leaf, aliasing its value buffer.
  CNode(CArena<T>& arena, const CArray<T>& src)
    : Node<T>(NodeKind::Constant, arena, CArray<T>(src)) {
    assert(src.arena() == &arena);
    this->set_requires_grad(false);
    this->mName = "const_" + s::to_string(arena.note_cnode());
  }
};

// ---------------------------------------------------------------------------
//  ONode<T>: operation node — holds the result, links to its input nodes.
//  mRight is null for unary operations (Neg, Exp, Log, Sin, reductions, ...).
//  mAxis is the reduced axis for Sum/Max/Min/Mean (a resolved non-negative
//  index), or -1 for a full reduction and every non-reduction op. Built by the
//  graph_* factories into the operands' arena; the input nodes it points at are
//  arena-owned too, so they outlive it.
// ---------------------------------------------------------------------------
template <typename T>
struct ONode : Node<T> {
  Op       mOp;
  Node<T>* mLeft;
  Node<T>* mRight;
  int64_t  mAxis;

  ONode(CArena<T>& arena, Op op, CArray<T>&& result,
        Node<T>* left, Node<T>* right = nullptr, int64_t axis = -1)
    : Node<T>(NodeKind::Operation, arena, s::move(result))
    , mOp(op), mLeft(left), mRight(right), mAxis(axis)
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
};

// ---------------------------------------------------------------------------
//  Internal loop helpers
// ---------------------------------------------------------------------------
namespace detail {

// Element-wise binary op. One operand may be a single value (size 1); it is
// broadcast against the other's shape. Any other shape mismatch asserts.
template <typename T, typename BinOp>
ONode<T>& make_binary(CArena<T>& arena, Op op, Node<T>* a, Node<T>* b, BinOp f) {
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
ONode<T>& make_unary(CArena<T>& arena, Op op, Node<T>* a, UnOp f) {
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
ONode<T>& make_dot(CArena<T>& arena, Node<T>* a, Node<T>* b) {
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

// Reduce `a` along one axis, or over every element when `axis` is absent. The
// accumulator is seeded with the first element of each group and folded with
// f(acc, x) over the rest, so one fold serves sum (plus), max, and min. When
// `average` is set the fold result is divided by the group size before it is
// stored — a plus-fold plus `average` is the mean. Output shape = input shape
// with `axis` removed; a full reduction — or removing the last remaining axis —
// yields Shape{1}. The resolved axis is stored on the ONode: a reverse pass
// broadcasts the Sum grad back along it (the Mean grad likewise, scaled by
// 1/count), and routes the Max/Min grad to each group's arg-extreme element
// (recomputed from mLeft).
template <typename T, typename Fold>
ONode<T>& make_reduce(CArena<T>& arena, Op op, Node<T>* a,
                      s::optional<int64_t> axis, Fold f, bool average = false) {
  assert(a->size() >= 1);
  const Shape& in = a->shape();
  const int64_t rank = static_cast<int64_t>(in.rank());
  const T* pin = a->data();

  if (not axis) {
    T acc = pin[0];
    for (s::size_t i = 1, n = a->size(); i < n; ++i)
      acc = f(acc, pin[i]);
    CArray<T> result(arena, Shape{1});
    result.data()[0] = average ? acc / static_cast<T>(a->size()) : acc;
    return arena.template adopt<ONode<T>>(arena, op, s::move(result), a, nullptr, -1);
  }

  int64_t ax = *axis;
  if (ax < 0) ax += rank;
  assert(ax >= 0 and ax < rank);

  int64_t outer = 1, inner = 1;
  for (int64_t d = 0;      d < ax;   ++d) outer *= in[d];
  for (int64_t d = ax + 1; d < rank; ++d) inner *= in[d];
  const int64_t axlen = in[ax];

  s::vector<int64_t> od;
  for (int64_t d = 0; d < rank; ++d)
    if (d != ax) od.push_back(in[d]);
  if (od.empty()) od.push_back(1);
  CArray<T> result(arena, Shape{s::move(od)});
  T* pr = result.data();

  for (int64_t o = 0; o < outer; ++o)
    for (int64_t j = 0; j < inner; ++j){
      T acc = pin[(o * axlen) * inner + j];
      for (int64_t k = 1; k < axlen; ++k)
        acc = f(acc, pin[(o * axlen + k) * inner + j]);
      pr[o * inner + j] = average ? acc / static_cast<T>(axlen) : acc;
    }
  return arena.template adopt<ONode<T>>(arena, op, s::move(result), a, nullptr, ax);
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

template <typename A, typename B>
using graph_value_t = typename s::remove_cvref_t<
  s::conditional_t<is_graph_operand<A>, A, B>>::value_type;

template <typename T, typename A, typename B>
CArena<T>& pick_arena(const A& a, const B& b) {
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
Node<T>& promote_scalar(CArena<T>& arena, T v) {
  // A numeric literal is always a constant, even when the arena's
  // auto_requires_grad policy is on — pass the flag explicitly.
  return arena.template adopt<CNode<T>>(arena, CArray<T>(arena, v, false));
}

template <typename T>
Node<T>& promote_leaf(CArena<T>& arena, const CArray<T>& src) {
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

template <typename T, typename X>
Node<T>& to_node(CArena<T>& arena, X&& x) {
  using U = s::remove_cvref_t<X>;
  if constexpr (s::is_base_of_v<Node<T>, U>) {
    static_assert(s::is_lvalue_reference_v<X>,
                  "a temporary graph node operand would dangle — name it or pass a CArray");
    return x;
  } else if constexpr (is_graph_operand<X>) {
    return promote_leaf<T>(arena, x);
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
#define AUTODIFF_GRAPH_BINOP(SYM, FN)                                              \
  template <typename A, typename B>                                               \
    requires (detail::is_graph_operand<A> or detail::is_graph_operand<B>)         \
  auto& operator SYM(A&& a, B&& b) {                                              \
    using T = detail::graph_value_t<A, B>;                                        \
    if constexpr (detail::is_graph_operand<A> and detail::is_graph_operand<B>)    \
      static_assert(s::is_same_v<typename s::remove_cvref_t<A>::value_type,       \
                                 typename s::remove_cvref_t<B>::value_type>,      \
                    "graph operands must share value_type");                      \
    CArena<T>& ar = detail::pick_arena<T>(a, b);                                  \
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
  CArena<T>& ar = detail::pick_arena<T>(a, b);
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
VNode<T>& graph_variable(CArena<T>& arena, const Shape& shape, T val = T{}) {
  return arena.template adopt<VNode<T>>(arena, shape, val);
}

template <typename T>
CNode<T>& graph_constant(CArena<T>& arena, const Shape& shape, T val = T{}) {
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

// Zero every gradient buffer in the arena's tape.
template <typename T>
void zero_grad(CArena<T>& arena) {
  for (int64_t i = 0, n = arena.node_count(); i < n; ++i)
    arena.node_at(i).zero_grad();
}

} // autodiff

#endif // C_GRAPH_H
