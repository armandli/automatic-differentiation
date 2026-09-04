#ifndef DREVERSE_AD
#define DREVERSE_AD

#include <c_graph.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Reverse-mode automatic differentiation over the define-by-run graph built in
// c_graph.h. Each node type carries a backward() member that maps the adjoint
// (the gradient flowing in from downstream) to gradients for its inputs:
//
//   VNode<T>::backward(adjoint)  -> void         stores adjoint as this->grad()
//   CNode<T>::backward(adjoint)  -> void         discards adjoint (no gradient)
//   ONode<T>::backward(adjoint)  -> GradList<T>  one (input, gradient) pair per
//                                                differentiable input
//
// autodiff::backward(root) drives the whole pass: it walks the graph rooted at
// `root` in reverse topological order, accumulates each node's adjoint, and
// routes ONode's returned gradients into its inputs. After it returns, every
// VNode reachable from `root` has its grad() populated (see grad_of).

namespace autodiff {

namespace s = std;

namespace detail {

// Sum `g` back down to `target` when a binary operand was broadcast up from a
// single element; otherwise `g` already matches `target` and is copied through.
// detail::make_binary only ever broadcasts a whole size-1 operand, so the sole
// shrink case is target.size() == 1 — the assert pins that assumption so a later
// generalisation of broadcasting fails loudly here instead of silently.
template <typename T>
CArray<T> unbroadcast(CArena& arena, const CArray<T>& g, const Shape& target) {
  if (g.shape() == target)
    return g.clone();
  assert(target.size() == 1 and
         "unbroadcast: forward pass only produces whole size-1 broadcasting");
  CArray<T> out(arena, target, T{});
  const T* pg = g.data();
  T acc{};
  for (s::size_t i = 0, n = g.size(); i < n; ++i)
    acc += pg[i];
  out.data()[0] = acc;
  return out;
}

// Post-order depth-first walk of the nodes feeding `n`, following the typed
// mLeft/mRight/mCond pointers (never the untyped arena tape, which may mix
// element types). A node is appended only after all of its inputs, so iterating
// `order` in reverse visits each node only after every node that consumes it —
// the ordering a reverse pass needs.
template <typename T>
void rev_collect(Node<T>& n, s::unordered_set<Node<T>*>& seen,
                 s::vector<Node<T>*>& order) {
  if (not seen.insert(&n).second)
    return;
  if (n.mKind == NodeKind::Operation) {
    ONode<T>& o = static_cast<ONode<T>&>(n);
    if (o.left())  rev_collect(*o.left(),  seen, order);
    if (o.right()) rev_collect(*o.right(), seen, order);
    if (o.cond())  rev_collect(*o.cond(),  seen, order);
  }
  order.push_back(&n);
}

// value of a binary operand at output position i (index 0 if it was a size-1
// operand broadcast against the other).
template <typename T>
T operand_at(const Node<T>& v, s::size_t i) {
  return v.data()[v.size() == 1 ? 0 : i];
}

// Inverse of a permutation: inv[axes[i]] = i. Transposing is a pure reindexing
// op, so its VJP is the same permute (see c_graph.h) applied with the inverted
// axis map — this is what routes a Transpose's adjoint back to its input.
inline s::vector<int64_t> invert_perm(const s::vector<int64_t>& axes) {
  s::vector<int64_t> inv(axes.size());
  for (s::size_t i = 0; i < axes.size(); ++i)
    inv[static_cast<s::size_t>(axes[i])] = static_cast<int64_t>(i);
  return inv;
}

// ---------------------------------------------------------------------------
//  Per-op gradient rules, mirroring the make_* builders in c_graph.h: each
//  takes the adjoint flowing in from downstream plus whatever forward-pass
//  data its derivative needs, and returns the gradient(s) for its input(s).
//  ONode<T>::backward below is pure dispatch over these.
// ---------------------------------------------------------------------------

// element-wise unary: gin[i] = adjoint[i] * deriv(x[i], y[i]), with x the
// input and y the forward result (the op node's own value).
template <typename T, typename Deriv>
CArray<T> grad_unary(CArena& arena, const Node<T>& x, const Node<T>& y,
                     const CArray<T>& adjoint, Deriv deriv) {
  CArray<T> gin(arena, x.shape());
  const T* pg = adjoint.data();
  const T* px = x.data();
  const T* py = y.data();
  T* pi = gin.data();
  for (s::size_t i = 0, n = gin.size(); i < n; ++i)
    pi[i] = pg[i] * deriv(px[i], py[i]);
  return gin;
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_add(CArena& arena, const Shape& a_shape,
                                       const Shape& b_shape, const CArray<T>& adjoint) {
  return { unbroadcast(arena, adjoint, a_shape), unbroadcast(arena, adjoint, b_shape) };
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_sub(CArena& arena, const Shape& a_shape,
                                       const Shape& b_shape, const CArray<T>& adjoint) {
  CArray<T> ng(arena, adjoint.shape());
  const T* pg = adjoint.data();
  for (s::size_t i = 0, n = ng.size(); i < n; ++i)
    ng.data()[i] = -pg[i];
  return { unbroadcast(arena, adjoint, a_shape), unbroadcast(arena, ng, b_shape) };
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_hadamard(CArena& arena, const Node<T>& a,
                                            const Node<T>& b, const CArray<T>& adjoint) {
  CArray<T> ta(arena, adjoint.shape()), tb(arena, adjoint.shape());
  const T* pg = adjoint.data();
  for (s::size_t i = 0, n = ta.size(); i < n; ++i) {
    ta.data()[i] = pg[i] * operand_at(b, i);
    tb.data()[i] = pg[i] * operand_at(a, i);
  }
  return { unbroadcast(arena, ta, a.shape()), unbroadcast(arena, tb, b.shape()) };
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_div(CArena& arena, const Node<T>& a,
                                       const Node<T>& b, const CArray<T>& adjoint) {
  // y = a / b ; dL/da = g / b ; dL/db = -g * a / b^2
  CArray<T> ta(arena, adjoint.shape()), tb(arena, adjoint.shape());
  const T* pg = adjoint.data();
  for (s::size_t i = 0, n = ta.size(); i < n; ++i) {
    const T av = operand_at(a, i), bv = operand_at(b, i);
    ta.data()[i] =  pg[i] / bv;
    tb.data()[i] = -pg[i] * av / (bv * bv);
  }
  return { unbroadcast(arena, ta, a.shape()), unbroadcast(arena, tb, b.shape()) };
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_pow(CArena& arena, const Node<T>& a, const Node<T>& b,
                                       const Node<T>& y, const CArray<T>& adjoint) {
  // y = a^b ; dL/da = g * b * a^(b-1) ; dL/db = g * y * log(a)
  const T* py = y.data();
  CArray<T> ta(arena, adjoint.shape()), tb(arena, adjoint.shape());
  const T* pg = adjoint.data();
  for (s::size_t i = 0, n = ta.size(); i < n; ++i) {
    const T av = operand_at(a, i), bv = operand_at(b, i);
    ta.data()[i] = pg[i] * bv * s::pow(av, bv - T{1});
    tb.data()[i] = pg[i] * py[i] * s::log(av);
  }
  return { unbroadcast(arena, ta, a.shape()), unbroadcast(arena, tb, b.shape()) };
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_dot(CArena& arena, const Node<T>& a, const Node<T>& b,
                                       const CArray<T>& adjoint) {
  // Same m,k,n split as detail::make_dot; every rank-1/rank-2 combination
  // collapses into these loops because the dropped axis has extent 1.
  //   dL/da = g . b^T   (shape a) ;  dL/db = a^T . g   (shape b)
  const int64_t m = (a.rank() == 2) ? a.shape()[0] : 1;
  const int64_t k = (a.rank() == 2) ? a.shape()[1] : a.shape()[0];
  const int64_t n = (b.rank() == 2) ? b.shape()[1] : 1;
  const T* pa = a.data();
  const T* pb = b.data();
  const T* pg = adjoint.data();
  CArray<T> ga(arena, a.shape());
  CArray<T> gb(arena, b.shape());
  T* pga = ga.data();
  T* pgb = gb.data();
  for (int64_t i = 0; i < m; ++i)
    for (int64_t p = 0; p < k; ++p) {
      T acc{};
      for (int64_t j = 0; j < n; ++j)
        acc += pg[i * n + j] * pb[p * n + j];
      pga[i * k + p] = acc;
    }
  for (int64_t p = 0; p < k; ++p)
    for (int64_t j = 0; j < n; ++j) {
      T acc{};
      for (int64_t i = 0; i < m; ++i)
        acc += pa[i * k + p] * pg[i * n + j];
      pgb[p * n + j] = acc;
    }
  return { s::move(ga), s::move(gb) };
}

template <typename T>
CArray<T> grad_reduce(CArena& arena, Op op, const Node<T>& a, const Node<T>& y,
                      int64_t axis, const CArray<T>& adjoint) {
  const AxisExtents ex = axis_extents(a.shape(), axis < 0 ? s::nullopt : s::optional<int64_t>(axis));
  CArray<T> gin(arena, a.shape(), T{});
  const T* pa = a.data();
  const T* py = y.data();
  const T* pg = adjoint.data();
  T* pi = gin.data();
  for (int64_t o = 0; o < ex.outer; ++o)
    for (int64_t j = 0; j < ex.inner; ++j) {
      const int64_t rb = o * ex.inner + j;
      const int64_t base = o * ex.axlen * ex.inner + j;
      if (op == Op::Sum) {
        for (int64_t c = 0; c < ex.axlen; ++c)
          pi[base + c * ex.inner] = pg[rb];
      } else if (op == Op::Mean) {
        const T share = pg[rb] / static_cast<T>(ex.axlen);
        for (int64_t c = 0; c < ex.axlen; ++c)
          pi[base + c * ex.inner] = share;
      } else {
        // route to every element tied for the extreme, split equally.
        const T ext = py[rb];
        int64_t ties = 0;
        for (int64_t c = 0; c < ex.axlen; ++c)
          if (pa[base + c * ex.inner] == ext) ++ties;
        const T share = pg[rb] / static_cast<T>(ties);
        for (int64_t c = 0; c < ex.axlen; ++c)
          pi[base + c * ex.inner] = (pa[base + c * ex.inner] == ext) ? share : T{};
      }
    }
  return gin;
}

template <typename T>
CArray<T> grad_softmax(CArena& arena, const Shape& a_shape, const Node<T>& y,
                       int64_t axis, const CArray<T>& adjoint) {
  // y = softmax(a) along axis ; dL/da = y * (g - sum_axis(y * g))
  const AxisExtents ex = axis_extents(a_shape, axis < 0 ? s::nullopt : s::optional<int64_t>(axis));
  CArray<T> gin(arena, a_shape);
  const T* py = y.data();
  const T* pg = adjoint.data();
  T* pi = gin.data();
  for (int64_t o = 0; o < ex.outer; ++o)
    for (int64_t j = 0; j < ex.inner; ++j) {
      const int64_t base = o * ex.axlen * ex.inner + j;
      T sdot{};
      for (int64_t c = 0; c < ex.axlen; ++c)
        sdot += py[base + c * ex.inner] * pg[base + c * ex.inner];
      for (int64_t c = 0; c < ex.axlen; ++c)
        pi[base + c * ex.inner] = py[base + c * ex.inner] * (pg[base + c * ex.inner] - sdot);
    }
  return gin;
}

template <typename T>
CArray<T> grad_cross_entropy(CArena& arena, const Node<T>& pred, const Node<T>& target,
                             int64_t axis, const CArray<T>& adjoint) {
  // loss = -log(clamp(pred[label], eps, 1)) per group; gradient only to the
  // looked-up probability, and zero where the forward clamp was flat.
  const AxisExtents ex = axis_extents(pred.shape(), axis < 0 ? s::nullopt : s::optional<int64_t>(axis));
  CArray<T> gin(arena, pred.shape(), T{});
  const T* pa = pred.data();
  const T* pt = target.data();
  const T* pg = adjoint.data();
  T* pi = gin.data();
  constexpr T kEps = static_cast<T>(1e-12);
  for (int64_t o = 0; o < ex.outer; ++o)
    for (int64_t j = 0; j < ex.inner; ++j) {
      const int64_t rb = o * ex.inner + j;
      const int64_t base = o * ex.axlen * ex.inner + j;
      const int64_t label = static_cast<int64_t>(s::llround(pt[rb]));
      const T p = pa[base + label * ex.inner];
      pi[base + label * ex.inner] = (p > kEps and p < T{1}) ? -pg[rb] / p : T{};
    }
  return gin;
}

template <typename T>
CArray<T> grad_softmax_cross_entropy(CArena& arena, const Node<T>& logits, const Node<T>& target,
                                     int64_t axis, const CArray<T>& adjoint) {
  // loss = logsumexp(logits) - logits[label] ; dL/dlogits = g * (softmax - onehot)
  const AxisExtents ex = axis_extents(logits.shape(), axis < 0 ? s::nullopt : s::optional<int64_t>(axis));
  CArray<T> gin(arena, logits.shape());
  const T* pl = logits.data();
  const T* pt = target.data();
  const T* pg = adjoint.data();
  T* pi = gin.data();
  for (int64_t o = 0; o < ex.outer; ++o)
    for (int64_t j = 0; j < ex.inner; ++j) {
      const int64_t rb = o * ex.inner + j;
      const int64_t base = o * ex.axlen * ex.inner + j;
      const int64_t label = static_cast<int64_t>(s::llround(pt[rb]));
      T mx = pl[base];
      for (int64_t c = 1; c < ex.axlen; ++c)
        mx = s::max(mx, pl[base + c * ex.inner]);
      T se{};
      for (int64_t c = 0; c < ex.axlen; ++c)
        se += s::exp(pl[base + c * ex.inner] - mx);
      for (int64_t c = 0; c < ex.axlen; ++c) {
        const T prob = s::exp(pl[base + c * ex.inner] - mx) / se;
        pi[base + c * ex.inner] = pg[rb] * (prob - (c == label ? T{1} : T{}));
      }
    }
  return gin;
}

template <typename T>
s::pair<CArray<T>, CArray<T>> grad_where(CArena& arena, const Node<T>& cond,
                                         const Shape& a_shape, const Shape& b_shape,
                                         const CArray<T>& adjoint) {
  // result[i] = cond[i] ? a[i] : b[i] ; grad follows the same selection, none
  // to the condition.
  const T* pc = cond.data();
  const T* pg = adjoint.data();
  CArray<T> ga(arena, a_shape);
  CArray<T> gb(arena, b_shape);
  T* pga = ga.data();
  T* pgb = gb.data();
  for (s::size_t i = 0, n = adjoint.size(); i < n; ++i) {
    const bool take_a = pc[i] != T{};
    pga[i] = take_a ? pg[i] : T{};
    pgb[i] = take_a ? T{}   : pg[i];
  }
  return { s::move(ga), s::move(gb) };
}

template <typename T>
CArray<T> grad_reshape(CArena& arena, const Shape& a_shape, const CArray<T>& adjoint) {
  // Same flat buffer, different Shape metadata — no element reordering, so the
  // adjoint just gets copied back into the input's own shape. Zero-filled
  // first because a -1-resolved reshape can shrink the visible extent
  // (resolve_reshape), leaving the untouched tail of gin at zero.
  CArray<T> gin(arena, a_shape, T{});
  s::copy_n(adjoint.data(), adjoint.size(), gin.data());
  return gin;
}

// transpose's VJP is itself a permute, run with the inverted axis map (see
// invert_perm above); permute() is c_graph.h's forward-pass primitive, shared
// unqualified here since this is the same autodiff::detail namespace.
template <typename T>
CArray<T> grad_transpose(CArena& arena, const CArray<T>& adjoint, const s::vector<int64_t>& axes) {
  return permute(arena, adjoint, invert_perm(axes));
}

} // detail

// ---------------------------------------------------------------------------
//  Leaf backward.
// ---------------------------------------------------------------------------
template <typename T>
void VNode<T>::backward(const CArray<T>& adjoint) {
  assert(adjoint.shape() == this->shape());
  T* pg = this->grad().data();
  const T* pa = adjoint.data();
  for (s::size_t i = 0, n = this->grad().size(); i < n; ++i)
    pg[i] = pa[i];
}

template <typename T>
void CNode<T>::backward(const CArray<T>&) {}

// ---------------------------------------------------------------------------
//  Operation backward — one gradient rule per Op.
// ---------------------------------------------------------------------------
template <typename T>
GradList<T> ONode<T>::backward(const CArray<T>& adjoint) {
  assert(mLeft != nullptr);
  assert(adjoint.shape() == this->shape());
  CArena& arena = *this->arena();
  GradList<T> out;

  switch (mOp) {
    case Op::Neg:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T,   T)   { return T{-1}; })); break;
    case Op::Exp:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T,   T y) { return y; })); break;
    case Op::Log:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T x, T)   { return T{1} / x; })); break;
    case Op::Sin:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T x, T)   { return s::cos(x); })); break;
    case Op::Cos:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T x, T)   { return -s::sin(x); })); break;
    case Op::Tan:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T,   T y) { return T{1} + y * y; })); break;
    case Op::Sqrt: out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T,   T y) { return T{1} / (T{2} * y); })); break;
    case Op::Abs:  out.emplace_back(mLeft, detail::grad_unary(arena, *mLeft, *this, adjoint,
                     [](T x, T)   { return x > T{} ? T{1} : (x < T{} ? T{-1} : T{}); })); break;

    case Op::Add: {
      auto [ga, gb] = detail::grad_add(arena, mLeft->shape(), mRight->shape(), adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }
    case Op::Sub: {
      auto [ga, gb] = detail::grad_sub(arena, mLeft->shape(), mRight->shape(), adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }
    case Op::Hadamard: {
      auto [ga, gb] = detail::grad_hadamard(arena, *mLeft, *mRight, adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }
    case Op::Div: {
      auto [ga, gb] = detail::grad_div(arena, *mLeft, *mRight, adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }
    case Op::Pow: {
      auto [ga, gb] = detail::grad_pow(arena, *mLeft, *mRight, *this, adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }
    case Op::Dot: {
      auto [ga, gb] = detail::grad_dot(arena, *mLeft, *mRight, adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }

    case Op::Sum: case Op::Mean: case Op::Max: case Op::Min:
      out.emplace_back(mLeft, detail::grad_reduce(arena, mOp, *mLeft, *this, mAxis, adjoint));
      break;

    case Op::Softmax:
      out.emplace_back(mLeft, detail::grad_softmax(arena, mLeft->shape(), *this, mAxis, adjoint));
      break;

    case Op::CrossEntropy:
      out.emplace_back(mLeft, detail::grad_cross_entropy(arena, *mLeft, *mRight, mAxis, adjoint));
      break;

    case Op::SoftmaxCrossEntropy:
      out.emplace_back(mLeft, detail::grad_softmax_cross_entropy(arena, *mLeft, *mRight, mAxis, adjoint));
      break;

    case Op::Where: {
      auto [ga, gb] = detail::grad_where(arena, *mCond, mLeft->shape(), mRight->shape(), adjoint);
      out.emplace_back(mLeft,  s::move(ga));
      out.emplace_back(mRight, s::move(gb));
      break;
    }

    case Op::Reshape: case Op::Squeeze: case Op::Unsqueeze:
      out.emplace_back(mLeft, detail::grad_reshape(arena, mLeft->shape(), adjoint));
      break;

    case Op::Transpose:
      out.emplace_back(mLeft, detail::grad_transpose(arena, adjoint, mAxes));
      break;
  }

  return out;
}

// ---------------------------------------------------------------------------
//  Reverse-pass driver.
// ---------------------------------------------------------------------------

// Run the reverse pass over the graph rooted at `root`, seeding root's adjoint
// with `seed` (which must match root's shape). Every VNode reachable from `root`
// has its grad() overwritten with the accumulated gradient; constant subtrees
// are skipped. Buffers allocated during the pass live in root's arena. Repeated
// calls overwrite rather than accumulate.
template <typename T>
void backward(Node<T>& root, const CArray<T>& seed) {
  assert(root.arena());
  assert(seed.shape() == root.shape());

  s::unordered_set<Node<T>*> seen;
  s::vector<Node<T>*> order;
  detail::rev_collect(root, seen, order);

  s::unordered_map<Node<T>*, CArray<T>> adj;
  adj.emplace(&root, seed.clone());

  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    Node<T>& n = **it;
    const auto found = adj.find(&n);
    if (found == adj.end())
      continue;                       // nothing flowed to this node
    const CArray<T>& a = found->second;

    switch (n.mKind) {
      case NodeKind::Variable:
        static_cast<VNode<T>&>(n).backward(a);
        break;
      case NodeKind::Constant:
        break;                        // CNode<T>::backward is a no-op
      case NodeKind::Operation: {
        GradList<T> edges = static_cast<ONode<T>&>(n).backward(a);
        for (auto& [in, g] : edges) {
          if (in == nullptr or not in->requires_grad())
            continue;
          const auto slot = adj.find(in);
          if (slot == adj.end()) {
            adj.emplace(in, s::move(g));
          } else {
            T* pd = slot->second.data();
            const T* ps = g.data();
            for (s::size_t q = 0, e = slot->second.size(); q < e; ++q)
              pd[q] += ps[q];
          }
        }
        break;
      }
    }
  }
}

// Seed root's adjoint with ones — the usual entry point for a scalar loss.
template <typename T>
void backward(Node<T>& root) {
  assert(root.arena());
  backward(root, CArray<T>(*root.arena(), root.shape(), T{1}));
}

} // autodiff

#endif//DREVERSE_AD
