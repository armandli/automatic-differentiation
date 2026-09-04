#ifndef OPTIMIZER
#define OPTIMIZER

#include <c_graph.h>

#include <cassert>
#include <cstddef>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
//  SGD: stochastic gradient descent over the parameter nodes of a
//  computation graph.
//
//  Usage:
//    SGD<double> opt(loss_node, 0.01);   // collect VNode leaves from graph
//    backward(loss_node);                 // populate gradients
//    opt.step();                          // param -= lr * grad
//    opt.zero_grad();                     // clear gradients before next pass
//
//  The optimizer borrows its arena from the root node; it must not outlive
//  that arena. No additional memory is allocated from the arena for plain SGD.
// ---------------------------------------------------------------------------

namespace autodiff {

namespace s = std;

namespace detail {

// Walk the graph from `n` and push every VNode (requires_grad leaf) into
// `vars`. Identical to rev_collect's traversal, kept separate so that
// optimizer.h does not depend on dreverse_ad.h.
template <typename T>
void collect_variables(Node<T>& n,
                       s::unordered_set<Node<T>*>& seen,
                       s::vector<VNode<T>*>& vars) {
  if (not seen.insert(&n).second) return;
  if (n.mKind == NodeKind::Variable) {
    vars.push_back(static_cast<VNode<T>*>(&n));
  } else if (n.mKind == NodeKind::Operation) {
    ONode<T>& o = static_cast<ONode<T>&>(n);
    if (o.left())  collect_variables(*o.left(),  seen, vars);
    if (o.right()) collect_variables(*o.right(), seen, vars);
    if (o.cond())  collect_variables(*o.cond(),  seen, vars);
    for (auto* inp : o.inputs()) collect_variables(*inp, seen, vars);
  }
}

} // detail

template <typename T>
struct SGD {
  SGD(Node<T>& root, T lr) : mArena(*root.arena()), mLr(lr) {
    assert(root.arena());
    s::unordered_set<Node<T>*> seen;
    detail::collect_variables(root, seen, mParams);
  }

  // Zero the gradient buffer of every managed parameter.
  void zero_grad() {
    for (VNode<T>* v : mParams)
      v->zero_grad();
  }

  // Apply one SGD step: param[i] -= lr * grad[i] for every managed parameter.
  void step() {
    for (VNode<T>* v : mParams) {
      T*       pd = v->data();
      const T* pg = v->grad().data();
      for (s::size_t i = 0, n = v->size(); i < n; ++i)
        pd[i] -= mLr * pg[i];
    }
  }

  T              lr()         const noexcept { return mLr; }
  void           set_lr(T lr) noexcept      { mLr = lr; }
  s::size_t      param_count()  const noexcept { return mParams.size(); }
  CArena&        arena()        const noexcept { return mArena; }

private:
  CArena&              mArena;
  T                    mLr;
  s::vector<VNode<T>*> mParams;
};

} // autodiff

#endif//OPTIMIZER
