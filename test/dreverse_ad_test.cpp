#include <c_arena.h>
#include <c_graph.h>
#include <dreverse_ad.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

namespace {

using autodiff::CArena;
using autodiff::CArray;
using autodiff::Node;
using autodiff::Shape;
using autodiff::backward;
using autodiff::grad_of;
using autodiff::graph_abs;
using autodiff::graph_constant;
using autodiff::graph_cos;
using autodiff::graph_cross_entropy;
using autodiff::graph_dot;
using autodiff::graph_exp;
using autodiff::graph_log;
using autodiff::graph_max;
using autodiff::graph_mean;
using autodiff::graph_min;
using autodiff::graph_neg;
using autodiff::graph_reshape;
using autodiff::graph_sin;
using autodiff::graph_softmax;
using autodiff::graph_softmax_cross_entropy;
using autodiff::graph_sqrt;
using autodiff::graph_squeeze;
using autodiff::graph_sum;
using autodiff::graph_tan;
using autodiff::graph_unsqueeze;
using autodiff::graph_variable;
using autodiff::graph_where;

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------

CArray<double> make_arr(CArena& arena, const Shape& shape,
                        const std::vector<double>& vals) {
  CArray<double> a(arena, shape);
  assert(a.size() == vals.size());
  for (std::size_t i = 0; i < vals.size(); ++i)
    a.data()[i] = vals[i];
  return a;
}

void expect_near(const double* actual, const std::vector<double>& expected,
                 double tol = 1e-6) {
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], tol) << "at index " << i;
}

// Central-difference gradient of a scalar loss w.r.t. each element of x0.
// `loss_of(arena, x)` rebuilds the whole graph from a length-x0.size() buffer
// (in a fresh arena) and returns the scalar loss value.
template <typename LossOf>
std::vector<double> numeric_grad(const std::vector<double>& x0, LossOf loss_of,
                                 double eps = 1e-6) {
  std::vector<double> g(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    std::vector<double> xp = x0, xm = x0;
    xp[i] += eps;
    xm[i] -= eps;
    CArena ap, am;
    g[i] = (loss_of(ap, xp) - loss_of(am, xm)) / (2.0 * eps);
  }
  return g;
}

// ---------------------------------------------------------------------------
//  element-wise unary ops
// ---------------------------------------------------------------------------

TEST(ReverseUnary, MatchNumeric) {
  using Fn = std::function<Node<double>&(Node<double>&)>;
  struct Case { const char* name; Fn op; std::vector<double> x; };
  const std::vector<Case> cases = {
    {"exp",  [](Node<double>& x) -> Node<double>& { return graph_exp(&x); },  {0.3, 1.0, -0.7, 2.1}},
    {"log",  [](Node<double>& x) -> Node<double>& { return graph_log(&x); },  {0.3, 1.0, 0.7, 2.1}},
    {"sin",  [](Node<double>& x) -> Node<double>& { return graph_sin(&x); },  {0.3, 1.0, -0.7, 2.1}},
    {"cos",  [](Node<double>& x) -> Node<double>& { return graph_cos(&x); },  {0.3, 1.0, -0.7, 2.1}},
    {"tan",  [](Node<double>& x) -> Node<double>& { return graph_tan(&x); },  {0.3, 1.0, -0.7, 1.2}},
    {"sqrt", [](Node<double>& x) -> Node<double>& { return graph_sqrt(&x); }, {0.3, 1.0, 0.7, 2.1}},
    {"abs",  [](Node<double>& x) -> Node<double>& { return graph_abs(&x); },  {0.3, -1.0, 0.7, -2.1}},
    {"neg",  [](Node<double>& x) -> Node<double>& { return graph_neg(&x); },  {0.3, -1.0, 0.7, -2.1}},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    CArena arena;
    CArray<double> xh = make_arr(arena, Shape{4}, c.x);
    auto& x = graph_variable(xh);
    auto& loss = graph_sum(&c.op(x));
    backward(loss);

    const Fn op = c.op;
    auto loss_of = [&op](CArena& ar, const std::vector<double>& xv) {
      CArray<double> h = make_arr(ar, Shape{4}, xv);
      auto& v = graph_variable(h);
      return graph_sum(&op(v)).item();
    };
    expect_near(grad_of(xh).data(), numeric_grad(c.x, loss_of), 1e-5);
  }
}

// ---------------------------------------------------------------------------
//  element-wise binary ops
// ---------------------------------------------------------------------------

TEST(ReverseBinary, AddSubNeg) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{2, 2}, {1, 2, 3, 4});
  CArray<double> bh = make_arr(arena, Shape{2, 2}, {5, 6, 7, 8});
  auto& a = graph_variable(ah);
  auto& b = graph_variable(bh);
  auto& y = a + b - b;                 // == a, but exercises Add and Sub
  auto& loss = graph_sum(&y);
  backward(loss);

  expect_near(grad_of(ah).data(), {1, 1, 1, 1});
  expect_near(grad_of(bh).data(), {0, 0, 0, 0});   // +1 from Add, -1 from Sub
}

TEST(ReverseBinary, HadamardGradientIsTheOtherOperand) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{3}, {2, 3, 4});
  CArray<double> bh = make_arr(arena, Shape{3}, {5, 6, 7});
  auto& a = graph_variable(ah);
  auto& b = graph_variable(bh);
  auto& loss = graph_sum(&(a * b));
  backward(loss);

  expect_near(grad_of(ah).data(), {5, 6, 7});
  expect_near(grad_of(bh).data(), {2, 3, 4});
}

TEST(ReverseBinary, DivGradient) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{3}, {6, 8, 9});
  CArray<double> bh = make_arr(arena, Shape{3}, {2, 4, 3});
  auto& a = graph_variable(ah);
  auto& b = graph_variable(bh);
  auto& loss = graph_sum(&(a / b));
  backward(loss);

  expect_near(grad_of(ah).data(), {1.0 / 2, 1.0 / 4, 1.0 / 3});
  expect_near(grad_of(bh).data(), {-6.0 / 4, -8.0 / 16, -9.0 / 9});
}

TEST(ReverseBinary, PowGradientMatchesNumeric) {
  const std::vector<double> a0 = {1.5, 2.0, 0.7};
  const std::vector<double> b0 = {2.0, 3.0, 1.5};

  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{3}, a0);
  CArray<double> bh = make_arr(arena, Shape{3}, b0);
  auto& a = graph_variable(ah);
  auto& b = graph_variable(bh);
  auto& loss = graph_sum(&(a ^ b));
  backward(loss);

  auto loss_a = [&b0](CArena& ar, const std::vector<double>& av) {
    CArray<double> h = make_arr(ar, Shape{3}, av);
    CArray<double> bb = make_arr(ar, Shape{3}, b0);
    auto& v = graph_variable(h);
    auto& w = graph_variable(bb);
    return graph_sum(&(v ^ w)).item();
  };
  auto loss_b = [&a0](CArena& ar, const std::vector<double>& bv) {
    CArray<double> aa = make_arr(ar, Shape{3}, a0);
    CArray<double> h = make_arr(ar, Shape{3}, bv);
    auto& v = graph_variable(aa);
    auto& w = graph_variable(h);
    return graph_sum(&(v ^ w)).item();
  };
  expect_near(grad_of(ah).data(), numeric_grad(a0, loss_a), 1e-5);
  expect_near(grad_of(bh).data(), numeric_grad(b0, loss_b), 1e-5);
}

// ---------------------------------------------------------------------------
//  scalar-broadcast unbroadcasting
// ---------------------------------------------------------------------------

TEST(ReverseBroadcast, AddScalarSumsGradIntoTheScalar) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  CArray<double> sh = make_arr(arena, Shape{1}, {10});
  auto& x = graph_variable(xh);
  auto& sc = graph_variable(sh);
  auto& loss = graph_sum(&(x + sc));
  backward(loss);

  expect_near(grad_of(xh).data(), {1, 1, 1, 1, 1, 1});
  EXPECT_NEAR(grad_of(sh).data()[0], 6.0, 1e-9);
}

TEST(ReverseBroadcast, HadamardScalarUnbroadcasts) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2}, {3, 4});
  CArray<double> sh = make_arr(arena, Shape{1}, {2});
  auto& x = graph_variable(xh);
  auto& sc = graph_variable(sh);
  auto& loss = graph_sum(&(x * sc));
  backward(loss);

  expect_near(grad_of(xh).data(), {2, 2});
  EXPECT_NEAR(grad_of(sh).data()[0], 7.0, 1e-9);   // 3 + 4
}

// ---------------------------------------------------------------------------
//  dot / matmul
// ---------------------------------------------------------------------------

TEST(ReverseDot, MatrixMatrix) {
  const std::vector<double> W0 = {1, 2, 3, 4, 5, 6};   // (2,3)
  const std::vector<double> X0 = {7, 8, 9, 10, 11, 12}; // (3,2)

  CArena arena;
  CArray<double> Wh = make_arr(arena, Shape{2, 3}, W0);
  CArray<double> Xh = make_arr(arena, Shape{3, 2}, X0);
  auto& W = graph_variable(Wh);
  auto& X = graph_variable(Xh);
  auto& loss = graph_sum(&graph_dot(&W, &X));
  backward(loss);

  auto loss_W = [&X0](CArena& ar, const std::vector<double>& wv) {
    CArray<double> h = make_arr(ar, Shape{2, 3}, wv);
    CArray<double> xx = make_arr(ar, Shape{3, 2}, X0);
    auto& w = graph_variable(h);
    auto& x = graph_variable(xx);
    return graph_sum(&graph_dot(&w, &x)).item();
  };
  auto loss_X = [&W0](CArena& ar, const std::vector<double>& xv) {
    CArray<double> ww = make_arr(ar, Shape{2, 3}, W0);
    CArray<double> h = make_arr(ar, Shape{3, 2}, xv);
    auto& w = graph_variable(ww);
    auto& x = graph_variable(h);
    return graph_sum(&graph_dot(&w, &x)).item();
  };
  EXPECT_EQ(grad_of(Wh).shape(), (Shape{2, 3}));
  EXPECT_EQ(grad_of(Xh).shape(), (Shape{3, 2}));
  expect_near(grad_of(Wh).data(), numeric_grad(W0, loss_W), 1e-5);
  expect_near(grad_of(Xh).data(), numeric_grad(X0, loss_X), 1e-5);
}

TEST(ReverseDot, MatrixVectorAndVectorMatrixAndVectorVector) {
  // matrix (2,3) . vector (3,)
  {
    const std::vector<double> A0 = {1, 2, 3, 4, 5, 6};
    const std::vector<double> v0 = {7, 8, 9};
    CArena arena;
    CArray<double> Ah = make_arr(arena, Shape{2, 3}, A0);
    CArray<double> vh = make_arr(arena, Shape{3}, v0);
    auto& A = graph_variable(Ah);
    auto& v = graph_variable(vh);
    auto& loss = graph_sum(&graph_dot(&A, &v));
    backward(loss);
    auto lA = [&v0](CArena& ar, const std::vector<double>& av) {
      CArray<double> h = make_arr(ar, Shape{2, 3}, av);
      CArray<double> vv = make_arr(ar, Shape{3}, v0);
      auto& a = graph_variable(h);
      auto& b = graph_variable(vv);
      return graph_sum(&graph_dot(&a, &b)).item();
    };
    auto lv = [&A0](CArena& ar, const std::vector<double>& vv) {
      CArray<double> aa = make_arr(ar, Shape{2, 3}, A0);
      CArray<double> h = make_arr(ar, Shape{3}, vv);
      auto& a = graph_variable(aa);
      auto& b = graph_variable(h);
      return graph_sum(&graph_dot(&a, &b)).item();
    };
    EXPECT_EQ(grad_of(Ah).shape(), (Shape{2, 3}));
    EXPECT_EQ(grad_of(vh).shape(), (Shape{3}));
    expect_near(grad_of(Ah).data(), numeric_grad(A0, lA), 1e-5);
    expect_near(grad_of(vh).data(), numeric_grad(v0, lv), 1e-5);
  }
  // vector (3,) . matrix (3,2)   and   vector (3,) . vector (3,)
  {
    const std::vector<double> u0 = {1, 2, 3};
    const std::vector<double> B0 = {4, 5, 6, 7, 8, 9};
    CArena arena;
    CArray<double> uh = make_arr(arena, Shape{3}, u0);
    CArray<double> Bh = make_arr(arena, Shape{3, 2}, B0);
    CArray<double> wh = make_arr(arena, Shape{3}, {2, -1, 0.5});
    auto& u = graph_variable(uh);
    auto& B = graph_variable(Bh);
    auto& w = graph_variable(wh);
    auto& loss = graph_sum(&graph_dot(&u, &B)) + graph_dot(&u, &w);
    backward(loss);
    EXPECT_EQ(grad_of(uh).shape(), (Shape{3}));
    EXPECT_EQ(grad_of(Bh).shape(), (Shape{3, 2}));

    auto lu = [&B0](CArena& ar, const std::vector<double>& uv) {
      CArray<double> h = make_arr(ar, Shape{3}, uv);
      CArray<double> bb = make_arr(ar, Shape{3, 2}, B0);
      CArray<double> ww = make_arr(ar, Shape{3}, {2, -1, 0.5});
      auto& a = graph_variable(h);
      auto& b = graph_variable(bb);
      auto& c = graph_variable(ww);
      return (graph_sum(&graph_dot(&a, &b)) + graph_dot(&a, &c)).item();
    };
    expect_near(grad_of(uh).data(), numeric_grad(u0, lu), 1e-5);
  }
}

// ---------------------------------------------------------------------------
//  reductions
// ---------------------------------------------------------------------------

TEST(ReverseReduce, SumAlongAxisBroadcastsSeedBack) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  auto& x = graph_variable(xh);
  auto& sm = graph_sum(&x, 1);                    // (2,)
  CArray<double> seed = make_arr(arena, Shape{2}, {10, 100});
  backward(sm, seed);
  expect_near(grad_of(xh).data(), {10, 10, 10, 100, 100, 100});
}

TEST(ReverseReduce, MeanScalesByGroupSize) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  auto& x = graph_variable(xh);
  auto& mn = graph_mean(&x, 1);                   // (2,)
  CArray<double> seed = make_arr(arena, Shape{2}, {3, 9});
  backward(mn, seed);
  expect_near(grad_of(xh).data(), {1, 1, 1, 3, 3, 3});
}

TEST(ReverseReduce, MaxSplitsGradientOverTies) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{1, 3}, {1.0, 3.0, 3.0});
  auto& x = graph_variable(xh);
  auto& mx = graph_max(&x, 1);
  CArray<double> seed = make_arr(arena, Shape{1}, {1.0});
  backward(mx, seed);
  expect_near(grad_of(xh).data(), {0.0, 0.5, 0.5});
}

TEST(ReverseReduce, MinSplitsGradientOverTies) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{1, 3}, {5.0, 2.0, 2.0});
  auto& x = graph_variable(xh);
  auto& mn = graph_min(&x, 1);
  CArray<double> seed = make_arr(arena, Shape{1}, {4.0});
  backward(mn, seed);
  expect_near(grad_of(xh).data(), {0.0, 2.0, 2.0});
}

// ---------------------------------------------------------------------------
//  softmax / cross-entropy
// ---------------------------------------------------------------------------

TEST(ReverseSoftmax, JacobianMatchesNumeric) {
  const std::vector<double> x0 = {1.0, 2.0, 0.5, -0.7, 0.1, 1.3};   // (2,3)
  const std::vector<double> w0 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2, 3}, x0);
  CArray<double> wh = make_arr(arena, Shape{2, 3}, w0);
  auto& x = graph_variable(xh);
  auto& w = graph_constant(wh);
  auto& sm = graph_softmax(&x, 1);
  auto& loss = graph_sum(&(sm * w));
  backward(loss);

  auto loss_of = [&w0](CArena& ar, const std::vector<double>& xv) {
    CArray<double> h = make_arr(ar, Shape{2, 3}, xv);
    CArray<double> ww = make_arr(ar, Shape{2, 3}, w0);
    auto& v = graph_variable(h);
    auto& c = graph_constant(ww);
    auto& s = graph_softmax(&v, 1);
    return graph_sum(&(s * c)).item();
  };
  expect_near(grad_of(xh).data(), numeric_grad(x0, loss_of), 1e-5);
}

TEST(ReverseCrossEntropy, GradientOnlyAtLabelProbabilities) {
  CArena arena;
  CArray<double> ph = make_arr(arena, Shape{2, 3}, {0.1, 0.7, 0.2, 0.3, 0.3, 0.4});
  CArray<double> th = make_arr(arena, Shape{2}, {1, 2});
  auto& p = graph_variable(ph);
  auto& t = graph_constant(th);
  auto& loss = graph_sum(&graph_cross_entropy(&p, &t, 1));
  backward(loss);

  expect_near(grad_of(ph).data(),
              {0.0, -1.0 / 0.7, 0.0, 0.0, 0.0, -1.0 / 0.4}, 1e-6);
  // the (constant) label operand keeps a zero gradient buffer
  expect_near(grad_of(th).data(), {0.0, 0.0});
}

TEST(ReverseSoftmaxCrossEntropy, GradientIsSoftmaxMinusOnehot) {
  const std::vector<double> l0 = {1.0, 2.0, 0.5, 0.2, -1.0, 3.0};   // (2,3)
  const std::vector<double> labels = {0, 2};

  CArena arena;
  CArray<double> lh = make_arr(arena, Shape{2, 3}, l0);
  CArray<double> th = make_arr(arena, Shape{2}, labels);
  auto& lg = graph_variable(lh);
  auto& t = graph_constant(th);
  auto& loss = graph_sum(&graph_softmax_cross_entropy(&lg, &t, 1));
  backward(loss);

  // analytic: softmax(row) - onehot(label)
  std::vector<double> expected(6);
  for (int r = 0; r < 2; ++r) {
    double mx = l0[r * 3];
    for (int c = 1; c < 3; ++c) mx = std::max(mx, l0[r * 3 + c]);
    double se = 0;
    for (int c = 0; c < 3; ++c) se += std::exp(l0[r * 3 + c] - mx);
    for (int c = 0; c < 3; ++c) {
      const double prob = std::exp(l0[r * 3 + c] - mx) / se;
      expected[r * 3 + c] = prob - (c == static_cast<int>(labels[r]) ? 1.0 : 0.0);
    }
  }
  expect_near(grad_of(lh).data(), expected, 1e-6);

  auto loss_of = [&labels](CArena& ar, const std::vector<double>& lv) {
    CArray<double> h = make_arr(ar, Shape{2, 3}, lv);
    CArray<double> tt = make_arr(ar, Shape{2}, labels);
    auto& v = graph_variable(h);
    auto& c = graph_constant(tt);
    return graph_sum(&graph_softmax_cross_entropy(&v, &c, 1)).item();
  };
  expect_near(grad_of(lh).data(), numeric_grad(l0, loss_of), 1e-5);
}

// ---------------------------------------------------------------------------
//  reshape / squeeze / unsqueeze
// ---------------------------------------------------------------------------

TEST(ReverseReshape, ReshapeRoutesAdjointBackInOriginalShape) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  auto& x = graph_variable(xh);
  auto& r = graph_reshape(&x, Shape{3, 2});
  CArray<double> seed = make_arr(arena, Shape{3, 2}, {1, 2, 3, 4, 5, 6});
  backward(r, seed);
  expect_near(grad_of(xh).data(), {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(grad_of(xh).shape(), (Shape{2, 3}));
}

TEST(ReverseReshape, SqueezeRoutesAdjointBackInOriginalShape) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{1, 3}, {1, 2, 3});
  auto& x = graph_variable(xh);
  auto& s = graph_squeeze(&x, 1);
  CArray<double> seed = make_arr(arena, Shape{3}, {10, 20, 30});
  backward(s, seed);
  expect_near(grad_of(xh).data(), {10, 20, 30});
  EXPECT_EQ(grad_of(xh).shape(), (Shape{1, 3}));
}

TEST(ReverseReshape, UnsqueezeRoutesAdjointBackInOriginalShape) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{3}, {1, 2, 3});
  auto& x = graph_variable(xh);
  auto& u = graph_unsqueeze(&x, 0);
  CArray<double> seed = make_arr(arena, Shape{1, 3}, {10, 20, 30});
  backward(u, seed);
  expect_near(grad_of(xh).data(), {10, 20, 30});
  EXPECT_EQ(grad_of(xh).shape(), (Shape{3}));
}

// ---------------------------------------------------------------------------
//  where
// ---------------------------------------------------------------------------

TEST(ReverseWhere, RoutesGradientByConditionAndNotToTheCondition) {
  CArena arena;
  CArray<double> ch = make_arr(arena, Shape{4}, {1, 0, 1, 0});
  CArray<double> ah = make_arr(arena, Shape{4}, {10, 20, 30, 40});
  CArray<double> bh = make_arr(arena, Shape{4}, {-1, -2, -3, -4});
  auto& c = graph_constant(ch);
  auto& a = graph_variable(ah);
  auto& b = graph_variable(bh);
  auto& w = graph_where(&c, &a, &b);
  CArray<double> seed = make_arr(arena, Shape{4}, {1, 2, 3, 4});
  backward(w, seed);

  expect_near(grad_of(ah).data(), {1, 0, 3, 0});
  expect_near(grad_of(bh).data(), {0, 2, 0, 4});
  expect_near(grad_of(ch).data(), {0, 0, 0, 0});
}

// ---------------------------------------------------------------------------
//  accumulation: diamonds and shared leaves
// ---------------------------------------------------------------------------

TEST(ReverseAccumulate, DiamondSumsBothPaths) {
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{1}, {3.0});
  auto& x = graph_variable(xh);
  auto& y = x * x + x;                 // d/dx (x^2 + x) = 2x + 1
  backward(y);
  EXPECT_NEAR(grad_of(xh).data()[0], 7.0, 1e-9);
}

TEST(ReverseAccumulate, SharedLeafUsedTwice) {
  CArena arena;
  CArray<double> wh = make_arr(arena, Shape{2, 2}, {1, 2, 3, 4});
  wh.set_requires_grad();
  auto& loss = graph_sum(&(wh + wh));   // both operands promote to one node
  backward(loss);
  expect_near(grad_of(wh).data(), {2, 2, 2, 2});
}

TEST(ReverseAccumulate, VariableUsedDirectlyAndThroughReshapeAccumulates) {
  // Before reshape/squeeze/unsqueeze were graph operators, a variable used
  // both directly and through a reshaped view of the same buffer collided in
  // the arena's pointer-keyed leaf cache (an assert in debug builds, silent
  // wrong-shape reuse in release). As ONodes, this is an ordinary
  // multi-consumer diamond, handled by the same accumulation as above.
  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  auto& x = graph_variable(xh);
  auto& r = graph_reshape(&x, Shape{6});
  auto& loss = graph_sum(&x) + graph_sum(&r);
  backward(loss);
  // d(sum(x))/dx = 1 everywhere; d(sum(reshape(x)))/dx = 1 everywhere too.
  expect_near(grad_of(xh).data(), {2, 2, 2, 2, 2, 2});
  EXPECT_EQ(grad_of(xh).shape(), (Shape{2, 3}));
}

// ---------------------------------------------------------------------------
//  composite: a one-layer classifier
// ---------------------------------------------------------------------------

TEST(ReverseComposite, LinearThenSoftmaxCrossEntropy) {
  const std::vector<double> x0 = {0.5, -0.2, 1.0};       // (3,)
  const std::vector<double> W0 = {0.1, 0.2, 0.3, -0.1, -0.2, 0.4};   // (3,2)
  const std::vector<double> b0 = {0.05, -0.05};          // (2,)
  const std::vector<double> label = {1};

  CArena arena;
  CArray<double> xh = make_arr(arena, Shape{3}, x0);
  CArray<double> Wh = make_arr(arena, Shape{3, 2}, W0);
  CArray<double> bh = make_arr(arena, Shape{2}, b0);
  CArray<double> lh = make_arr(arena, Shape{1}, label);
  auto& xc = graph_constant(xh);
  auto& W = graph_variable(Wh);
  auto& b = graph_variable(bh);
  auto& lbl = graph_constant(lh);
  auto& logits = graph_dot(&xc, &W) + b;                 // (2,)
  auto& loss = graph_softmax_cross_entropy(&logits, &lbl, 0);
  backward(loss);

  auto loss_W = [&](CArena& ar, const std::vector<double>& wv) {
    CArray<double> xx = make_arr(ar, Shape{3}, x0);
    CArray<double> h = make_arr(ar, Shape{3, 2}, wv);
    CArray<double> bb = make_arr(ar, Shape{2}, b0);
    CArray<double> ll = make_arr(ar, Shape{1}, label);
    auto& xn = graph_constant(xx);
    auto& wn = graph_variable(h);
    auto& bn = graph_variable(bb);
    auto& ln = graph_constant(ll);
    auto& lg = graph_dot(&xn, &wn) + bn;
    return graph_softmax_cross_entropy(&lg, &ln, 0).item();
  };
  auto loss_b = [&](CArena& ar, const std::vector<double>& bv) {
    CArray<double> xx = make_arr(ar, Shape{3}, x0);
    CArray<double> ww = make_arr(ar, Shape{3, 2}, W0);
    CArray<double> h = make_arr(ar, Shape{2}, bv);
    CArray<double> ll = make_arr(ar, Shape{1}, label);
    auto& xn = graph_constant(xx);
    auto& wn = graph_variable(ww);
    auto& bn = graph_variable(h);
    auto& ln = graph_constant(ll);
    auto& lg = graph_dot(&xn, &wn) + bn;
    return graph_softmax_cross_entropy(&lg, &ln, 0).item();
  };
  EXPECT_EQ(grad_of(Wh).shape(), (Shape{3, 2}));
  EXPECT_EQ(grad_of(bh).shape(), (Shape{2}));
  expect_near(grad_of(Wh).data(), numeric_grad(W0, loss_W), 1e-5);
  expect_near(grad_of(bh).data(), numeric_grad(b0, loss_b), 1e-5);
}

// ---------------------------------------------------------------------------
//  non-scalar root via the explicit-seed overload
// ---------------------------------------------------------------------------

TEST(ReverseDriver, NonScalarRootUsesExplicitSeed) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{3}, {1, 2, 3});
  CArray<double> bh = make_arr(arena, Shape{3}, {4, 5, 6});
  auto& a = graph_variable(ah);
  auto& b = graph_variable(bh);
  auto& y = a + b;
  CArray<double> seed = make_arr(arena, Shape{3}, {2, 3, 4});
  backward(y, seed);
  expect_near(grad_of(ah).data(), {2, 3, 4});
  expect_near(grad_of(bh).data(), {2, 3, 4});
}

} // namespace
