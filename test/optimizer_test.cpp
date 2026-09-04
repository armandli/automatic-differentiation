#include <c_graph.h>
#include <dreverse_ad.h>
#include <optimizer.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace {

using autodiff::CArena;
using autodiff::CArray;
using autodiff::Node;
using autodiff::Shape;
using autodiff::SGD;
using autodiff::VNode;
using autodiff::backward;
using autodiff::graph_constant;
using autodiff::graph_variable;

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------

CArray<double> make_arr(CArena& arena, const Shape& shape,
                        const std::vector<double>& vals) {
  CArray<double> a(arena, shape);
  for (std::size_t i = 0; i < vals.size(); ++i)
    a.data()[i] = vals[i];
  return a;
}

void expect_near(const double* ptr, const std::vector<double>& expected,
                 double tol = 1e-9) {
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(ptr[i], expected[i], tol);
}

// ---------------------------------------------------------------------------
//  SGD construction
// ---------------------------------------------------------------------------

TEST(SGD, CollectsVariablesFromGraph) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{3}, {1, 2, 3});
  CArray<double> bh = make_arr(arena, Shape{3}, {4, 5, 6});
  auto& a  = graph_variable(ah);
  auto& b  = graph_variable(bh);
  auto& cn = graph_constant(arena, Shape{1}, 1.0);
  auto& y  = a + b + cn;

  SGD<double> opt(y, 0.1);
  EXPECT_EQ(opt.param_count(), 2u);
}

TEST(SGD, IgnoresConstantNodes) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{2}, {1, 2});
  auto& a = graph_variable(ah);
  auto& c = graph_constant(arena, Shape{2}, 3.0);
  auto& y = a + c;

  SGD<double> opt(y, 0.01);
  EXPECT_EQ(opt.param_count(), 1u);
}

TEST(SGD, BorrowsArenaFromRoot) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{2}, {1, 2});
  auto& a = graph_variable(ah);

  SGD<double> opt(a, 0.1);
  EXPECT_EQ(&opt.arena(), &arena);
}

TEST(SGD, LrAccessors) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{2}, {1, 2});
  auto& a = graph_variable(ah);

  SGD<double> opt(a, 0.05);
  EXPECT_DOUBLE_EQ(opt.lr(), 0.05);
  opt.set_lr(0.001);
  EXPECT_DOUBLE_EQ(opt.lr(), 0.001);
}

// ---------------------------------------------------------------------------
//  step() — gradient descent update
// ---------------------------------------------------------------------------

TEST(SGD, StepUpdatesParamByLrTimesGrad) {
  CArena arena;
  // loss = sum(a) ; grad = [1, 1, 1]
  CArray<double> ah = make_arr(arena, Shape{3}, {2, 3, 4});
  auto& a = graph_variable(ah);
  auto& y = graph_sum(&a);

  SGD<double> opt(y, 0.1);
  backward(y);
  opt.step();

  // param = [2,3,4] - 0.1*[1,1,1] = [1.9, 2.9, 3.9]
  expect_near(ah.data(), {1.9, 2.9, 3.9});
}

TEST(SGD, StepWithTwoParams) {
  CArena arena;
  // W: {2,2}, b: {2}
  // loss = sum(W * x + b) where x is constant
  CArray<double> Wh = make_arr(arena, Shape{2, 2}, {1, 0, 0, 1});
  CArray<double> bh = make_arr(arena, Shape{2},    {0, 0});
  CArray<double> xh = make_arr(arena, Shape{2},    {3, 4});
  auto& W  = graph_variable(Wh);
  auto& b  = graph_variable(bh);
  auto& xn = graph_constant(xh);
  auto& dot = graph_dot(&W, &xn);
  auto& y  = graph_sum(&(dot + b));

  SGD<double> opt(y, 0.5);
  backward(y);
  opt.step();

  // grad_W = outer(ones, x) = [[3,4],[3,4]]
  // grad_b = [1, 1]
  expect_near(Wh.data(), {1 - 0.5*3, 0 - 0.5*4, 0 - 0.5*3, 1 - 0.5*4});
  expect_near(bh.data(), {0 - 0.5, 0 - 0.5});
}

TEST(SGD, StepMovesParamTowardMinimum) {
  CArena arena;
  // loss = (w - 1)^2 ; minimum at w=1
  // w=3 -> grad = 2*(3-1) = 4 -> w' = 3 - 0.1*4 = 2.6  (closer to 1)
  CArray<double> wh = make_arr(arena, Shape{1}, {3.0});
  auto& w    = graph_variable(wh);
  auto& one  = graph_constant(arena, Shape{1}, 1.0);
  auto& diff = w - one;
  auto& sq   = diff * diff;
  auto& loss = graph_sum(&sq);

  SGD<double> opt(loss, 0.1);
  backward(loss);
  opt.step();

  EXPECT_NEAR(wh.data()[0], 2.6, 1e-9);
}

// ---------------------------------------------------------------------------
//  zero_grad()
// ---------------------------------------------------------------------------

TEST(SGD, ZeroGradClearsGradients) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{3}, {1, 2, 3});
  auto& a = graph_variable(ah);
  auto& y = graph_sum(&a);

  SGD<double> opt(y, 0.1);
  backward(y);
  // grad is [1,1,1] after backward
  opt.zero_grad();
  expect_near(a.grad().data(), {0, 0, 0});
}

TEST(SGD, ZeroGradDoesNotAffectParamValues) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{2}, {5, 6});
  auto& a = graph_variable(ah);
  auto& y = graph_sum(&a);

  SGD<double> opt(y, 0.1);
  backward(y);
  opt.zero_grad();

  // param values unchanged
  expect_near(ah.data(), {5, 6});
}

TEST(SGD, MultipleStepsViaFreshArenas) {
  // loss = w^2 ; grad = 2w ; step: w' = w*(1 - 2*lr)
  // With lr=0.1: w' = 0.8*w. After 20 steps: w = 4 * 0.8^20 ≈ 0.029
  double w_val = 4.0;
  for (int i = 0; i < 20; ++i) {
    CArena arena;
    CArray<double> wh = make_arr(arena, Shape{1}, {w_val});
    auto& w    = graph_variable(wh);
    auto& wsq  = w * w;
    auto& loss = graph_sum(&wsq);
    SGD<double> opt(loss, 0.1);
    backward(loss);
    opt.step();
    w_val = wh.data()[0];
  }
  EXPECT_LT(std::abs(w_val), 0.1);
}

// ---------------------------------------------------------------------------
//  Shared subexpression: variable reachable by multiple paths counted once
// ---------------------------------------------------------------------------

TEST(SGD, SharedVariableCountedOnce) {
  CArena arena;
  CArray<double> ah = make_arr(arena, Shape{2}, {1, 2});
  auto& a = graph_variable(ah);
  auto& y = a + a;   // a appears twice but is one VNode

  SGD<double> opt(y, 0.1);
  EXPECT_EQ(opt.param_count(), 1u);
}

} // namespace
