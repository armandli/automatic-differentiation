#include <c_graph.h>
#include <dreverse_ad.h>
#include <optimizer.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
//  Integration test: linear regression on a 2-D synthetic dataset.
//
//  Dataset: N points evenly placed on the unit circle.
//    x0_i = cos(2π i / N),  x1_i = sin(2π i / N)
//
//  Because cos and sin are orthogonal over equal-spaced angles, X^T X = N/2 · I.
//  This means W and b decouple and converge independently, so the test result
//  is deterministic and easy to bound analytically:
//    W_error decays by factor (1 - lr) per step,
//    b_error decays by factor (1 - 2·lr) per step.
//  With lr = 0.1 and 200 steps both errors are negligible (< 1e-8).
// ---------------------------------------------------------------------------

namespace {

using autodiff::CArena;
using autodiff::CArray;
using autodiff::Shape;
using autodiff::SGD;
using autodiff::backward;
using autodiff::graph_constant;
using autodiff::graph_dot;
using autodiff::graph_mean;
using autodiff::graph_variable;

static constexpr int    kN    = 50;
static constexpr int    kD    = 2;
static constexpr double kW0   =  2.0;
static constexpr double kW1   = -3.0;
static constexpr double kBias =  1.0;

struct Dataset {
  std::vector<double> X;   // row-major, shape {N, D}
  std::vector<double> y;   // shape {N}
};

Dataset make_dataset() {
  Dataset ds;
  ds.X.resize(kN * kD);
  ds.y.resize(kN);
  for (int i = 0; i < kN; ++i) {
    const double theta = 2.0 * std::numbers::pi * i / kN;
    ds.X[i * kD + 0] = std::cos(theta);
    ds.X[i * kD + 1] = std::sin(theta);
    ds.y[i] = kW0 * std::cos(theta) + kW1 * std::sin(theta) + kBias;
  }
  return ds;
}

// Build one forward pass, run backward, apply SGD step, and return the MSE
// loss value from before the step. W_buf and b_val are updated in-place.
double train_step(const Dataset& ds, std::vector<double>& W_buf,
                  double& b_val, double lr) {
  CArena arena;

  // Dataset in this arena (constants — no gradient needed)
  CArray<double> X_arr(arena, Shape{kN, kD});
  CArray<double> y_arr(arena, Shape{kN});
  for (int i = 0; i < kN * kD; ++i) X_arr.data()[i] = ds.X[i];
  for (int i = 0; i < kN;      ++i) y_arr.data()[i] = ds.y[i];

  // Parameters in this arena (variables — gradients flow here)
  CArray<double> W_arr(arena, Shape{kD});
  CArray<double> b_arr(arena, Shape{1});
  for (int j = 0; j < kD; ++j) W_arr.data()[j] = W_buf[j];
  b_arr.data()[0] = b_val;

  auto& X_node = graph_constant(X_arr);
  auto& y_node = graph_constant(y_arr);
  auto& W_node = graph_variable(W_arr);
  auto& b_node = graph_variable(b_arr);

  // y_pred = X @ W + b   (shape {N})
  auto& y_pred = graph_dot(&X_node, &W_node) + b_node;

  // MSE loss = mean((y_pred - y_true)^2)
  auto& diff = y_pred - y_node;
  auto& sq   = diff * diff;
  auto& loss = graph_mean(&sq);

  const double loss_val = loss.data()[0];

  SGD<double> opt(loss, lr);
  backward(loss);
  opt.step();

  // Copy updated parameter values out before the arena is destroyed
  for (int j = 0; j < kD; ++j) W_buf[j] = W_arr.data()[j];
  b_val = b_arr.data()[0];

  return loss_val;
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

TEST(LinearRegression, LossDecreasesOnEveryStep) {
  const Dataset ds = make_dataset();
  std::vector<double> W(kD, 0.0);
  double b = 0.0;

  double prev_loss = train_step(ds, W, b, 0.1);
  for (int step = 1; step < 20; ++step) {
    const double cur_loss = train_step(ds, W, b, 0.1);
    EXPECT_LT(cur_loss, prev_loss)
        << "loss increased at step " << step
        << " (" << cur_loss << " >= " << prev_loss << ")";
    prev_loss = cur_loss;
  }
}

TEST(LinearRegression, ConvergesToTrueWeights) {
  const Dataset ds = make_dataset();
  std::vector<double> W(kD, 0.0);
  double b = 0.0;

  for (int epoch = 0; epoch < 200; ++epoch)
    train_step(ds, W, b, 0.1);

  EXPECT_NEAR(W[0], kW0,   1e-6);
  EXPECT_NEAR(W[1], kW1,   1e-6);
  EXPECT_NEAR(b,    kBias, 1e-6);
}

TEST(LinearRegression, FinalLossNearZero) {
  const Dataset ds = make_dataset();
  std::vector<double> W(kD, 0.0);
  double b = 0.0;

  double loss = 0.0;
  for (int epoch = 0; epoch < 200; ++epoch)
    loss = train_step(ds, W, b, 0.1);

  EXPECT_LT(loss, 1e-10);
}

TEST(LinearRegression, GradientDirectionIsCorrectOnFirstStep) {
  // At W=0, b=0:
  //   d_loss/d_W = -(2/N) * X^T * y_true = -W_true (by circular-data orthogonality)
  //   d_loss/d_b = -2 * b_true
  // One gradient-descent step with lr should move each param toward its target.
  const Dataset ds = make_dataset();
  std::vector<double> W(kD, 0.0);
  double b = 0.0;

  train_step(ds, W, b, 0.1);

  // W should have moved from 0 toward kW0, kW1
  EXPECT_GT(W[0], 0.0);     // kW0 > 0, so W[0] moves positive
  EXPECT_LT(W[1], 0.0);     // kW1 < 0, so W[1] moves negative
  EXPECT_GT(b,    0.0);     // kBias > 0, so b moves positive
}

TEST(LinearRegression, RandomInitConverges) {
  // Non-zero initialization to verify the optimizer doesn't depend on zero-init.
  const Dataset ds = make_dataset();
  std::vector<double> W = {0.5, 0.5};
  double b = -0.5;

  for (int epoch = 0; epoch < 200; ++epoch)
    train_step(ds, W, b, 0.1);

  EXPECT_NEAR(W[0], kW0,   1e-5);
  EXPECT_NEAR(W[1], kW1,   1e-5);
  EXPECT_NEAR(b,    kBias, 1e-5);
}

} // namespace
