#include <autodiff/autodiff.hpp>

#include <cmath>

#include <gtest/gtest.h>

namespace {

using ad::derivative;
using ad::dual;
using ad::value_and_derivative;
using ad::variable;

TEST(Dual, ImplicitScalarIsAConstant) {
  const dual<double> c = 3.0;
  EXPECT_DOUBLE_EQ(c.value(), 3.0);
  EXPECT_DOUBLE_EQ(c.derivative(), 0.0);
}

TEST(Dual, VariableSeedsUnitDerivative) {
  const dual<double> x = variable(5.0);
  EXPECT_DOUBLE_EQ(x.value(), 5.0);
  EXPECT_DOUBLE_EQ(x.derivative(), 1.0);
}

TEST(Dual, PolynomialDerivative) {
  // f(x) = x^2 + 3x + 2   ->   f'(x) = 2x + 3   ->   f'(2) = 7
  const auto f = [](auto v) { return v * v + 3.0 * v + 2.0; };
  const auto [value, slope] = value_and_derivative(f, 2.0);
  EXPECT_DOUBLE_EQ(value, 12.0);
  EXPECT_DOUBLE_EQ(slope, 7.0);
}

TEST(Dual, QuotientRule) {
  // f(x) = (x + 1) / (x - 1)   ->   f'(x) = -2 / (x - 1)^2   ->   f'(3) = -0.5
  const auto f = [](auto v) { return (v + 1.0) / (v - 1.0); };
  EXPECT_DOUBLE_EQ(derivative(f, 3.0), -0.5);
}

TEST(Dual, ChainRuleThroughSin) {
  // d/dx sin(x^2) = 2x * cos(x^2)
  const double x = 1.3;
  const auto f = [](auto v) { return ad::sin(v * v); };
  EXPECT_NEAR(derivative(f, x), 2.0 * x * std::cos(x * x), 1e-12);
}

TEST(Dual, ExpAndLogAreInverses) {
  const auto f = [](auto v) { return ad::log(ad::exp(v)); };
  const auto [value, slope] = value_and_derivative(f, 0.7);
  EXPECT_NEAR(value, 0.7, 1e-12);
  EXPECT_NEAR(slope, 1.0, 1e-12);
}

TEST(Dual, SqrtDerivative) {
  // d/dx sqrt(x) = 1 / (2 * sqrt(x))   ->   at x = 4   ->   0.25
  EXPECT_DOUBLE_EQ(derivative([](auto v) { return ad::sqrt(v); }, 4.0), 0.25);
}

TEST(Dual, PowDerivative) {
  // d/dx x^3 = 3x^2   ->   at x = 2   ->   12
  EXPECT_DOUBLE_EQ(derivative([](auto v) { return ad::pow(v, 3.0); }, 2.0), 12.0);
}

}  // namespace
