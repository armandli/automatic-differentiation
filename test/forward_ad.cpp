#include <forward_ad.h>
#include <dual_number.h>

#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace {

using autodiff::derivative;
using autodiff::gradient;
using autodiff::check_derivative;

// Written as generic lambdas so the same body evaluates on plain scalars and on
// DualNumber<T>.
constexpr auto cube = [](auto x) { return x * x * x; };
constexpr auto poly = [](auto x) { return x * x + 3. * x + 2.; };
constexpr auto prod2 = [](auto x, auto y) { return x * y; };
constexpr auto prod3 = [](auto x, auto y, auto z) { return x * y * z; };
constexpr auto sumsq = [](auto x, auto y) { return x * x + y * y; };
const auto expsin = [](auto x) { return autodiff::exp(autodiff::sin(x * x)); };
const auto powcube = [](auto x) { return autodiff::pow(x, 3); };
const auto xsiny = [](auto x, auto y) { return x * autodiff::sin(y); };

// ---------------------------------------------------------------------------
//  derivative: single variable
// ---------------------------------------------------------------------------

TEST(Derivative, SingleVariableCube) {
  // d/dx x^3 = 3x^2  ->  at x = 2  ->  12
  EXPECT_DOUBLE_EQ(derivative(cube, 2.0), 12.0);
}

TEST(Derivative, DefaultIndexEqualsZero) {
  EXPECT_DOUBLE_EQ(derivative(cube, 2.0), derivative<0>(cube, 2.0));
}

TEST(Derivative, PolynomialWithScalarLiterals) {
  // f(x) = x^2 + 3x + 2  ->  f'(x) = 2x + 3  ->  at x = 2  ->  7
  EXPECT_DOUBLE_EQ(derivative(poly, 2.0), 7.0);
}

TEST(Derivative, PowConstantExponent) {
  // d/dx x^3 = 3x^2  ->  at x = 2  ->  12
  EXPECT_DOUBLE_EQ(derivative(powcube, 2.0), 12.0);
}

TEST(Derivative, ChainRuleExpSin) {
  // f(x) = exp(sin(x^2)),  f'(x) = exp(sin(x^2)) * cos(x^2) * 2x
  const double v = 1.3;
  const double expected = std::exp(std::sin(v * v)) * std::cos(v * v) * 2.0 * v;
  EXPECT_NEAR(derivative(expsin, v), expected, 1e-12);
}

// ---------------------------------------------------------------------------
//  derivative: partials of a multi-argument function
// ---------------------------------------------------------------------------

TEST(Derivative, PartialWrtFirstArgument) {
  // d/dx (x*y) = y  ->  at (3, 5)  ->  5
  EXPECT_DOUBLE_EQ(derivative<0>(prod2, 3.0, 5.0), 5.0);
}

TEST(Derivative, PartialWrtSecondArgument) {
  // d/dy (x*y) = x  ->  at (3, 5)  ->  3
  EXPECT_DOUBLE_EQ(derivative<1>(prod2, 3.0, 5.0), 3.0);
}

TEST(Derivative, ThreeArgumentProductRule) {
  // f(x,y,z) = x*y*z at (2, 3, 4)
  EXPECT_DOUBLE_EQ(derivative<0>(prod3, 2.0, 3.0, 4.0), 12.0);  // y*z
  EXPECT_DOUBLE_EQ(derivative<1>(prod3, 2.0, 3.0, 4.0), 8.0);   // x*z
  EXPECT_DOUBLE_EQ(derivative<2>(prod3, 2.0, 3.0, 4.0), 6.0);   // x*y
}

TEST(Derivative, MixedNonlinearPartials) {
  // f(x,y) = x*sin(y)  ->  df/dx = sin(y),  df/dy = x*cos(y)
  const double x = 1.5;
  const double y = 0.7;
  EXPECT_NEAR(derivative<0>(xsiny, x, y), std::sin(y), 1e-12);
  EXPECT_NEAR(derivative<1>(xsiny, x, y), x * std::cos(y), 1e-12);
}

// ---------------------------------------------------------------------------
//  derivative: non-double instantiation and constexpr usability
// ---------------------------------------------------------------------------

TEST(Derivative, FloatInstantiation) {
  // d/dx x^3 = 3x^2  ->  at x = 1.5f  ->  6.75f
  EXPECT_FLOAT_EQ(derivative(cube, 1.5f), 6.75f);
  EXPECT_FLOAT_EQ(derivative<1>(prod2, 2.0f, 3.0f), 2.0f);
}

TEST(Derivative, UsableInConstantExpression) {
  // Only arithmetic is involved, so the whole evaluation is a constant
  // expression (the <cmath>-backed elementary functions are not constexpr).
  static_assert(derivative(cube, 2.0) == 12.0);
  static_assert(derivative(poly, 2.0) == 7.0);
  static_assert(derivative<1>(prod2, 3.0, 5.0) == 3.0);
  EXPECT_DOUBLE_EQ(derivative(poly, 2.0), 7.0);
}

// ---------------------------------------------------------------------------
//  gradient
// ---------------------------------------------------------------------------

TEST(Gradient, ReturnsAllPartials) {
  // f(x,y) = x^2 + y^2  ->  grad = (2x, 2y)  ->  at (3, 4)  ->  (6, 8)
  const std::array<double, 2> g = gradient(sumsq, 3.0, 4.0);
  EXPECT_DOUBLE_EQ(g[0], 6.0);
  EXPECT_DOUBLE_EQ(g[1], 8.0);
}

TEST(Gradient, SingleArgumentGradient) {
  // f(x) = x^3  ->  grad = (3x^2)  ->  at x = 2  ->  (12)
  const std::array<double, 1> g = gradient(cube, 2.0);
  EXPECT_DOUBLE_EQ(g[0], 12.0);
}

TEST(Gradient, SizeMatchesArity) {
  constexpr auto g = gradient(prod3, 2.0, 3.0, 4.0);
  static_assert(g.size() == 3);
  EXPECT_DOUBLE_EQ(g[0], 12.0);
  EXPECT_DOUBLE_EQ(g[1], 8.0);
  EXPECT_DOUBLE_EQ(g[2], 6.0);
}

TEST(Gradient, MixedNonlinear) {
  // f(x,y) = x*sin(y)  ->  grad = (sin(y), x*cos(y))
  const double x = 1.5;
  const double y = 0.7;
  const std::array<double, 2> g = gradient(xsiny, x, y);
  EXPECT_NEAR(g[0], std::sin(y), 1e-12);
  EXPECT_NEAR(g[1], x * std::cos(y), 1e-12);
}

TEST(Gradient, UsableInConstantExpression) {
  constexpr std::array<double, 2> g = gradient(sumsq, 3.0, 4.0);
  static_assert(g[0] == 6.0);
  static_assert(g[1] == 8.0);
  EXPECT_DOUBLE_EQ(g[0], 6.0);
}

// ---------------------------------------------------------------------------
//  check_derivative: finite-difference sanity check
// ---------------------------------------------------------------------------

TEST(CheckDerivative, AcceptsCorrectDerivative) {
  EXPECT_TRUE(check_derivative<0>(cube, 12.0, 2.0));
}

TEST(CheckDerivative, RejectsWrongDerivative) {
  EXPECT_FALSE(check_derivative<0>(cube, 999.0, 2.0));
  EXPECT_FALSE(check_derivative<0>(cube, 11.5, 2.0));
}

TEST(CheckDerivative, MultiVariablePartial) {
  // d/dy (x*y) = x  ->  at (3, 5)  ->  3
  EXPECT_TRUE(check_derivative<1>(prod2, 3.0, 3.0, 5.0));
  EXPECT_FALSE(check_derivative<1>(prod2, 5.0, 3.0, 5.0));  // that is df/dx, not df/dy
}

TEST(CheckDerivative, AgreesWithDerivative) {
  const double v = 0.7;
  EXPECT_TRUE(check_derivative<0>(expsin, derivative(expsin, v), v));

  const double x = 1.5;
  const double y = 0.9;
  EXPECT_TRUE(check_derivative<0>(xsiny, derivative<0>(xsiny, x, y), x, y));
  EXPECT_TRUE(check_derivative<1>(xsiny, derivative<1>(xsiny, x, y), x, y));
}

TEST(CheckDerivative, CustomTolerances) {
  // A coarser step still passes for a smooth polynomial.
  EXPECT_TRUE((check_derivative<0, 1e-4, 1e-4, 1e-7>(cube, 12.0, 2.0)));
}

}  // namespace
