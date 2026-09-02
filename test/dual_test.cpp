#include <autodiff/dual_number.h>

#include <cmath>
#include <compare>

#include <gtest/gtest.h>

namespace {

using autodiff::DualNumber;

// ---------------------------------------------------------------------------
//  Construction and accessors
// ---------------------------------------------------------------------------

TEST(Dual, DefaultConstructsToZero) {
  const DualNumber<double> d;
  EXPECT_DOUBLE_EQ(d.real(), 0.0);
  EXPECT_DOUBLE_EQ(d.dual(), 0.0);
}

TEST(Dual, ScalarConstructorHasZeroDerivative) {
  const DualNumber<double> c = 2.5;
  EXPECT_DOUBLE_EQ(c.real(), 2.5);
  EXPECT_DOUBLE_EQ(c.dual(), 0.0);
}

TEST(Dual, TwoArgConstructorSetsBothParts) {
  const DualNumber<double> x(3.0, 1.0);
  EXPECT_DOUBLE_EQ(x.real(), 3.0);
  EXPECT_DOUBLE_EQ(x.dual(), 1.0);
}

// ---------------------------------------------------------------------------
//  Unary operators
// ---------------------------------------------------------------------------

TEST(Dual, UnaryPlusIsIdentity) {
  const DualNumber<double> x(3.0, 2.0);
  EXPECT_DOUBLE_EQ((+x).real(), 3.0);
  EXPECT_DOUBLE_EQ((+x).dual(), 2.0);
}

TEST(Dual, UnaryMinusNegatesBothParts) {
  const DualNumber<double> x(3.0, 2.0);
  EXPECT_DOUBLE_EQ((-x).real(), -3.0);
  EXPECT_DOUBLE_EQ((-x).dual(), -2.0);
}

// ---------------------------------------------------------------------------
//  Binary arithmetic
// ---------------------------------------------------------------------------

TEST(Dual, AddsComponentwise) {
  const auto c = DualNumber<double>(2.0, 1.0) + DualNumber<double>(5.0, 3.0);
  EXPECT_DOUBLE_EQ(c.real(), 7.0);
  EXPECT_DOUBLE_EQ(c.dual(), 4.0);
}

TEST(Dual, SubtractsComponentwise) {
  const auto c = DualNumber<double>(2.0, 1.0) - DualNumber<double>(5.0, 3.0);
  EXPECT_DOUBLE_EQ(c.real(), -3.0);
  EXPECT_DOUBLE_EQ(c.dual(), -2.0);
}

TEST(Dual, MultipliesByProductRule) {
  // (uv)' = u'v + uv'  ->  1*5 + 2*3 = 11
  const auto c = DualNumber<double>(2.0, 1.0) * DualNumber<double>(5.0, 3.0);
  EXPECT_DOUBLE_EQ(c.real(), 10.0);
  EXPECT_DOUBLE_EQ(c.dual(), 11.0);
}

TEST(Dual, DividesByQuotientRule) {
  // (u/v)' = (u'v - uv') / v^2  ->  (1*3 - 6*1) / 9 = -1/3
  const auto c = DualNumber<double>(6.0, 1.0) / DualNumber<double>(3.0, 1.0);
  EXPECT_DOUBLE_EQ(c.real(), 2.0);
  EXPECT_DOUBLE_EQ(c.dual(), -1.0 / 3.0);
}

TEST(Dual, MixesWithScalarOnEitherSide) {
  const DualNumber<double> x(4.0, 1.0);

  const auto right = 10.0 + x - 2.0;  // real 12, dual 1
  EXPECT_DOUBLE_EQ(right.real(), 12.0);
  EXPECT_DOUBLE_EQ(right.dual(), 1.0);

  const auto left = 1.0 - x;  // real -3, dual -1
  EXPECT_DOUBLE_EQ(left.real(), -3.0);
  EXPECT_DOUBLE_EQ(left.dual(), -1.0);

  const auto scaled = 4.0 * x;  // real 16, dual 4
  EXPECT_DOUBLE_EQ(scaled.real(), 16.0);
  EXPECT_DOUBLE_EQ(scaled.dual(), 4.0);

  const auto inv = 1.0 / x;  // real 0.25, dual (0*4 - 1*1)/16 = -1/16
  EXPECT_DOUBLE_EQ(inv.real(), 0.25);
  EXPECT_DOUBLE_EQ(inv.dual(), -1.0 / 16.0);
}

// ---------------------------------------------------------------------------
//  Compound assignment
// ---------------------------------------------------------------------------

TEST(Dual, CompoundAssignmentMutatesInPlace) {
  DualNumber<double> x(2.0, 1.0);

  x += DualNumber<double>(3.0, 2.0);  // (5, 3)
  EXPECT_DOUBLE_EQ(x.real(), 5.0);
  EXPECT_DOUBLE_EQ(x.dual(), 3.0);

  x -= DualNumber<double>(1.0, 1.0);  // (4, 2)
  EXPECT_DOUBLE_EQ(x.real(), 4.0);
  EXPECT_DOUBLE_EQ(x.dual(), 2.0);

  x *= DualNumber<double>(2.0, 0.0);  // real 8, dual 4*0 + 2*2 = 4
  EXPECT_DOUBLE_EQ(x.real(), 8.0);
  EXPECT_DOUBLE_EQ(x.dual(), 4.0);

  x /= DualNumber<double>(4.0, 0.0);  // real 2, dual (4*4 - 8*0)/16 = 1
  EXPECT_DOUBLE_EQ(x.real(), 2.0);
  EXPECT_DOUBLE_EQ(x.dual(), 1.0);
}

TEST(Dual, CompoundAssignmentReturnsReference) {
  DualNumber<double> x(1.0, 1.0);
  (x += DualNumber<double>(1.0, 0.0)) += DualNumber<double>(1.0, 0.0);
  EXPECT_DOUBLE_EQ(x.real(), 3.0);
  EXPECT_DOUBLE_EQ(x.dual(), 1.0);
}

// ---------------------------------------------------------------------------
//  Comparison: real part only
// ---------------------------------------------------------------------------

TEST(Dual, RelationalOperatorsCompareRealPart) {
  const DualNumber<double> a(1.0, 5.0);
  const DualNumber<double> b(2.0, -5.0);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b >= a);
  EXPECT_FALSE(a > b);
}

TEST(Dual, EqualityIgnoresDerivative) {
  const DualNumber<double> a(2.0, 1.0);
  const DualNumber<double> b(2.0, 99.0);
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE((a <=> b) == std::partial_ordering::equivalent);
}

TEST(Dual, ComparesAgainstScalar) {
  const DualNumber<double> x(3.0, 1.0);
  EXPECT_TRUE(x == 3.0);
  EXPECT_TRUE(3.0 == x);
  EXPECT_TRUE(x < 4.0);
  EXPECT_TRUE(2.0 < x);
}

// ---------------------------------------------------------------------------
//  pow
// ---------------------------------------------------------------------------

TEST(Dual, PowWithDualExponent) {
  // f(x) = x^x  ->  f'(x) = x^x (ln x + 1)
  const DualNumber<double> x(2.0, 1.0);
  const auto r = autodiff::pow(x, x);
  EXPECT_DOUBLE_EQ(r.real(), 4.0);
  EXPECT_NEAR(r.dual(), 4.0 * (std::log(2.0) + 1.0), 1e-12);
}

TEST(Dual, PowWithConstantExponent) {
  // d/dx x^3 = 3x^2  ->  at x = 2  ->  12
  const DualNumber<double> x(2.0, 1.0);
  const auto r = autodiff::pow(x, 3.0);
  EXPECT_DOUBLE_EQ(r.real(), 8.0);
  EXPECT_DOUBLE_EQ(r.dual(), 12.0);
}

TEST(Dual, PowWithConstantExponentNegativeBase) {
  // Regression: a constant exponent must not route through 0 * log(negative).
  const DualNumber<double> x(-3.0, 1.0);
  const auto r = autodiff::pow(x, 2.0);  // (-3)^2 = 9, d/dx = 2x = -6
  EXPECT_FALSE(std::isnan(r.dual()));
  EXPECT_NEAR(r.real(), 9.0, 1e-12);
  EXPECT_NEAR(r.dual(), -6.0, 1e-12);

  const DualNumber<double> y(-2.0, 1.0);
  const auto q = autodiff::pow(y, 3.0);  // (-2)^3 = -8, d/dx = 3x^2 = 12
  EXPECT_NEAR(q.real(), -8.0, 1e-12);
  EXPECT_NEAR(q.dual(), 12.0, 1e-12);
}

TEST(Dual, PowWithIntegerLiteralExponent) {
  // Exercises the non-deduced type_identity_t parameter.
  const DualNumber<double> x(2.0, 1.0);
  const auto r = autodiff::pow(x, 2);
  EXPECT_DOUBLE_EQ(r.real(), 4.0);
  EXPECT_DOUBLE_EQ(r.dual(), 4.0);
}

TEST(Dual, PowWithConstantBase) {
  // f(x) = 2^x  ->  f'(x) = 2^x ln 2
  const DualNumber<double> x(3.0, 1.0);
  const auto r = autodiff::pow(2.0, x);
  EXPECT_DOUBLE_EQ(r.real(), 8.0);
  EXPECT_NEAR(r.dual(), 8.0 * std::log(2.0), 1e-12);
}

// ---------------------------------------------------------------------------
//  Elementary functions
// ---------------------------------------------------------------------------

TEST(Dual, Log) {
  const DualNumber<double> x(2.0, 1.0);
  const auto r = autodiff::log(x);
  EXPECT_NEAR(r.real(), std::log(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(r.dual(), 0.5);
}

TEST(Dual, LogChainRule) {
  // d/dx log(x^2) = 2/x  ->  at x = 3  ->  2/3
  const DualNumber<double> x(3.0, 1.0);
  const auto r = autodiff::log(x * x);
  EXPECT_NEAR(r.real(), std::log(9.0), 1e-12);
  EXPECT_NEAR(r.dual(), 2.0 / 3.0, 1e-12);
}

TEST(Dual, Exp) {
  const DualNumber<double> x(1.0, 1.0);
  const auto r = autodiff::exp(x);
  EXPECT_NEAR(r.real(), std::exp(1.0), 1e-12);
  EXPECT_NEAR(r.dual(), std::exp(1.0), 1e-12);
}

TEST(Dual, ExpAndLogAreInverses) {
  const DualNumber<double> x(0.7, 1.0);
  const auto r = autodiff::log(autodiff::exp(x));
  EXPECT_NEAR(r.real(), 0.7, 1e-12);
  EXPECT_NEAR(r.dual(), 1.0, 1e-12);
}

TEST(Dual, Sin) {
  const DualNumber<double> x(0.5, 1.0);
  const auto r = autodiff::sin(x);
  EXPECT_NEAR(r.real(), std::sin(0.5), 1e-12);
  EXPECT_NEAR(r.dual(), std::cos(0.5), 1e-12);
}

TEST(Dual, Cos) {
  const DualNumber<double> x(0.5, 1.0);
  const auto r = autodiff::cos(x);
  EXPECT_NEAR(r.real(), std::cos(0.5), 1e-12);
  EXPECT_NEAR(r.dual(), -std::sin(0.5), 1e-12);
}

TEST(Dual, Tan) {
  const DualNumber<double> x(0.5, 1.0);
  const auto r = autodiff::tan(x);
  const double sec = 1.0 / std::cos(0.5);
  EXPECT_NEAR(r.real(), std::tan(0.5), 1e-12);
  EXPECT_NEAR(r.dual(), sec * sec, 1e-12);
}

TEST(Dual, Sqrt) {
  const DualNumber<double> x(4.0, 1.0);
  const auto r = autodiff::sqrt(x);
  EXPECT_DOUBLE_EQ(r.real(), 2.0);
  EXPECT_DOUBLE_EQ(r.dual(), 0.25);
}

TEST(Dual, Tanh) {
  const DualNumber<double> x(0.5, 1.0);
  const auto r = autodiff::tanh(x);
  const double t = std::tanh(0.5);
  EXPECT_NEAR(r.real(), t, 1e-12);
  EXPECT_NEAR(r.dual(), 1.0 - t * t, 1e-12);
}

TEST(Dual, AbsNegative) {
  const DualNumber<double> x(-3.0, 1.0);
  const auto r = autodiff::abs(x);
  EXPECT_DOUBLE_EQ(r.real(), 3.0);
  EXPECT_DOUBLE_EQ(r.dual(), -1.0);
}

TEST(Dual, AbsPositive) {
  const DualNumber<double> x(2.0, 1.0);
  const auto r = autodiff::abs(x);
  EXPECT_DOUBLE_EQ(r.real(), 2.0);
  EXPECT_DOUBLE_EQ(r.dual(), 1.0);
}

TEST(Dual, AbsAtZeroTakesRightDerivative) {
  // |x| is not differentiable at 0; the implementation picks +1.
  const DualNumber<double> x(0.0, 1.0);
  const auto r = autodiff::abs(x);
  EXPECT_DOUBLE_EQ(r.real(), 0.0);
  EXPECT_DOUBLE_EQ(r.dual(), 1.0);
}

// ---------------------------------------------------------------------------
//  Composed expressions
// ---------------------------------------------------------------------------

TEST(Dual, PolynomialWithScalarLiterals) {
  // f(x) = x^2 + 3x + 2  ->  f'(x) = 2x + 3  ->  at x = 2  ->  (12, 7)
  const DualNumber<double> x(2.0, 1.0);
  const auto f = x * x + 3.0 * x + 2.0;
  EXPECT_DOUBLE_EQ(f.real(), 12.0);
  EXPECT_DOUBLE_EQ(f.dual(), 7.0);
}

TEST(Dual, QuotientRuleExpression) {
  // f(x) = (x + 1) / (x - 1)  ->  f'(x) = -2/(x-1)^2  ->  at x = 3  ->  (2, -0.5)
  const DualNumber<double> x(3.0, 1.0);
  const auto f = (x + 1.0) / (x - 1.0);
  EXPECT_DOUBLE_EQ(f.real(), 2.0);
  EXPECT_DOUBLE_EQ(f.dual(), -0.5);
}

TEST(Dual, ChainRuleThroughComposite) {
  // f(x) = exp(sin(x^2)),  f'(x) = exp(sin(x^2)) * cos(x^2) * 2x
  const double v = 1.3;
  const DualNumber<double> x(v, 1.0);
  const auto r = autodiff::exp(autodiff::sin(x * x));
  const double expected_value = std::exp(std::sin(v * v));
  const double expected_slope = expected_value * std::cos(v * v) * 2.0 * v;
  EXPECT_NEAR(r.real(), expected_value, 1e-12);
  EXPECT_NEAR(r.dual(), expected_slope, 1e-12);
}

// ---------------------------------------------------------------------------
//  Non-double instantiation and constexpr usability
// ---------------------------------------------------------------------------

TEST(Dual, FloatInstantiation) {
  const DualNumber<float> x(2.0f, 1.0f);
  const auto poly = x * x + x;  // real 6, dual 2*2 + 1 = 5
  EXPECT_FLOAT_EQ(poly.real(), 6.0f);
  EXPECT_FLOAT_EQ(poly.dual(), 5.0f);

  const auto sq = autodiff::sqrt(DualNumber<float>(4.0f, 1.0f));
  EXPECT_FLOAT_EQ(sq.real(), 2.0f);
  EXPECT_FLOAT_EQ(sq.dual(), 0.25f);

  const auto sn = autodiff::sin(DualNumber<float>(0.5f, 1.0f));
  EXPECT_NEAR(sn.real(), std::sin(0.5f), 1e-6f);
  EXPECT_NEAR(sn.dual(), std::cos(0.5f), 1e-6f);
}

TEST(Dual, UsableInConstantExpressions) {
  constexpr DualNumber<double> a(2.0, 1.0);
  constexpr DualNumber<double> b(3.0, 1.0);

  constexpr auto product = a * b;  // real 6, dual 2*1 + 1*3 = 5
  static_assert(product.real() == 6.0);
  static_assert(product.dual() == 5.0);

  constexpr auto affine = a + b - 1.0;  // real 4, dual 2
  static_assert(affine.real() == 4.0);
  static_assert(affine.dual() == 2.0);

  EXPECT_DOUBLE_EQ(product.real(), 6.0);
  EXPECT_DOUBLE_EQ(affine.dual(), 2.0);
}

}  // namespace
