#ifndef AUTODIFF_DUAL_HPP
#define AUTODIFF_DUAL_HPP

#include <cmath>
#include <compare>
#include <type_traits>
#include <utility>

namespace ad {

// ---------------------------------------------------------------------------
//  dual<T>
//
//  A forward-mode automatic differentiation number.  It carries the value of a
//  subexpression together with the derivative of that subexpression with
//  respect to the single input that was seeded with `variable()`.  Building an
//  expression out of dual numbers therefore evaluates both f(x) and f'(x) in
//  one pass, with the chain rule applied automatically by the operators below.
// ---------------------------------------------------------------------------
template <typename T>
class dual {
  static_assert(std::is_floating_point_v<T>,
                "ad::dual<T> requires a floating-point T");

public:
  using value_type = T;

  constexpr dual() noexcept = default;

  // Non-explicit on purpose: a bare scalar is a constant (derivative 0), which
  // lets you write `x * 2.0 + 1.0` without wrapping every literal.
  constexpr dual(T value) noexcept : value_(value) {}
  constexpr dual(T value, T derivative) noexcept
    : value_(value), derivative_(derivative) {}

  constexpr T value() const noexcept { return value_; }
  constexpr T derivative() const noexcept { return derivative_; }

  constexpr dual operator+() const noexcept { return *this; }
  constexpr dual operator-() const noexcept {
    return dual(-value_, -derivative_);
  }

  constexpr dual& operator+=(dual rhs) noexcept { return *this = *this + rhs; }
  constexpr dual& operator-=(dual rhs) noexcept { return *this = *this - rhs; }
  constexpr dual& operator*=(dual rhs) noexcept { return *this = *this * rhs; }
  constexpr dual& operator/=(dual rhs) noexcept { return *this = *this / rhs; }

  // Hidden friends: found by ADL, and because the parameters are `dual` a plain
  // scalar on either side is implicitly promoted to a constant dual.
  friend constexpr dual operator+(dual a, dual b) noexcept {
    return dual(a.value_ + b.value_, a.derivative_ + b.derivative_);
  }
  friend constexpr dual operator-(dual a, dual b) noexcept {
    return dual(a.value_ - b.value_, a.derivative_ - b.derivative_);
  }
  friend constexpr dual operator*(dual a, dual b) noexcept {
    // (uv)' = u'v + uv'
    return dual(a.value_ * b.value_,
                a.derivative_ * b.value_ + a.value_ * b.derivative_);
  }
  friend constexpr dual operator/(dual a, dual b) noexcept {
    // (u/v)' = (u'v - uv') / v^2
    const T inv = T{1} / b.value_;
    return dual(a.value_ * inv,
                (a.derivative_ * b.value_ - a.value_ * b.derivative_) * inv * inv);
  }

  // Ordering / equality compare the underlying values only.
  friend constexpr bool operator==(dual a, dual b) noexcept {
    return a.value_ == b.value_;
  }
  friend constexpr std::partial_ordering operator<=>(dual a, dual b) noexcept {
    return a.value_ <=> b.value_;
  }

private:
  T value_{};
  T derivative_{};
};

// ---------------------------------------------------------------------------
//  Seeding helpers
// ---------------------------------------------------------------------------

// The independent variable: d(v)/d(v) = 1.
template <typename T>
constexpr dual<T> variable(T v) noexcept {
  return dual<T>(v, T{1});
}

// An explicit constant: d(c)/d(x) = 0 (same as the implicit conversion).
template <typename T>
constexpr dual<T> constant(T c) noexcept {
  return dual<T>(c, T{0});
}

// ---------------------------------------------------------------------------
//  Elementary functions.  Each applies the chain rule: d f(u) = f'(u) * u'.
//  Namespace-scoped so `ad::sin(x)` works and ADL finds them for `sin(x)`.
// ---------------------------------------------------------------------------

template <typename T>
dual<T> sin(dual<T> x) {
  return dual<T>(std::sin(x.value()), std::cos(x.value()) * x.derivative());
}

template <typename T>
dual<T> cos(dual<T> x) {
  return dual<T>(std::cos(x.value()), -std::sin(x.value()) * x.derivative());
}

template <typename T>
dual<T> tan(dual<T> x) {
  const T c = std::cos(x.value());
  return dual<T>(std::tan(x.value()), x.derivative() / (c * c));
}

template <typename T>
dual<T> exp(dual<T> x) {
  const T e = std::exp(x.value());
  return dual<T>(e, e * x.derivative());
}

template <typename T>
dual<T> log(dual<T> x) {
  return dual<T>(std::log(x.value()), x.derivative() / x.value());
}

template <typename T>
dual<T> sqrt(dual<T> x) {
  const T s = std::sqrt(x.value());
  return dual<T>(s, x.derivative() / (T{2} * s));
}

template <typename T>
dual<T> tanh(dual<T> x) {
  const T t = std::tanh(x.value());
  return dual<T>(t, (T{1} - t * t) * x.derivative());
}

template <typename T>
dual<T> abs(dual<T> x) {
  const T sign = (x.value() < T{0}) ? T{-1} : T{1};
  return dual<T>(std::abs(x.value()), sign * x.derivative());
}

// Power with a constant real exponent: d/dx u^k = k * u^(k-1) * u'.
template <typename T>
dual<T> pow(dual<T> base, T exponent) {
  const T value = std::pow(base.value(), exponent);
  const T slope =
    exponent * std::pow(base.value(), exponent - T{1}) * base.derivative();
  return dual<T>(value, slope);
}

// ---------------------------------------------------------------------------
//  Convenience drivers for scalar functions f : dual<T> -> dual<T>
// ---------------------------------------------------------------------------

// f'(x)
template <typename T, typename F>
T derivative(F&& f, T x) {
  return std::forward<F>(f)(variable(x)).derivative();
}

// (f(x), f'(x))
template <typename T, typename F>
std::pair<T, T> value_and_derivative(F&& f, T x) {
  const dual<T> result = std::forward<F>(f)(variable(x));
  return {result.value(), result.derivative()};
}

}  // namespace ad

#endif  // AUTODIFF_DUAL_HPP
