#ifndef DUAL_NUMBER
#define DUAL_NUMBER

#include <cassert>
#include <cmath>
#include <compare>
#include <type_traits>

namespace autodiff {

namespace s = std;

template <typename T>
struct DualNumber {
  static_assert(s::is_floating_point_v<T>,
                "DualNumber<T> requires a floating-point T");

  constexpr DualNumber() noexcept : mReal(0.), mDual(0.) {}
  constexpr DualNumber(T real) noexcept : mReal(real), mDual(0.) {}
  constexpr DualNumber(T real, T dual) noexcept : mReal(real), mDual(dual) {}

  constexpr T real() const noexcept { return mReal; }
  constexpr T dual() const noexcept { return mDual; }

  constexpr DualNumber operator+() const noexcept { return *this; }
  constexpr DualNumber operator-() const noexcept { return DualNumber(-mReal, -mDual); }

  constexpr DualNumber& operator+=(const DualNumber& b) noexcept { return *this = *this + b; }
  constexpr DualNumber& operator-=(const DualNumber& b) noexcept { return *this = *this - b; }
  constexpr DualNumber& operator*=(const DualNumber& b) noexcept { return *this = *this * b; }
  constexpr DualNumber& operator/=(const DualNumber& b) noexcept { return *this = *this / b; }

  friend constexpr DualNumber operator+(const DualNumber& a, const DualNumber& b) noexcept {
    return DualNumber(a.mReal + b.mReal, a.mDual + b.mDual);
  }
  friend constexpr DualNumber operator-(const DualNumber& a, const DualNumber& b) noexcept {
    return DualNumber(a.mReal - b.mReal, a.mDual - b.mDual);
  }
  friend constexpr DualNumber operator*(const DualNumber& a, const DualNumber& b) noexcept {
    // (uv)' = u'v + uv'
    return DualNumber(a.mReal * b.mReal, a.mReal * b.mDual + a.mDual * b.mReal);
  }
  friend constexpr DualNumber operator/(const DualNumber& a, const DualNumber& b) noexcept {
    // (u/v)' = (u'v - uv') / v^2
    return DualNumber(a.mReal / b.mReal,
                      (a.mDual * b.mReal - a.mReal * b.mDual) / (b.mReal * b.mReal));
  }

  constexpr bool operator==(const DualNumber& b) const noexcept { return mReal == b.mReal; }
  constexpr s::partial_ordering operator<=>(const DualNumber& b) const noexcept {
    return mReal <=> b.mReal;
  }

private:
  T mReal;
  T mDual;
};

// General power: both base and exponent may carry a derivative.
//   d/dx a^b = a^b * (b' * ln(a) + b * a'/a)
template <typename T>
DualNumber<T> pow(const DualNumber<T>& a, const DualNumber<T>& b){
  T p = s::pow(a.real(), b.real());
  return DualNumber<T>(p, p * (b.dual() * s::log(a.real()) + a.dual() * b.real() / a.real()));
}

// Constant real exponent: d/dx a^k = k * a^(k-1) * a'. Stays finite for any sign
// of a.real(), unlike routing a constant through the general formula above.
// type_identity_t keeps k out of template deduction so pow(x, 2) also compiles.
template <typename T>
DualNumber<T> pow(const DualNumber<T>& a, s::type_identity_t<T> k){
  T p = s::pow(a.real(), k);
  return DualNumber<T>(p, k * s::pow(a.real(), k - T{1.}) * a.dual());
}

// Constant real base: d/dx c^b = c^b * ln(c) * b'.
template <typename T>
DualNumber<T> pow(s::type_identity_t<T> c, const DualNumber<T>& b){
  T p = s::pow(c, b.real());
  return DualNumber<T>(p, p * s::log(c) * b.dual());
}

template <typename T>
DualNumber<T> log(const DualNumber<T>& a){
  assert(a.real() != T{0.});
  return DualNumber<T>(s::log(a.real()), a.dual() / a.real());
}

template <typename T>
DualNumber<T> exp(const DualNumber<T>& a){
  T e = s::exp(a.real());
  return DualNumber<T>(e, e * a.dual());
}

template <typename T>
DualNumber<T> sin(const DualNumber<T>& a){
  return DualNumber<T>(s::sin(a.real()), s::cos(a.real()) * a.dual());
}

template <typename T>
DualNumber<T> cos(const DualNumber<T>& a){
  return DualNumber<T>(s::cos(a.real()), T{-1.} * s::sin(a.real()) * a.dual());
}

template <typename T>
DualNumber<T> tan(const DualNumber<T>& a){
  T sec = T{1.} / s::cos(a.real());
  return DualNumber<T>(s::tan(a.real()), a.dual() * sec * sec);
}

template <typename T>
DualNumber<T> sqrt(const DualNumber<T>& a){
  T sqrta = s::sqrt(a.real());
  return DualNumber<T>(sqrta, a.dual() / (T{2.} * sqrta));
}

template <typename T>
DualNumber<T> tanh(const DualNumber<T>& a){
  T t = s::tanh(a.real());
  return DualNumber<T>(t, (T{1.} - t * t) * a.dual());
}

template <typename T>
DualNumber<T> abs(const DualNumber<T>& a){
  T sign = (a.real() < T{0.}) ? T{-1.} : T{1.};
  return DualNumber<T>(s::abs(a.real()), sign * a.dual());
}

} // autodiff

#endif//DUAL_NUMBER
