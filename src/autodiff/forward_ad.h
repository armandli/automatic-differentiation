#ifndef FORWARD_AD
#define FORWARD_AD

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <dual_number.h>

namespace autodiff {

namespace s = std;

// derivative<IDX>(f, x0, x1, ...) evaluates d f / d x_IDX at the given point.
// IDX defaults to 0, so the single-variable case reads derivative(f, x).
// f must be callable with DualNumber<T> arguments; a generic lambda
// [](auto...){ ... } is the usual form and may use autodiff::sin / exp / pow.
template <s::size_t IDX = 0, typename F, typename A, typename... As>
constexpr A derivative(F func, A arg0, As... args){
  static_assert(IDX <= sizeof...(As), "IDX is past the end of f's argument list");
  static_assert((s::is_same_v<A, As> and ...), "every argument must have the same floating-point type");

  const A xs[]{arg0, args...};
  return [&]<s::size_t... I>(s::index_sequence<I...>){
    return DualNumber<A>(func(DualNumber<A>(xs[I], I == IDX ? A{1.} : A{0.})...)).dual();
  }(s::make_index_sequence<1 + sizeof...(As)>{});
}

// gradient(f, x0, x1, ...) returns { d f / d x0, d f / d x1, ... }.
template <typename F, typename A, typename... As>
constexpr s::array<A, 1 + sizeof...(As)> gradient(F func, A arg0, As... args){
  return [&]<s::size_t... I>(s::index_sequence<I...>){
    return s::array<A, 1 + sizeof...(As)>{derivative<I>(func, arg0, args...)...};
  }(s::make_index_sequence<1 + sizeof...(As)>{});
}

// check_derivative<IDX>(f, suspect, x0, x1, ...) reports whether suspect matches
// a central finite-difference estimate of d f / d x_IDX. The finite-difference
// evaluation runs through DualNumber<A> with zero dual parts, so any f usable
// with derivative() works here too. Tolerances are overridable as
// check_derivative<IDX, H, RERR, AERR>.
template <s::size_t IDX, double H = 1e-6, double RERR = 1e-5, double AERR = 1e-8, typename F, typename A, typename... As>
constexpr bool check_derivative(F func, A suspect, A arg0, As... args){
  static_assert(IDX <= sizeof...(As), "IDX is past the end of f's argument list");
  static_assert((s::is_same_v<A, As> and ...), "every argument must have the same floating-point type");

  const A xs[]{arg0, args...};
  auto value_at = [&](A step){
    return [&]<s::size_t... I>(s::index_sequence<I...>){
      return DualNumber<A>(func(DualNumber<A>(xs[I] + (I == IDX ? step : A{0.}), A{0.})...)).real();
    }(s::make_index_sequence<1 + sizeof...(As)>{});
  };
  const A h = static_cast<A>(H);
  const double numerical = (value_at(h) - value_at(-h)) / (2. * static_cast<double>(h));
  const double tolerance = AERR + s::abs(numerical) * RERR;
  return s::abs(static_cast<double>(suspect) - numerical) <= tolerance;
}

} // autodiff

#endif//FORWARD_AD
