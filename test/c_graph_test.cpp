#include <c_arena.h>
#include <c_graph.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

using autodiff::CArena;
using autodiff::CArray;
using autodiff::CNode;
using autodiff::Node;
using autodiff::ONode;
using autodiff::Op;
using autodiff::Shape;
using autodiff::VNode;
using autodiff::graph_abs;
using autodiff::graph_add;
using autodiff::graph_constant;
using autodiff::graph_cos;
using autodiff::graph_cross_entropy;
using autodiff::graph_div;
using autodiff::graph_dot;
using autodiff::graph_exp;
using autodiff::graph_hadamard;
using autodiff::graph_log;
using autodiff::graph_max;
using autodiff::graph_mean;
using autodiff::graph_min;
using autodiff::graph_neg;
using autodiff::graph_pow;
using autodiff::graph_softmax;
using autodiff::graph_softmax_cross_entropy;
using autodiff::graph_sum;
using autodiff::graph_sin;
using autodiff::graph_sqrt;
using autodiff::graph_sub;
using autodiff::graph_tan;
using autodiff::graph_variable;
using autodiff::graph_where;
using autodiff::grad_of;
using autodiff::zero_grad;

// ---------------------------------------------------------------------------
//  Counters
// ---------------------------------------------------------------------------

TEST(Counters, VNodeIncrementsVNodeCounter) {
  CArena arena;
  const int64_t before = arena.vnode_count();
  VNode<double> v(arena, Shape{2, 3}, 1.0);
  EXPECT_EQ(arena.vnode_count(), before + 1);
}

TEST(Counters, CNodeIncrementsCNodeCounter) {
  CArena arena;
  const int64_t before = arena.cnode_count();
  CNode<double> c(arena, Shape{2, 3}, 1.0);
  EXPECT_EQ(arena.cnode_count(), before + 1);
}

TEST(Counters, AddIncrementsAddCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_add_count();
  graph_add(&a, &b);
  EXPECT_EQ(arena.onode_add_count(), before + 1);
}

TEST(Counters, SubIncrementsSubCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_sub_count();
  graph_sub(&a, &b);
  EXPECT_EQ(arena.onode_sub_count(), before + 1);
}

TEST(Counters, HadamardIncrementsHadamardCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_hadamard_count();
  graph_hadamard(&a, &b);
  EXPECT_EQ(arena.onode_hadamard_count(), before + 1);
}

TEST(Counters, DotIncrementsDotCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 2}, 2.0);
  const int64_t before = arena.onode_dot_count();
  graph_dot(&a, &b);
  EXPECT_EQ(arena.onode_dot_count(), before + 1);
}

TEST(Counters, DivIncrementsDivCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_div_count();
  graph_div(&a, &b);
  EXPECT_EQ(arena.onode_div_count(), before + 1);
}

TEST(Counters, ExpIncrementsExpCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  const int64_t before = arena.onode_exp_count();
  graph_exp(&a);
  EXPECT_EQ(arena.onode_exp_count(), before + 1);
}

TEST(Counters, LogIncrementsLogCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  const int64_t before = arena.onode_log_count();
  graph_log(&a);
  EXPECT_EQ(arena.onode_log_count(), before + 1);
}

TEST(Counters, SinIncrementsSinCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  const int64_t before = arena.onode_sin_count();
  graph_sin(&a);
  EXPECT_EQ(arena.onode_sin_count(), before + 1);
}

TEST(Counters, CosIncrementsCosCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  const int64_t before = arena.onode_cos_count();
  graph_cos(&a);
  EXPECT_EQ(arena.onode_cos_count(), before + 1);
}

TEST(Counters, TanIncrementsTanCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  const int64_t before = arena.onode_tan_count();
  graph_tan(&a);
  EXPECT_EQ(arena.onode_tan_count(), before + 1);
}

TEST(Counters, SqrtIncrementsSqrtCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 4.0);
  const int64_t before = arena.onode_sqrt_count();
  graph_sqrt(&a);
  EXPECT_EQ(arena.onode_sqrt_count(), before + 1);
}

TEST(Counters, AbsIncrementsAbsCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, -3.0);
  const int64_t before = arena.onode_abs_count();
  graph_abs(&a);
  EXPECT_EQ(arena.onode_abs_count(), before + 1);
}

TEST(Counters, PowIncrementsPowCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  const int64_t before = arena.onode_pow_count();
  graph_pow(&a, &b);
  EXPECT_EQ(arena.onode_pow_count(), before + 1);
}

TEST(Counters, VNodeBumpsCarrayCountByTwo) {
  CArena arena;
  EXPECT_EQ(arena.carray_count(), 0);
  VNode<double> v(arena, Shape{3}, 1.0);
  EXPECT_EQ(arena.carray_count(), 2);   // value buffer + grad buffer
}

TEST(Counters, BinaryOpBumpsCarrayCountByTwo) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 1.0), b(arena, Shape{3}, 2.0);
  EXPECT_EQ(arena.carray_count(), 4);
  graph_add(&a, &b);
  EXPECT_EQ(arena.carray_count(), 6);   // op result + op grad
}

TEST(Counters, AliasingViewDoesNotAllocate) {
  CArena arena;
  VNode<double> a(arena, Shape{6}, 1.0);
  const int64_t before = arena.carray_count();
  CArray<double> r = a.reshape(Shape{2, 3});
  CArray<double> sv = a.sub(0);
  CArray<double> c = a;
  EXPECT_EQ(r.data(), a.data());
  EXPECT_EQ(sv.data(), a.data());
  EXPECT_EQ(c.data(), a.data());
  EXPECT_EQ(arena.carray_count(), before);
}

// ---------------------------------------------------------------------------
//  ONodeShape
// ---------------------------------------------------------------------------

TEST(ONodeShape, BinaryOpPreservesShape) {
  CArena arena;
  VNode<double> a(arena, Shape{3, 4}, 1.0), b(arena, Shape{3, 4}, 2.0);
  auto& n = graph_add(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{3, 4}));
}

TEST(ONodeShape, UnaryOpPreservesShape) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3, 4}, 1.0);
  auto& n = graph_exp(&a);
  EXPECT_EQ(n.shape(), (Shape{2, 3, 4}));
}

TEST(ONodeShape, PowPreservesShape) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3, 4}, 2.0), b(arena, Shape{2, 3, 4}, 3.0);
  auto& n = graph_pow(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2, 3, 4}));
}

TEST(ONodeShape, DotMatMatContractsInnerDim) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 4}, 1.0);
  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2, 4}));
}

TEST(ONodeShape, DotMatVecDropsTrailingAxis) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3}, 1.0);
  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2}));
}

TEST(ONodeShape, DotVecMatDropsLeadingAxis) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 1.0), b(arena, Shape{3, 4}, 1.0);
  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{4}));
}

TEST(ONodeShape, DotVecVecYieldsSingleValue) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 1.0), b(arena, Shape{3}, 1.0);
  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{1}));
}

// ---------------------------------------------------------------------------
//  ONodeLinks
// ---------------------------------------------------------------------------

TEST(ONodeLinks, BinaryOpLinksInputNodes) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  auto& n = graph_add(&a, &b);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), static_cast<Node<double>*>(&b));
}

TEST(ONodeLinks, UnaryOpHasNullRight) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  auto& n = graph_exp(&a);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), nullptr);
}

TEST(ONodeLinks, OpEnumMatchesOperation) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  auto& add_n = graph_add(&a, &b);
  auto& sub_n = graph_sub(&a, &b);
  auto& mul_n = graph_hadamard(&a, &b);
  auto& div_n = graph_div(&a, &b);
  auto& exp_n = graph_exp(&a);
  auto& log_n = graph_log(&a);
  EXPECT_EQ(add_n.op(), Op::Add);
  EXPECT_EQ(sub_n.op(), Op::Sub);
  EXPECT_EQ(mul_n.op(), Op::Hadamard);
  EXPECT_EQ(div_n.op(), Op::Div);
  EXPECT_EQ(exp_n.op(), Op::Exp);
  EXPECT_EQ(log_n.op(), Op::Log);
}

TEST(ONodeLinks, DotLinksBothInputs) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 2}, 2.0);
  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.op(), Op::Dot);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), static_cast<Node<double>*>(&b));
}

TEST(ONodeLinks, NewOpsEnumMatchesOperation) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 0.5), b(arena, Shape{2}, 4.0);
  auto& sin_n = graph_sin(&a);
  auto& cos_n = graph_cos(&a);
  auto& tan_n = graph_tan(&a);
  auto& sqrt_n = graph_sqrt(&b);
  auto& abs_n = graph_abs(&a);
  auto& pow_n = graph_pow(&b, &a);
  EXPECT_EQ(sin_n.op(),  Op::Sin);
  EXPECT_EQ(cos_n.op(),  Op::Cos);
  EXPECT_EQ(tan_n.op(),  Op::Tan);
  EXPECT_EQ(sqrt_n.op(), Op::Sqrt);
  EXPECT_EQ(abs_n.op(),  Op::Abs);
  EXPECT_EQ(pow_n.op(),  Op::Pow);
}

TEST(ONodeLinks, NewUnaryOpsHaveNullRight) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  auto& n = graph_sin(&a);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), nullptr);
}

TEST(ONodeLinks, PowLinksBothInputs) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  auto& n = graph_pow(&a, &b);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), static_cast<Node<double>*>(&b));
}

// ---------------------------------------------------------------------------
//  ONodeValues
// ---------------------------------------------------------------------------

TEST(ONodeValues, AddIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 3.0), b(arena, Shape{3}, 2.0);
  auto& n = graph_add(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 5.0);
}

TEST(ONodeValues, SubIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 5.0), b(arena, Shape{3}, 2.0);
  auto& n = graph_sub(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 3.0);
}

TEST(ONodeValues, HadamardIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 3.0), b(arena, Shape{3}, 4.0);
  auto& n = graph_hadamard(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 12.0);
}

TEST(ONodeValues, DivIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 6.0), b(arena, Shape{3}, 2.0);
  auto& n = graph_div(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 3.0);
}

TEST(ONodeValues, ExpIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 2.0);
  auto& n = graph_exp(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::exp(2.0));
}

TEST(ONodeValues, LogIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, std::exp(1.0));
  auto& n = graph_log(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 1.0);
}

TEST(ONodeValues, SinIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 0.5);
  auto& n = graph_sin(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::sin(0.5));
}

TEST(ONodeValues, CosIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 0.5);
  auto& n = graph_cos(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::cos(0.5));
}

TEST(ONodeValues, TanIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 0.5);
  auto& n = graph_tan(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::tan(0.5));
}

TEST(ONodeValues, SqrtIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 4.0);
  auto& n = graph_sqrt(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 2.0);
}

TEST(ONodeValues, AbsIsElementWiseOnNegativeInput) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, -3.0);
  auto& n = graph_abs(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 3.0);
}

TEST(ONodeValues, PowIsElementWise) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 2.0), b(arena, Shape{3}, 3.0);
  auto& n = graph_pow(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::pow(2.0, 3.0));
}

TEST(ONodeValues, DotComputesMatrixProduct) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 0.0), b(arena, Shape{3, 2}, 0.0);
  const double av[2][3] = {{1, 2, 3}, {4, 5, 6}};
  const double bv[3][2] = {{7, 8}, {9, 10}, {11, 12}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) a[i, j] = av[i][j];
  for (int64_t i = 0; i < 3; ++i)
    for (int64_t j = 0; j < 2; ++j) b[i, j] = bv[i][j];

  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_DOUBLE_EQ((n[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((n[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((n[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((n[1, 1]), 154.0);
}

TEST(ONodeValues, DotVecVecIsInnerProduct) {
  CArena arena;
  VNode<double> a(arena, Shape{3}, 0.0), b(arena, Shape{3}, 0.0);
  for (int64_t i = 0; i < 3; ++i) { a[i] = i + 1.0; b[i] = i + 4.0; }
  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_DOUBLE_EQ(n.item(), 32.0);   // 1*4 + 2*5 + 3*6
}

TEST(ONodeValues, DotMatVecIsMatrixVectorProduct) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 0.0), b(arena, Shape{3}, 0.0);
  const double av[2][3] = {{1, 2, 3}, {4, 5, 6}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) a[i, j] = av[i][j];
  for (int64_t i = 0; i < 3; ++i) b[i] = i + 1.0;   // {1, 2, 3}

  auto& n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2}));
  EXPECT_DOUBLE_EQ(n[0], 14.0);   // 1 + 4 + 9
  EXPECT_DOUBLE_EQ(n[1], 32.0);   // 4 + 10 + 18
}

// ---------------------------------------------------------------------------
//  GradBuffer
// ---------------------------------------------------------------------------

TEST(GradBuffer, GradHasSameShapeAsValue) {
  CArena arena;
  VNode<double> v(arena, Shape{3, 4}, 1.0);
  EXPECT_EQ(v.grad().shape(), (Shape{3, 4}));
}

TEST(GradBuffer, GradIsZeroInitialized) {
  CArena arena;
  VNode<double> v(arena, Shape{4}, 5.0);
  const double* pg = v.grad().data();
  for (int64_t i = 0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(pg[i], 0.0);
}

TEST(GradBuffer, ZeroGradResetsToZero) {
  CArena arena;
  VNode<double> v(arena, Shape{3}, 1.0);
  v.grad().data()[0] = 9.9;
  v.grad().data()[1] = 8.8;
  v.zero_grad();
  const double* pg = v.grad().data();
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(pg[i], 0.0);
}

TEST(GradBuffer, ONodeGradHasSameShapeAsValue) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{2, 3}, 1.0);
  auto& n = graph_add(&a, &b);
  EXPECT_EQ(n.grad().shape(), n.shape());
}

// ---------------------------------------------------------------------------
//  Operator overloads
// ---------------------------------------------------------------------------

TEST(OperatorOverload, PlusBuildsAddNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 3.0), b(arena, Shape{2}, 4.0);
  auto& n = a + b;
  EXPECT_EQ(n.op(), Op::Add);
  EXPECT_DOUBLE_EQ(n.data()[0], 7.0);
}

TEST(OperatorOverload, MinusBuildsSubNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 5.0), b(arena, Shape{2}, 2.0);
  auto& n = a - b;
  EXPECT_EQ(n.op(), Op::Sub);
  EXPECT_DOUBLE_EQ(n.data()[0], 3.0);
}

TEST(OperatorOverload, StarBuildsHadamardNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 3.0), b(arena, Shape{2}, 4.0);
  auto& n = a * b;
  EXPECT_EQ(n.op(), Op::Hadamard);
  EXPECT_DOUBLE_EQ(n.data()[0], 12.0);
}

TEST(OperatorOverload, SlashBuildsDivNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 6.0), b(arena, Shape{2}, 3.0);
  auto& n = a / b;
  EXPECT_EQ(n.op(), Op::Div);
  EXPECT_DOUBLE_EQ(n.data()[0], 2.0);
}

TEST(OperatorOverload, CaretBuildsPowNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  auto& n = a ^ b;
  EXPECT_EQ(n.op(), Op::Pow);
  EXPECT_DOUBLE_EQ(n.data()[0], 8.0);
}

TEST(OperatorOverload, AmpersandBuildsDotNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 2}, 1.0);
  auto& n = a & b;
  EXPECT_EQ(n.op(), Op::Dot);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_DOUBLE_EQ(n.data()[0], 3.0);
}

TEST(OperatorOverload, ChainedOpsLinkCorrectly) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  auto& ab = a + b;        // 5
  auto& abxa = ab * a;     // 10
  EXPECT_DOUBLE_EQ(abxa.data()[0], 10.0);
  EXPECT_EQ(abxa.left(),  static_cast<Node<double>*>(&ab));
  EXPECT_EQ(abxa.right(), static_cast<Node<double>*>(&a));
}

// ---------------------------------------------------------------------------
//  NodeName
// ---------------------------------------------------------------------------

TEST(NodeName, VNodeAutoNameUsesCounter) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  VNode<double> b(arena, Shape{2}, 1.0);
  EXPECT_EQ(a.name(), "var_0");
  EXPECT_EQ(b.name(), "var_1");
}

TEST(NodeName, VNodeCustomNameIsUsed) {
  CArena arena;
  VNode<double> v(arena, Shape{2}, "weights", 1.0);
  EXPECT_EQ(v.name(), "weights");
}

TEST(NodeName, VNodeNamesAreDistinct) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  VNode<double> b(arena, Shape{2}, 1.0);
  EXPECT_NE(a.name(), b.name());
}

TEST(NodeName, CNodeAutoNameUsesCounter) {
  CArena arena;
  CNode<double> a(arena, Shape{2}, 1.0);
  CNode<double> b(arena, Shape{2}, 1.0);
  EXPECT_EQ(a.name(), "const_0");
  EXPECT_EQ(b.name(), "const_1");
}

TEST(NodeName, CNodeCustomNameIsUsed) {
  CArena arena;
  CNode<double> c(arena, Shape{2}, "bias", 0.0);
  EXPECT_EQ(c.name(), "bias");
}

TEST(NodeName, VNodeAndCNodeCountersAreIndependent) {
  CArena arena;
  VNode<double> v(arena, Shape{2}, 1.0);
  CNode<double> c(arena, Shape{2}, 1.0);
  EXPECT_EQ(v.name(), "var_0");
  EXPECT_EQ(c.name(), "const_0");
}

// ---------------------------------------------------------------------------
//  Promotion — CArray and literal operands
// ---------------------------------------------------------------------------

TEST(Promotion, CArrayDefaultsToConstant) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 3.0);
  auto& n = x + x;
  EXPECT_EQ(arena.cnode_count(), 1);            // one memoized leaf
  EXPECT_EQ(arena.vnode_count(), 0);
  EXPECT_EQ(n.left(), n.right());               // same node, not two
  EXPECT_EQ(n.left()->mKind, autodiff::NodeKind::Constant);
}

TEST(Promotion, RequiresGradCArrayBecomesVariable) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 3.0);
  x.set_requires_grad();
  auto& n = x * x;
  EXPECT_EQ(arena.vnode_count(), 1);
  EXPECT_EQ(arena.cnode_count(), 0);
  EXPECT_EQ(n.left(), n.right());
  EXPECT_TRUE(n.left()->requires_grad());
  EXPECT_TRUE(n.requires_grad());               // propagated to the op node
}

TEST(Promotion, SameArrayAcrossStatementsMapsToOneNode) {
  CArena arena;
  CArray<double> x(arena, Shape{2}, 1.0);
  x.set_requires_grad();
  auto& p = x + 1.0;
  auto& q = x * 2.0;
  EXPECT_EQ(p.left(), q.left());
  EXPECT_EQ(arena.vnode_count(), 1);
}

TEST(Promotion, ArenaPolicyPromotesCArrayToVariable) {
  CArena arena;
  arena.set_auto_requires_grad(true);
  CArray<double> x(arena, Shape{3}, 2.0);       // no explicit flag
  auto& n = x * x;
  EXPECT_EQ(arena.vnode_count(), 1);
  EXPECT_EQ(arena.cnode_count(), 0);
  EXPECT_EQ(n.left()->mKind, autodiff::NodeKind::Variable);
}

TEST(Promotion, LiteralStaysConstantUnderArenaPolicy) {
  CArena arena;
  arena.set_auto_requires_grad(true);
  CArray<double> x(arena, Shape{3}, 1.0);
  auto& n = x + 2.0;
  EXPECT_EQ(n.left()->mKind, autodiff::NodeKind::Variable);    // x
  EXPECT_EQ(n.right()->mKind, autodiff::NodeKind::Constant);   // the literal 2.0
}

TEST(Promotion, ScalarLiteralBecomesConstantScalar) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 1.0);
  ONode<double>& n = x + 2.0;
  EXPECT_EQ(n.op(), Op::Add);
  EXPECT_EQ(n.right()->mKind, autodiff::NodeKind::Constant);
  EXPECT_EQ(n.right()->shape(), (Shape{1}));
  EXPECT_DOUBLE_EQ(n.right()->item(), 2.0);
}

TEST(Promotion, ScalarBroadcastsAgainstArrayShape) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 3.0);
  auto& n = x * 2.0;
  EXPECT_EQ(n.shape(), (Shape{2, 3}));
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_DOUBLE_EQ(n.data()[i], 6.0);
}

TEST(Promotion, ScalarLiteralOnLeft) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 4.0);
  auto& n = 10.0 - x;
  EXPECT_EQ(n.op(), Op::Sub);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n.data()[i], 6.0);
}

TEST(Promotion, IntLiteralIsCastToValueType) {
  CArena arena;
  CArray<double> x(arena, Shape{2}, 5.0);
  auto& n = x + 2;                              // int literal
  for (int64_t i = 0; i < 2; ++i)
    EXPECT_DOUBLE_EQ(n.data()[i], 7.0);
}

TEST(Promotion, ChainWithoutNamingIntermediates) {
  CArena arena;
  CArray<double> a(arena, Shape{3}, 2.0);
  CArray<double> x(arena, Shape{3}, 3.0);
  CArray<double> b(arena, Shape{3}, 1.0);
  auto& y = a * x + b;
  EXPECT_EQ(y.op(), Op::Add);
  EXPECT_EQ(arena.node_count(), 5);             // a, x, a*x, b, +
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(y.data()[i], 7.0);
}

TEST(Promotion, MixedNodeAndCArrayOperands) {
  CArena arena;
  VNode<double> v(arena, Shape{3}, 2.0);
  CArray<double> c(arena, Shape{3}, 3.0);
  auto& n = v * c;
  EXPECT_EQ(n.left(), static_cast<Node<double>*>(&v));
  EXPECT_EQ(n.right()->mKind, autodiff::NodeKind::Constant);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n.data()[i], 6.0);
}

TEST(Promotion, DotAndPowThroughOperators) {
  CArena arena;
  CArray<double> w(arena, Shape{3}, 0.5);
  w.set_requires_grad();
  CArray<double> x(arena, Shape{3}, 2.0);
  auto& y = (w & x) + 1.0;
  EXPECT_EQ(y.op(), Op::Add);
  EXPECT_EQ(static_cast<ONode<double>*>(y.left())->op(), Op::Dot);
  EXPECT_DOUBLE_EQ(y.item(), 0.5 * 2.0 * 3 + 1.0);
}

TEST(Promotion, AliasingPromotionAllocatesOnlyGradBuffer) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 1.0);
  const int64_t before = arena.carray_count();  // x's value buffer only
  graph_constant(x);
  EXPECT_EQ(arena.carray_count(), before + 1);  // grad buffer, value aliased
}

// ---------------------------------------------------------------------------
//  Unary operators / math on graph operands
// ---------------------------------------------------------------------------

TEST(UnaryPromotion, NegateBuildsNegNode) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 2.0);
  auto& n = -x;
  EXPECT_EQ(n.op(), Op::Neg);
  EXPECT_EQ(n.right(), nullptr);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n.data()[i], -2.0);
}

TEST(UnaryPromotion, GraphNegDirect) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 3.0);
  const int64_t before = arena.onode_neg_count();
  auto& n = graph_neg(&a);
  EXPECT_EQ(arena.onode_neg_count(), before + 1);
  EXPECT_DOUBLE_EQ(n.data()[0], -3.0);
}

TEST(UnaryPromotion, MathFunctionsOnCArray) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 0.5);
  auto& e = exp(x);
  auto& s = sin(x);
  EXPECT_EQ(e.op(), Op::Exp);
  EXPECT_EQ(s.op(), Op::Sin);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(e.data()[i], std::exp(0.5));
    EXPECT_DOUBLE_EQ(s.data()[i], std::sin(0.5));
  }
}

TEST(UnaryPromotion, PowFreeFunction) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 3.0);
  auto& n = pow(x, 2.0);
  EXPECT_EQ(n.op(), Op::Pow);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n.data()[i], 9.0);
}

// ---------------------------------------------------------------------------
//  Explicit leaf factories and gradient helpers
// ---------------------------------------------------------------------------

TEST(Leaf, GraphVariablePromotesAndNames) {
  CArena arena;
  CArray<double> w(arena, Shape{3}, 0.5);
  VNode<double>& v = graph_variable(w);
  EXPECT_TRUE(w.requires_grad());
  EXPECT_EQ(v.name(), "var_0");
  auto& n = w + w;                              // reuses the same leaf
  EXPECT_EQ(n.left(), static_cast<Node<double>*>(&v));
}

TEST(Leaf, GraphConstantConvenienceOverload) {
  CArena arena;
  CNode<double>& c = graph_constant(arena, Shape{2}, 4.0);
  EXPECT_EQ(c.name(), "const_0");
  EXPECT_DOUBLE_EQ(c.data()[0], 4.0);
}

TEST(Leaf, GradOfReturnsPromotedLeafGradient) {
  CArena arena;
  CArray<double> w(arena, Shape{2, 3}, 1.0);
  w.set_requires_grad();
  auto& y = w * w;
  (void)y;
  EXPECT_EQ(grad_of(w).shape(), (Shape{2, 3}));
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_DOUBLE_EQ(grad_of(w).data()[i], 0.0);
}

TEST(Leaf, ZeroGradClearsEveryNode) {
  CArena arena;
  auto& x = graph_variable(arena, Shape{3}, 2.0);   // arena-owned leaf
  auto& y = x + x;
  x.grad().data()[0] = 5.0;
  y.grad().data()[1] = 7.0;
  zero_grad(arena);
  EXPECT_DOUBLE_EQ(x.grad().data()[0], 0.0);
  EXPECT_DOUBLE_EQ(y.grad().data()[1], 0.0);
}

// ---------------------------------------------------------------------------
//  Reduction — sum / mean / max / min, full and per-axis
// ---------------------------------------------------------------------------

// Fill a tensor row-major with 1, 2, 3, ... so reductions are easy to check.
static void fill_iota(CArray<double>& a) {
  for (int64_t i = 0, n = static_cast<int64_t>(a.size()); i < n; ++i)
    a.data()[i] = static_cast<double>(i) + 1.0;
}

TEST(Reduction, SumOverEverythingYieldsScalar) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);                                   // 1..6
  auto& n = sum(x);
  EXPECT_EQ(n.op(), Op::Sum);
  EXPECT_EQ(n.axis(), -1);
  EXPECT_EQ(n.right(), nullptr);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_DOUBLE_EQ(n.item(), 21.0);
}

TEST(Reduction, SumAlongAxisDropsThatAxis) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);                                   // [[1,2,3],[4,5,6]]

  auto& c = sum(x, 0);
  EXPECT_EQ(c.shape(), (Shape{3}));
  EXPECT_EQ(c.axis(), 0);
  EXPECT_DOUBLE_EQ(c.data()[0], 5.0);
  EXPECT_DOUBLE_EQ(c.data()[1], 7.0);
  EXPECT_DOUBLE_EQ(c.data()[2], 9.0);

  auto& r = sum(x, 1);
  EXPECT_EQ(r.shape(), (Shape{2}));
  EXPECT_EQ(r.axis(), 1);
  EXPECT_DOUBLE_EQ(r.data()[0], 6.0);
  EXPECT_DOUBLE_EQ(r.data()[1], 15.0);
}

TEST(Reduction, NegativeAxisIsSameAsCountingFromEnd) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);
  auto& a = sum(x, -1);
  auto& b = sum(x, 1);
  EXPECT_EQ(a.axis(), 1);
  EXPECT_TRUE(autodiff::content_equal(a, b));
}

TEST(Reduction, ThreeDimAxisReduction) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3, 4}, 0.0);
  fill_iota(x);                                   // element (i,j,k) = i*12 + j*4 + k + 1
  auto& n = sum(x, 1);
  EXPECT_EQ(n.shape(), (Shape{2, 4}));
  // output (i,k) = sum over j of (i*12 + j*4 + k + 1) = 3*(i*12+k+1) + 4*(0+1+2)
  EXPECT_DOUBLE_EQ((n[0, 0]), 3.0 * 1.0 + 12.0);
  EXPECT_DOUBLE_EQ((n[1, 3]), 3.0 * (12.0 + 3.0 + 1.0) + 12.0);
}

TEST(Reduction, RankOneAxisReductionYieldsScalar) {
  CArena arena;
  CArray<double> x(arena, Shape{4}, 0.0);
  fill_iota(x);                                   // 1,2,3,4
  auto& n = sum(x, 0);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_DOUBLE_EQ(n.item(), 10.0);
}

TEST(Reduction, MeanOverEverythingYieldsScalar) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);                                   // 1..6
  auto& n = mean(x);
  EXPECT_EQ(n.op(), Op::Mean);
  EXPECT_EQ(n.axis(), -1);
  EXPECT_EQ(n.right(), nullptr);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_DOUBLE_EQ(n.item(), 3.5);               // 21 / 6
}

TEST(Reduction, MeanAlongAxisDropsThatAxis) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);                                   // [[1,2,3],[4,5,6]]

  auto& c = mean(x, 0);
  EXPECT_EQ(c.shape(), (Shape{3}));
  EXPECT_EQ(c.axis(), 0);
  EXPECT_DOUBLE_EQ(c.data()[0], 2.5);
  EXPECT_DOUBLE_EQ(c.data()[1], 3.5);
  EXPECT_DOUBLE_EQ(c.data()[2], 4.5);

  auto& r = mean(x, 1);
  EXPECT_EQ(r.shape(), (Shape{2}));
  EXPECT_EQ(r.axis(), 1);
  EXPECT_DOUBLE_EQ(r.data()[0], 2.0);
  EXPECT_DOUBLE_EQ(r.data()[1], 5.0);
}

TEST(Reduction, MaxAndMinFullAndPerAxis) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  const double v[2][3] = {{3, 1, 4}, {1, 5, 9}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) x[i, j] = v[i][j];

  EXPECT_DOUBLE_EQ(max(x).item(), 9.0);
  EXPECT_DOUBLE_EQ(min(x).item(), 1.0);

  auto& rmax = max(x, 1);
  EXPECT_EQ(rmax.op(), Op::Max);
  EXPECT_EQ(rmax.shape(), (Shape{2}));
  EXPECT_DOUBLE_EQ(rmax.data()[0], 4.0);
  EXPECT_DOUBLE_EQ(rmax.data()[1], 9.0);

  auto& cmin = min(x, 0);
  EXPECT_EQ(cmin.op(), Op::Min);
  EXPECT_EQ(cmin.shape(), (Shape{3}));
  EXPECT_DOUBLE_EQ(cmin.data()[0], 1.0);
  EXPECT_DOUBLE_EQ(cmin.data()[1], 1.0);
  EXPECT_DOUBLE_EQ(cmin.data()[2], 4.0);
}

TEST(Reduction, CounterBumpsPerReduction) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0);
  graph_sum(&a);
  graph_mean(&a, 1);
  graph_mean(&a);
  graph_max(&a, 1);
  graph_min(&a, 0);
  graph_min(&a);
  EXPECT_EQ(arena.onode_sum_count(), 1);
  EXPECT_EQ(arena.onode_mean_count(), 2);
  EXPECT_EQ(arena.onode_max_count(), 1);
  EXPECT_EQ(arena.onode_min_count(), 2);
}

TEST(Reduction, PromotesCArrayOperand) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 2.0);
  auto& n = sum(x);
  EXPECT_EQ(n.left()->mKind, autodiff::NodeKind::Constant);
  EXPECT_FALSE(n.requires_grad());

  CArray<double> w(arena, Shape{3}, 2.0);
  w.set_requires_grad();
  auto& m = sum(w, 0);
  EXPECT_EQ(m.left()->mKind, autodiff::NodeKind::Variable);
  EXPECT_TRUE(m.requires_grad());
}

TEST(Reduction, LowLevelGraphMaxOnNode) {
  CArena arena;
  VNode<double> a(arena, Shape{2, 3}, 0.0);
  fill_iota(a);
  ONode<double>& n = graph_max(&a, 0);
  EXPECT_EQ(n.shape(), (Shape{3}));
  EXPECT_DOUBLE_EQ(n.data()[0], 4.0);
  EXPECT_DOUBLE_EQ(n.data()[2], 6.0);
}

// ---------------------------------------------------------------------------
//  Softmax — shape-preserving normalization, whole-array and per-axis
// ---------------------------------------------------------------------------

TEST(Softmax, NormalizesWholeArrayAndKeepsShape) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);                                   // 1..6
  auto& n = softmax(x);
  EXPECT_EQ(n.op(), Op::Softmax);
  EXPECT_EQ(n.axis(), -1);
  EXPECT_EQ(n.right(), nullptr);
  EXPECT_EQ(n.shape(), (Shape{2, 3}));
  double total = 0.0;
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_GT(n.data()[i], 0.0);
    total += n.data()[i];
  }
  EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(Softmax, KnownValuesAlongLastAxis) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 0.0);
  x.data()[0] = 1.0; x.data()[1] = 2.0; x.data()[2] = 3.0;
  auto& n = softmax(x, 0);
  EXPECT_EQ(n.axis(), 0);
  EXPECT_EQ(n.shape(), (Shape{3}));
  EXPECT_NEAR(n.data()[0], 0.09003057317038046, 1e-12);
  EXPECT_NEAR(n.data()[1], 0.24472847105479764, 1e-12);
  EXPECT_NEAR(n.data()[2], 0.66524095577482200, 1e-12);
}

TEST(Softmax, PerAxisRowsAndColumnsSumToOne) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 0.0);
  fill_iota(x);                                   // [[1,2,3],[4,5,6]]

  auto& r = softmax(x, 1);                        // each row is a distribution
  EXPECT_EQ(r.axis(), 1);
  EXPECT_EQ(r.shape(), (Shape{2, 3}));
  EXPECT_NEAR((r[0, 0]) + (r[0, 1]) + (r[0, 2]), 1.0, 1e-12);
  EXPECT_NEAR((r[1, 0]) + (r[1, 1]) + (r[1, 2]), 1.0, 1e-12);
  EXPECT_NEAR((r[0, 2]), 0.66524095577482200, 1e-12);

  auto& c = softmax(x, 0);                        // each column is a distribution
  EXPECT_NEAR((c[0, 0]) + (c[1, 0]), 1.0, 1e-12);
  EXPECT_NEAR((c[0, 1]), 0.04742587317756678, 1e-12);
}

TEST(Softmax, StableForLargeLogitsAndUniformForEqualInputs) {
  CArena arena;
  CArray<double> big(arena, Shape{3}, 1000.0);
  auto& n = softmax(big);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(std::isnan(n.data()[i]));
    EXPECT_NEAR(n.data()[i], 1.0 / 3.0, 1e-12);
  }
}

// ---------------------------------------------------------------------------
//  CrossEntropy — categorical CE against integer class labels
// ---------------------------------------------------------------------------

TEST(CrossEntropy, MatchesNegLogProbOfLabelPerRow) {
  CArena arena;
  CArray<double> p(arena, Shape{2, 3}, 0.0);
  const double v[2][3] = {{0.7, 0.2, 0.1}, {0.1, 0.3, 0.6}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) p[i, j] = v[i][j];
  CArray<double> y(arena, Shape{2}, 0.0);
  y.data()[0] = 0.0; y.data()[1] = 2.0;           // labels: row 0 -> class 0, row 1 -> class 2

  auto& n = cross_entropy(p, y, 1);
  EXPECT_EQ(n.op(), Op::CrossEntropy);
  EXPECT_EQ(n.axis(), 1);
  EXPECT_NE(n.left(), nullptr);
  EXPECT_NE(n.right(), nullptr);
  EXPECT_EQ(n.shape(), (Shape{2}));
  EXPECT_NEAR(n.data()[0], -std::log(0.7), 1e-12);
  EXPECT_NEAR(n.data()[1], -std::log(0.6), 1e-12);
}

TEST(CrossEntropy, WholeArrayReducesToScalar) {
  CArena arena;
  CArray<double> p(arena, Shape{3}, 0.0);
  p.data()[0] = 0.2; p.data()[1] = 0.5; p.data()[2] = 0.3;
  CArray<double> y(arena, Shape{1}, 1.0);         // one label for the whole array
  auto& n = cross_entropy(p, y);
  EXPECT_EQ(n.axis(), -1);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_NEAR(n.item(), -std::log(0.5), 1e-12);
}

TEST(CrossEntropy, ClampsZeroProbabilityToFiniteLoss) {
  CArena arena;
  CArray<double> p(arena, Shape{2}, 0.0);         // p[label] == 0 exactly
  p.data()[0] = 0.0; p.data()[1] = 1.0;
  CArray<double> y(arena, Shape{1}, 0.0);
  auto& n = cross_entropy(p, y);
  EXPECT_FALSE(std::isinf(n.item()));
  EXPECT_NEAR(n.item(), -std::log(1e-12), 1e-9);
}

// ---------------------------------------------------------------------------
//  SoftmaxCrossEntropy — fused, numerically stable classification loss
// ---------------------------------------------------------------------------

TEST(SoftmaxCrossEntropy, EqualsNegLogSoftmaxOfLabel) {
  CArena arena;
  CArray<double> z(arena, Shape{3}, 0.0);
  z.data()[0] = 1.0; z.data()[1] = 2.0; z.data()[2] = 3.0;
  CArray<double> y(arena, Shape{1}, 2.0);
  auto& n = softmax_cross_entropy(z, y);
  EXPECT_EQ(n.op(), Op::SoftmaxCrossEntropy);
  EXPECT_EQ(n.axis(), -1);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_NEAR(n.item(), 0.4076059644443806, 1e-12);   // -log(softmax(z)[2])
}

TEST(SoftmaxCrossEntropy, AgreesWithCrossEntropyOfSoftmax) {
  CArena arena;
  CArray<double> z(arena, Shape{2, 3}, 0.0);
  const double v[2][3] = {{0.5, -1.0, 2.0}, {3.0, 0.0, -0.5}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) z[i, j] = v[i][j];
  CArray<double> y(arena, Shape{2}, 0.0);
  y.data()[0] = 2.0; y.data()[1] = 0.0;

  auto& fused = softmax_cross_entropy(z, y, 1);
  EXPECT_EQ(fused.shape(), (Shape{2}));

  CArray<double> z2(arena, Shape{2, 3}, 0.0);
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) z2[i, j] = v[i][j];
  CArray<double> y2(arena, Shape{2}, 0.0);
  y2.data()[0] = 2.0; y2.data()[1] = 0.0;
  auto& viaSoftmax = cross_entropy(softmax(z2, 1), y2, 1);

  EXPECT_NEAR(fused.data()[0], viaSoftmax.data()[0], 1e-12);
  EXPECT_NEAR(fused.data()[1], viaSoftmax.data()[1], 1e-12);
}

TEST(SoftmaxCrossEntropy, StableForLargeLogits) {
  CArena arena;
  CArray<double> z(arena, Shape{3}, 0.0);
  z.data()[0] = 1000.0; z.data()[1] = 1001.0; z.data()[2] = 999.0;
  CArray<double> y(arena, Shape{1}, 1.0);
  auto& n = softmax_cross_entropy(z, y);
  EXPECT_FALSE(std::isnan(n.item()));
  EXPECT_FALSE(std::isinf(n.item()));
  EXPECT_GT(n.item(), 0.0);
}

TEST(NeuralOps, CountersBumpPerOp) {
  CArena arena;
  VNode<double> z(arena, Shape{2, 3}, 1.0);
  CNode<double> y(arena, Shape{2}, 0.0);
  graph_softmax(&z);
  graph_softmax(&z, 1);
  graph_cross_entropy(&z, &y, 1);
  graph_softmax_cross_entropy(&z, &y, 1);
  graph_softmax_cross_entropy(&z, &y, 1);
  EXPECT_EQ(arena.onode_softmax_count(), 2);
  EXPECT_EQ(arena.onode_cross_entropy_count(), 1);
  EXPECT_EQ(arena.onode_softmax_cross_entropy_count(), 2);
}

// ---------------------------------------------------------------------------
//  MixedType — one arena holds many element types; binary ops promote operands
//  to the C++ usual-arithmetic-conversion result type.
// ---------------------------------------------------------------------------

template <typename N>
using node_value_t = typename std::remove_reference_t<N>::value_type;

TEST(MixedType, FloatTimesBoolMask) {
  CArena arena;
  CArray<float> x(arena, Shape{2, 2}, 0.0f);
  const float xv[2][2] = {{1, 2}, {3, 4}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 2; ++j) x[i, j] = xv[i][j];
  CArray<bool> m(arena, Shape{2, 2}, false);
  m[0, 0] = true; m[0, 1] = false; m[1, 0] = true; m[1, 1] = true;

  auto& n = x * m;
  static_assert(std::is_same_v<node_value_t<decltype(n)>, float>);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_FLOAT_EQ((n[0, 0]), 1.0f);
  EXPECT_FLOAT_EQ((n[0, 1]), 0.0f);
  EXPECT_FLOAT_EQ((n[1, 0]), 3.0f);
  EXPECT_FLOAT_EQ((n[1, 1]), 4.0f);
}

TEST(MixedType, DoubleMatmulIntMatrix) {
  CArena arena;
  CArray<double> d(arena, Shape{2, 3}, 0.0);
  CArray<int>    w(arena, Shape{3, 2}, 0);
  for (int64_t i = 0, v = 1; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j, ++v) d[i, j] = static_cast<double>(v);
  for (int64_t i = 0, v = 1; i < 3; ++i)
    for (int64_t j = 0; j < 2; ++j, ++v) w[i, j] = static_cast<int>(v);

  auto& n = d & w;                                // [[1,2,3],[4,5,6]] * [[1,2],[3,4],[5,6]]
  static_assert(std::is_same_v<node_value_t<decltype(n)>, double>);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_DOUBLE_EQ((n[0, 0]), 22.0);
  EXPECT_DOUBLE_EQ((n[0, 1]), 28.0);
  EXPECT_DOUBLE_EQ((n[1, 0]), 49.0);
  EXPECT_DOUBLE_EQ((n[1, 1]), 64.0);
}

TEST(MixedType, ResultIsCommonType) {
  CArena arena;
  CArray<float>  a(arena, Shape{1}, 1.0f);
  CArray<double> b(arena, Shape{1}, 1e-10);
  auto& n = a + b;
  static_assert(std::is_same_v<node_value_t<decltype(n)>, double>);
  EXPECT_DOUBLE_EQ(n.item(), 1.0 + 1e-10);        // double precision survived
}

TEST(MixedType, HeterogeneousGraphsShareOneArena) {
  CArena arena;
  CArray<float>  xf(arena, Shape{3}, 2.0f);
  CArray<double> xd(arena, Shape{3}, 3.0);
  auto& yf = xf + xf;
  auto& yd = xd * xd;
  static_assert(std::is_same_v<node_value_t<decltype(yf)>, float>);
  static_assert(std::is_same_v<node_value_t<decltype(yd)>, double>);
  EXPECT_FLOAT_EQ(yf.data()[0], 4.0f);
  EXPECT_DOUBLE_EQ(yd.data()[0], 9.0);
}

TEST(MixedType, SameTypeStillMemoizesOneLeaf) {
  CArena arena;
  CArray<double> x(arena, Shape{2}, 3.0);
  auto& n = x * x;                                // no conversion — one shared leaf
  EXPECT_EQ(n.left(), n.right());
  EXPECT_DOUBLE_EQ(n.data()[0], 9.0);
}

TEST(MixedType, ConvertedOperandIsAConstantLeaf) {
  CArena arena;
  CArray<double> x(arena, Shape{2}, 5.0);
  CArray<int>    k(arena, Shape{2}, 2);
  auto& n = x * k;
  EXPECT_EQ(n.right()->mKind, autodiff::NodeKind::Constant);
  EXPECT_FALSE(n.right()->requires_grad());
  EXPECT_DOUBLE_EQ(n.data()[0], 10.0);
}

// ---------------------------------------------------------------------------
//  Where — element-wise conditional select
// ---------------------------------------------------------------------------

TEST(Where, SelectsPerElement) {
  CArena arena;
  CArray<bool>   c(arena, Shape{4}, false);
  c.data()[0] = true; c.data()[2] = true;          // [T, F, T, F]
  CArray<double> a(arena, Shape{4}, 0.0);
  CArray<double> b(arena, Shape{4}, 0.0);
  for (int64_t i = 0; i < 4; ++i) {
    a.data()[i] = static_cast<double>(i) + 1.0;     // [1, 2, 3, 4]
    b.data()[i] = 10.0 * (static_cast<double>(i) + 1.0);   // [10, 20, 30, 40]
  }

  auto& n = where(c, a, b);
  EXPECT_EQ(n.op(), Op::Where);
  EXPECT_EQ(n.shape(), (Shape{4}));
  EXPECT_NE(n.left(), nullptr);
  EXPECT_NE(n.right(), nullptr);
  EXPECT_NE(n.cond(), nullptr);
  EXPECT_DOUBLE_EQ(n.data()[0], 1.0);               // cond true  -> a
  EXPECT_DOUBLE_EQ(n.data()[1], 20.0);              // cond false -> b
  EXPECT_DOUBLE_EQ(n.data()[2], 3.0);
  EXPECT_DOUBLE_EQ(n.data()[3], 40.0);
}

TEST(Where, TwoDimensional) {
  CArena arena;
  CArray<bool>   c(arena, Shape{2, 2}, true);
  c[0, 1] = false; c[1, 0] = false;
  CArray<double> a(arena, Shape{2, 2}, 1.0);
  CArray<double> b(arena, Shape{2, 2}, 9.0);
  auto& n = where(c, a, b);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_DOUBLE_EQ((n[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((n[0, 1]), 9.0);
  EXPECT_DOUBLE_EQ((n[1, 0]), 9.0);
  EXPECT_DOUBLE_EQ((n[1, 1]), 1.0);
}

TEST(Where, ConditionConvertsToConstantLeaf) {
  CArena arena;
  CArray<bool>   c(arena, Shape{2}, true);
  CArray<double> a(arena, Shape{2}, 1.0);
  a.set_requires_grad();
  CArray<double> b(arena, Shape{2}, 2.0);
  auto& n = where(c, a, b);
  EXPECT_EQ(n.cond()->mKind, autodiff::NodeKind::Constant);
  EXPECT_FALSE(n.cond()->requires_grad());
  EXPECT_TRUE(n.requires_grad());                   // a is a variable
}

TEST(Where, NumericConditionAlsoWorks) {
  CArena arena;
  CArray<double> c(arena, Shape{3}, 0.0);
  c.data()[1] = 1.0;                                // only index 1 selects a
  CArray<double> a(arena, Shape{3}, 5.0);
  CArray<double> b(arena, Shape{3}, 6.0);
  auto& n = where(c, a, b);
  EXPECT_DOUBLE_EQ(n.data()[0], 6.0);
  EXPECT_DOUBLE_EQ(n.data()[1], 5.0);
  EXPECT_DOUBLE_EQ(n.data()[2], 6.0);
}

TEST(Where, CounterBumps) {
  CArena arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  VNode<double> b(arena, Shape{2}, 2.0);
  CNode<double> c(arena, Shape{2}, 1.0);
  graph_where(&c, &a, &b);
  graph_where(&c, &a, &b);
  EXPECT_EQ(arena.onode_where_count(), 2);
}

} // namespace
