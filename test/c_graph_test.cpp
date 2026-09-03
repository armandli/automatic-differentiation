#include <c_arena.h>
#include <c_graph.h>

#include <cmath>
#include <cstdint>

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
using autodiff::graph_cos;
using autodiff::graph_div;
using autodiff::graph_dot;
using autodiff::graph_exp;
using autodiff::graph_hadamard;
using autodiff::graph_log;
using autodiff::graph_pow;
using autodiff::graph_sin;
using autodiff::graph_sqrt;
using autodiff::graph_sub;
using autodiff::graph_tan;

// ---------------------------------------------------------------------------
//  Counters
// ---------------------------------------------------------------------------

TEST(Counters, VNodeIncrementsVNodeCounter) {
  CArena<double> arena;
  const int64_t before = arena.vnode_count();
  VNode<double> v(arena, Shape{2, 3}, 1.0);
  EXPECT_EQ(arena.vnode_count(), before + 1);
}

TEST(Counters, CNodeIncrementsCNodeCounter) {
  CArena<double> arena;
  const int64_t before = arena.cnode_count();
  CNode<double> c(arena, Shape{2, 3}, 1.0);
  EXPECT_EQ(arena.cnode_count(), before + 1);
}

TEST(Counters, AddIncrementsAddCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_add_count();
  ONode<double> n = graph_add(&a, &b);
  EXPECT_EQ(arena.onode_add_count(), before + 1);
}

TEST(Counters, SubIncrementsSubCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_sub_count();
  ONode<double> n = graph_sub(&a, &b);
  EXPECT_EQ(arena.onode_sub_count(), before + 1);
}

TEST(Counters, HadamardIncrementsHadamardCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_hadamard_count();
  ONode<double> n = graph_hadamard(&a, &b);
  EXPECT_EQ(arena.onode_hadamard_count(), before + 1);
}

TEST(Counters, DotIncrementsDotCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 2}, 2.0);
  const int64_t before = arena.onode_dot_count();
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(arena.onode_dot_count(), before + 1);
}

TEST(Counters, DivIncrementsDivCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  const int64_t before = arena.onode_div_count();
  ONode<double> n = graph_div(&a, &b);
  EXPECT_EQ(arena.onode_div_count(), before + 1);
}

TEST(Counters, ExpIncrementsExpCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  const int64_t before = arena.onode_exp_count();
  ONode<double> n = graph_exp(&a);
  EXPECT_EQ(arena.onode_exp_count(), before + 1);
}

TEST(Counters, LogIncrementsLogCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  const int64_t before = arena.onode_log_count();
  ONode<double> n = graph_log(&a);
  EXPECT_EQ(arena.onode_log_count(), before + 1);
}

TEST(Counters, SinIncrementsSinCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  const int64_t before = arena.onode_sin_count();
  ONode<double> n = graph_sin(&a);
  EXPECT_EQ(arena.onode_sin_count(), before + 1);
}

TEST(Counters, CosIncrementsCosCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  const int64_t before = arena.onode_cos_count();
  ONode<double> n = graph_cos(&a);
  EXPECT_EQ(arena.onode_cos_count(), before + 1);
}

TEST(Counters, TanIncrementsTanCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  const int64_t before = arena.onode_tan_count();
  ONode<double> n = graph_tan(&a);
  EXPECT_EQ(arena.onode_tan_count(), before + 1);
}

TEST(Counters, SqrtIncrementsSqrtCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 4.0);
  const int64_t before = arena.onode_sqrt_count();
  ONode<double> n = graph_sqrt(&a);
  EXPECT_EQ(arena.onode_sqrt_count(), before + 1);
}

TEST(Counters, AbsIncrementsAbsCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, -3.0);
  const int64_t before = arena.onode_abs_count();
  ONode<double> n = graph_abs(&a);
  EXPECT_EQ(arena.onode_abs_count(), before + 1);
}

TEST(Counters, PowIncrementsPowCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  const int64_t before = arena.onode_pow_count();
  ONode<double> n = graph_pow(&a, &b);
  EXPECT_EQ(arena.onode_pow_count(), before + 1);
}

TEST(Counters, VNodeBumpsCarrayCountByTwo) {
  CArena<double> arena;
  EXPECT_EQ(arena.carray_count(), 0);
  VNode<double> v(arena, Shape{3}, 1.0);
  EXPECT_EQ(arena.carray_count(), 2);   // value buffer + grad buffer
}

TEST(Counters, BinaryOpBumpsCarrayCountByTwo) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 1.0), b(arena, Shape{3}, 2.0);
  EXPECT_EQ(arena.carray_count(), 4);
  ONode<double> n = graph_add(&a, &b);
  EXPECT_EQ(arena.carray_count(), 6);   // op result + op grad
}

TEST(Counters, AliasingViewDoesNotAllocate) {
  CArena<double> arena;
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
  CArena<double> arena;
  VNode<double> a(arena, Shape{3, 4}, 1.0), b(arena, Shape{3, 4}, 2.0);
  ONode<double> n = graph_add(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{3, 4}));
}

TEST(ONodeShape, UnaryOpPreservesShape) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3, 4}, 1.0);
  ONode<double> n = graph_exp(&a);
  EXPECT_EQ(n.shape(), (Shape{2, 3, 4}));
}

TEST(ONodeShape, PowPreservesShape) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3, 4}, 2.0), b(arena, Shape{2, 3, 4}, 3.0);
  ONode<double> n = graph_pow(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2, 3, 4}));
}

TEST(ONodeShape, DotMatMatContractsInnerDim) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 4}, 1.0);
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2, 4}));
}

TEST(ONodeShape, DotMatVecDropsTrailingAxis) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3}, 1.0);
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2}));
}

TEST(ONodeShape, DotVecMatDropsLeadingAxis) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 1.0), b(arena, Shape{3, 4}, 1.0);
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{4}));
}

TEST(ONodeShape, DotVecVecYieldsSingleValue) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 1.0), b(arena, Shape{3}, 1.0);
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{1}));
}

// ---------------------------------------------------------------------------
//  ONodeLinks
// ---------------------------------------------------------------------------

TEST(ONodeLinks, BinaryOpLinksInputNodes) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  ONode<double> n = graph_add(&a, &b);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), static_cast<Node<double>*>(&b));
}

TEST(ONodeLinks, UnaryOpHasNullRight) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  ONode<double> n = graph_exp(&a);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), nullptr);
}

TEST(ONodeLinks, OpEnumMatchesOperation) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0), b(arena, Shape{2}, 2.0);
  ONode<double> add_n = graph_add(&a, &b);
  ONode<double> sub_n = graph_sub(&a, &b);
  ONode<double> mul_n = graph_hadamard(&a, &b);
  ONode<double> div_n = graph_div(&a, &b);
  ONode<double> exp_n = graph_exp(&a);
  ONode<double> log_n = graph_log(&a);
  EXPECT_EQ(add_n.op(), Op::Add);
  EXPECT_EQ(sub_n.op(), Op::Sub);
  EXPECT_EQ(mul_n.op(), Op::Hadamard);
  EXPECT_EQ(div_n.op(), Op::Div);
  EXPECT_EQ(exp_n.op(), Op::Exp);
  EXPECT_EQ(log_n.op(), Op::Log);
}

TEST(ONodeLinks, DotLinksBothInputs) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 2}, 2.0);
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.op(), Op::Dot);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), static_cast<Node<double>*>(&b));
}

TEST(ONodeLinks, NewOpsEnumMatchesOperation) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 0.5), b(arena, Shape{2}, 4.0);
  ONode<double> sin_n = graph_sin(&a);
  ONode<double> cos_n = graph_cos(&a);
  ONode<double> tan_n = graph_tan(&a);
  ONode<double> sqrt_n = graph_sqrt(&b);
  ONode<double> abs_n = graph_abs(&a);
  ONode<double> pow_n = graph_pow(&b, &a);
  EXPECT_EQ(sin_n.op(),  Op::Sin);
  EXPECT_EQ(cos_n.op(),  Op::Cos);
  EXPECT_EQ(tan_n.op(),  Op::Tan);
  EXPECT_EQ(sqrt_n.op(), Op::Sqrt);
  EXPECT_EQ(abs_n.op(),  Op::Abs);
  EXPECT_EQ(pow_n.op(),  Op::Pow);
}

TEST(ONodeLinks, NewUnaryOpsHaveNullRight) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 0.5);
  ONode<double> n = graph_sin(&a);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), nullptr);
}

TEST(ONodeLinks, PowLinksBothInputs) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  ONode<double> n = graph_pow(&a, &b);
  EXPECT_EQ(n.left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n.right(), static_cast<Node<double>*>(&b));
}

// ---------------------------------------------------------------------------
//  ONodeValues
// ---------------------------------------------------------------------------

TEST(ONodeValues, AddIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 3.0), b(arena, Shape{3}, 2.0);
  ONode<double> n = graph_add(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 5.0);
}

TEST(ONodeValues, SubIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 5.0), b(arena, Shape{3}, 2.0);
  ONode<double> n = graph_sub(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 3.0);
}

TEST(ONodeValues, HadamardIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 3.0), b(arena, Shape{3}, 4.0);
  ONode<double> n = graph_hadamard(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 12.0);
}

TEST(ONodeValues, DivIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 6.0), b(arena, Shape{3}, 2.0);
  ONode<double> n = graph_div(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 3.0);
}

TEST(ONodeValues, ExpIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 2.0);
  ONode<double> n = graph_exp(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::exp(2.0));
}

TEST(ONodeValues, LogIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, std::exp(1.0));
  ONode<double> n = graph_log(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 1.0);
}

TEST(ONodeValues, SinIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 0.5);
  ONode<double> n = graph_sin(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::sin(0.5));
}

TEST(ONodeValues, CosIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 0.5);
  ONode<double> n = graph_cos(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::cos(0.5));
}

TEST(ONodeValues, TanIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 0.5);
  ONode<double> n = graph_tan(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::tan(0.5));
}

TEST(ONodeValues, SqrtIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 4.0);
  ONode<double> n = graph_sqrt(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 2.0);
}

TEST(ONodeValues, AbsIsElementWiseOnNegativeInput) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, -3.0);
  ONode<double> n = graph_abs(&a);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], 3.0);
}

TEST(ONodeValues, PowIsElementWise) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 2.0), b(arena, Shape{3}, 3.0);
  ONode<double> n = graph_pow(&a, &b);
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(n[i], std::pow(2.0, 3.0));
}

TEST(ONodeValues, DotComputesMatrixProduct) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 0.0), b(arena, Shape{3, 2}, 0.0);
  const double av[2][3] = {{1, 2, 3}, {4, 5, 6}};
  const double bv[3][2] = {{7, 8}, {9, 10}, {11, 12}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) a[i, j] = av[i][j];
  for (int64_t i = 0; i < 3; ++i)
    for (int64_t j = 0; j < 2; ++j) b[i, j] = bv[i][j];

  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_DOUBLE_EQ((n[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((n[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((n[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((n[1, 1]), 154.0);
}

TEST(ONodeValues, DotVecVecIsInnerProduct) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{3}, 0.0), b(arena, Shape{3}, 0.0);
  for (int64_t i = 0; i < 3; ++i) { a[i] = i + 1.0; b[i] = i + 4.0; }
  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{1}));
  EXPECT_DOUBLE_EQ(n.item(), 32.0);   // 1*4 + 2*5 + 3*6
}

TEST(ONodeValues, DotMatVecIsMatrixVectorProduct) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 0.0), b(arena, Shape{3}, 0.0);
  const double av[2][3] = {{1, 2, 3}, {4, 5, 6}};
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j) a[i, j] = av[i][j];
  for (int64_t i = 0; i < 3; ++i) b[i] = i + 1.0;   // {1, 2, 3}

  ONode<double> n = graph_dot(&a, &b);
  EXPECT_EQ(n.shape(), (Shape{2}));
  EXPECT_DOUBLE_EQ(n[0], 14.0);   // 1 + 4 + 9
  EXPECT_DOUBLE_EQ(n[1], 32.0);   // 4 + 10 + 18
}

// ---------------------------------------------------------------------------
//  GradBuffer
// ---------------------------------------------------------------------------

TEST(GradBuffer, GradHasSameShapeAsValue) {
  CArena<double> arena;
  VNode<double> v(arena, Shape{3, 4}, 1.0);
  EXPECT_EQ(v.grad().shape(), (Shape{3, 4}));
}

TEST(GradBuffer, GradIsZeroInitialized) {
  CArena<double> arena;
  VNode<double> v(arena, Shape{4}, 5.0);
  const double* pg = v.grad().data();
  for (int64_t i = 0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(pg[i], 0.0);
}

TEST(GradBuffer, ZeroGradResetsToZero) {
  CArena<double> arena;
  VNode<double> v(arena, Shape{3}, 1.0);
  v.grad().data()[0] = 9.9;
  v.grad().data()[1] = 8.8;
  v.zero_grad();
  const double* pg = v.grad().data();
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(pg[i], 0.0);
}

TEST(GradBuffer, ONodeGradHasSameShapeAsValue) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{2, 3}, 1.0);
  ONode<double> n = graph_add(&a, &b);
  EXPECT_EQ(n.grad().shape(), n.shape());
}

// ---------------------------------------------------------------------------
//  Operator overloads
// ---------------------------------------------------------------------------

TEST(OperatorOverload, PlusBuildsAddNode) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 3.0), b(arena, Shape{2}, 4.0);
  ONode<double> n = a + b;
  EXPECT_EQ(n.op(), Op::Add);
  EXPECT_DOUBLE_EQ(n.data()[0], 7.0);
}

TEST(OperatorOverload, MinusBuildsSubNode) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 5.0), b(arena, Shape{2}, 2.0);
  ONode<double> n = a - b;
  EXPECT_EQ(n.op(), Op::Sub);
  EXPECT_DOUBLE_EQ(n.data()[0], 3.0);
}

TEST(OperatorOverload, StarBuildsHadamardNode) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 3.0), b(arena, Shape{2}, 4.0);
  ONode<double> n = a * b;
  EXPECT_EQ(n.op(), Op::Hadamard);
  EXPECT_DOUBLE_EQ(n.data()[0], 12.0);
}

TEST(OperatorOverload, SlashBuildsDivNode) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 6.0), b(arena, Shape{2}, 3.0);
  ONode<double> n = a / b;
  EXPECT_EQ(n.op(), Op::Div);
  EXPECT_DOUBLE_EQ(n.data()[0], 2.0);
}

TEST(OperatorOverload, CaretBuildsPowNode) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  ONode<double> n = a ^ b;
  EXPECT_EQ(n.op(), Op::Pow);
  EXPECT_DOUBLE_EQ(n.data()[0], 8.0);
}

TEST(OperatorOverload, AmpersandBuildsDotNode) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2, 3}, 1.0), b(arena, Shape{3, 2}, 1.0);
  ONode<double> n = a & b;
  EXPECT_EQ(n.op(), Op::Dot);
  EXPECT_EQ(n.shape(), (Shape{2, 2}));
  EXPECT_DOUBLE_EQ(n.data()[0], 3.0);
}

TEST(OperatorOverload, ChainedOpsLinkCorrectly) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 2.0), b(arena, Shape{2}, 3.0);
  ONode<double> ab = a + b;        // 5
  ONode<double> abxa = ab * a;     // 10
  EXPECT_DOUBLE_EQ(abxa.data()[0], 10.0);
  EXPECT_EQ(abxa.left(),  static_cast<Node<double>*>(&ab));
  EXPECT_EQ(abxa.right(), static_cast<Node<double>*>(&a));
}

// ---------------------------------------------------------------------------
//  NodeName
// ---------------------------------------------------------------------------

TEST(NodeName, VNodeAutoNameUsesCounter) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  VNode<double> b(arena, Shape{2}, 1.0);
  EXPECT_EQ(a.name(), "var_0");
  EXPECT_EQ(b.name(), "var_1");
}

TEST(NodeName, VNodeCustomNameIsUsed) {
  CArena<double> arena;
  VNode<double> v(arena, Shape{2}, "weights", 1.0);
  EXPECT_EQ(v.name(), "weights");
}

TEST(NodeName, VNodeNamesAreDistinct) {
  CArena<double> arena;
  VNode<double> a(arena, Shape{2}, 1.0);
  VNode<double> b(arena, Shape{2}, 1.0);
  EXPECT_NE(a.name(), b.name());
}

TEST(NodeName, CNodeAutoNameUsesCounter) {
  CArena<double> arena;
  CNode<double> a(arena, Shape{2}, 1.0);
  CNode<double> b(arena, Shape{2}, 1.0);
  EXPECT_EQ(a.name(), "const_0");
  EXPECT_EQ(b.name(), "const_1");
}

TEST(NodeName, CNodeCustomNameIsUsed) {
  CArena<double> arena;
  CNode<double> c(arena, Shape{2}, "bias", 0.0);
  EXPECT_EQ(c.name(), "bias");
}

TEST(NodeName, VNodeAndCNodeCountersAreIndependent) {
  CArena<double> arena;
  VNode<double> v(arena, Shape{2}, 1.0);
  CNode<double> c(arena, Shape{2}, 1.0);
  EXPECT_EQ(v.name(), "var_0");
  EXPECT_EQ(c.name(), "const_0");
}

} // namespace
