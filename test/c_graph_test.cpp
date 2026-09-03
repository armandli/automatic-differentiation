#include <c_graph.h>

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

namespace {

using autodiff::CArray;
using autodiff::CNode;
using autodiff::Node;
using autodiff::ONode;
using autodiff::Op;
using autodiff::Shape;
using autodiff::VNode;
using autodiff::graph_add;
using autodiff::graph_div;
using autodiff::graph_exp;
using autodiff::graph_log;
using autodiff::graph_mul;
using autodiff::graph_sub;
namespace stats = autodiff::graph_stats;

// RAII guard for heap-allocated ONodes so tests do not leak.
template <typename T>
struct ONodeGuard {
  ONode<T>* ptr;
  explicit ONodeGuard(ONode<T>* p) : ptr(p) {}
  ~ONodeGuard() { delete ptr; }
  ONodeGuard(const ONodeGuard&) = delete;
  ONodeGuard& operator=(const ONodeGuard&) = delete;
  ONode<T>* operator->() const { return ptr; }
  ONode<T>& operator*()  const { return *ptr; }
};

// ---------------------------------------------------------------------------
//  Counters
// ---------------------------------------------------------------------------

TEST(Counters, VNodeIncrementsVNodeCounter) {
  const int64_t before = stats::vnode_count.load();
  VNode<double> v(Shape{2, 3}, 1.0);
  EXPECT_EQ(stats::vnode_count.load(), before + 1);
}

TEST(Counters, CNodeIncrementsCNodeCounter) {
  const int64_t before = stats::cnode_count.load();
  CNode<double> c(Shape{2, 3}, 1.0);
  EXPECT_EQ(stats::cnode_count.load(), before + 1);
}

TEST(Counters, AddIncrementsAddCounter) {
  VNode<double> a(Shape{2}, 1.0), b(Shape{2}, 2.0);
  const int64_t before = stats::onode_add_count.load();
  ONodeGuard<double> n(graph_add(&a, &b));
  EXPECT_EQ(stats::onode_add_count.load(), before + 1);
}

TEST(Counters, SubIncrementsSubCounter) {
  VNode<double> a(Shape{2}, 1.0), b(Shape{2}, 2.0);
  const int64_t before = stats::onode_sub_count.load();
  ONodeGuard<double> n(graph_sub(&a, &b));
  EXPECT_EQ(stats::onode_sub_count.load(), before + 1);
}

TEST(Counters, MulIncrementsMulCounter) {
  VNode<double> a(Shape{2}, 1.0), b(Shape{2}, 2.0);
  const int64_t before = stats::onode_mul_count.load();
  ONodeGuard<double> n(graph_mul(&a, &b));
  EXPECT_EQ(stats::onode_mul_count.load(), before + 1);
}

TEST(Counters, DivIncrementsDivCounter) {
  VNode<double> a(Shape{2}, 1.0), b(Shape{2}, 2.0);
  const int64_t before = stats::onode_div_count.load();
  ONodeGuard<double> n(graph_div(&a, &b));
  EXPECT_EQ(stats::onode_div_count.load(), before + 1);
}

TEST(Counters, ExpIncrementsExpCounter) {
  VNode<double> a(Shape{2}, 1.0);
  const int64_t before = stats::onode_exp_count.load();
  ONodeGuard<double> n(graph_exp(&a));
  EXPECT_EQ(stats::onode_exp_count.load(), before + 1);
}

TEST(Counters, LogIncrementsLogCounter) {
  VNode<double> a(Shape{2}, 1.0);
  const int64_t before = stats::onode_log_count.load();
  ONodeGuard<double> n(graph_log(&a));
  EXPECT_EQ(stats::onode_log_count.load(), before + 1);
}

// ---------------------------------------------------------------------------
//  ONodeShape
// ---------------------------------------------------------------------------

TEST(ONodeShape, BinaryOpPreservesShape) {
  VNode<double> a(Shape{3, 4}, 1.0), b(Shape{3, 4}, 2.0);
  ONodeGuard<double> n(graph_add(&a, &b));
  EXPECT_EQ(n->shape(), (Shape{3, 4}));
}

TEST(ONodeShape, UnaryOpPreservesShape) {
  VNode<double> a(Shape{2, 3, 4}, 1.0);
  ONodeGuard<double> n(graph_exp(&a));
  EXPECT_EQ(n->shape(), (Shape{2, 3, 4}));
}

// ---------------------------------------------------------------------------
//  ONodeLinks
// ---------------------------------------------------------------------------

TEST(ONodeLinks, BinaryOpLinksInputNodes) {
  VNode<double> a(Shape{2}, 1.0), b(Shape{2}, 2.0);
  ONodeGuard<double> n(graph_add(&a, &b));
  EXPECT_EQ(n->left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n->right(), static_cast<Node<double>*>(&b));
}

TEST(ONodeLinks, UnaryOpHasNullRight) {
  VNode<double> a(Shape{2}, 1.0);
  ONodeGuard<double> n(graph_exp(&a));
  EXPECT_EQ(n->left(),  static_cast<Node<double>*>(&a));
  EXPECT_EQ(n->right(), nullptr);
}

TEST(ONodeLinks, OpEnumMatchesOperation) {
  VNode<double> a(Shape{2}, 1.0), b(Shape{2}, 2.0);
  ONodeGuard<double> add_n(graph_add(&a, &b));
  ONodeGuard<double> sub_n(graph_sub(&a, &b));
  ONodeGuard<double> mul_n(graph_mul(&a, &b));
  ONodeGuard<double> div_n(graph_div(&a, &b));
  ONodeGuard<double> exp_n(graph_exp(&a));
  ONodeGuard<double> log_n(graph_log(&a));
  EXPECT_EQ(add_n->op(), Op::Add);
  EXPECT_EQ(sub_n->op(), Op::Sub);
  EXPECT_EQ(mul_n->op(), Op::Mul);
  EXPECT_EQ(div_n->op(), Op::Div);
  EXPECT_EQ(exp_n->op(), Op::Exp);
  EXPECT_EQ(log_n->op(), Op::Log);
}

// ---------------------------------------------------------------------------
//  ONodeValues
// ---------------------------------------------------------------------------

TEST(ONodeValues, AddIsElementWise) {
  VNode<double> a(Shape{3}, 3.0), b(Shape{3}, 2.0);
  ONodeGuard<double> n(graph_add(&a, &b));
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ((*n)[i], 5.0);
}

TEST(ONodeValues, SubIsElementWise) {
  VNode<double> a(Shape{3}, 5.0), b(Shape{3}, 2.0);
  ONodeGuard<double> n(graph_sub(&a, &b));
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ((*n)[i], 3.0);
}

TEST(ONodeValues, MulIsElementWise) {
  VNode<double> a(Shape{3}, 3.0), b(Shape{3}, 4.0);
  ONodeGuard<double> n(graph_mul(&a, &b));
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ((*n)[i], 12.0);
}

TEST(ONodeValues, DivIsElementWise) {
  VNode<double> a(Shape{3}, 6.0), b(Shape{3}, 2.0);
  ONodeGuard<double> n(graph_div(&a, &b));
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ((*n)[i], 3.0);
}

TEST(ONodeValues, ExpIsElementWise) {
  VNode<double> a(Shape{3}, 2.0);
  ONodeGuard<double> n(graph_exp(&a));
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ((*n)[i], std::exp(2.0));
}

TEST(ONodeValues, LogIsElementWise) {
  VNode<double> a(Shape{3}, std::exp(1.0));
  ONodeGuard<double> n(graph_log(&a));
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ((*n)[i], 1.0);
}

// ---------------------------------------------------------------------------
//  GradBuffer
// ---------------------------------------------------------------------------

TEST(GradBuffer, GradHasSameShapeAsValue) {
  VNode<double> v(Shape{3, 4}, 1.0);
  EXPECT_EQ(v.grad().shape(), (Shape{3, 4}));
}

TEST(GradBuffer, GradIsZeroInitialized) {
  VNode<double> v(Shape{4}, 5.0);
  const double* pg = v.grad().data();
  for (int64_t i = 0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(pg[i], 0.0);
}

TEST(GradBuffer, ZeroGradResetsToZero) {
  VNode<double> v(Shape{3}, 1.0);
  v.grad().data()[0] = 9.9;
  v.grad().data()[1] = 8.8;
  v.zero_grad();
  const double* pg = v.grad().data();
  for (int64_t i = 0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(pg[i], 0.0);
}

TEST(GradBuffer, ONodeGradHasSameShapeAsValue) {
  VNode<double> a(Shape{2, 3}, 1.0), b(Shape{2, 3}, 1.0);
  ONodeGuard<double> n(graph_add(&a, &b));
  EXPECT_EQ(n->grad().shape(), n->shape());
}

// ---------------------------------------------------------------------------
//  Operator overloads
// ---------------------------------------------------------------------------

TEST(OperatorOverload, PlusBuildsAddNode) {
  VNode<double> a(Shape{2}, 3.0), b(Shape{2}, 4.0);
  ONodeGuard<double> n(a + b);
  EXPECT_EQ(n->op(), Op::Add);
  EXPECT_DOUBLE_EQ(n->data()[0], 7.0);
}

TEST(OperatorOverload, MinusBuildsSubNode) {
  VNode<double> a(Shape{2}, 5.0), b(Shape{2}, 2.0);
  ONodeGuard<double> n(a - b);
  EXPECT_EQ(n->op(), Op::Sub);
  EXPECT_DOUBLE_EQ(n->data()[0], 3.0);
}

TEST(OperatorOverload, StarBuildsMulNode) {
  VNode<double> a(Shape{2}, 3.0), b(Shape{2}, 4.0);
  ONodeGuard<double> n(a * b);
  EXPECT_EQ(n->op(), Op::Mul);
  EXPECT_DOUBLE_EQ(n->data()[0], 12.0);
}

TEST(OperatorOverload, SlashBuildsDivNode) {
  VNode<double> a(Shape{2}, 6.0), b(Shape{2}, 3.0);
  ONodeGuard<double> n(a / b);
  EXPECT_EQ(n->op(), Op::Div);
  EXPECT_DOUBLE_EQ(n->data()[0], 2.0);
}

TEST(OperatorOverload, ChainedOpsLinkCorrectly) {
  VNode<double> a(Shape{2}, 2.0), b(Shape{2}, 3.0);
  ONodeGuard<double> ab(a + b);        // 5
  ONodeGuard<double> abxa((*ab) * a);  // 10
  EXPECT_DOUBLE_EQ(abxa->data()[0], 10.0);
  EXPECT_EQ(abxa->left(),  static_cast<Node<double>*>(ab.ptr));
  EXPECT_EQ(abxa->right(), static_cast<Node<double>*>(&a));
}

} // namespace
