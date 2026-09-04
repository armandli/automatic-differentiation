#include <c_graph.h>
#include <c_graph_diagram.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using autodiff::CArena;
using autodiff::CArray;
using autodiff::Node;
using autodiff::Shape;
using autodiff::VNode;
using autodiff::create_diagram;
using autodiff::write_mermaid;

std::string mermaid_of(const Node<double>& root) {
  std::ostringstream os;
  write_mermaid(os, root);
  return os.str();
}

int count_substr(const std::string& hay, const std::string& needle) {
  int n = 0;
  for (std::size_t p = hay.find(needle); p != std::string::npos;
       p = hay.find(needle, p + needle.size()))
    ++n;
  return n;
}

bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------

TEST(GraphDiagram, HeaderHasFlowchartAndClassDefs) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 1.0);
  auto& n = x + x;
  const std::string m = mermaid_of(n);
  EXPECT_TRUE(contains(m, "flowchart LR"));
  EXPECT_TRUE(contains(m, "classDef var "));
  EXPECT_TRUE(contains(m, "classDef const "));
  EXPECT_TRUE(contains(m, "classDef op "));
  EXPECT_TRUE(contains(m, "%% autodiff computation graph, root = Add"));
}

TEST(GraphDiagram, LinearExpressionNodesAndEdges) {
  CArena arena;
  CArray<double> w(arena, Shape{3}, 0.5);
  w.set_requires_grad();
  CArray<double> x(arena, Shape{3}, 2.0);
  auto& y = w * x + 1.0;
  const std::string m = mermaid_of(y);

  EXPECT_EQ(count_substr(m, "]):::var"), 1);     // w
  EXPECT_EQ(count_substr(m, "]:::const"), 2);    // x and the literal 1
  EXPECT_EQ(count_substr(m, "}}:::op"), 2);      // Hadamard, Add
  EXPECT_TRUE(contains(m, "\"Hadamard<br/>(3)\""));
  EXPECT_TRUE(contains(m, "\"Add<br/>(3)\""));
  EXPECT_TRUE(contains(m, "var_0<br/>(3)"));

  EXPECT_EQ(count_substr(m, "-->"), 4);
  EXPECT_EQ(count_substr(m, "-->|L|"), 2);
  EXPECT_EQ(count_substr(m, "-->|R|"), 2);
}

TEST(GraphDiagram, SharedSubexpressionDrawnOnce) {
  CArena arena;
  CArray<double> x(arena, Shape{2}, 3.0);
  x.set_requires_grad();
  auto& sq = x * x;
  const std::string m = mermaid_of(sq);

  EXPECT_EQ(count_substr(m, "]):::var"), 1);     // one leaf, not two
  EXPECT_EQ(count_substr(m, "}}:::op"), 1);
  EXPECT_EQ(count_substr(m, "-->"), 2);
  // both edges originate from the same node id and target the same node id
  EXPECT_TRUE(contains(m, "n1 -->|L| n0"));
  EXPECT_TRUE(contains(m, "n1 -->|R| n0"));
}

TEST(GraphDiagram, UnaryOpEdgeIsUnlabeled) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 0.5);
  auto& n = exp(x);
  const std::string m = mermaid_of(n);
  EXPECT_TRUE(contains(m, "\"Exp<br/>(3)\""));
  EXPECT_EQ(count_substr(m, "-->"), 1);
  EXPECT_EQ(count_substr(m, "-->|"), 0);         // no L/R label on a unary edge
}

TEST(GraphDiagram, ReductionLabelShowsAxis) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(sum(x, 1)), "\"Sum axis=1<br/>(2)\""));

  CArray<double> z(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(sum(z)), "\"Sum (all)<br/>(1)\""));

  CArray<double> w(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(mean(w, 1)), "\"Mean axis=1<br/>(2)\""));

  CArray<double> v(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(mean(v)), "\"Mean (all)<br/>(1)\""));

  CArray<double> s(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(softmax(s, 1)), "\"Softmax axis=1<br/>(2, 3)\""));

  CArray<double> t(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(softmax(t)), "\"Softmax (all)<br/>(2, 3)\""));
}

TEST(GraphDiagram, TransposeLabelShowsNoAxes) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(transpose(x)), "\"Transpose<br/>(3, 2)\""));

  CArray<double> y(arena, Shape{2, 3, 4}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(transpose(y)), "\"Transpose<br/>(4, 3, 2)\""));
}

TEST(GraphDiagram, PermuteLabelShowsAxes) {
  CArena arena;
  CArray<double> x(arena, Shape{2, 3}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(permute(x, std::vector<int64_t>{1, 0})),
                       "\"Permute axes=[1,0]<br/>(3, 2)\""));

  CArray<double> y(arena, Shape{2, 3, 4}, 1.0);
  EXPECT_TRUE(contains(mermaid_of(permute(y, std::vector<int64_t>{1, 2, 0})),
                       "\"Permute axes=[1,2,0]<br/>(3, 4, 2)\""));
}

TEST(GraphDiagram, CrossEntropyIsBinarySoftmaxIsUnary) {
  CArena arena;
  CArray<double> p(arena, Shape{2, 3}, 0.3);
  CArray<double> y(arena, Shape{2}, 0.0);
  const std::string ce = mermaid_of(cross_entropy(p, y, 1));
  EXPECT_TRUE(contains(ce, "\"CrossEntropy axis=1<br/>(2)\""));
  EXPECT_EQ(count_substr(ce, "-->|L|"), 1);
  EXPECT_EQ(count_substr(ce, "-->|R|"), 1);

  CArray<double> x(arena, Shape{3}, 0.5);
  const std::string sm = mermaid_of(softmax(x));
  EXPECT_EQ(count_substr(sm, "-->|"), 0);          // unary: no L/R label
}

TEST(GraphDiagram, ConstantScalarShowsValue) {
  CArena arena;
  CArray<double> x(arena, Shape{3}, 1.0);
  auto& n = x + 2.0;
  EXPECT_TRUE(contains(mermaid_of(n), " = 2<br/>(1)"));
}

TEST(GraphDiagram, CustomNameIsEscaped) {
  CArena arena;
  VNode<double> v(arena, Shape{2}, "L<R>&\"q\"", 1.0);
  const std::string m = mermaid_of(v);
  EXPECT_TRUE(contains(m, "L&lt;R&gt;&amp;&quot;q&quot;"));
  EXPECT_FALSE(contains(m, "L<R>"));
}

TEST(GraphDiagram, CreateDiagramRoundTrips) {
  CArena arena;
  CArray<double> a(arena, Shape{2}, 1.0);
  CArray<double> b(arena, Shape{2}, 2.0);
  auto& y = a * b - 1.0;

  const std::string path = ::testing::TempDir() + "autodiff_graph.mmd";
  create_diagram(path, y);

  std::ifstream in(path);
  ASSERT_TRUE(in.is_open());
  std::stringstream buf;
  buf << in.rdbuf();
  EXPECT_FALSE(buf.str().empty());
  EXPECT_EQ(buf.str(), mermaid_of(y));
}

TEST(GraphDiagram, MixedTypeGraphRendersConvertedOperandAsConstant) {
  CArena arena;
  CArray<float> x(arena, Shape{2, 2}, 1.0f);
  x.set_requires_grad();
  CArray<bool> mask(arena, Shape{2, 2}, true);
  auto& y = x * mask;                      // bool operand converts to a const leaf

  std::ostringstream os;
  write_mermaid(os, y);
  const std::string m = os.str();
  EXPECT_TRUE(contains(m, "\"Hadamard<br/>(2, 2)\""));
  EXPECT_TRUE(contains(m, "]:::const\n"));   // the converted mask
  EXPECT_TRUE(contains(m, "]):::var\n"));    // the float variable
  EXPECT_EQ(count_substr(m, "-->|L|"), 1);
  EXPECT_EQ(count_substr(m, "-->|R|"), 1);
}

TEST(GraphDiagram, WhereHasThreeLabeledEdges) {
  CArena arena;
  CArray<bool>   c(arena, Shape{2}, true);
  CArray<double> a(arena, Shape{2}, 1.0);
  CArray<double> b(arena, Shape{2}, 2.0);
  auto& n = where(c, a, b);
  const std::string m = mermaid_of(n);
  EXPECT_TRUE(contains(m, "\"Where<br/>(2)\""));
  EXPECT_EQ(count_substr(m, "-->|L|"), 1);   // a
  EXPECT_EQ(count_substr(m, "-->|R|"), 1);   // b
  EXPECT_EQ(count_substr(m, "-->|C|"), 1);   // cond
}

} // namespace
