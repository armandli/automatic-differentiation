#ifndef C_GRAPH_DIAGRAM
#define C_GRAPH_DIAGRAM

#include <c_graph.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace autodiff {

namespace s = std;

// ---------------------------------------------------------------------------
//  Mermaid visualization of a computation graph.
//
//  write_mermaid(out, root) / create_diagram(path, root) walk every node
//  reachable from `root` (the final node of a reverse-AD graph) and emit a
//  Mermaid `flowchart LR`: leaves on the left, the root on the right, edges
//  pointing operand -> result. Shared subexpressions are drawn once. Node
//  shape encodes the kind — variable ([stadium]), constant ([rect]),
//  operation ({{hexagon}}) — and each carries a `:::var` / `:::const` /
//  `:::op` class for coloring.
//
//  Paste the output between ```mermaid fences, or render the file with
//  `mmdc -i graph.mmd -o graph.svg`.
// ---------------------------------------------------------------------------
namespace detail {

inline const char* op_name(Op op) {
  switch (op) {
    case Op::Add:      return "Add";
    case Op::Sub:      return "Sub";
    case Op::Neg:      return "Neg";
    case Op::Hadamard: return "Hadamard";
    case Op::Dot:      return "Dot";
    case Op::Div:      return "Div";
    case Op::Sum:      return "Sum";
    case Op::Max:      return "Max";
    case Op::Min:      return "Min";
    case Op::Mean:     return "Mean";
    case Op::Softmax:             return "Softmax";
    case Op::CrossEntropy:        return "CrossEntropy";
    case Op::SoftmaxCrossEntropy: return "SoftmaxCrossEntropy";
    case Op::Where:              return "Where";
    case Op::Exp:      return "Exp";
    case Op::Log:      return "Log";
    case Op::Sin:      return "Sin";
    case Op::Cos:      return "Cos";
    case Op::Tan:      return "Tan";
    case Op::Sqrt:     return "Sqrt";
    case Op::Abs:      return "Abs";
    case Op::Pow:      return "Pow";
  }
  return "?";   // unreachable; satisfies -Wreturn-type
}

// Make a node name safe to drop inside a quoted Mermaid label. Single pass, so
// the '&' -> "&amp;" rewrite never re-escapes text it just inserted.
inline s::string escape(s::string_view text) {
  s::string out;
  out.reserve(text.size());
  for (char c : text)
    switch (c) {
      case '&': out += "&amp;";  break;
      case '"': out += "&quot;"; break;
      case '<': out += "&lt;";   break;
      case '>': out += "&gt;";   break;
      default:  out += c;        break;
    }
  return out;
}

inline s::string shape_str(const Shape& shape) {
  s::string out = "(";
  const s::vector<int64_t>& dims = shape.dims();
  for (s::size_t i = 0; i < dims.size(); ++i) {
    if (i) out += ", ";
    out += s::to_string(dims[i]);
  }
  out += ")";
  return out;
}

template <typename T>
struct mermaid_writer {
  s::ostream& out;
  s::unordered_map<const Node<T>*, int> ids;
  int next = 0;

  explicit mermaid_writer(s::ostream& o) : out(o) {}

  // Emit `n`'s declaration and its incoming edges (recursing into inputs first);
  // return the small integer id assigned to it. A node already seen is not
  // re-emitted — its stored id is returned so a shared subexpression links once.
  int visit(const Node<T>& n) {
    if (auto it = ids.find(&n); it != ids.end())
      return it->second;
    const int id = next++;
    ids.emplace(&n, id);

    out << "  n" << id;
    switch (n.mKind) {
      case NodeKind::Variable:
        out << "([\"" << escape(n.name()) << "<br/>" << shape_str(n.shape())
            << "\"]):::var\n";
        break;
      case NodeKind::Constant:
        out << "[\"" << escape(n.name());
        if (n.size() == 1) out << " = " << n.data()[0];
        out << "<br/>" << shape_str(n.shape()) << "\"]:::const\n";
        break;
      case NodeKind::Operation: {
        const ONode<T>& o = static_cast<const ONode<T>&>(n);
        out << "{{\"" << op_name(o.op());
        if (o.op() == Op::Sum or o.op() == Op::Max or o.op() == Op::Min or
            o.op() == Op::Mean or o.op() == Op::Softmax or
            o.op() == Op::CrossEntropy or o.op() == Op::SoftmaxCrossEntropy) {
          if (o.axis() >= 0) out << " axis=" << o.axis();
          else               out << " (all)";
        }
        out << "<br/>" << shape_str(n.shape()) << "\"}}:::op\n";

        const int l = visit(*o.left());
        if (o.right()) {
          const int r = visit(*o.right());
          out << "  n" << l << " -->|L| n" << id << "\n";
          out << "  n" << r << " -->|R| n" << id << "\n";
        } else {
          out << "  n" << l << " --> n" << id << "\n";
        }
        if (o.cond()) {
          const int c = visit(*o.cond());
          out << "  n" << c << " -->|C| n" << id << "\n";
        }
        break;
      }
    }
    return id;
  }
};

} // detail

// Write a Mermaid flowchart of the graph rooted at `root` to `out`.
template <typename T>
void write_mermaid(s::ostream& out, const Node<T>& root) {
  out << "%% autodiff computation graph";
  if (not root.name().empty())
    out << ", root = " << detail::escape(root.name());
  else if (root.mKind == NodeKind::Operation)
    out << ", root = " << detail::op_name(static_cast<const ONode<T>&>(root).op());
  out << "\n";
  out << "flowchart LR\n";
  out << "  classDef var fill:#dceccb,stroke:#5a8a3c,color:#1e2b12\n";
  out << "  classDef const fill:#f2e9c9,stroke:#a8862f,color:#2b2410\n";
  out << "  classDef op fill:#cfe0f2,stroke:#3d72ab,color:#0f2036\n";
  detail::mermaid_writer<T>{out}.visit(root);
}

// Write that same diagram to the file at `path` (created / truncated).
template <typename T>
void create_diagram(const s::string& path, const Node<T>& root) {
  s::ofstream ofs(path);
  assert(ofs.is_open() and "create_diagram: could not open output file");
  write_mermaid(ofs, root);
}

} // autodiff

#endif//C_GRAPH_DIAGRAM
