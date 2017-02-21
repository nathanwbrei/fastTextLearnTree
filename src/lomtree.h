#ifndef FASTTEXT_MODEL_H
#define FASTTEXT_MODEL_H

#include <vector>
#include <utility>
#include <memory>
#include <boost/unordered_map.hpp>
#include <queue>

//~ #include "args.h"
//~ #include "matrix.h"
#include "vector.h"
#include "real.h"

namespace fasttext {

struct AuxTriple {
    int32_t i;
    int32_t j;
    real    v;

    AuxTriple(int32_t n1, int32_t n2, real x) : i(n1), j(n2), v(x) {}

    bool operator<(const struct S& other) const
    {
        return v < other.v;
    }
};

struct NodeLOM {
  int32_t parent;
  std::vector<int32_t> children;
  int64_t count;
  int32_t pindex;
  real * probas;
  std::vector<real> q;
  std::vector<real> p;
  std::vector<std::vector<real>> p_cond;
  std::vector<std::vector<real>> grad_j;
  std::priority_queue<AuxTriple> sort_queue;
};


class LOMtree {
  private:
    int32_t arity_;
    int32_t nlabels_;
    int32_t nleaves_;
    int32_t nnodes_;
    // NODE STATS:
    // acc_probas maps (node, label) to \sum_{(x, label)} p(child|label)
    // acc_counts maps (node, label) to \sum_{(x, label)} 1
    // node_labels[node] is the set of labels that currently reach a node
    boost::unordered_map< std::pair<int32_t, int32_t>, real *> acc_probas;
    boost::unordered_map<std::pair<int32_t, int32_t>, real> acc_counts;
    std::vector<std::vector<int32_t>> node_labels;
    // TREE STRUCTURE
    std::vector<NodeLOM> treeLOM;
    // top-down tree paths
    std::vector< std::vector<int32_t> > paths;
    std::vector< std::vector<int32_t> > codesLOM;

  public:
    LOMtree();
    ~LOMtree();
  
    void buildLOMTree(const std::vector<int64_t>&, int32_t, int32_t);
    void updateNode(int32_t);
    void updateTree();
    void updateStats(int32_t, int32_t, real*);
};

}

#endif
