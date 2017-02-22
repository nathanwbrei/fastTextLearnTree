#ifndef FASTTEXT_LOMTREE_H
#define FASTTEXT_LOMTREE_H

#include <vector>
#include <utility>
#include <memory>
#include <boost/unordered_map.hpp>
#include <queue>

#include <assert.h> 

#include "vector.h"
#include "real.h"

namespace fasttext {


struct AuxPair {
    int32_t n;
    real    v;
    AuxPair(int32_t a, real x) : n(a), v(x) {}
    bool operator<(const struct AuxPair& other) const { return v < other.v; }
};


struct AuxTriple {
    int32_t i;
    int32_t j;
    real    v;
    AuxTriple(int32_t n1, int32_t n2, real x) : i(n1), j(n2), v(x) {}
    bool operator<(const struct AuxTriple& other) const { return v < other.v; }
};


struct NodeLOM {
  int32_t parent;
  std::vector<int32_t> children;
  int64_t count;
  int32_t pindex;
  real * probas;
  // Node stats
  std::vector<real> q;
  std::vector<real> p;
  std::vector<std::vector<real>> p_cond;
  std::vector<std::vector<real>> grad_j;
  // Auxiliary variables
  std::vector<bool> assigned;
  std::priority_queue<AuxPair> children_queue;
  std::priority_queue<AuxTriple> sort_queue;
  std::vector<std::vector<int32_t>> children_labels;
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
    // AUXILIARY VARIABLES
    std::vector<int32_t> freeNodeIds;

  public:
    LOMtree();
    ~LOMtree();
  
    void updateStats(int32_t, int32_t, Vector&);
    void updatePaths();
    void initNodeStats(int32_t);
    void buildLOMTree(const std::vector<int64_t>&, int32_t);
    
    void computeNodeStats(int32_t);
    void assignNodeChildren(int32_t);
    void updateNode(int32_t);
    void updateTree();
    
    int32_t getNLeaves();
    int32_t getNNodes();
    NodeLOM getNode(int32_t);
};

}

#endif
