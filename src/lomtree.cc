#include "lomtree.h"

#include <algorithm>


namespace fasttext {


LOMtree::LOMtree() {
}


LOMtree::~LOMtree() {
  // remove probas and acc_probas
  if (treeLOM.size() > 0) {
    for (int32_t i = nleaves_; i < nnodes_; i++) {
      delete[] treeLOM[i].probas;
    }
  }
  for (auto it = acc_probas.begin(); it!= acc_probas.end(); ++it) {
    delete[] it->second;
  }
}


void LOMtree::updateNode(int32_t node) {
  int32_t ndesc = node_labels[node].size();
  if (ndesc == 0) return;
  std::pair<int32_t, int32_t> key;
  // q_i : proportion of label i = acc_counts[(node, i)] / sum_{i' \in node_labels[node]} acc_counts[(node, i)]
  real q_sum = 0;
  for (int32_t i = 0; i < ndesc; i++){
    key = std::make_pair(node, node_labels[node][i]);
    treeLOM[node].q[i] = acc_counts[key];
    q_sum += treeLOM[node].q[i];
  }
  for (int32_t i = 0; i < ndesc; i++){
    treeLOM[node].q[i] /= q_sum;
  }
  // p_ji: proportion of child j for label i
  // p_j : proportion of child j
  real p_sum = 0;
  for (int32_t i = 0; i < ndesc; i++){
    for (int32_t j = 0; j < arity_; j++) {
      key = std::make_pair(node, node_labels[node][i]);
      treeLOM[node].p_cond[i][j] = acc_probas[key][j] / acc_counts[key];
      treeLOM[node].p[j] += acc_probas[key][j];
      p_sum += acc_probas[key][j];
    }
  }
  for (int32_t j = 0; j < arity_; j++) {
    treeLOM[node].p[j] /= p_sum;
  }
  // sort by:
  // q_i * (1 - q_i) * sign(p_ji - pi) /?/ * p_ji
  
  // recurse
  for (int32_t i = 0; i < arity_; i++) {
    updateNode(treeLOM[node].children[i]);
  }
}

// update tree structure as per LOMtree algorithm
void LOMtree::updateTree() {
  updateNode(nleaves_);
}


// update node stats during forwards pass
void LOMtree::updateStats(int32_t node, int32_t label, real * probas) {
  std::pair<int32_t, int32_t> key(node, label);
  for (int32_t i = 0; i < arity_; i++) acc_probas[key][i] += probas[i];
}


// Build initial tree with huffman coding
void LOMtree::buildLOMTree(const std::vector<int64_t>& counts,
                           int32_t arity, int32_t max_nodes) {
  // add 0 weight symbol to have well-formed n-ary tree
  arity_   = arity;
  nlabels_ = counts.size();
  if (max_nodes < nlabels_ / (arity_ - 1)) {
    nleaves_ = (nlabels_ / (arity_ - 1) + 1) * (arity_ - 1);
  } else {
    nleaves_ = max_nodes * (arity_ - 1);
  }
  nnodes_ = nleaves_ + (nleaves_ - 1) / (arity_ - 1);
  // initialize the tree
  treeLOM.resize(nnodes_);
  node_labels.resize(nnodes_);
  for (int32_t i = 0; i < nnodes_; i++) {
    treeLOM[i].parent = -1;
    treeLOM[i].count = 1e15;
    treeLOM[i].pindex = -1;
  }
  for (int32_t i = 0; i < nlabels_; i++) {
    treeLOM[i].count = counts[i];
    node_labels[i].insert(node_labels[i].end(), i);
  }
  for (int32_t i = nlabels_; i < nleaves_; i++) {
    treeLOM[i].count = 0;
  }
  // start n-ary huffman coding
  int32_t leaf = nleaves_ - 1;
  int32_t node = nleaves_;
  for (int32_t i = nleaves_; i < nnodes_; i++) {
    treeLOM[i].children.resize(arity_);
    treeLOM[i].probas = new real[arity_];
    int32_t ncount = 0;
    int32_t cid;
    for (int32_t j = 0; j < arity_; j++) {
      if (leaf >= 0 && treeLOM[leaf].count < treeLOM[node].count) {
        cid = leaf--;
      } else {
        cid = node++;
      }
      treeLOM[i].children[j] = cid;
      ncount += treeLOM[cid].count;
      treeLOM[cid].parent = i;
      treeLOM[cid].pindex = j;
    }
  }
  // build paths and codes
  for (int32_t i = 0; i < nlabels_; i++) {
    std::vector<int32_t> path;
    std::vector<int32_t> codeLOM;
    int32_t j = i;
    int32_t parent;
    while (treeLOM[j].parent != -1) {
      path.push_back(treeLOM[j].parent - nleaves_);
      codeLOM.push_back(treeLOM[j].pindex);
      parent = treeLOM[j].parent;
      node_labels[parent].insert(node_labels[parent].end(),
                                 node_labels[j].begin(),
                                 node_labels[j].end());
      j = parent;
    }
    paths.push_back(path);
    codesLOM.push_back(codeLOM);
  }
  // initialize node stats
  for (int32_t i = 0; i < nnodes_; i++) {
    for (int32_t j = 0; j < node_labels[i].size(); j++) {
      std::pair<int32_t, int32_t> key(i, node_labels[i][j]);
      acc_counts[key] = 0;
      acc_probas[key] = new real[arity];
      for (int32_t k = 0; k < arity_; k++) acc_probas[key][k] = 0.;
    }
  }
}

}
