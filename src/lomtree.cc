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


// update node stats during forwards pass
void LOMtree::updateStats(int32_t node, int32_t label, Vector& probas) {
  //~ printf("updating %d %d\n", node, label);
  std::pair<int32_t, int32_t> key(node, label);
  assert(acc_probas.count(key));
  acc_counts[key] += 1.;
  for (int32_t i = 0; i < arity_; i++) acc_probas[key][i] += probas[i];
  //~ printf("updated %d %d\n", node, label);
}


void LOMtree::printPath(int32_t node) {
  printf("PATH TO NODE %d : ", node);
  for (int32_t i = paths[node].size() - 1; i >= 0 ; i--) {
    printf("(%d, %d) ", paths[node][i], codesLOM[node][i]);
  }
  printf("\n");
}


// build paths and codes
void LOMtree::updatePaths() {
  paths.resize(0);
  codesLOM.resize(0);
  int32_t max_depth = 0;
  real tot_depth = 0;
  real weight_depth = 0;
  real tot_weight = 0;
  for (int32_t i = 0; i < nlabels_; i++) {
    std::vector<int32_t> path;
    std::vector<int32_t> codeLOM;
    int32_t j = i;
    int32_t parent;
    while (treeLOM[j].parent != -1) {
      path.push_back(treeLOM[j].parent - nleaves_);
      codeLOM.push_back(treeLOM[j].pindex);
      parent = treeLOM[j].parent;
      j = parent;
    }
    max_depth = (max_depth > path.size()) ? max_depth : path.size();
    tot_depth += path.size() + 1;
    weight_depth += (path.size() + 1) * treeLOM[i].count;
    tot_weight += treeLOM[i].count;
    paths.push_back(path);
    codesLOM.push_back(codeLOM);
  }
  printf("Tree built, depth %d, average %f, weighted average %f, %d leaves, %d labels, %d nodes \n",
         max_depth + 1, tot_depth / nlabels_, weight_depth / tot_weight, nleaves_, nlabels_, nnodes_);
  printPath(1105);
  printPath(20000);
}


void LOMtree::initNodeStats(int32_t node) {
  for (int32_t j = 0; j < node_labels[node].size(); j++) {
    std::pair<int32_t, int32_t> key(node, node_labels[node][j]);
    if (not acc_probas.count(key)) {
      acc_counts[key] = 1.;
      acc_probas[key] = new real[arity_];
      for (int32_t k = 0; k < arity_; k++) acc_probas[key][k] = 0.5 / (arity_);
      acc_probas[key][j % arity_] += 0.5;
    }
  }
}


// Build initial tree with huffman coding
void LOMtree::buildLOMTree(const std::vector<int64_t>& counts,
                           int32_t arity) {
  // add 0 weight symbol to have well-formed n-ary tree
  arity_   = arity;
  nlabels_ = counts.size();
  nleaves_ = (nlabels_ / (arity_ - 1) + 1) * (arity_ - 1) + 1;
  nnodes_  = nleaves_ + ((nleaves_ - 1) / (arity_ - 1));
  // random tree
  std::vector<int32_t> shuffle_vec;
  for (int32_t i = 0; i < nleaves_; i++) shuffle_vec.push_back(i);
  std::random_shuffle(shuffle_vec.begin(), shuffle_vec.end());
  std::bernoulli_distribution bern(1. - 1./(5 * arity_));
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
    node_labels[i].resize(0);
  }
  // start n-ary huffman coding
  int32_t leaf = nleaves_ - 1;
  int32_t node = nleaves_;
  for (int32_t i = nleaves_; i < nnodes_; i++) {
    treeLOM[i].children.resize(arity_);
    treeLOM[i].children_labels.resize(arity_);
    treeLOM[i].probas = new real[arity_];
    int32_t ncount = 0, curr_labels = 0;
    int32_t cid;
    for (int32_t j = 0; j < arity_; j++) {
      //~ if (leaf >= 0 && (node >= i or bern(generator))) { //TODO: random-ish tree
      if (leaf >= 0 && treeLOM[leaf].count < treeLOM[node].count) { //TODO: huffman option
        cid = leaf--;
        //~ cid = shuffle_vec[leaf--];
      } else {
        cid = node++;
      }
      assert(i >= node);
      treeLOM[i].children[j] = cid;
      ncount += treeLOM[cid].count;
      curr_labels += node_labels[cid].size();
      treeLOM[cid].parent = i;
      treeLOM[cid].pindex = j;
      node_labels[i].insert(node_labels[i].end(),
                            node_labels[cid].begin(),
                            node_labels[cid].end());
    }
    treeLOM[i].count = ncount;
    assert(node_labels[i].size() == curr_labels);
  }
  // initialize node stats
  for (int32_t i = 0; i < nnodes_; i++) {
    initNodeStats(i);
  }
  // build paths
  printf("building paths");
  updatePaths();
}


// compute useful node stats (q, p, p_cond, J gradients) based on acc_probas and acc_counts
// and pre-sorts (label, child) pairs according to dJ / dp_{j|i}
void LOMtree::computeNodeStats(int32_t node) {
  int32_t ndesc = node_labels[node].size();
  // resize stats
  treeLOM[node].q.resize(ndesc);
  treeLOM[node].p_cond.resize(ndesc);
  treeLOM[node].grad_j.resize(ndesc);
  //
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
  // p_{j|i}: proportion of child j for label i
  // p_j    : proportion of child j
  real p_sum = 0;
  treeLOM[node].p.resize(0);
  treeLOM[node].p.resize(arity_, 0.);
  for (int32_t i = 0; i < ndesc; i++){
    treeLOM[node].p_cond[i].resize(arity_);
    assert(acc_probas.count(key));
    key = std::make_pair(node, node_labels[node][i]);
    for (int32_t j = 0; j < arity_; j++) {
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
  for (int32_t i = 0; i < ndesc; i++){
    treeLOM[node].grad_j[i].resize(arity_);
    for (int32_t j = 0; j < arity_; j++) {
      treeLOM[node].grad_j[i][j] = treeLOM[node].q[i] * (1 - treeLOM[node].q[i]);
      treeLOM[node].grad_j[i][j] *= ((treeLOM[node].p_cond[i][j] > treeLOM[node].p[j]) ? 1.0 : -1.0);
      // TODO: decide
      //~ treeLOM[node].grad_j[i][j] *= treeLOM[node].p_cond[i][j]; // or not, depending on objective
      // pre-sort (label, child) pairs
      treeLOM[node].sort_queue.push(AuxTriple(i, j, treeLOM[node].grad_j[i][j]));
    }
  }
}


// assign labels to children based on stats
void LOMtree::assignNodeChildren(int32_t node) {
  int32_t ndesc = node_labels[node].size();
  // assign to children
  treeLOM[node].assigned.resize(0);
  treeLOM[node].assigned.resize(ndesc, false);
  for (int32_t j = 0; j < arity_; j++) {
    treeLOM[node].children_labels[j].resize(0);
  }
  int32_t needed = arity_;
  int32_t remaining = node_labels[node].size();
  int32_t i, cid, child_n_labels;
  while (not treeLOM[node].sort_queue.empty()){
    auto triple = treeLOM[node].sort_queue.top();
    treeLOM[node].sort_queue.pop();
    i = triple.i; cid = triple.j;
    child_n_labels = treeLOM[node].children_labels[cid].size();
    // keep proper number of labels in sub-trees
    if (not (treeLOM[node].assigned[i] or
             remaining <= needed and child_n_labels % (arity_ - 1) == 1)) {
      treeLOM[node].children_labels[cid].push_back(node_labels[node][i]);
      treeLOM[node].assigned[i] = true;
      needed += (child_n_labels  % (arity_ - 1) == 1) ? (arity_ - 2) : -1;
      remaining -= 1;
    }
  }
}


// compute node stats, assign labels to children, then update children
void LOMtree::updateNode(int32_t node) {
  int32_t ndesc = node_labels[node].size();
  //~ printf("node %d sees %d labels\n", node, ndesc);
  if (ndesc == 0) return;
  int32_t j, n_old, n_new, old_leaf, leaf, new_node;
  // compute relevant stats and pre-sort (label, child) pairs
  computeNodeStats(node);
  assignNodeChildren(node);
  // recursion
  // sort by decrease in number of labels
  for (int32_t j = 0; j < arity_; j++) {
    n_new = treeLOM[node].children_labels[j].size();
    n_old = node_labels[treeLOM[node].children[j]].size();
    // ASSERT
    assert(node >= nleaves_ or node_labels[node].size() <= 1);
    treeLOM[node].children_queue.push(AuxPair(j, n_old - n_new));
  }
  // treat children
  // sorted children in children_queue ensure that there always are
  // free nodes available when necessary
  while (not treeLOM[node].children_queue.empty()) {
    auto pair = treeLOM[node].children_queue.top();
    treeLOM[node].children_queue.pop();
    j = pair.n;
    n_new = treeLOM[node].children_labels[j].size();
    n_old = node_labels[treeLOM[node].children[j]].size();
    if (n_new <= 1 and n_old  <= 1) {
      leaf = (n_new == 1) ? treeLOM[node].children_labels[j][0] : nleaves_ - 1;
      old_leaf = treeLOM[node].children[j];
      treeLOM[node].children[j] = leaf;
      treeLOM[leaf].parent = node;
      treeLOM[leaf].pindex = j;
      //~ printf("a-> child %d of %d was %d is now %d\n", j, node, old_leaf, leaf);
    } else if (n_new <= 1 and n_old > 1) {
      // free up child j of current node and replace with leaf
      leaf = (n_new == 1) ? treeLOM[node].children_labels[j][0] : nleaves_ - 1;
      freeNodeIds.push_back(treeLOM[node].children[j]);
      node_labels[treeLOM[node].children[j]].resize(0);
      treeLOM[node].children[j] = leaf;
      treeLOM[leaf].parent = node;
      treeLOM[leaf].pindex = j;
    } else if (n_new > 1 and n_old <= 1) {
      // fetch unused node
      new_node = freeNodeIds.back();
      freeNodeIds.pop_back();
      // replace child j (leaf or empty) with new_node
      node_labels[new_node].resize(0);
      node_labels[new_node].insert(node_labels[new_node].end(),
                                   treeLOM[node].children_labels[j].begin(),
                                   treeLOM[node].children_labels[j].end());
      // ASSERT
      assert(new_node >= nleaves_ or node_labels[new_node].size());
      initNodeStats(new_node);
      treeLOM[node].children[j] = new_node;
      treeLOM[new_node].parent = node;
      treeLOM[new_node].pindex = j;
      // continue recursion
      //~ printf("b-> child %d of %d now has %d labels\n", j, node, node_labels[new_node].size());
      updateNode(treeLOM[node].children[j]);
    } else {
      // simple recursion
      new_node = treeLOM[node].children[j];
      node_labels[new_node].resize(0);
      node_labels[new_node].insert(node_labels[new_node].end(),
                                   treeLOM[node].children_labels[j].begin(),
                                   treeLOM[node].children_labels[j].end());
      initNodeStats(new_node);
      //~ printf("c-> child %d of %d now has %d labels\n", j, node, node_labels[new_node].size());
      updateNode(treeLOM[node].children[j]);
    }
  }
}


// update tree structure as per LOMtree algorithm
void LOMtree::updateTree() {
  updateNode(nnodes_ - 1);
  updatePaths();
}


// auxiliary functions
int32_t LOMtree::getNLeaves() {
  return nleaves_;
}

int32_t LOMtree::getNNodes() {
  return nnodes_;
}

int32_t LOMtree::getNLabels() {
  return nlabels_;
}

NodeLOM LOMtree::getNode(int32_t node){
  return treeLOM[node];
}

// save / load functions

void LOMtree::save(std::ostream& out) {
  out.write((char*) &arity_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &nleaves_, sizeof(int32_t));
  out.write((char*) &nnodes_, sizeof(int32_t));
  for (int32_t i = 0; i < nnodes_; i++) {
    auto node_lm = treeLOM[i];
    out.write((char*) &node_lm.parent, sizeof(int32_t));
    out.write((char*) &node_lm.pindex, sizeof(int32_t));
    out.write((char*) &node_lm.count, sizeof(int32_t));
    int32_t nchildren = node_lm.children.size();
    out.write((char*) &nchildren, sizeof(int32_t));
    for (int32_t j = 0; j < nchildren; j++) {
      out.write((char*) &node_lm.children[j], sizeof(int32_t));
    }
  }
}

void LOMtree::load(std::istream& in) {
  arity_ = 0;
  in.read((char*) &arity_, sizeof(int32_t));
  printf("arity %d\n", arity_);
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &nleaves_, sizeof(int32_t));
  in.read((char*) &nnodes_, sizeof(int32_t));
  printf("nnodes %d\n", nnodes_);
  treeLOM.resize(nnodes_);
  for (int32_t i = 0; i < nnodes_; i++) {
    in.read((char*) &treeLOM[i].parent, sizeof(int32_t));
    in.read((char*) &treeLOM[i].pindex, sizeof(int32_t));
    in.read((char*) &treeLOM[i].count, sizeof(int32_t));
    //~ if (i == 0) printf("%d %d %d \n", treeLOM[i].parent, treeLOM[i].pindex, treeLOM[i].count);
    int32_t nchildren;
    in.read((char*) &nchildren, sizeof(int32_t));
    treeLOM[i].children.resize(nchildren);
    for (int32_t j = 0; j < nchildren; j++) {
      in.read((char*) &treeLOM[i].children[j], sizeof(int32_t));
    }
  }
  printf("rebuilding tree\n");
  updatePaths();
}

}
