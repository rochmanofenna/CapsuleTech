#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace enn {

using F   = double;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

struct State { 
    Vec psi; 
    Vec h;     // optional hidden state
    
    State() = default;
    State(int k, int hidden_dim) : psi(Vec::Zero(k)), h(Vec::Zero(hidden_dim)) {}
};

struct Input { 
    Vec x; 
    
    Input() = default;
    explicit Input(const Vec& x_) : x(x_) {}
};

// Training batch structure
struct Batch {
    std::vector<Vec> inputs;   // [batch_size] of input vectors
    std::vector<F> targets;    // [batch_size] of target scalars
    
    size_t size() const { return inputs.size(); }
};

// Sequence batch for BPTT
struct SeqBatch {
    std::vector<std::vector<Vec>> sequences;  // [batch_size][seq_len] 
    std::vector<std::vector<F>> targets;      // [batch_size][seq_len]
    
    size_t batch_size() const { return sequences.size(); }
    size_t seq_len() const { return sequences.empty() ? 0 : sequences[0].size(); }
};

} // namespace enn