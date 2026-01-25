#pragma once
#include "types.hpp"

namespace enn {

struct CollapseCache { 
    Vec scores; 
    Vec alpha; 
    Vec collapsed;
    Vec gated;
    F temperature = 1.0;
    
    CollapseCache() = default;
    CollapseCache(int k) : scores(Vec::Zero(k)), alpha(Vec::Zero(k)),
                           collapsed(Vec::Zero(k)), gated(Vec::Zero(k)) {}
};

struct Collapse {
    Mat Wq;         // [k x k] attention query weights
    Vec Wout;       // [k] projection weights
    F bout = 0.0;   // scalar bias for output
    F log_temp;     // learned log-temperature
    int k;          // entanglement dimension
    
    explicit Collapse(int k_, unsigned seed = 123);
    
    // Numerically stable softmax helper
    Vec softmax(const Vec& z) const;
    Vec softmax_jacobian_matvec(const Vec& alpha, const Vec& vec) const;
    
    // Forward pass returning scalar prediction
    F forward(const Vec& psi, CollapseCache& cache) const;
    
    struct Grads {
        Mat dWq;
        Vec dWout;
        F dbias = 0.0;
        F dlog_temp = 0.0;

        explicit Grads(int k) : dWq(Mat::Zero(k, k)), dWout(Vec::Zero(k)) {}

        void zero() {
            dWq.setZero();
            dWout.setZero();
            dbias = 0.0;
            dlog_temp = 0.0;
        }

        void add_scaled(const Grads& other, F scale) {
            dWq += scale * other.dWq;
            dWout += scale * other.dWout;
            dbias += scale * other.dbias;
            dlog_temp += scale * other.dlog_temp;
        }

        void scale(F s) {
            dWq *= s;
            dWout *= s;
            dbias *= s;
            dlog_temp *= s;
        }
    };
    
    void backward(F dL_dpred, const Vec& psi, const CollapseCache& cache,
                  Vec& dpsi, Grads& grads) const;
};

} // namespace enn
