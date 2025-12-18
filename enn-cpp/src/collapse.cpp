#include "enn/collapse.hpp"
#include <random>
#include <algorithm>
#include <cmath>

namespace enn {

Collapse::Collapse(int k_, unsigned seed) : k(k_) {
    std::mt19937 gen(seed);
    std::normal_distribution<F> dist(0.0, 0.05);
    
    Wg = Mat::NullaryExpr(k, k, [&]() { return dist(gen); });
}

Vec Collapse::softmax(const Vec& z) const {
    // Numerically stable softmax: exp(z - max(z)) / sum(exp(z - max(z)))
    F max_z = z.maxCoeff();
    Vec exp_z = (z.array() - max_z).exp();
    return exp_z / exp_z.sum();
}

F Collapse::forward(const Vec& psi, CollapseCache& cache) const {
    cache.logits = Wg * psi;
    cache.alpha = softmax(cache.logits);
    return cache.alpha.dot(psi);
}

void Collapse::backward(F dL_dpred, const Vec& psi, const CollapseCache& cache,
                       Vec& dpsi, Mat& dWg) const {
    const Vec& alpha = cache.alpha;
    
    // Jacobian of softmax: J = diag(alpha) - alpha * alpha^T
    Mat J = Mat(alpha.asDiagonal()) - (alpha * alpha.transpose());
    
    // d(pred)/d(psi) = alpha + Wg^T * J * psi  
    dpsi = dL_dpred * (alpha + (Wg.transpose() * (J * psi)));
    
    // d(pred)/d(Wg) = J * psi * psi^T
    dWg = dL_dpred * (J * psi) * psi.transpose();
}

// Multi-head collapse implementation
Collapse::MultiHead::MultiHead(int k, int num_heads, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<F> dist(0.0, 0.05);
    
    heads.resize(num_heads);
    for (auto& head : heads) {
        head = Mat::NullaryExpr(k, k, [&]() { return dist(gen); });
    }
    
    Wout = Mat::NullaryExpr(1, num_heads, [&]() { return dist(gen); });
}

F Collapse::MultiHead::forward(const Vec& psi, std::vector<CollapseCache>& caches) const {
    caches.resize(heads.size());
    
    F total_output = 0.0;
    for (size_t i = 0; i < heads.size(); ++i) {
        caches[i].logits = heads[i] * psi;
        caches[i].alpha = Vec((caches[i].logits.array() - caches[i].logits.maxCoeff()).exp());
        caches[i].alpha /= caches[i].alpha.sum();
        
        F head_output = caches[i].alpha.dot(psi);
        total_output += Wout(0, i) * head_output;
    }
    
    return total_output;
}

void Collapse::MultiHead::backward(F dL_dpred, const Vec& psi,
                                  const std::vector<CollapseCache>& caches,
                                  Vec& dpsi, std::vector<Mat>& dheads, Mat& dWout) const {
    dpsi.setZero();
    dheads.resize(heads.size());
    dWout.setZero();
    
    for (size_t i = 0; i < heads.size(); ++i) {
        const Vec& alpha = caches[i].alpha;
        Mat J = Mat(alpha.asDiagonal()) - (alpha * alpha.transpose());
        
        F head_output = alpha.dot(psi);
        F w_i = Wout(0, i);
        
        // Gradient w.r.t. output weights
        dWout(0, i) = dL_dpred * head_output;
        
        // Gradient w.r.t. psi (accumulate from all heads)
        Vec dpsi_head = w_i * (alpha + (heads[i].transpose() * (J * psi)));
        dpsi += dL_dpred * dpsi_head;
        
        // Gradient w.r.t. head weights
        dheads[i] = dL_dpred * w_i * (J * psi) * psi.transpose();
    }
}

} // namespace enn