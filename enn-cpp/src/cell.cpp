#include "enn/cell.hpp"
#include <random>
#include <Eigen/Eigenvalues>

namespace enn {

EntangledCell::EntangledCell(int k_, int input_dim_, int hidden_dim_, F lambda_, unsigned seed)
    : k(k_), input_dim(input_dim_), hidden_dim(hidden_dim_), lambda(lambda_) {
    
    std::mt19937 gen(seed);
    std::normal_distribution<F> dist(0.0, 0.05);
    
    // Initialize weights with small random values
    Wx = Mat::NullaryExpr(k, input_dim, [&]() { return dist(gen); });
    Wh = Mat::NullaryExpr(k, hidden_dim, [&]() { return dist(gen); });
    L = Mat::NullaryExpr(k, k, [&]() { return dist(gen); });
    b = Vec::Zero(k);
}

Vec EntangledCell::forward(const Vec& x, const Vec& h, const Vec& psi_in, CellCache& cache) const {
    // Cache inputs for backprop
    cache.x = x;
    cache.h = h;
    cache.psi_in = psi_in;
    
    // Compute entanglement matrix E = L * L^T
    cache.E = L * L.transpose();
    
    // Forward pass: s = Wx*x + Wh*h + (E - lambda*I)*psi_in + b
    cache.s = Wx * x + Wh * h + (cache.E * psi_in) - lambda * psi_in + b;
    
    // Apply activation: psi = tanh(s)
    cache.psi = cache.s.array().tanh();
    
    return cache.psi;
}

void EntangledCell::backward(const Vec& dpsi_out, const CellCache& cache,
                           Grads& grads, Vec& dpsi_in, Vec& dh) const {
    // Backprop through tanh: ds = dpsi_out * (1 - tanh^2(s))
    Vec ds = dpsi_out.cwiseProduct((1.0 - cache.psi.array().square()).matrix());
    
    // Gradients w.r.t. parameters
    grads.dWx += ds * cache.x.transpose();
    grads.dWh += ds * cache.h.transpose();
    grads.db += ds;
    
    // Gradients w.r.t. E and lambda from the term (E - lambda*I)*psi_in
    Mat dE = ds * cache.psi_in.transpose();
    grads.dlambda += -(ds.dot(cache.psi_in));
    
    // Chain rule: dE -> dL using E = L * L^T
    // dL/dL = (dE + dE^T) * L (symmetrized because E should be symmetric)
    grads.dL += (dE + dE.transpose()) * L;
    
    // Upstream gradients
    dpsi_in = cache.E.transpose() * ds - lambda * ds;
    dh = Wh.transpose() * ds;
}

Mat EntangledCell::get_entanglement_matrix() const {
    return L * L.transpose();
}

bool EntangledCell::is_entanglement_psd(F tolerance) const {
    Mat E = get_entanglement_matrix();
    Eigen::SelfAdjointEigenSolver<Mat> solver(E);
    
    if (solver.info() != Eigen::Success) {
        return false; // Could not compute eigenvalues
    }
    
    Vec eigenvals = solver.eigenvalues();
    return eigenvals.minCoeff() >= -tolerance;
}

F EntangledCell::compute_psd_regularizer_loss() const {
    // Since we use E = L*L^T, PSD is automatically enforced
    // We can add a small regularizer on ||L||_F to prevent explosion
    return 0.5 * 1e-6 * L.squaredNorm();
}

F EntangledCell::compute_param_l2_loss() const {
    return 0.5 * (Wx.squaredNorm() + Wh.squaredNorm() + L.squaredNorm() + b.squaredNorm());
}

} // namespace enn