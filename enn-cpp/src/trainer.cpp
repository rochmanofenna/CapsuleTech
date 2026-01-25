#include "enn/trainer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace enn {

SequenceTrainer::SequenceTrainer(int k, int raw_input_dim, int embed_dim, int hidden_dim,
                                F lambda, const TrainConfig& config)
    : config_(config), embed_dim_(embed_dim), raw_input_dim_(raw_input_dim) {
    frontend_ = std::make_unique<SpatialTemporalCNN>(raw_input_dim_, embed_dim_,
                                                     config.frontend_filters,
                                                     config.frontend_temporal_kernel,
                                                     config.frontend_depth_kernel);
    cell_ = std::make_unique<EntangledCell>(k, embed_dim_, hidden_dim,
                                            lambda, config.use_layer_norm);
    collapse_ = std::make_unique<Collapse>(k);
    optimizer_ = std::make_unique<AdamW>(config.learning_rate, 0.9, 0.999, 1e-8,
                                         config.weight_decay);
    opt_state_ = std::make_unique<OptimizerState>(k, embed_dim_, hidden_dim,
                                                  config.frontend_filters, raw_input_dim_,
                                                  config.frontend_temporal_kernel,
                                                  config.frontend_depth_kernel);
}

F SequenceTrainer::train_epoch(const SeqBatch& data) {
    F total_loss = 0.0;
    int num_batches = 0;

    for (size_t start = 0; start < data.batch_size(); start += config_.batch_size) {
        size_t end = std::min(start + config_.batch_size, data.batch_size());
        if (start >= end) break;

        EntangledCell::Grads cell_acc(cell_->k, embed_dim_, cell_->hidden_dim);
        cell_acc.zero();
        FrontendGrads front_acc(config_.frontend_filters, raw_input_dim_,
                                config_.frontend_temporal_kernel,
                                config_.frontend_depth_kernel, embed_dim_);
        front_acc.zero();
        Collapse::Grads collapse_acc(cell_->k);
        collapse_acc.zero();
        F batch_loss = 0.0;

        for (size_t i = start; i < end; ++i) {
            SequenceCache cache;
            F seq_loss = train_sequence(data.sequences[i], data.targets[i], cache);
            batch_loss += seq_loss;

            EntangledCell::Grads seq_cell(cell_->k, embed_dim_, cell_->hidden_dim);
            seq_cell.zero();
            FrontendGrads seq_front(config_.frontend_filters, raw_input_dim_,
                                    config_.frontend_temporal_kernel,
                                    config_.frontend_depth_kernel, embed_dim_);
            seq_front.zero();
            Collapse::Grads seq_collapse(cell_->k);
            seq_collapse.zero();

            backward_through_time(data.sequences[i], data.targets[i], cache,
                                  seq_cell, seq_front, seq_collapse);

            cell_acc.add_scaled(seq_cell, 1.0);
            front_acc.add_scaled(seq_front, 1.0);
            collapse_acc.add_scaled(seq_collapse, 1.0);
        }

        F scale = 1.0 / static_cast<F>(end - start);
        cell_acc.scale(scale);
        front_acc.scale(scale);
        collapse_acc.scale(scale);

        F reg_loss = compute_regularization_loss();
        apply_gradients(cell_acc, front_acc, collapse_acc, reg_loss);

        total_loss += batch_loss * scale;
        num_batches++;
    }

    return (num_batches > 0) ? total_loss / num_batches : total_loss;
}

F SequenceTrainer::train_sequence(const std::vector<Vec>& inputs,
                                  const std::vector<F>& targets,
                                  SequenceCache& cache) {
    const int seq_len = static_cast<int>(inputs.size());

    cache.cell_caches.resize(seq_len);
    cache.collapse_caches.resize(seq_len);
    cache.psi_history.resize(seq_len);
    cache.h_history.resize(seq_len);
    cache.embeddings.resize(seq_len);
    cache.embed_grads.assign(seq_len, Vec::Zero(embed_dim_));
    cache.predictions.resize(seq_len);
    cache.initial_psi = Vec::Zero(cell_->k);
    cache.initial_h = Vec::Zero(cell_->hidden_dim);

    frontend_->forward_sequence(inputs, cache.embeddings, cache.frontend_cache);

    Vec psi = cache.initial_psi;
    Vec h = cache.initial_h;
    F total_loss = 0.0;

    for (int t = 0; t < seq_len; ++t) {
        psi = cell_->forward(cache.embeddings[t], h, psi, cache.cell_caches[t]);
        cache.psi_history[t] = psi;
        cache.h_history[t] = h;

        F pred = collapse_->forward(psi, cache.collapse_caches[t]);
        cache.predictions[t] = pred;
        F diff = pred - targets[t];
        total_loss += 0.5 * diff * diff;
    }

    return total_loss;
}

void SequenceTrainer::backward_through_time(const std::vector<Vec>& inputs,
                                            const std::vector<F>& targets,
                                            SequenceCache& cache,
                                            EntangledCell::Grads& cell_grads,
                                            FrontendGrads& front_grads,
                                            Collapse::Grads& collapse_grads) {
    const int seq_len = static_cast<int>(targets.size());
    Vec dpsi_future = Vec::Zero(cell_->k);
    Vec dh_future = Vec::Zero(cell_->hidden_dim);

    for (int t = seq_len - 1; t >= 0; --t) {
        F dL_dpred = cache.predictions[t] - targets[t];

        Vec dpsi_collapse;
        collapse_->backward(dL_dpred, cache.psi_history[t],
                            cache.collapse_caches[t], dpsi_collapse,
                            collapse_grads);

        Vec dpsi_total = dpsi_collapse + dpsi_future;
        Vec dpsi_in, dh, dx;
        cell_->backward(dpsi_total, cache.cell_caches[t], cell_grads,
                        dpsi_in, dh, dx);
        cache.embed_grads[t] = dx;

        dpsi_future = dpsi_in;
        dh_future = dh;
        (void)dh_future; // hidden state not yet recurrent, placeholder for future use
    }

    frontend_->backward_sequence(inputs, cache.frontend_cache,
                                 cache.embed_grads, front_grads);
}

void SequenceTrainer::apply_gradients(const EntangledCell::Grads& cell_grads,
                                      const FrontendGrads& front_grads,
                                      const Collapse::Grads& collapse_grads,
                                      F /*reg_loss*/) {
    optimizer_->step(cell_->Wx, opt_state_->m_Wx, opt_state_->v_Wx, cell_grads.dWx);
    optimizer_->step(cell_->Wh, opt_state_->m_Wh, opt_state_->v_Wh, cell_grads.dWh);
    optimizer_->step(cell_->L, opt_state_->m_L, opt_state_->v_L, cell_grads.dL);
    optimizer_->step(cell_->b, opt_state_->m_b, opt_state_->v_b, cell_grads.db);
    optimizer_->step(cell_->ln_gamma, opt_state_->m_ln_gamma, opt_state_->v_ln_gamma, cell_grads.dgamma);
    optimizer_->step(cell_->ln_beta, opt_state_->m_ln_beta, opt_state_->v_ln_beta, cell_grads.dbeta);
    optimizer_->step(cell_->log_lambda, opt_state_->m_log_lambda, opt_state_->v_log_lambda,
                     cell_grads.dlog_lambda);

    optimizer_->step(frontend_->W_temporal(), opt_state_->m_front_temporal,
                     opt_state_->v_front_temporal, front_grads.dW_temporal);
    optimizer_->step(frontend_->b_temporal(), opt_state_->m_front_temporal_b,
                     opt_state_->v_front_temporal_b, front_grads.db_temporal);
    optimizer_->step(frontend_->W_spatial(), opt_state_->m_front_spatial,
                     opt_state_->v_front_spatial, front_grads.dW_spatial);
    optimizer_->step(frontend_->b_spatial(), opt_state_->m_front_spatial_b,
                     opt_state_->v_front_spatial_b, front_grads.db_spatial);
    optimizer_->step(frontend_->W_depthwise(), opt_state_->m_front_depthwise,
                     opt_state_->v_front_depthwise, front_grads.dW_depthwise);
    optimizer_->step(frontend_->b_depthwise(), opt_state_->m_front_depthwise_b,
                     opt_state_->v_front_depthwise_b, front_grads.db_depthwise);
    optimizer_->step(frontend_->W_proj(), opt_state_->m_front_proj,
                     opt_state_->v_front_proj, front_grads.dW_proj);
    optimizer_->step(frontend_->b_proj(), opt_state_->m_front_proj_b,
                     opt_state_->v_front_proj_b, front_grads.db_proj);

    optimizer_->step(collapse_->Wq, opt_state_->m_Wq, opt_state_->v_Wq, collapse_grads.dWq);
    optimizer_->step(collapse_->Wout, opt_state_->m_Wout, opt_state_->v_Wout, collapse_grads.dWout);
    optimizer_->step(collapse_->bout, opt_state_->m_collapse_bias, opt_state_->v_collapse_bias,
                     collapse_grads.dbias);
    optimizer_->step(collapse_->log_temp, opt_state_->m_log_temp, opt_state_->v_log_temp,
                     collapse_grads.dlog_temp);
}

F SequenceTrainer::compute_regularization_loss() {
    F reg_loss = 0.0;
    if (config_.reg_beta > 0) {
        reg_loss += config_.reg_beta * cell_->compute_psd_regularizer_loss();
    }
    if (config_.reg_eta > 0) {
        reg_loss += config_.reg_eta * cell_->compute_param_l2_loss();
    }
    return reg_loss;
}

F SequenceTrainer::evaluate(const SeqBatch& data, Metrics& metrics) {
    metrics.reset();
    F total_loss = 0.0;

    for (size_t i = 0; i < data.batch_size(); ++i) {
        std::vector<F> preds = forward_sequence(data.sequences[i]);
        F seq_loss = 0.0;
        for (size_t t = 0; t < data.targets[i].size(); ++t) {
            F diff = preds[t] - data.targets[i][t];
            F loss = 0.5 * diff * diff;
            seq_loss += loss;
            if (t == data.targets[i].size() - 1) {
                metrics.update(preds[t], data.targets[i][t], loss);
            }
        }
        total_loss += seq_loss;
    }

    metrics.finalize();
    return total_loss / std::max<size_t>(1, data.batch_size());
}

std::vector<F> SequenceTrainer::forward_sequence(const std::vector<Vec>& sequence,
                                                 Vec* final_psi, Vec* final_h,
                                                 Vec* final_alpha, F* final_temperature) const {
    const int seq_len = static_cast<int>(sequence.size());
    std::vector<Vec> embeddings(seq_len);
    FrontendCache front_cache;
    frontend_->forward_sequence(sequence, embeddings, front_cache);

    std::vector<F> predictions;
    predictions.reserve(seq_len);
    Vec psi = Vec::Zero(cell_->k);
    Vec h = Vec::Zero(cell_->hidden_dim);
    Vec last_alpha = Vec::Zero(cell_->k);
    F last_temp = std::exp(collapse_->log_temp);

    for (int t = 0; t < seq_len; ++t) {
        CellCache cache;
        psi = cell_->forward(embeddings[t], h, psi, cache);
        CollapseCache collapse_cache;
        F pred = collapse_->forward(psi, collapse_cache);
        predictions.push_back(pred);
        last_alpha = collapse_cache.alpha;
        last_temp = collapse_cache.temperature;
    }

    if (final_psi) *final_psi = psi;
    if (final_h) *final_h = h;
    if (final_alpha) *final_alpha = last_alpha;
    if (final_temperature) *final_temperature = last_temp;
    return predictions;
}

TrainerWithScheduler::TrainerWithScheduler(std::unique_ptr<SequenceTrainer> trainer,
                                           F base_lr, F min_lr, int total_steps)
    : trainer_(std::move(trainer)) {
    scheduler_ = std::make_unique<CosineScheduler>(base_lr, min_lr, total_steps);
}

F TrainerWithScheduler::train_epoch(const SeqBatch& data) {
    update_learning_rate();
    return trainer_->train_epoch(data);
}

F TrainerWithScheduler::evaluate(const SeqBatch& data, Metrics& metrics) {
    return trainer_->evaluate(data, metrics);
}

void TrainerWithScheduler::update_learning_rate() {
    F new_lr = (*scheduler_)(current_step_);
    trainer_->set_learning_rate(new_lr);
    current_step_++;
}

F TrainerWithScheduler::get_current_lr() const {
    return (*scheduler_)(current_step_);
}

} // namespace enn
