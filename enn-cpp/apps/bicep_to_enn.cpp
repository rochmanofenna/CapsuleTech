#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <optional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "enn/trainer.hpp"
#include "enn/calibrator.hpp"

using namespace enn;

// Simple CSV reader for BICEP parquet data (converted to CSV)
struct StepFeature {
    F mean = 0.0;
    F std = 0.0;
    F q10 = 0.0;
    F q90 = 0.0;
    F aleatoric = 0.0;
    F epistemic = 0.0;
};

struct TrajectoryData {
    std::vector<std::vector<Vec>> sequences;
    std::vector<std::vector<F>> targets;
    std::vector<std::vector<StepFeature>> features;
    std::vector<uint64_t> sequence_ids;
};

struct CliOptions {
    std::string csv_path;
    std::string telemetry_path = "enn_predictions.csv";
    std::optional<std::string> calibrator_path;
    std::optional<std::string> metadata_path;
};

namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime  = 1099511628211ULL;

uint64_t fnv1a_bytes(const uint8_t* data, size_t n) {
    uint64_t h = kFnvOffset;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= kFnvPrime;
    }
    return h;
}

uint64_t hash_value(F value) {
    return fnv1a_bytes(reinterpret_cast<const uint8_t*>(&value), sizeof(F));
}

uint64_t hash_sequence_inputs(const std::vector<Vec>& seq) {
    uint64_t h = kFnvOffset;
    for (const auto& vec : seq) {
        for (int j = 0; j < vec.size(); ++j) {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&vec[j]);
            for (size_t b = 0; b < sizeof(F); ++b) {
                h ^= bytes[b];
                h *= kFnvPrime;
            }
        }
    }
    return h;
}

struct FeatureSummary {
    std::vector<double> sum;
    std::vector<double> sum_sq;
    size_t count = 0;
};

void update_summary(FeatureSummary& summary, const Vec& vec) {
    if (summary.sum.empty()) {
        summary.sum.resize(vec.size(), 0.0);
        summary.sum_sq.resize(vec.size(), 0.0);
    }
    for (int d = 0; d < vec.size(); ++d) {
        double val = static_cast<double>(vec[d]);
        summary.sum[d] += val;
        summary.sum_sq[d] += val * val;
    }
    summary.count += 1;
}

void log_dataset_stats(const TrajectoryData& data) {
    if (data.sequences.empty()) {
        std::cout << "[Debug] No sequences loaded" << std::endl;
        return;
    }

    FeatureSummary summary;
    std::unordered_set<uint64_t> seq_hashes;
    std::vector<uint64_t> sample_hashes;

    for (size_t i = 0; i < data.sequences.size(); ++i) {
        const auto& seq = data.sequences[i];
        uint64_t h = hash_sequence_inputs(seq);
        seq_hashes.insert(h);
        if (sample_hashes.size() < 3) {
            sample_hashes.push_back(h);
        }
        for (const auto& vec : seq) {
            update_summary(summary, vec);
        }
    }

    std::map<F, size_t> label_counts;
    double label_sum = 0.0;
    size_t label_total = 0;
    for (const auto& tgt_seq : data.targets) {
        if (tgt_seq.empty()) continue;
        F y = tgt_seq.back();
        label_counts[y]++;
        label_sum += y;
        label_total++;
    }

    std::cout << "[Debug] sequences=" << data.sequences.size()
              << " unique_input_hashes=" << seq_hashes.size() << std::endl;
    if (!sample_hashes.empty()) {
        std::cout << "[Debug] sample input hashes:";
        for (auto h : sample_hashes) {
            std::cout << " 0x" << std::hex << h << std::dec;
        }
        std::cout << std::endl;
    }

    if (summary.count > 0) {
        std::cout << "[Debug] feature stats (mean ± std):" << std::endl;
        for (size_t d = 0; d < summary.sum.size(); ++d) {
            double mean = summary.sum[d] / summary.count;
            double var = summary.sum_sq[d] / summary.count - mean * mean;
            var = std::max(var, 0.0);
            std::cout << "  dim " << d << ": " << mean << " ± " << std::sqrt(var) << std::endl;
        }
    }

    if (label_total > 0) {
        std::cout << "[Debug] label mean=" << label_sum / label_total
                  << " counts:";
        for (const auto& kv : label_counts) {
            std::cout << " [" << kv.first << " -> " << kv.second << "]";
        }
        std::cout << std::endl;
    }
}

void print_usage(const char* exe) {
    std::cerr << "Usage: " << exe << " <bicep_csv_file> [--telemetry output.csv] [--calibrator calibrator.json]" << std::endl;
}

void validate_metadata(const std::optional<std::string>& path) {
    if (!path) {
        return;
    }
    std::ifstream meta_file(*path);
    if (!meta_file.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + *path);
    }
    std::string json((std::istreambuf_iterator<char>(meta_file)), std::istreambuf_iterator<char>());
    std::string compact;
    compact.reserve(json.size());
    for (char ch : json) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            compact.push_back(ch);
        }
    }
    if (compact.find("\"std_ddof\":1") == std::string::npos) {
        throw std::runtime_error("Metadata must contain std_ddof = 1");
    }
    if (compact.find("type7") == std::string::npos) {
        throw std::runtime_error("Metadata must specify quantile_method type7");
    }
}

CliOptions parse_cli(int argc, char* argv[]) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--telemetry") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --telemetry");
            }
            opts.telemetry_path = argv[++i];
        } else if (arg == "--calibrator") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --calibrator");
            }
            opts.calibrator_path = std::string(argv[++i]);
        } else if (arg == "--metadata") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --metadata");
            }
            opts.metadata_path = std::string(argv[++i]);
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + arg);
        } else if (opts.csv_path.empty()) {
            opts.csv_path = arg;
        } else {
            throw std::runtime_error("Unexpected positional argument: " + arg);
        }
    }

    if (opts.csv_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("CSV file path required");
    }

    return opts;
}

F sigmoid(F x) {
    if (x >= 0) {
        F z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    F z = std::exp(x);
    return z / (1.0f + z);
}

struct AlphaStats {
    F entropy = 0.0f;
    F alpha_max = 0.0f;
    int argmax = 0;
};

AlphaStats summarize_alpha(const Vec& alpha) {
    AlphaStats stats;
    for (int j = 0; j < alpha.size(); ++j) {
        F p = std::max(alpha[j], static_cast<F>(1e-9));
        stats.entropy -= p * std::log(p);
        if (p > stats.alpha_max) {
            stats.alpha_max = p;
            stats.argmax = j;
        }
    }
    return stats;
}

} // namespace

TrajectoryData load_bicep_data(const std::string& csv_file) {
    TrajectoryData data;
    std::ifstream file(csv_file);
    std::string line;
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + csv_file);
    }
    
    // Read header and build column index map
    std::getline(file, line);
    std::vector<std::string> header;
    {
        std::istringstream header_stream(line);
        std::string htok;
        while (std::getline(header_stream, htok, ',')) {
            header.push_back(htok);
        }
    }

    auto col_index = [&](const std::string& name) -> size_t {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) {
            throw std::runtime_error("Missing column in CSV: " + name);
        }
        return static_cast<size_t>(std::distance(header.begin(), it));
    };

    auto col_if_present = [&](const std::string& name) -> std::optional<size_t> {
        auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) {
            return std::nullopt;
        }
        return static_cast<size_t>(std::distance(header.begin(), it));
    };

    const size_t idx_sequence_id = col_index("sequence_id");
    const size_t idx_step = col_index("step");
    const auto idx_state0_opt = col_if_present("state_0");
    const auto idx_state_col = col_if_present("state");
    const auto idx_input_opt = col_if_present("input");
    const size_t idx_input = idx_input_opt
        ? *idx_input_opt
        : (idx_state0_opt ? *idx_state0_opt : col_index("state_0"));
    const size_t idx_target = col_index("target");

    const auto idx_state_mean_col = col_if_present("state_mean");
    const auto idx_state_std_col = col_if_present("state_std");
    const auto idx_state_q10_col = col_if_present("state_q10");
    const auto idx_state_q90_col = col_if_present("state_q90");
    const auto idx_aleatoric_col = col_if_present("aleatoric_unc");
    const auto idx_epistemic_col = col_if_present("epistemic_unc");

    auto resolve_with_fallback = [&](const std::optional<size_t>& primary,
                                     const std::optional<size_t>& secondary,
                                     size_t fallback) -> size_t {
        if (primary) return *primary;
        if (secondary) return *secondary;
        return fallback;
    };

    const size_t idx_state_mean = idx_state_mean_col
        ? *idx_state_mean_col
        : resolve_with_fallback(idx_state0_opt, idx_state_col, idx_input);
    const size_t idx_state_std = idx_state_std_col
        ? *idx_state_std_col
        : idx_state_mean;
    const size_t idx_state_q10 = idx_state_q10_col
        ? *idx_state_q10_col
        : idx_state_mean;
    const size_t idx_state_q90 = idx_state_q90_col
        ? *idx_state_q90_col
        : idx_state_mean;
    const size_t idx_aleatoric = idx_aleatoric_col
        ? *idx_aleatoric_col
        : idx_state_mean;
    const size_t idx_epistemic = idx_epistemic_col
        ? *idx_epistemic_col
        : idx_state_mean;
    
    std::map<uint64_t, std::vector<Vec>> sequence_map;
    std::map<uint64_t, std::vector<F>> target_map;
    std::map<uint64_t, std::vector<StepFeature>> feature_map;
    std::unordered_map<uint64_t, int> last_step_seen;
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() <= idx_target) continue;
        
        auto parse_value = [&](size_t idx) -> F {
            if (idx >= tokens.size() || tokens[idx].empty()) {
                return 0.0;
            }
            return static_cast<F>(std::stod(tokens[idx]));
        };

        uint64_t sequence_id = std::stoull(tokens[idx_sequence_id]);
        uint32_t step = static_cast<uint32_t>(std::stoul(tokens[idx_step]));
        F input = parse_value(idx_input);
        F state_mean = parse_value(idx_state_mean);
        F state_std = parse_value(idx_state_std);
        F state_q10 = parse_value(idx_state_q10);
        F state_q90 = parse_value(idx_state_q90);
        F aleatoric = parse_value(idx_aleatoric);
        F epistemic = parse_value(idx_epistemic);
        F target = parse_value(idx_target);
        
        // Invariants
        if (state_std < 0 || aleatoric < 0 || epistemic < 0) {
            throw std::runtime_error("Negative variance/uncertainty encountered");
        }
        if (state_q10 > state_q90) {
            throw std::runtime_error("state_q10 greater than state_q90");
        }
        if (!(state_q10 - 1e-6 <= state_mean && state_mean <= state_q90 + 1e-6)) {
            throw std::runtime_error("state_mean outside [q10,q90]");
        }
        if (!(0.0 - 1e-6 <= target && target <= 1.0 + 1e-6)) {
            throw std::runtime_error("target outside [0,1]");
        }
        int expected_step = 0;
        auto it_step = last_step_seen.find(sequence_id);
        if (it_step != last_step_seen.end()) {
            expected_step = it_step->second + 1;
        }
        if (static_cast<int>(step) != expected_step) {
            throw std::runtime_error("Non-consecutive step for sequence " + std::to_string(sequence_id));
        }
        last_step_seen[sequence_id] = step;

        // Build feature vector [base input, mean, std, q10, q90, aleatoric, epistemic]
        const int feature_dim = 7;
        Vec input_vec(feature_dim);
        input_vec << input, state_mean, state_std, state_q10, state_q90, aleatoric, epistemic;
        
        if (sequence_map.find(sequence_id) == sequence_map.end()) {
            sequence_map[sequence_id] = std::vector<Vec>();
            target_map[sequence_id] = std::vector<F>();
        }
        
        // Ensure vectors are large enough
        if (sequence_map[sequence_id].size() <= step) {
            sequence_map[sequence_id].resize(step + 1);
            target_map[sequence_id].resize(step + 1);
        }

        sequence_map[sequence_id][step] = input_vec;
        target_map[sequence_id][step] = target;

        auto& feature_seq = feature_map[sequence_id];
        if (feature_seq.size() <= step) {
            feature_seq.resize(step + 1);
        }

        StepFeature feat;
        feat.mean = state_mean;
        feat.std = state_std;
        feat.q10 = state_q10;
        feat.q90 = state_q90;
        feat.aleatoric = aleatoric;
        feat.epistemic = epistemic;
        feature_seq[step] = feat;
    }
    
    // Convert map to vectors
    for (const auto& pair : sequence_map) {
        auto seq_id = pair.first;
        data.sequences.push_back(pair.second);
        data.targets.push_back(target_map[seq_id]);
        data.features.push_back(feature_map[seq_id]);
        data.sequence_ids.push_back(seq_id);
    }
    
    return data;
}

int main(int argc, char* argv[]) {
    CliOptions options;
    try {
        options = parse_cli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    try {
        validate_metadata(options.metadata_path);
    } catch (const std::exception& e) {
        std::cerr << "Metadata validation failed: " << e.what() << std::endl;
        return 1;
    }

    std::string csv_file = options.csv_path;
    std::cout << "=== BICEP -> ENN-C++ Integration ===" << std::endl;
    
    try {
        // Load BICEP trajectory data
        std::cout << "Loading BICEP trajectory data from: " << csv_file << std::endl;
        TrajectoryData traj_data = load_bicep_data(csv_file);
        log_dataset_stats(traj_data);
        
        std::cout << "Loaded " << traj_data.sequences.size() << " sequences" << std::endl;
        if (traj_data.sequences.empty()) {
            std::cerr << "No data loaded!" << std::endl;
            return 1;
        }
        
        // Print sample data
        std::cout << "Sample sequence (first 5 steps):" << std::endl;
        for (size_t i = 0; i < std::min(5UL, traj_data.sequences[0].size()); ++i) {
            std::cout << "  Step " << i << ": features=" << traj_data.sequences[0][i].transpose()
                      << ", target=" << traj_data.targets[0][i] << std::endl;
        }
        
        // Convert to ENN SeqBatch format
        SeqBatch train_data;
        train_data.sequences = traj_data.sequences;
        train_data.targets = traj_data.targets;
        
        // Configure ENN trainer for parity task
        TrainConfig config;
        config.learning_rate = 1e-3;
        config.weight_decay = 1e-6;
        config.batch_size = 16;
        config.epochs = 100;
        config.reg_beta = 1e-4;
        config.reg_eta = 1e-4;
        config.verbose = true;
        config.print_every = 10;
        
        const int k = 32;
        const int feature_dim = 7;
        const int hidden_dim = 64;
        const F lambda = 0.05;
        config.frontend_filters = 32;
        config.frontend_temporal_kernel = 5;
        config.frontend_depth_kernel = 3;
        config.embed_dim = 32;
        config.use_layer_norm = true;
        
        std::cout << "\n=== Training ENN on BICEP Trajectories ===" << std::endl;
        SequenceTrainer trainer(k, feature_dim, config.embed_dim, hidden_dim, lambda, config);
        
        // Training loop
        F best_loss = std::numeric_limits<F>::max();
        
        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            F train_loss = trainer.train_epoch(train_data);
            
            if (train_loss < best_loss) {
                best_loss = train_loss;
            }
            
            if (epoch % config.print_every == 0) {
                // Test on first sequence
                auto predictions = trainer.forward_sequence(train_data.sequences[0]);
                
                std::cout << "Epoch " << std::setw(3) << epoch
                          << " | Loss: " << std::fixed << std::setprecision(6) << train_loss
                          << " | Final pred: " << std::setprecision(3) << predictions.back()
                          << " | Target: " << train_data.targets[0].back() << std::endl;
            }
        }
        
        std::cout << "\n=== ENN Training Complete ===" << std::endl;
        std::cout << "Best loss: " << best_loss << std::endl;
        
        // Test on a few sequences
        std::cout << "\nTesting on sample sequences:" << std::endl;
        for (size_t i = 0; i < std::min(5UL, train_data.sequences.size()); ++i) {
            auto predictions = trainer.forward_sequence(train_data.sequences[i]);
            F final_pred = predictions.back();
            F target = train_data.targets[i].back();
            bool correct = (final_pred > 0.5) == (target > 0.5);
            
            std::cout << "Seq " << i << ": pred=" << std::setprecision(3) << final_pred
                      << ", target=" << target << ", correct=" << (correct ? "YES" : "NO") << std::endl;
        }
        
        // Save ENN predictions for FusionAlpha
        std::cout << "\n=== Saving ENN Outputs for FusionAlpha ===" << std::endl;

        Calibrator calibrator = options.calibrator_path
            ? Calibrator::from_json_file(*options.calibrator_path)
            : Calibrator::identity();
        std::cout << "Calibrator: " << calibrator.calibrator_id << std::endl;
        
        std::ofstream enn_output(options.telemetry_path);
        enn_output << "sequence_id,step,margin,q_pred,obs_reliability,alpha_entropy,alpha_max,attention_argmax,collapse_temperature,state_mean,state_std,state_q10,state_q90,aleatoric_unc,epistemic_unc,target,calibrator_id\n";

        double margin_sum = 0.0;
        double margin_sq = 0.0;
        size_t margin_count = 0;
        std::map<int, size_t> margin_hist;

        for (size_t i = 0; i < train_data.sequences.size(); ++i) {
            Vec final_alpha;
            F collapse_temp = 1.0;
            auto predictions = trainer.forward_sequence(
                train_data.sequences[i], nullptr, nullptr, &final_alpha, &collapse_temp);
            F margin = predictions.back();
            F q_pred = sigmoid(margin);
            F obs_reliability = calibrator.calibrate(margin);
            F target = train_data.targets[i].back();
            const StepFeature& feat = traj_data.features[i].back();
            AlphaStats stats = summarize_alpha(final_alpha);
            size_t final_step = train_data.sequences[i].empty() ? 0 : (train_data.sequences[i].size() - 1);

            enn_output << traj_data.sequence_ids[i] << ","
                       << final_step << ","
                       << margin << ","
                       << q_pred << ","
                       << obs_reliability << ","
                       << stats.entropy << ","
                       << stats.alpha_max << ","
                       << stats.argmax << ","
                       << collapse_temp << ","
                       << feat.mean << ","
                       << feat.std << ","
                       << feat.q10 << ","
                       << feat.q90 << ","
                       << feat.aleatoric << ","
                       << feat.epistemic << ","
                       << target << ","
                       << calibrator.calibrator_id << "\n";

            margin_sum += margin;
            margin_sq += margin * margin;
            margin_count += 1;
            int bucket = static_cast<int>(std::round(margin * 1000.0));
            margin_hist[bucket]++;
        }

        enn_output.close();
        if (margin_count > 0) {
            double mean = margin_sum / margin_count;
            double var = margin_sq / margin_count - mean * mean;
            if (var < 0.0) var = 0.0;
            std::cout << "[Debug] Margin mean=" << mean << " std=" << std::sqrt(var)
                      << " samples=" << margin_count << std::endl;
            std::cout << "[Debug] Margin histogram (scaled x1000):";
            for (const auto& kv : margin_hist) {
                std::cout << " [" << kv.first << " -> " << kv.second << "]";
            }
            std::cout << std::endl;
        }
        std::cout << "Saved ENN predictions to: " << options.telemetry_path << std::endl;
        
        std::cout << "\n✅ BICEP -> ENN-C++ pipeline completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
