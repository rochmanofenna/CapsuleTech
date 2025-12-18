#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <map>
#include <iomanip>
#include "enn/trainer.hpp"

using namespace enn;

// Simple CSV reader for BICEP parquet data (converted to CSV)
struct TrajectoryData {
    std::vector<std::vector<Vec>> sequences;
    std::vector<std::vector<F>> targets;
    std::vector<uint64_t> sequence_ids;
};

TrajectoryData load_bicep_data(const std::string& csv_file) {
    TrajectoryData data;
    std::ifstream file(csv_file);
    std::string line;
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + csv_file);
    }
    
    // Skip header
    std::getline(file, line);
    
    std::map<uint64_t, std::vector<Vec>> sequence_map;
    std::map<uint64_t, std::vector<F>> target_map;
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 12) continue;
        
        uint64_t sequence_id = std::stoull(tokens[6]); // sequence_id column
        uint32_t step = std::stoul(tokens[7]);         // step column  
        F input = std::stod(tokens[9]);                // input column (index 9)
        F target = std::stod(tokens[10]);              // target column (index 10)
        
        // Store input as 1D vector
        Vec input_vec(1);
        input_vec << input;
        
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
    }
    
    // Convert map to vectors
    for (const auto& pair : sequence_map) {
        data.sequences.push_back(pair.second);
        data.targets.push_back(target_map[pair.first]);
        data.sequence_ids.push_back(pair.first);
    }
    
    return data;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <bicep_csv_file>" << std::endl;
        return 1;
    }
    
    std::string csv_file = argv[1];
    std::cout << "=== BICEP -> ENN-C++ Integration ===" << std::endl;
    
    try {
        // Load BICEP trajectory data
        std::cout << "Loading BICEP trajectory data from: " << csv_file << std::endl;
        TrajectoryData traj_data = load_bicep_data(csv_file);
        
        std::cout << "Loaded " << traj_data.sequences.size() << " sequences" << std::endl;
        if (traj_data.sequences.empty()) {
            std::cerr << "No data loaded!" << std::endl;
            return 1;
        }
        
        // Print sample data
        std::cout << "Sample sequence (first 5 steps):" << std::endl;
        for (size_t i = 0; i < std::min(5UL, traj_data.sequences[0].size()); ++i) {
            std::cout << "  Step " << i << ": input=" << traj_data.sequences[0][i].transpose()
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
        const int input_dim = 1;
        const int hidden_dim = 64;
        const F lambda = 0.05;
        
        std::cout << "\n=== Training ENN on BICEP Trajectories ===" << std::endl;
        SequenceTrainer trainer(k, input_dim, hidden_dim, lambda, config);
        
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
        
        std::ofstream enn_output("enn_predictions.csv");
        enn_output << "sequence_id,final_prediction,target,confidence\n";
        
        for (size_t i = 0; i < train_data.sequences.size(); ++i) {
            auto predictions = trainer.forward_sequence(train_data.sequences[i]);
            F final_pred = predictions.back();
            F target = train_data.targets[i].back();
            F confidence = std::abs(final_pred - 0.5) * 2.0; // Distance from 0.5, scaled to [0,1]
            
            enn_output << traj_data.sequence_ids[i] << ","
                       << final_pred << ","
                       << target << ","
                       << confidence << "\n";
        }
        
        enn_output.close();
        std::cout << "Saved ENN predictions to: enn_predictions.csv" << std::endl;
        
        std::cout << "\n✅ BICEP -> ENN-C++ pipeline completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}