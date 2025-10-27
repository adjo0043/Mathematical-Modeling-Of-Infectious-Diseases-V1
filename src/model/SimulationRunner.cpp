#include "model/SimulationRunner.hpp"
#include "model/AgeSEPAIHRDsimulator.hpp"
#include "utils/Logger.hpp"
#include <functional>
#include <sstream>

namespace epidemic {

SimulationRunner::SimulationRunner(
    std::shared_ptr<AgeSEPAIHRDModel> model_template,
    std::shared_ptr<IOdeSolverStrategy> solver)
    : model_template_(model_template),
      solver_(solver) {
    
    if (!model_template_) {
        throw std::invalid_argument("SimulationRunner: Model template cannot be null");
    }
    if (!solver_) {
        throw std::invalid_argument("SimulationRunner: Solver strategy cannot be null");
    }
}

SimulationResult SimulationRunner::runSimulation(
    const SEPAIHRDParameters& params,
    const Eigen::VectorXd& initial_state,
    const std::vector<double>& time_points) {
    
    total_calls_++;
    
    // Generate cache key from parameters
    Eigen::VectorXd param_vec = parametersToVector(params);
    size_t cache_key = hashParameterVector(param_vec);
    
    // Check cache
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
        cache_hits_++;
        Logger::getInstance().debug("SimulationRunner", 
            "Cache hit! (Hit rate: " + std::to_string(cache_hits_) + "/" + 
            std::to_string(total_calls_) + ")");
        return it->second;
    }
    
    // Cache miss - run simulation
    auto run_npi_strategy = model_template_->getNpiStrategy()->clone();
    auto run_model = std::make_shared<AgeSEPAIHRDModel>(params, run_npi_strategy);
    
    AgeSEPAIHRDSimulator simulator(
        run_model, 
        solver_, 
        time_points.front(), 
        time_points.back(), 
        1.0,  // dt
        1e-6, // atol
        1e-6  // rtol
    );
    
    SimulationResult result = simulator.run(initial_state, time_points);
    
    // Store in cache
    if (result.isValid()) {
        cache_[cache_key] = result;
    } else {
        Logger::getInstance().warning("SimulationRunner", "Invalid simulation result - not cached");
    }
    
    return result;
}

void SimulationRunner::clearCache() {
    cache_.clear();
    cache_hits_ = 0;
    total_calls_ = 0;
    Logger::getInstance().info("SimulationRunner", "Cache cleared");
}

std::pair<size_t, size_t> SimulationRunner::getCacheStats() const {
    return {cache_hits_, total_calls_};
}

size_t SimulationRunner::hashParameterVector(const Eigen::VectorXd& param_vec) const {
    size_t hash = 0;
    for (int i = 0; i < param_vec.size(); ++i) {
        // Combine hash values using XOR and bit rotation
        size_t value_hash = std::hash<double>{}(param_vec(i));
        hash ^= value_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

Eigen::VectorXd SimulationRunner::parametersToVector(const SEPAIHRDParameters& params) const {
    // Flatten all parameters into a single vector for hashing
    std::vector<double> flat_params;
    
    // Basic scalars
    flat_params.push_back(params.beta);
    flat_params.push_back(params.sigma);
    flat_params.push_back(params.gamma_p);
    flat_params.push_back(params.gamma_A);
    flat_params.push_back(params.gamma_I);
    flat_params.push_back(params.gamma_H);
    flat_params.push_back(params.gamma_ICU);
    flat_params.push_back(params.theta);
    flat_params.push_back(params.contact_matrix_scaling_factor);
    
    // Age-specific vectors
    for (int i = 0; i < params.p.size(); ++i) flat_params.push_back(params.p(i));
    for (int i = 0; i < params.h.size(); ++i) flat_params.push_back(params.h(i));
    for (int i = 0; i < params.icu.size(); ++i) flat_params.push_back(params.icu(i));
    for (int i = 0; i < params.d_H.size(); ++i) flat_params.push_back(params.d_H(i));
    for (int i = 0; i < params.d_ICU.size(); ++i) flat_params.push_back(params.d_ICU(i));
    for (int i = 0; i < params.a.size(); ++i) flat_params.push_back(params.a(i));
    for (int i = 0; i < params.h_infec.size(); ++i) flat_params.push_back(params.h_infec(i));
    for (int i = 0; i < params.N.size(); ++i) flat_params.push_back(params.N(i));
    
    // Contact matrix (flatten row-major)
    for (int i = 0; i < params.M_baseline.rows(); ++i) {
        for (int j = 0; j < params.M_baseline.cols(); ++j) {
            flat_params.push_back(params.M_baseline(i, j));
        }
    }
    
    // Kappa values
    for (const auto& kappa : params.kappa_values) {
        flat_params.push_back(kappa);
    }
    
    // Kappa end times
    for (const auto& t : params.kappa_end_times) {
        flat_params.push_back(t);
    }
    
    // Beta values  
    for (const auto& beta_val : params.beta_values) {
        flat_params.push_back(beta_val);
    }
    
    // Beta end times
    for (const auto& t : params.beta_end_times) {
        flat_params.push_back(t);
    }
    
    // Convert to Eigen vector
    Eigen::VectorXd param_vec(flat_params.size());
    for (size_t i = 0; i < flat_params.size(); ++i) {
        param_vec(i) = flat_params[i];
    }
    
    return param_vec;
}

} // namespace epidemic
