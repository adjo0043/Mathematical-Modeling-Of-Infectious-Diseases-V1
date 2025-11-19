#ifndef SIMULATION_CACHE_HPP
#define SIMULATION_CACHE_HPP

#include "sir_age_structured/interfaces/ISimulationCache.hpp"
#include <unordered_map>
#include <string>
#include <map>
#include <list>
#include <vector>
#include <optional>
#include <Eigen/Dense>

namespace epidemic {

/**
 * @brief An implementation of ISimulationCache using LFU eviction with LRU tie-breaking.
 *        Optimized with Zobrist-style hashing for Eigen::VectorXd keys.
 */
class SimulationCache : public ISimulationCache {
public:
    /**
     * @brief Constructor.
     * @param max_size Maximum number of entries in the cache.
     */
    explicit SimulationCache(size_t max_size = 1000);

    // Implementation of the ISimulationCache interface
    std::optional<double> get(const Eigen::VectorXd& parameters) override;
    void set(const Eigen::VectorXd& parameters, double result) override;
    void clear() override;
    size_t size() const override;
    
    // Note: This method name is kept for interface compatibility but now returns a stringified hash
    std::string createCacheKey(const Eigen::VectorXd& parameters) const override;
    
    // Overloaded for direct hash access (internal optimization)
    size_t computeHash(const Eigen::VectorXd& parameters) const;

    bool getLikelihood(const std::string& key, double& value) override;
    void storeLikelihood(const std::string& key, double value) override;

private:
    struct CacheNode {
        double value;
        int frequency;
        std::list<size_t>::iterator freq_list_iter; // Stores iterator to hash key
    };

    /** @brief Main cache storage mapping hash keys to cache nodes. */
    std::unordered_map<size_t, CacheNode> cache_;

    /** @brief Frequency map organizing hash keys by access frequency. */
    std::map<int, std::list<size_t>> freq_map_;

    size_t max_size_;
    int min_frequency_;

    void updateFrequency(size_t key);
};

} // namespace epidemic

#endif // SIMULATION_CACHE_HPP