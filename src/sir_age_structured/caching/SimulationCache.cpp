#include "sir_age_structured/caching/SimulationCache.hpp"
#include <functional>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace epidemic {

// Helper: Hash combiner (Boost-like) to mix bits effectively
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

SimulationCache::SimulationCache(size_t max_size)
    : max_size_(max_size), min_frequency_(0)
{
    if (max_size_ == 0) {
        throw std::invalid_argument("SimulationCache: max_size must be > 0.");
    }
}

// Optimized Hash Computation O(N) - No Heap Allocation
size_t SimulationCache::computeHash(const Eigen::VectorXd& params) const {
    std::size_t seed = 0;
    std::hash<long long> hasher;
    
    // Precision factor for quantization (e.g., 1e6 for 6 decimal places)
    // This allows fuzzy matching for floating point values within epsilon
    constexpr double precision_scale = 1e8;

    for (int i = 0; i < params.size(); ++i) {
        // Quantize to integer to handle floating point non-determinism
        // std::round ensures 1.00000001 and 0.99999999 map to same bucket
        long long quantized = static_cast<long long>(std::round(params[i] * precision_scale));
        
        // Combine hash
        hash_combine(seed, hasher(quantized));
    }
    return seed;
}

std::string SimulationCache::createCacheKey(const Eigen::VectorXd& parameters) const {
    // For interface compatibility, return string representation of the hash
    return std::to_string(computeHash(parameters));
}

void SimulationCache::updateFrequency(size_t key) {
    auto& node = cache_.at(key);
    int old_freq = node.frequency;

    freq_map_[old_freq].erase(node.freq_list_iter);

    if (freq_map_[old_freq].empty()) {
        freq_map_.erase(old_freq);
        if (old_freq == min_frequency_) {
            min_frequency_++; // Simplified update for LFU
            // If freq_map_ is empty (shouldn't happen here), reset to 0
            if (freq_map_.empty()) min_frequency_ = 0; 
            else min_frequency_ = freq_map_.begin()->first;
        }
    }

    node.frequency++;
    int new_freq = node.frequency;

    freq_map_[new_freq].push_back(key);
    node.freq_list_iter = std::prev(freq_map_[new_freq].end());
}

std::optional<double> SimulationCache::get(const Eigen::VectorXd& parameters) {
    size_t key = computeHash(parameters);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    updateFrequency(key);
    return it->second.value;
}

void SimulationCache::set(const Eigen::VectorXd& parameters, double result) {
    size_t key = computeHash(parameters);
    
    if (cache_.find(key) != cache_.end()) {
        cache_[key].value = result;
        updateFrequency(key);
    } else {
        if (cache_.size() >= max_size_) {
            // Evict LFU/LRU
            auto it = freq_map_.find(min_frequency_);
            if (it != freq_map_.end() && !it->second.empty()) {
                size_t key_to_evict = it->second.front();
                it->second.pop_front();
                cache_.erase(key_to_evict);
                if (it->second.empty()) {
                    freq_map_.erase(min_frequency_);
                }
            }
        }
        
        int initial_freq = 1;
        freq_map_[initial_freq].push_back(key);
        auto list_it = std::prev(freq_map_[initial_freq].end());
        cache_[key] = {result, initial_freq, list_it};
        min_frequency_ = 1;
    }
}

void SimulationCache::clear() {
    cache_.clear();
    freq_map_.clear();
    min_frequency_ = 0;
}

size_t SimulationCache::size() const {
    return cache_.size();
}

// For interface compatibility: converts string key back to size_t if possible
// Note: This assumes the string key passed in IS the hash string from createCacheKey
bool SimulationCache::getLikelihood(const std::string& keyStr, double& value) {
    try {
        size_t key = std::stoull(keyStr);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            updateFrequency(key);
            value = it->second.value;
            return true;
        }
    } catch (...) {
        // Key wasn't a valid size_t string, or overflow
        return false;
    }
    return false;
}

void SimulationCache::storeLikelihood(const std::string& keyStr, double value) {
    try {
        size_t key = std::stoull(keyStr);
        if (cache_.find(key) != cache_.end()) {
            cache_[key].value = value;
            updateFrequency(key);
        } else {
            if (cache_.size() >= max_size_) {
                auto freq_it = freq_map_.find(min_frequency_);
                if (freq_it != freq_map_.end() && !freq_it->second.empty()) {
                    size_t key_to_evict = freq_it->second.front();
                    freq_it->second.pop_front();
                    cache_.erase(key_to_evict);
                    if (freq_it->second.empty()) {
                        freq_map_.erase(min_frequency_);
                    }
                }
            }
            int initial_freq = 1;
            freq_map_[initial_freq].push_back(key);
            auto list_it = std::prev(freq_map_[initial_freq].end());
            cache_[key] = {value, initial_freq, list_it};
            min_frequency_ = 1;
        }
    } catch (...) {
        // Invalid key string, ignore or log error
    }
}

} // namespace epidemic