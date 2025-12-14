#pragma once

#include "model/interfaces/INpiStrategy.hpp"
#include "exceptions/Exceptions.hpp"
#include "model/ModelConstants.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <limits>

namespace epidemic {

class PiecewiseConstantParameterStrategy {
public:
    PiecewiseConstantParameterStrategy(
        const std::string& param_name,
        const std::vector<double>& end_times,
        const std::vector<double>& values,
        double baseline_value,
        double baseline_end_time);

    double getValue(double time) const;

private:
    std::string parameter_name_;
    std::vector<double> period_end_times_;
    std::vector<double> parameter_values_;
    double baseline_value_;
    double baseline_period_end_time_;

    // Cache for fast interval lookup when times are (mostly) non-decreasing.
    mutable size_t cached_period_index_ = 0;
    mutable double cached_time_ = std::numeric_limits<double>::lowest();
};

} // namespace epidemic