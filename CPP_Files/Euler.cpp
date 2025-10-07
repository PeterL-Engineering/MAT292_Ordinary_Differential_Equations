#include <vector>
#include <functional>

class EulerSolver {
private:
    std::vector<double> solutions_;
    size_t num_steps_;
    size_t num_vars_;

    size_t idx(size_t step, size_t var) const {
        return step * num_vars_ + var;
    }

public:
    EulerSolver(size_t num_steps, size_t num_vars)
        : num_steps_(num_steps), num_vars_(num_vars) {
            solutions_.resize(num_steps * num_vars, 0.0);
        }

    void solve(const std::vector<double>& initial_conditions,
               double dt,
               std::function<std::vector<double>(double, const std::vector<double>&)> derivative) {
        
        // Set initial conditions
        for (size_t i = 0; i < num_vars_; ++i) {
            at(0, i) = initial_conditions[i];
        }

        // Euler integration
        for (size_t t = 0; t < num_steps_ - 1; ++t) {
            double current_time = t * dt;  
            auto deriv = derivative(current_time, get_state(t)); 

            for (size_t i = 0; i < num_vars_; ++i) {
                at(t + 1, i) = at(t, i) + dt * deriv[i]; 
            }
        }
    }

    double& at(size_t step, size_t var) { 
        return solutions_[idx(step, var)];
    }

    double at(size_t step, size_t var) const {  
        return solutions_[idx(step, var)];
    }

    std::vector<double> get_state(size_t step) const {
        std::vector<double> state(num_vars_);
        for (size_t i = 0; i < num_vars_; ++i) {
            state[i] = solutions_[idx(step, i)];
        }
        return state;
    }
};