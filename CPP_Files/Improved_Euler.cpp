#include <vector>
#include <functional>
#include <stdexcept>

/**
 * @class ImprovedEulerSolver
 * @brief A numerical solver for systems of ordinary differential equations (ODEs) using the Improved Euler method (Heun's method).
 * 
 * This class implements the Improved Euler method, also known as Heun's method or the explicit trapezoidal rule,
 * which is a second-order Runge-Kutta method. It provides predictor-corrector functionality for more accurate
 * numerical integration compared to the basic Euler method.
 * 
 * The solver stores the complete solution history and can handle systems of multiple variables.
 */
class ImprovedEulerSolver {
private:
    std::vector<double> solutions_;  ///< Flat storage for all solution values [step0_var0, step0_var1, ..., step1_var0, ...]
    size_t num_steps_;               ///< Total number of time steps to store
    size_t num_vars_;                ///< Number of variables in the system of ODEs

    /**
     * @brief Converts 2D indices (step, variable) to a 1D index in the solutions_ array.
     * @param step Time step index (0 to num_steps_-1)
     * @param var Variable index (0 to num_vars_-1)
     * @return Linear index in the solutions_ vector
     */
    size_t idx(size_t step, size_t var) const {
        return step * num_vars_ + var;
    }

public:
    /**
     * @brief Constructs an ImprovedEulerSolver with specified dimensions.
     * @param num_steps Number of time steps to allocate storage for
     * @param num_vars Number of variables in the system of ODEs
     */
    ImprovedEulerSolver(size_t num_steps, size_t num_vars)
        : num_steps_(num_steps), num_vars_(num_vars) {
            solutions_.resize(num_steps * num_vars, 0.0);
        }

    /**
     * @brief Solves the system of ODEs using the Improved Euler method.
     * 
     * The Improved Euler method uses a predictor-corrector approach:
     * 1. Predictor step: Compute Euler method estimate
     * 2. Corrector step: Average slopes at current and predicted points
     * 
     * This results in a second-order accurate method with error O(dt²).
     * 
     * @param initial_conditions Initial values for all variables at time t=0
     * @param dt Time step size
     * @param derivative Function that computes derivatives dy/dt = f(t, y)
     *                   Takes current time and state vector, returns derivative vector
     * 
     * @throws std::invalid_argument if initial_conditions size doesn't match num_vars_
     */
    void solve(const std::vector<double>& initial_conditions,
               double dt,
               std::function<std::vector<double>(double, const std::vector<double>&)> derivative) {
        
        // Validate initial conditions
        if (initial_conditions.size() != num_vars_) {
            throw std::invalid_argument("Initial conditions size must match number of variables");
        }
        
        // Set initial conditions
        for (size_t i = 0; i < num_vars_; ++i) {
            at(0, i) = initial_conditions[i];
        }

        // Improved Euler integration
        for (size_t t = 0; t < num_steps_ - 1; ++t) {
            double current_time = t * dt;

            // Get current state
            auto current_state = get_state(t);

            // Predictor step (Euler method): y* = y_n + dt * f(t_n, y_n)
            auto k1 = derivative(current_time, current_state);
            std::vector<double> predictor(num_vars_);
            for (size_t i = 0; i < num_vars_; ++i) {
                predictor[i] = current_state[i] + dt * k1[i];
            }

            // Corrector step: y_{n+1} = y_n + dt/2 * [f(t_n, y_n) + f(t_{n+1}, y*)]
            double next_time = current_time + dt;
            auto k2 = derivative(next_time, predictor);

            for (size_t i = 0; i < num_vars_; ++i) {
                at(t + 1, i) = current_state[i] + dt * 0.5 * (k1[i] + k2[i]);
            }
        }
    }

    /**
     * @brief Access solution value for a specific time step and variable (mutable version).
     * @param step Time step index (0 to num_steps_-1)
     * @param var Variable index (0 to num_vars_-1)
     * @return Reference to the solution value
     * 
     * @throws std::out_of_range if step or var indices are out of bounds
     */
    double& at(size_t step, size_t var) { 
        if (step >= num_steps_ || var >= num_vars_) {
            throw std::out_of_range("Step or variable index out of bounds");
        }
        return solutions_[idx(step, var)];
    }

    /**
     * @brief Access solution value for a specific time step and variable (const version).
     * @param step Time step index (0 to num_steps_-1)
     * @param var Variable index (0 to num_vars_-1)
     * @return The solution value
     * 
     * @throws std::out_of_range if step or var indices are out of bounds
     */
    double at(size_t step, size_t var) const {  
        if (step >= num_steps_ || var >= num_vars_) {
            throw std::out_of_range("Step or variable index out of bounds");
        }
        return solutions_[idx(step, var)];
    }

    /**
     * @brief Get the complete state vector at a specific time step.
     * @param step Time step index (0 to num_steps_-1)
     * @return Vector containing all variable values at the specified time step
     * 
     * @throws std::out_of_range if step index is out of bounds
     */
    std::vector<double> get_state(size_t step) const {
        if (step >= num_steps_) {
            throw std::out_of_range("Step index out of bounds");
        }
        std::vector<double> state(num_vars_);
        for (size_t i = 0; i < num_vars_; ++i) {
            state[i] = solutions_[idx(step, i)];
        }
        return state;
    }

    /**
     * @brief Get the number of time steps allocated.
     * @return Total number of time steps
     */
    size_t get_num_steps() const { return num_steps_; }

    /**
     * @brief Get the number of variables in the system.
     * @return Number of variables
     */
    size_t get_num_vars() const { return num_vars_; }
};