#include <vector>
#include <functional>
#include <stdexcept>

/**
 * @class RungeKutta4Solver
 * @brief A numerical solver for systems of ordinary differential equations (ODEs) using the 4th-order Runge-Kutta method.
 * 
 * This class implements the classic 4th-order Runge-Kutta method (RK4), which provides
 * fourth-order accuracy with error O(dt⁴). It uses four slope evaluations per time step
 * to achieve higher accuracy than lower-order methods.
 * 
 * The solver stores the complete solution history and can handle systems of multiple variables.
 */
class RungeKutta4Solver {
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

    /**
     * @brief Adds two vectors element-wise.
     * @param a First vector
     * @param b Second vector
     * @return Vector sum a + b
     */
    std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b) const {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    /**
     * @brief Multiplies a vector by a scalar.
     * @param scalar Scalar multiplier
     * @param vec Vector to multiply
     * @return Scaled vector
     */
    std::vector<double> vector_scale(double scalar, const std::vector<double>& vec) const {
        std::vector<double> result(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = scalar * vec[i];
        }
        return result;
    }

public:
    /**
     * @brief Constructs a RungeKutta4Solver with specified dimensions.
     * @param num_steps Number of time steps to allocate storage for
     * @param num_vars Number of variables in the system of ODEs
     */
    RungeKutta4Solver(size_t num_steps, size_t num_vars)
        : num_steps_(num_steps), num_vars_(num_vars) {
            solutions_.resize(num_steps * num_vars, 0.0);
        }

    /**
     * @brief Solves the system of ODEs using the 4th-order Runge-Kutta method.
     * 
     * The RK4 method uses four slope evaluations per time step:
     * 1. k1 = f(t, y_n)
     * 2. k2 = f(t + dt/2, y_n + (dt/2)*k1)
     * 3. k3 = f(t + dt/2, y_n + (dt/2)*k2) 
     * 4. k4 = f(t + dt, y_n + dt*k3)
     * 
     * Final update: y_{n+1} = y_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
     * 
     * This results in a fourth-order accurate method with error O(dt⁴).
     * 
     * @param initial_conditions Initial values for all variables at time t=0
     * @param dt Time step size
     * @param derivative Function that computes derivatives dy/dt = f(t, y)
     *                   Takes current time and state vector, returns derivative vector
     * 
     * @throws std::invalid_argument if initial_conditions size doesn't match num_vars_
     * 
     * @example
     * // Solve dy/dt = -y with y(0) = 1.0
     * RungeKutta4Solver solver(100, 1);
     * auto derivative = [](double t, const std::vector<double>& y) {
     *     return std::vector<double>{-y[0]};
     * };
     * solver.solve({1.0}, 0.01, derivative);
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

        // Runge-Kutta 4 integration
        for (size_t t = 0; t < num_steps_ - 1; ++t) {
            double current_time = t * dt;
            auto current_state = get_state(t);

            // RK4 stages
            auto k1 = derivative(current_time, current_state);
            
            auto k2_input = vector_add(current_state, vector_scale(dt/2.0, k1));
            auto k2 = derivative(current_time + dt/2.0, k2_input);
            
            auto k3_input = vector_add(current_state, vector_scale(dt/2.0, k2));
            auto k3 = derivative(current_time + dt/2.0, k3_input);
            
            auto k4_input = vector_add(current_state, vector_scale(dt, k3));
            auto k4 = derivative(current_time + dt, k4_input);

            // Combine slopes for final update
            for (size_t i = 0; i < num_vars_; ++i) {
                at(t + 1, i) = current_state[i] + 
                               (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
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