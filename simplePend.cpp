#include<Eigen/dense>

// pendulum dynamics
Eigen::VectorXd pendDyn(
    double t_in,
    Eigen::VectorXd y_in,
    Eigen::VectorXd params
) {
    double g = params[0], l = params[1];
    return Eigen::VectorXd({
        y_in[1],
        -(g/l) * std::sin(y_in[0])
    });
}

int main() {
    return 0;
}
