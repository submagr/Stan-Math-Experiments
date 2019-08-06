#include<stan/math.hpp>
#include<vector>
#include<iostream>

using namespace stan::math;

struct pend_dyn {
    std::vector<stan::return_type<double, double>::type>
    operator() (
        const double& t, const std::vector<double>& y, const std::vector<double>& theta,
        const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
    ) const {
        std::vector<stan::return_type<double, double>::type> dydt(2);
        dydt[0] = y[1];
        dydt[1] = -(theta[0]/theta[1])*sin(y[0]);
        return dydt;
    }

};

int main() {
    const pend_dyn dyn;
    std::vector<double> y0 = {pi()/4, 0};
    double t0 = 0;
    std::vector<double> ts;
    for (double i=0.1; i<10; i+=0.1) {
       ts.push_back(i);
    }

    std::vector<double> theta = {9.8, 1};
    std::vector<double> x;
    std::vector<int> x_int;
    std::vector<std::vector<double>> y = integrate_ode_rk45(dyn, y0, t0, ts, theta, x, x_int);
    return 0;
}