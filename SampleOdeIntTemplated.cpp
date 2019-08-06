#include<stan/math.hpp>
#include<vector>
#include<iostream>

using namespace stan::math;

struct pend_dyn {
    template<typename T0, typename T1, typename T2>
    std::vector<typename stan::return_type<T1, T2>::type>
    operator() (
        const T0& t, const std::vector<T1>& y, const std::vector<T2>& theta,
        const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
    ) const {
        std::vector<typename stan::return_type<T1, T2>::type> dydt(2);
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
    for (auto yi: y) {
        for (auto yij: yi) {
            std::cout << yij << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}