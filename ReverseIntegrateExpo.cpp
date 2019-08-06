#include<stan/math.hpp>
#include<vector>
#include<iostream>
using namespace std;

namespace pend_sys {
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

    struct pend_dyn_backward {
        template<typename T0, typename T1, typename T2>
        std::vector<typename stan::return_type<T1, T2>::type>
        operator() (
                const T0& t, const std::vector<T1>& y, const std::vector<T2>& theta,
                const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
        ) const {
            std::vector<typename stan::return_type<T1, T2>::type> dydt(2);
            dydt[0] = -y[1];
            dydt[1] = (theta[0]/theta[1])*sin(y[0]);
            return dydt;
        }
    };
}


int main() {
    pend_sys::pend_dyn dyn;
    pend_sys::pend_dyn_backward dyn_backward;
    vector<double> y0 = {stan::math::pi()/4, 0};
    double t0 = 0, t1 = 0.2;
    vector<double> ts = {t1};
    vector<double> theta = {9.8, 1};

    cout << "Initial" << endl;
    for (auto yi: y0)
        cout << yi << " ";
    cout << endl;

    vector<vector<double>> y1 = stan::math::integrate_ode_rk45(dyn, y0, t0, ts, theta, vector<double>(), vector<int>(), &(cout));
    cout << "Forward from 0 to 0.2" << endl;
    for (auto yi: y1[0])
        cout << yi << " ";
    cout << endl;


    cout << "Backward from 0.2 to 0" << endl;
    ts = {t1-t0};
    vector<vector<double>> y0p = stan::math::integrate_ode_rk45(dyn_backward, y1[0], 0, ts, theta, vector<double>(), vector<int>(), &(cout));
    for (auto yi: y0p[0])
        cout << yi << " ";
    cout << endl;

    return 0;
}
