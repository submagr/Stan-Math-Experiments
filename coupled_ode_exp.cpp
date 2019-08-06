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

// struct pend_dyn {
//     std::vector<var>
//     operator() (
//         const double& t, const std::vector<double>& y, const std::vector<var>& theta,
//         const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
//     ) const {
//         std::vector<var> dydt(2);
//         dydt[0] = y[1];
//         dydt[1] = -(theta[0]/theta[1])*sin(y[0]);
//         return dydt;
//     }

//     std::vector<double>
//     operator() (
//         const double& t, const std::vector<double>& y, const std::vector<double>& theta,
//         const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
//     ) const {
//         std::vector<double> dydt(2);
//         dydt[0] = y[1];
//         dydt[1] = -(theta[0]/theta[1])*sin(y[0]);
//         return dydt;
//     }

// };

int main() {
    const pend_dyn dyn;
    const std::vector<double> y0 = {pi()/4, 0};

    double t0 = 0;
    std::vector<double> ts;
    for (double i=0.1; i<10; i+=0.1) {
       ts.push_back(i);
    }

    const std::vector<var> theta = {9.8, 2};
    const std::vector<double> x;
    const std::vector<int> x_int;

    // coupled_ode_system<pend_dyn, double, var> coupled_system(dyn, y0, theta, x, x_int, &std::cout);

    auto y = integrate_ode_rk45(dyn, y0, t0, ts, theta, x, x_int);
    for (int i=0; i<2; i++) {
        set_zero_all_adjoints();
        y[9][i].grad();
        std::cout << theta[1].adj() << ", " <<theta[1].adj() << std::endl;
    }
    return 0;
}

(const ids::ParameterizedRigidBodyDynamics<
    ids::models::CompoundPendulum<
        ids::Model<
            stan::math::var, 3, 3,
            stan::math::var, stan::math::var, stan::math::var
        >,
        2, true
    >,
    stan::math::var
>)(
    double,
    ids::math::Vector<stan::math::var, 6> &,
    ids::math::Vector<stan::math::var, 2> &
)’ is ambiguous
    math::Vector<stan::math::var, StateDim> yRetV = dyn_(double(t), yV, thetaV)