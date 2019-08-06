//
// Created by resl on 7/15/19.
//

#ifndef IDS_ODE_INTEGRATE_ADJOINT_H
#define IDS_ODE_INTEGRATE_ADJOINT_H

#include<stan/math.hpp>
#include<vector>

using namespace stan{
    using namespace math{
        struct aug_dyn_ {
            int theta_size;
            int y_size;
            int t1;
            F dyn;

            aug_dyn_(F dyn, int theta_size, int y_size, int t1);

            template<typename T0, typename T1, typename T2>
            std::vector<typename stan::return_type<T1, T2>::type>
            operator()(
                    const T0 &t, const std::vector <T1> &aug_y, const std::vector <T2> &theta,
                    const std::vector<double> &x, const std::vector<int> &x_i, std::ostream *msgs
            );
        };

        std::vector<double> ode_adjoint_integrate_rk45(
                const std::vector<double>& y0,
                const std::vector<double>& theta0,
                const double& t0,
                const double& t1
        );

        std::vector<var> ode_adjoint_integrate_rk45(
                const std::vector<var>& y0,
                const std::vector<var>& theta0,
                const var& t0,
                const var& t1
        );
    }
}


#endif //IDS_ODE_INTEGRATE_ADJOINT_H
