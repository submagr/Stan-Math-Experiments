//
// Created by resl on 7/15/19.
//

#include "ode_integrate_adjoint.h"
#include<stan/math.hpp>
#include<vector>

using namespace stan{
    using namespace math{
        struct aug_dyn_ {
            int theta_size;
            int y_size;
            int t1;
            F dyn;

            aug_dyn_(F dyn, int theta_size, int y_size, int t1) :
                dyn(dyn),
                theta_size(theta_size),
                y_size(y_size),
                t1(t1)
            {};

            template<typename T0, typename T1, typename T2>
            std::vector<typename stan::return_type<T1, T2>::type>
            operator()(
                    const T0 &t, const std::vector <T1> &aug_y, const std::vector <T2> &theta,
                    const std::vector<double> &x, const std::vector<int> &x_i, std::ostream *msgs
            ) {
                std::vector<double> yt1;
                std::vector<std::vector<double>> at1;
                std::vector<std::vector<double>> dl_dtheta0;
                std::vector<double> dl_dt0;

                inflate(aug_y, theta_size, y_size, yt1, at1, dl_dtheta0, dl_dt0);

                stan::math::var tVar = t1 - t;
                std::vector<stan::math::var> yt1Var(yt1.begin(), yt1.end());
                std::vector<stan::math::var> thetaVar(theta.begin(), theta.end());
                std::vector<stan::math::var> f = dyn(tVar, yt1Var, thetaVar, std::vector<double>(), std::vector<int>(), &(std::cout));
                std::vector<double> fVal = value_of(f);

                // Calculate -a(t)
                std::vector<std::vector<double>> at1T (at1.size(), std::vector<double>(at1[0].size()));
                for (int i=0; i<at1T.size(); i++) {
                    for (int j=0; j<at1T[0].size(); j++) {
                        at1T[i][j] = -at1[i][j];
                    }
                }


                std::vector<std::vector<double>> df_dyt1(yt1.size(), std::vector<double>(yt1.size()));
                for (int i=0; i<yt1.size(); i++) {
                    f[i].grad();
                    for (int j=0; j<yt1.size(); j++) {
                        df_dyt1[i][j] = yt1Var[j].adj();
                    }
                    stan::math::set_zero_all_adjoints();
                }
                std::vector<std::vector<double>> ret_ayt1(at1T.size(), std::vector<double>(df_dyt1[0].size()));
                for (int i=0; i<ret_ayt1.size(); i++) {
                    for (int j=0; j<ret_ayt1[0].size(); j++) {
                        double sum = 0;
                        for (int k=0; k<df_dyt1.size(); k++) {
                            sum += at1T[i][k]*df_dyt1[k][j];
                        }
                        ret_ayt1[i][j] = sum;
                    }
                }

                std::vector<std::vector<double>> df_dtheta(yt1.size(), std::vector<double>(theta.size()));
                for (int i=0; i<yt1.size(); i++) {
                    f[i].grad();
                    for (int j=0; j<theta.size(); j++) {
                        df_dtheta[i][j] = thetaVar[j].adj();
                    }
                    stan::math::set_zero_all_adjoints();
                }
                std::vector<std::vector<double>> ret_atheta(at1T.size(), std::vector<double>(df_dtheta[0].size()));
                for (int i=0; i<ret_atheta.size(); i++) {
                    for (int j=0; j<ret_atheta[0].size(); j++) {
                        double sum = 0;
                        for (int k=0; k<df_dtheta.size(); k++) {
                            sum += at1T[i][k]*df_dtheta[k][j];
                        }
                        ret_atheta[i][j] = sum;
                    }
                }

                std::vector<double> df_dt(yt1.size());
                for (int i=0; i<yt1.size(); i++){
                    f[i].grad();
                    df_dt[i] = tVar.adj();
                    stan::math::set_zero_all_adjoints();
                }
                std::vector<double> ret_at(df_dt.size());
                for (int i=0; i<at1T.size(); i++) {
                    double sum = 0;
                    for (int k=0; k<df_dt.size(); k++) {
                        sum += at1T[i][k]*df_dt[k];
                    }
                    ret_at[i] = sum;
                }

                // flatten aug_y0 into a vector
                std::vector<typename stan::return_type<T1, T2>::type> ret_aug_y;
                flatten(fVal, ret_ayt1, ret_atheta, ret_at, ret_aug_y);

                return ret_aug_y;
            }
        };

        template<typename F>
        std::vector<double> ode_adjoint_integrate_rk45(
                F dyn,
                const std::vector<double>& y,
                const double& t,
                const std::vector<double>& ts,
                const std::vector<double>& theta,
                const std::vector<double>& x,
                const std::vector<int>& x_int
        ){
            std::vector<double> ts = {t1};
            std::vector<double> x;
            std::vector<int> xInt;
            std::vector<std::vector<double>> f = integrate_ode_rk45(dyn, y0, t0, ts, theta0, x, xInt);
            return f[0];
        };

        std::vector<var> ode_adjoint_integrate_rk45(
                const std::vector<var>& y0,
                const std::vector<var>& theta0,
                const var& t0,
                const var& t1
        );
    }
}
