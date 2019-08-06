#include<stan/math.hpp>
#include<vector>

void flatten(
        std::vector<double>& yt1,
        std::vector<std::vector<double>>& dl_dyt1,
        std::vector<std::vector<double>>& aug_theta,
        std::vector<double>& dl_dt1,
        std::vector<double>& ret
) {
    ret.clear();
    ret.insert(ret.end(), yt1.begin(), yt1.end());
    for (auto dl_dyt1i: dl_dyt1)
        ret.insert(ret.end(), dl_dyt1i.begin(), dl_dyt1i.end());
    for (auto aug_thetai: aug_theta)
        ret.insert(ret.end(), aug_thetai.begin(), aug_thetai.end());

    ret.insert(ret.end(), dl_dt1.begin(), dl_dt1.end());
}

void inflate(
        const std::vector<double>& ret,
        double theta_size,
        double y_size,
        std::vector<double>& yt1,
        std::vector<std::vector<double>>& dl_dyt1,
        std::vector<std::vector<double>>& aug_theta,
        std::vector<double>& dl_dt1
) {
    yt1.clear();
    yt1.insert(yt1.end(), ret.begin(), ret.begin()+y_size);

    int c = y_size;

    dl_dyt1 = std::vector<std::vector<double>>(y_size, std::vector<double>(y_size));
    for (int i=0; i<y_size; i++) {
        for (int j=0; j<y_size; j++) {
            dl_dyt1[i][j] = ret[c++];
        }
    }

    aug_theta = std::vector<std::vector<double>>(y_size, std::vector<double>(theta_size));
    for (int i=0; i<y_size; i++) {
        for (int j=0; j<theta_size; j++) {
            aug_theta[i][j] = ret[c++];
        }
    }

    dl_dt1.clear();
    dl_dt1.insert(dl_dt1.end(), ret.begin()+c, ret.end());
}

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

    struct pend_aug_dyn {
        int theta_size;
        int y_size;
        int t1;
        pend_aug_dyn(int theta_size, int y_size, int t1): theta_size(theta_size), y_size(y_size), t1(t1) {};

        template<typename T0, typename T1, typename T2>
        std::vector<typename stan::return_type<T1, T2>::type>
        operator() (
                const T0& t, const std::vector<T1>& aug_y, const std::vector<T2>& theta,
                const std::vector<double>& x, const std::vector<int>& x_i, std::ostream* msgs
        ) const {
            std::vector<double> yt1;
            std::vector<std::vector<double>> at1;
            std::vector<std::vector<double>> dl_dtheta0;
            std::vector<double> dl_dt0;

            inflate(
                    aug_y,
                    theta_size,
                    y_size,
                    yt1,
                    at1,
                    dl_dtheta0,
                    dl_dt0
            );

//            std::cout << "Printing aug_y0" << std::endl;
//            std::cout << "y" << std::endl;
//            for (auto yi: yt1)
//                std::cout << yi << " ";
//            std::cout << std::endl;
//
//            std::cout << "a_y" << std::endl;
//            for (auto ayi: at1) {
//                for (auto ayij: ayi)
//                    std::cout << ayij << " ";
//                std::cout << std::endl;
//            }
//
//            std::cout << "a_theta" << std::endl;
//            for (auto athetai: dl_dtheta0) {
//                for (auto athetaij: athetai)
//                    std::cout << athetaij << " ";
//                std::cout << std::endl;
//            }
//
//            std::cout << "a_t" << std::endl;
//            for (auto ati: dl_dt0) {
//                std::cout << ati << " ";
//            }
//            std::cout << std::endl;

            pend_dyn_backward forward_dyn;

            stan::math::var tVar = t1 - t;
            std::vector<stan::math::var> yt1Var(yt1.begin(), yt1.end());
            std::vector<stan::math::var> thetaVar(theta.begin(), theta.end());
            std::vector<stan::math::var> f = forward_dyn(tVar, yt1Var, thetaVar, std::vector<double>(), std::vector<int>(), &(std::cout));
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
}

namespace stan {
    namespace math {
        template<typename F, typename T1>
        std::vector<std::vector<T1>>
        integrate_ode_euler(
                F dyn,
                std::vector<T1> y0,
                double t0,
                std::vector<double> ts,
                std::vector<T1> theta,
                std::vector<double> x,
                std::vector<int> xInt
        ) {
            const double tstep = 0.001;
            std::vector<T1> y1;
            double t = t0;
            while(t < ts[0]) {
                // std::cout << "\t time: " << t << std::endl;
                std::vector<T1> dy_dt = dyn(t, y0, theta, x, xInt, &(std::cout));
                for(int i=0; i<y0.size(); i++) {
                    y0[i] += dy_dt[i]*tstep;
                }
                t += tstep;
            }
            return std::vector<std::vector<T1>>({y0});
        }
        
        std::vector<double> pend_forward(
                const std::vector<double>& y0,
                const std::vector<double>& theta0,
                const double& t0,
                const double& t1
        ) {
            pend_sys::pend_dyn dyn;
            std::vector<double> ts = {t1};
            std::vector<double> x;
            std::vector<int> xInt;
            std::vector<std::vector<double>> f = integrate_ode_rk45(dyn, y0, t0, ts, theta0, x, xInt);
            return f[0];
        }

        std::vector<var> pend_forward(
                const std::vector<var>& y0,
                const std::vector<var>& theta0,
                const var& t0,
                const var& t1
        ) {
            std::vector<double> y0Val = value_of(y0);
            std::vector<double> theta0Val = value_of(theta0);
            double t0Val = value_of(t0);
            double t1Val = value_of(t1);

            const pend_sys::pend_dyn dyn;
            const pend_sys::pend_aug_dyn aug_dyn(theta0.size(), y0.size(), t1Val);


            // y(t1)
            std::vector<double> yt1 = pend_forward(y0Val, theta0Val, t0Val, t1Val);

            // dL/dy(t1):: L here is same as y(t1) - Hence, identity Jacobian.
            std::vector<std::vector<double>> dl_dyt1(y0.size(), std::vector<double>(y0.size(), 0));
            for (int i=0; i<dl_dyt1.size(); i++) {
                for (int j=0; j<dl_dyt1[0].size(); j++) {
                    if (i == j) {
                        dl_dyt1[i][j] = 1;
                    }
                }
            }

            // aug_theta
            std::vector<std::vector<double>> aug_theta(y0.size(), std::vector<double>(theta0.size(), 0));

            // -dL/dt1 = -Transpose(dL/dy(t1)) * dyn(y(t1), t1, theta)
            std::vector<double> dl_dt1 = dyn(t1Val, yt1, theta0Val, std::vector<double>(), std::vector<int>(), &(std::cout));
            for (int i=0; i<dl_dt1.size(); i++) {
                dl_dt1[i] *= -1;
            }

            std::vector<double> aug_y0;
            flatten(yt1, dl_dyt1, aug_theta, dl_dt1, aug_y0);
            std::vector<double> ts = {t1Val - t0Val};

            std::vector<std::vector<double>> aug_y_ = integrate_ode_rk45(
                    aug_dyn,
                    aug_y0,
                    0,
                    ts,
                    theta0Val,
                    std::vector<double>(),
                    std::vector<int>()
            );

            std::vector<double> aug_y = aug_y_[0];

            std::vector<double> yt1_temp;
            std::vector<std::vector<double>> dl_dyt0;
            std::vector<std::vector<double>> dl_dtheta0;
            std::vector<double> dl_dt0;

            inflate(
                    aug_y,
                    theta0.size(),
                    y0.size(),
                    yt1_temp,
                    dl_dyt0,
                    dl_dtheta0,
                    dl_dt0
            );

            std::cout << "Computed Value of y(t0): " << yt1_temp[0] << ", " << yt1_temp[1] << std::endl;

            for (int i=0; i<dl_dt1.size(); i++)
                dl_dt1[i] *= -1;

            std::vector<var> params;
            params.insert(params.end(), y0.begin(), y0.end());
            params.insert(params.end(), theta0.begin(), theta0.end());
            params.insert(params.end(), t0);
            params.insert(params.end(), t1);

            std::vector<var> yt1Var(yt1.size());
            for (int i=0; i<yt1Var.size(); i++) {
                std::vector<double> val_jacob_i;
                val_jacob_i.insert(val_jacob_i.end(), dl_dyt0[i].begin(), dl_dyt0[i].end());
                val_jacob_i.insert(val_jacob_i.end(), dl_dtheta0[i].begin(), dl_dtheta0[i].end());
                val_jacob_i.insert(val_jacob_i.end(), dl_dt0[i]);
                val_jacob_i.insert(val_jacob_i.end(), dl_dt1[i]);
                yt1Var[i] = precomputed_gradients(yt1[i], params, val_jacob_i);
            }

            return yt1Var;
        }

    }
}

void numerical_jacobian (
        std::vector<double> y0,
        double t0, double t1,
        std::vector<double> theta0,
        const double& eps
) {
    std::vector<double> yt1 = stan::math::pend_forward(y0, theta0, t0, t1);
    std::vector<std::vector<double>> num_jacob(y0.size(), std::vector<double>(1 + 1 + theta0.size()));

    pend_sys::pend_dyn dyn;

    // t0
    double tl = t0 - eps/2;
    std::vector<std::vector<double>> yt1_tl = stan::math::integrate_ode_rk45(dyn, y0, tl, std::vector<double>({t1}), theta0, std::vector<double>(), std::vector<int>());

    double tr = t0 + eps/2;
    std::vector<std::vector<double>> yt1_tr = stan::math::integrate_ode_rk45(dyn, y0, tr, std::vector<double>({t1}), theta0, std::vector<double>(), std::vector<int>());

    for (int i=0; i<y0.size(); i++) {
        num_jacob[i][0] = (yt1_tr[0][i] - yt1_tl[0][i])/eps;
    }

    // t1
    double t1l = t1 - eps/2;
    std::vector<std::vector<double>> yt1_t1l = stan::math::integrate_ode_rk45(dyn, y0, t0, std::vector<double>({t1l}), theta0, std::vector<double>(), std::vector<int>());

    double t1r = t1 + eps/2;
    std::vector<std::vector<double>> yt1_t1r = stan::math::integrate_ode_rk45(dyn, y0, t0, std::vector<double>({t1r}), theta0, std::vector<double>(), std::vector<int>());

    for (int i=0; i<y0.size(); i++) {
        num_jacob[i][1] = (yt1_t1r[0][i] - yt1_t1l[0][i])/eps;
    }

    // theta
    std::vector<double> theta0Copy = theta0;
    for (int j=0; j<theta0.size(); j++) {
        double thetajl = theta0[j] - eps/2;
        theta0Copy[j] = thetajl;
        std::vector<std::vector<double>> yt1_thetajl = stan::math::integrate_ode_rk45(dyn, y0, t0, std::vector<double>({t1}), theta0Copy, std::vector<double>(), std::vector<int>());
        theta0Copy[j] = theta0[j];

        double thetajr = theta0[j] + eps/2;
        theta0Copy[j] = thetajr;
        std::vector<std::vector<double>> yt1_thetajr = stan::math::integrate_ode_rk45(dyn, y0, t0, std::vector<double>({t1}), theta0Copy, std::vector<double>(), std::vector<int>());
        theta0Copy[j] = theta0[j];

        for (int i=0; i<y0.size(); i++) {
            num_jacob[i][j+2] = (yt1_thetajr[0][i] - yt1_thetajl[0][i])/eps;
        }
    }

    for (int i=0; i<y0.size(); i++) {
        for (int j=0; j<theta0.size()+2; j++) {
            std::cout << num_jacob[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void numerical_jacobian_exact (
        std::vector<double> y0,
        double t0, double t1,
        std::vector<double> theta0,
        const double& eps
) {
    double root_g_l = stan::math::sqrt(theta0[0]/theta0[1]);
    double temp = (stan::math::pi()/8)*(root_g_l)*stan::math::sin(root_g_l*t1)*t1;
    std::cout << "dy1[0]/dg = " << -1*temp/theta0[0] << std::endl;
    std::cout << "dy1[0]/dl = " << temp/theta0[1] << std::endl;
}

int main() {
    std::vector<stan::math::var> y0 = {stan::math::pi()/4, 0};
    std::vector<stan::math::var> theta0 = {9.8, 1};
    stan::math::var t0 = 0;
    stan::math::var t1 = 0.3;
    std::vector<stan::math::var> y1 = stan::math::pend_forward(y0, theta0, t0, t1);

    std::cout << "Computed Value of y(t1): ";
    for (int i=0; i<y1.size(); i++) {
        std::cout << y1[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "\t Analytical Gradient Transpose" << std::endl;
//    std::cout << "\tGradient with respect to t0" << std::endl;
    for (int i=0; i<y1.size(); i++) {
        y1[i].grad();
        std::cout << t0.adj() << " " << t1.adj() << " ";
        for (int j=0; j<theta0.size(); j++) {
            std::cout << theta0[j].adj() << " ";
        }
        std::cout << std::endl;
        stan::math::set_zero_all_adjoints();
    }
    std::cout << std::endl << std::endl;

    std::cout << "\t Numerical Gradient Transpose" << std::endl;
    numerical_jacobian(value_of(y0), t0.val(), t1.val(), value_of(theta0), 0.001);
    std::cout << std::endl;

    std::cout << "\t Numerical Gradient Exact" << std::endl;
    numerical_jacobian_exact(value_of(y0), t0.val(), t1.val(), value_of(theta0), 0.000001);
    std::cout << std::endl;

    return 0;
}

