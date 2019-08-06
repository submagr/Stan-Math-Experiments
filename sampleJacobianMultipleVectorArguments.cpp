#include <stan/math.hpp>
#include <vector>
#include <numeric>
#include <iostream>

namespace bar {
    using namespace std;

    template<typename T, typename T1>
    vector<T> f(
            const vector<T> &x,
            const vector<T> &y,
            const vector<T> &z,
            vector<T1> &theta
    ) {
        T1 mu = accumulate(theta.begin(), theta.end(), 0)/theta.size();
        vector<T> ret(x.size());
        for(int i=0; i<ret.size(); i++) {
            ret[i] = mu*(x[i] + 2*y[i] + 3*z[i]);
        }
        return ret;
    }

    template<typename T, typename T1>
    vector<vector<T>> f_jacobian(
            const vector<T> &x,
            const vector<T> &y,
            const vector<T> &z,
            vector<T1> &theta
    ) {
        T1 mu = accumulate(theta.begin(), theta.end(), 0)/theta.size();
        vector<vector<T>> ret(x.size(), vector<T>(x.size()*3, 0));
        for (int i=0; i<ret.size(); i++) {
            for (int k=0; k<3; k++) {
                for (int j=k*x.size(); j<(k+1)*x.size(); j++) {
                    if (i == j-k*x.size()) {
                        ret[i][j] = (k+1)*mu;
                    }
                }
            }
        }
        return ret;
    }
}

namespace stan{
    namespace math {
        std::vector<var> forward(
                const std::vector<var>& x,
                const std::vector<var>& y,
                const std::vector<var>& z,
                std::vector<double> theta
        ) {
            return bar::f(x,y,z,theta);
        }

        std::vector<var> backward(
                const std::vector<var>& x,
                const std::vector<var>& y,
                const std::vector<var>& z,
                std::vector<double> theta
        ) {
            std::cout << "Backward called " << std::endl;
            // operands
            std::vector<var> params;
            params.insert(params.end(), x.begin(), x.end());
            params.insert(params.end(), y.begin(), y.end());
            params.insert(params.end(), z.begin(), z.end());

            // value
            std::vector<double> val = value_of(bar::f(x,y,z,theta));

            // gradients
            std::vector<std::vector<var>> val_jacobian_var = bar::f_jacobian(x,y,z,theta);
            std::vector<std::vector<double>> val_jacobian(val_jacobian_var.size());
            for (int i=0; i<val_jacobian_var.size(); i++){
                val_jacobian[i] = value_of(val_jacobian_var[i]);
            }
            std::vector<var> df_dparmas(val.size());
            for (int i=0; i<df_dparmas.size(); i++) {
                df_dparmas[i] = precomputed_gradients(val[i], params, val_jacobian[i]);
            }

            return df_dparmas;
        }
    }
}
int main() {
    using namespace std;
    cout << "\tmanual jacobian: " << endl;
    vector<double> x = {1,2,3,4};
    vector<double> y = {11,12,13,14};
    vector<double> z = {21,22,23,24};
    vector<double> theta = {2,2};

    vector<double> f = bar::f(x,y,z,theta);
    for (auto fi: f)
        cout << fi << " ";
    cout << endl;

    vector<vector<double>> f_jacobian = bar::f_jacobian(x,y,z,theta);
    for (auto f_jacobi: f_jacobian) {
        for (auto f_jacobij: f_jacobi) {
            cout << f_jacobij << ", ";
        }
        cout << endl;
    }


    cout << endl << "\tusing stan math" << endl;

    std::vector<stan::math::var> xVar(x.begin(), x.end());
    std::vector<stan::math::var> yVar(y.begin(), y.end());
    std::vector<stan::math::var> zVar(z.begin(), z.end());

    vector<stan::math::var> val = stan::math::forward(xVar,yVar,zVar,theta);
    for (auto vali: val)
        cout << vali << " ";
    cout << endl;

//    std::vector<stan::math::var> val_jacobian = backward(xVar, yVar, zVar, theta);
    for (int i=0; i<val.size(); i++) {
        val[i].grad();
        for (int k=0; k<3; k++) {
            for (int j=0; j<x.size(); j++) {
                if (k==0) {
                    cout << xVar[j].adj() << ", ";
                } else if (k==1) {
                    cout << yVar[j].adj() << ", ";
                } else {
                    cout << zVar[j].adj() << ", ";
                }
            }
        }
        stan::math::set_zero_all_adjoints();
        cout << endl;
    }



    cout << "\tgradient using auto-diff functionality" << endl;
    stan::math::var L = 0;
    for (int i=0; i<val.size(); i++){
        L += val[i]*val[i];
    }
    L.grad();
    for (int k=0; k<3; k++) {
        for (int j=0; j<xVar.size(); j++) {
            if (k==0) {
                cout << xVar[j].adj() << ", ";
            } else if (k==1) {
                cout << yVar[j].adj() << ", ";
            } else {
                cout << zVar[j].adj() << ", ";
            }
        }
        cout << endl;
    }
    // Tears rolling down the eyes
    return 0;
}
