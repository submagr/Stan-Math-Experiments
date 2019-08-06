#include <stan/math.hpp>
#include <vector>
#include <iostream>
using namespace stan::math;

namespace bar {
    template<typename T>
    Eigen::Matrix<T, -1, 1> foo(const Eigen::Matrix<T, -1, 1>& x){
        return x;
    }

    template<typename T>
    Eigen::Matrix<T, -1, -1> foo_jacobian(const Eigen::Matrix<T, -1, 1>& x) {
        Eigen::Matrix<T, -1, -1> J(x.size(), x.size());
        for (int i=0; i<x.size(); i++) {
            for (int j=0; j<x.size(); j++){
                if (i==j)
                   J(i,j)  = 1;
                else
                    J(i, j) = 0;
            }
        }
        return J;
    }
}

namespace stan {
    namespace math {
        template<typename T>
        Eigen::Matrix<T, -1, 1> foo(const Eigen::Matrix<T, -1, 1> &x) {
            return bar::foo(x);
        }

        Eigen::Matrix<var, -1, -1> foo(
                const Eigen::Matrix<var, -1, 1> &x,
                std::ostream *pstream__
        ) {
            // extract parameter values from x
            const var *x_data = x.data();
            int x_size = x.size();
            std::vector<var> x_std_var(x_data, x_data + x_size);
            std::cout << "x_std_var:" << std::endl;
            for (auto xi: x_std_var)
                std::cout << xi << " ";
            std::cout << std::endl;

            Eigen::Matrix<var, -1, 1> f_val = bar::foo(x); // evaluate f
            Eigen::Matrix<var, -1, -1> f_val_jacobian_ = bar::foo_jacobian(x);

            // copy f_val_jacobian matrix into a vector of vector.
            std::vector<std::vector<double>> f_val_jacobian(
                f_val_jacobian_.rows(),
                std::vector<double>(f_val_jacobian_.cols())
            );
            for (int i = 0; i < f_val_jacobian_.rows(); i++) {
                for (int j=0; j<f_val_jacobian_.cols(); j++) {
                   f_val_jacobian[i][j]  = f_val_jacobian_(i, j);
                }
            }

            std::cout << "f_val_jacobian: " << std::endl;
            for (auto r: f_val_jacobian) {
                for (auto c: r)
                    std::cout << c << " ";
                std::cout<< std::endl;
            }
            // f_val_jacobian[i][j] is the partial derivative of f_i w.r.t. parameter j

            int f_size = f_val_jacobian.size();
            Eigen::Matrix<var, -1, -1> f_var_jacobian(f_val_jacobian.rows(), f_val_jacobian.cols());
            for (int i = 0; i < f_val_jacobian.size(); i++) {
                for (int j=0; j<f_val_jacobian[0].size(); j++) {
                    f_var_jacobian(i, j) = precomputed_gradients(
                        f_val(i),
                        {x_std_var[j].vi_},
                        {f_val_jacobian[i][j]}
                    );
                }
            }
            return f_var_jacobian;
        }
    }
}

int main() {
    Eigen::Matrix<var, -1, 1> x(4);
    x << 1,2,3,4;
    Eigen::Matrix<var, -1, 1> z = foo(x);
    for (int i=0; i<z.size(); i++) {
        z[i].grad();
        std::cout << x[i].adj() << std::endl;
    }
//    Eigen::Matrix<var, -1, -1> zP = bar::foo_jacobian(z);
    return 0;
}

