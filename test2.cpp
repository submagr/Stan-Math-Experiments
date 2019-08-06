#include <stan/math.hpp>
#include <vector>
#include <iostream>
using namespace stan::math;

namespace bar {
//    Example 2:
//    double foo(double x) {
//       return x;
//    }
//
//    double d_foo(double x) {
//        return 100;
//    }

// Example 3
//    double foo(double x, double y) {
//       return x*(std::pow(y,3));
//    }
//    double dfoo_dx(double x, double y) {
//       return std::pow(y,3);
//    };
//    double dfoo_dy(double x, double y) {
//       return x*3*std::pow(y,2);
//    };

// Example 4
//    double foo(const vector<double>& x) {
//       return accumulate(x.begin(), x.end(), 0);
//    }
//
//    vector<double> grad_foo(const vector<double>& x) {
//        return vector<double>(x.size(), 1);
//    }

// Example 5
    Eigen::VectorXd foo(const Eigen::VectorXd& x){
        return x;
    }

//    Eigen::MatrixXd foo_jacobian(const Eigen::VectorXd& x) {
//        std::cout << x.size() << std::endl;
//        Eigen::MatrixXd J;// (/*x.size(), x.size()*/);
//        std::cout << J.rows() << " " << J.cols() << " " << J.size() << std::endl;
//        for (int i=0; i<x.size(); i++) {
//            for (int j=0; j<x.size(); j++){
//                if (i==j)
//                   J(i,j)  = 1;
//                else
//                    J(i, j) = 0;
//            }
//        }
//        return J;
//    }
}

/*namespace stan {
    namespace math {
//      Function using standard operations. Don't need to define gradients
//        template <typename T>
//        std::vector<T> z(const std::vector<T>& x) {
//            T mean = stan::math::mean(x);
//            T sd = stan::math::sd(x);
//            std::vector<T> result(x.size());
//            for (size_t i = 0; i < x.size(); ++i) {
//                result[i] = (x[i] - mean)/ sd;
//            }
//            return result;
//        }

//      Example 2: Single argument functions
//        double foo(double x) {
//            return bar::foo(x);
//        }
//
//        stan::math::var foo(const stan::math::var& x) {
//            double a = x.val();
//            double fa = bar::foo(a);
//            double dfa_da = bar::d_foo(a);
//            return stan::math::precomputed_gradients(fa, {x.vi_}, {dfa_da});
//        }

// Example 3
//        double foo(double x, double y) {
//            return bar::foo(x, y);
//        }
//
//        var foo(const var& x, double y) {
//            double a = x.val();
//            double fay = bar::foo(a, y);
//            double dfay_da = bar::dfoo_dx(a, y);
//            return precomputed_gradients(fay, {x.vi_}, {dfay_da});
//        }
//
//        var foo(double x, const var& y) {
//            double b = y.val();
//            double fxb = bar::foo(x, b);
//            double dfxb_db = bar::dfoo_dy(x, b);
//            return precomputed_gradients(fxb, {y.vi_}, {dfxb_db});
//        }
//
//        var foo(const var& x, const var& y) {
//            double a = x.val();
//            double b = y.val();
//            double fab = bar::foo(a, b);
//            double dfab_da = bar::dfoo_dx(a, b);
//            double dfab_db = bar::dfoo_dy(a, b);
//            return precomputed_gradients(fab, {x.vi_, y.vi_}, {dfab_da, dfab_db});
//        }

// Example 4
//        vector<double> foo(const vector<double>& x) {
//            return bar::foo(x);
//        }
//
//        vector<var> foo(const vector<var>& x) {
//            vector<double> a = value_of(a);
//            double fa = bar::foo(a);
//            vector<double> grad_fa = bar::grad_foo(a);
//            return precomputed_gradients(fa, x, grad_fa);
//        }
//    }

// Example 5
        Eigen::VectorXd foo(const Eigen::VectorXd &x) {
            return bar::foo(x);
        }

        Eigen::Matrix<var, -1, 1> foo(
                const Eigen::Matrix<var, -1, 1> &x,
                std::ostream *pstream__
        ) {
            // extract parameter values from x
            const var *x_data = x.data();
            int x_size = x.size();
            std::vector <var> x_std_var(x_data, x_data + x_size);

            Eigen::VectorXd a = value_of(x);
            Eigen::VectorXd f_val = bar::foo(a); // evaluate f

            Eigen::MatrixXd f_val_jacobian_ = bar::foo_jacobian(a);
            std::vector <std::vector<double>> f_val_jacobian(f_val_jacobian_.rows());
            for (int i = 0; i < f_val_jacobian_.rows(); i++) {
                f_val_jacobian[i] = std::vector<double>(
                        f_val_jacobian_.row(i).data(),
                        f_val_jacobian_.row(i).data() + f_val_jacobian_.row(i).size()
                );
            }
            // f_val_jacobian[i][j] is the partial derivative of f_i w.r.t. parameter j

            int f_size = f_val_jacobian.size();
            Eigen::Matrix<var, -1, 1> f_var_jacobian(f_size);
            for (int i = 0; i < f_size; i++) {
                f_var_jacobian(i) = precomputed_gradients(f_val(i), x_std_var, f_val_jacobian[i]);
            }
            return f_var_jacobian;
        }
    }
}*/

int main() {
//  Example 1:
//    std::vector<stan::math::var> x = {1,2,3,4};
//    std::vector<stan::math::var> z = stan::math::z(x);
//
//    z[0].grad();
//    for (auto xi: x) {
//        std::cout << xi.adj() << std::endl;
//    }

//  Example 2: Single Argument Functions
//    stan::math::var x = 4;
//    stan::math::var z1 = pow(stan::math::foo(x), 3);
//
//    z1.grad();
//    std::cout << x.adj() << std::endl;


// Example 3: Function with multiple arguments
//    var x=3, y=4;
//    var z=pow(
//        3,
//        sin(foo(x,y))
//    );
//    z.grad();
//    std::cout << x.adj() << " " << y.adj() << std::endl;

// Example 4: Function which takes a vector, returns an scalar
//    std::vector<var>  x = {1,2,3,4};
//    var z = foo(x);
//    z.grad();
//    for (auto xi: x) {
//        std::cout << xi.adj() << std::endl;
//    }


// Example 5: In Vector, Out Vector
//    Eigen::Matrix<var, -1, 1> x;
    Eigen::Matrix<double, -1, 1> x(4);
    x << 1,2,3,4;
//    Eigen::Matrix<var, Eigen::Dynamic, 1> z = stan::math::foo(x);
    Eigen::Matrix<double, -1, 1> z = bar::foo(x);
//    Eigen::Matrix<double, -1, -1>  zP = bar::foo_jacobian(z);
    /*for (int i=0; i<x.size(); i++) {
        z[i].grad();
        std::cout << x[i].adj() << std::endl;
//        Eigen::VectorXd xi_adj = x[i].adj();
//        for (int i=0; i<xi_adj.rows(); i++) {
//            std::cout << xi_adj(i, 0) << " ";
//        }
//        std::cout << std::endl;
    }*/
    return 0;
}
