#include <stan/math.hpp>
#include <vector>
#include <iostream>
using namespace stan::math;

namespace bar {
    using namespace Eigen;
    VectorXd f(VectorXd x) {
       return x.cwiseProduct(x);
    }

    MatrixXd f_jacobian(VectorXd x) {
       MatrixXd df_dx(x.size(), x.size());
       for (int i=0; i<x.size(); i++) {
           for (int j=0; j<x.size(); j++) {
               df_dx(i, j) = 0;
               if (i == j) df_dx(i, j) = 2*x(i);
           }
       }
       return df_dx;
    }
}

namespace stan {
    namespace math {
        Eigen::VectorXd f(const Eigen::VectorXd& x) {
            return bar::f(x);
        }

        Eigen::Matrix<var, -1, 1> f_jacobian(const Eigen::Matrix<var, -1, 1> xEigenVar) {
            // operands
            Eigen::VectorXd x = value_of(xEigenVar);
            std::vector<var> xStdVar(xEigenVar.size());
            for (int i=0; i<xEigenVar.size(); i++) {
                xStdVar[i] = xEigenVar(i);
            }

            // value
            Eigen::VectorXd fEvalxEigen = bar::f(x);

            // gradients
            Eigen::MatrixXd df_dxEigen = bar::f_jacobian(x);
            std::vector<std::vector<double>> df_dxStd (df_dxEigen.rows(), std::vector<double>(df_dxEigen.cols()));
            for (int i=0; i<df_dxEigen.rows(); i++) {
                for (int j=0; j<df_dxEigen.cols(); j++) {
                    df_dxStd[i][j] = df_dxEigen(i, j);
                }
            }
            Eigen::Matrix<var, -1, 1> df_dxEigenVar(df_dxEigen.rows());
            for (int i=0; i<df_dxEigen.rows(); i++) {
                df_dxEigenVar(i) = precomputed_gradients(fEvalxEigen(i), xStdVar, df_dxStd[i]);
            }
            return df_dxEigenVar;
        }
    }
}

int main() {
    using namespace Eigen;

    VectorXd x(4);
    x << 1,2,3,4;
    std::cout << "x: " << std::endl << x << std::endl;

    VectorXd fEvalx = bar::f(x);
    std::cout << "f(x)" << std::endl << fEvalx << std::endl;

    MatrixXd df_dx = bar::f_jacobian(x);
    std::cout << "Direct df(x)/dx = " << std::endl << df_dx << std::endl;


    std::cout << "Using stan math" << std::endl;

    Matrix<var, -1, 1> xEigenVar(x.size());
    std::vector<var> xStdVar(x.size());
    for (int i=0; i<x.size(); i++) {
       xEigenVar(i) = stan::math::var(x(i));
       xStdVar[i] = xEigenVar(i);
    }
    Eigen::Matrix<var, -1, 1> df_dxEigenVar = stan::math::f_jacobian(xEigenVar);
    std::cout << "f(x) = " << std::endl << df_dxEigenVar << std::endl;

    std::cout << "df(x)/dx = " << std::endl;
    for (int i=0; i<df_dxEigenVar.size(); i++) {
        df_dxEigenVar(i).grad();

        for (int j=0; j<xStdVar.size(); j++)
            std::cout << xStdVar[j].adj() << ", ";
        std::cout << std::endl;
        stan::math::set_zero_all_adjoints();
    }
    return 0;
}
