#include<Eigen/Core>
#include<iostream>
#include<vector>
#include<stan/math.hpp>
using namespace std;

int main() {
    // we have a vector. Extract a vector, matrix and a vector out of it
    // Eigen::Matrix<double, 22, 1> flatMat;
    // flatMat << 
    //     1, 2, 3, 
    //     4, 5, 6, 7,
    //     8, 9, 1, 2,
    //     3, 4, 5, 6,
    //     7, 8, 9, 1,
    //     2, 3, 4
    // ;
    // Eigen::Matrix<double, 3, 1> v1;
    // v1 = Eigen::Map<Eigen::Matrix<double, 3, 1>>(flatMat.data());
    // auto m1(Eigen::Map<Eigen::Matrix<double, 4, 4>>(flatMat.data() + 3));
    // Eigen::Matrix<double, 3, 1> v2(Eigen::Map<Eigen::Matrix<double, 3, 1>>(flatMat.data()+19));
    // cout << v1 << endl;
    // cout << m1 << endl;
    // cout << v2 << endl;

    // Eigen::Matrix<double, 22, 1> flatMat2;
    // flatMat2.segment(0, 3) = v1;
    // flatMat2.segment(3, 16) = Eigen::Map<Eigen::Matrix<double, 16, 1>>(m1.data());
    // flatMat2.segment(19, 3) = v2;
    // cout << endl << flatMat2 << endl;

    // assign an std vector to eigen vector
    // std::vector<double> vStd(4, 5);
    // Eigen::VectorXd v3 = Eigen::Map<Eigen::VectorXd>(vStd.data(), 4);
    // cout << v3 << endl;

    // assign a std vector matrix to eigen matrix
    // std::vector<std::vector<double>> mStd(4, std::vector<double>(4, 0));
    // mStd[3][2] = -3;
    // Eigen::Matrix<double, 4, 4> m = Eigen::Map<Eigen::Matrix<double, 4, 4>>(mStd.data(), 16);
    // std::cout << m << std::endl;

    // Eigen::MatrixXd m;
    // std::cout << m.size() << std::endl;

    // auto m2 = (Eigen::Matrix<double, 10, 1>::Zero(10).array() + 1.).matrix().asDiagonal();
    // cout << m2 << endl;
    // cout << m2(3,4) << endl;

    stan::math::var a;
    cout << stan::math::value_of(a) << endl;
    return 0;
}

// Numerical:
//          0 -0.0353553
//          0   0.173241
// Analytical:
//          0 -0.0353553
//          0   0.173241
