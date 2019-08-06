#include<iostream>
#include<vector>
using namespace std;

class temp{
public:
  temp() {
    a = 0;
  }
  void inc() {
    a++;
  }
  static int a;
};

int temp::a = 1;

int main() {
    temp t1;
    temp t2;
    t1.inc();
    t2.inc();
    cout << t1.a << endl;
    cout << t2.a << endl;
    return 0;
}