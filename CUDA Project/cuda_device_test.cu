#include <iostream>

using namespace std;

__device__ bool flag = true;
__device__ int d_num = 2;

__global__ void test();

int main(){
  test<<<2,1>>>();
  int num;
  cudaMemcpyFromSymbol(&num, d_num, sizeof(num), 0, cudaMemcpyDeviceToHost);
  cout << num << endl;
  return 0;
}

__global__ void test(){
  // if (blockIdx.x == 0){
  //   // wait for block 1
  //   while (flag){}
  //   d_num *= 2;
  // } else if (blockIdx.x == 1){
  //   d_num += 1;
  //   flag = false;
  // }
}
