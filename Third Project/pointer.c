#include <stdio.h>

void test(int *);

int main(int argc, char * argv []){
  int blah [5];
  test(blah);
  printf("%d\n", blah[4]);
}

void test(int * blah){
  int a;
  int yo [5] = {1,2,3,4,5};
  for (a=0; a<5; a++){
    blah[a] = yo[a];
  }
  // blah = yo;
}
