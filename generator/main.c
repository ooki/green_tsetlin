

//#include "tm.h"
#include "out.h"
#include <stdio.h>




int main() {
    printf("Hello world\n");

    uint8_t x[] = {1, 1};

    int y_hat = predict_tm(x);
    printf("prediction: %d\n", y_hat);

    return 0;
};

