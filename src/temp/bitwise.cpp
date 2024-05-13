

#include <iostream>
#include <immintrin.h> // intrics
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <bitset>

int main() {

    const int n_ta = 8;
    int8_t in_data[n_ta] = {-1, 0, -1, -1, -1, -1, -1, -1};
    int8_t data[n_ta] = {0};


    std::cout << "in_data: ";
    for (int i = 0; i < n_ta; ++i)
        std::cout << (int)in_data[i] << " ";
    std::cout << std::endl;

    const int n_bits = 8;
    for (int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < n_bits; ++j)
        {
            int8_t bit = (in_data[i] >> j) & 1;
            data[i] = (data[j] << 1) | bit;            
        }                
    }

    for (size_t i = 0; i < n_bits; ++i) {
        std::cout << "in_data[" << i << "]: " << std::bitset<8>(in_data[i]) << std::endl;
    }

    for (size_t i = 0; i < n_bits; ++i) {
        std::cout << "data[" << i << "]: " << std::bitset<8>(data[i]) << std::endl;
    }
    
    // output is wrong.


    return 0;
}






