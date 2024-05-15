

#include <iostream>
#include <immintrin.h> // intrics
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <bitset>



void encode_decode_and_reverse_order()
{
    const int n_ta = 8;
    int8_t in_data[n_ta] = {-3, -2, -1, 0, 1, 2, 3, 4};
    int8_t data[n_ta] = {0};
    int8_t out_data[n_ta] = {0};

    for (int i = 0; i < n_ta; ++i)
        std::cout << (int)in_data[i] << " ";
    std::cout << std::endl;

    const int n_bits = 8;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            data[j] |= ((in_data[i] >> j) & 1) << (7 - i);
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            out_data[7 - j] |= ((data[i] >> j) & 1) << i;
        }
    }

    for (size_t i = 0; i < n_bits; ++i) {
       std::cout << "in_data[" << i << "]: " << std::bitset<8>(in_data[i]) << std::endl;
    }

    for (size_t i = 0; i < n_bits; ++i) {
        std::cout << "data[" << i << "]: " << std::bitset<8>(data[i]) << std::endl;
    }

    for (size_t i = 0; i < n_bits; ++i) {
        std::cout << "data_out[" << i << "]: " << std::bitset<8>(out_data[i]) << std::endl;
    }
}


void inc(uint16_t* base, uint16_t add)
{
    std::cout << "add:" << std::bitset<16>(add) << std::endl;
    uint16_t carry = add;
    for(int i = 0; i < 8; i++)
    {        
        uint16_t carry_next = base[i] & carry;
        base[i] = (base[i] ^ carry);
        carry = carry_next;
    }

    if(carry > 0)
    {
        for(int i = 0; i < 8; i++)
            base[i] |= carry;
    }
}

int main() {


    const int n_ta = 16;
    const int n_bits = 8;

    // uint8_t in_data[n_ta] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, (uint8_t)-1};
    uint8_t in_data[n_ta] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 127, 128, (uint8_t)-1};
    uint16_t data[n_bits] = {0};
    uint8_t data_out[n_ta] = {0};

    for (int i = 0; i < n_ta; ++i)
        std::cout << (uint32_t)in_data[i] << " ";
    std::cout << std::endl;
    


    for(int ta_i = 0; ta_i < n_ta; ta_i++)
    {
        for(int bit_j = 0; bit_j < n_bits; bit_j++)
        {
            uint16_t bit_ij = (in_data[ta_i] >> bit_j) & 1;
            data[bit_j] |= bit_ij << ta_i;

            // if(ta_i == 0)
            // {
            //     std::cout << "i:" << ta_i << " j:" << bit_j << " bit_ji = " << bit_ij << std::endl;
            //     std::cout << "or mask: " << std::bitset<16>(data[bit_j]) << std::endl;
            // }
        }
    }


    for (size_t i = 0; i < n_bits; ++i) {
        std::cout << "POST data[" << i << "]: " << std::bitset<16>(data[i]) << std::endl;
    }

    for(int i = 0; i < 1000; i++)
        inc(data, (uint16_t)-1);

    for (size_t i = 0; i < n_bits; ++i) {
        std::cout << "POST data[" << i << "]: " << std::bitset<16>(data[i]) << std::endl;
    }

    for(int bit_i = 0; bit_i < n_bits; bit_i++)
    {
        for(int ta_j = 0; ta_j < n_ta; ta_j++)
        {
            uint16_t bit_ij = (data[bit_i] >> ta_j) & 1;
            data_out[ta_j] |= bit_ij << bit_i;
        }
    }

    


    



    // for (size_t i = 0; i < n_ta; ++i) {
    //    std::cout << "in_data[" << i << "]: " << std::bitset<8>(in_data[i]) << std::endl;
    // }

    for (int i = 0; i < n_ta; ++i)
        std::cout << (uint32_t)data_out[i] << " ";

    std::cout << std::endl;

    return 0;
}








