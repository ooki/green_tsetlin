

#include <iostream>
#include <immintrin.h> // intrics
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <bitset>


void inc(uint32_t* base, uint32_t add)
{
    // std::cout << "add:" << std::bitset<16>(add) << std::endl;
    uint32_t carry = add;
    for(int i = 0; i < 8; i++)
    {        
        uint32_t carry_next = base[i] & carry;
        base[i] = (base[i] ^ carry);
        carry = carry_next;
    }

    if(carry > 0)
    {
        for(int i = 0; i < 8; i++)
            base[i] |= carry;
    }
}


void  dec(uint32_t* base, uint32_t dec)
{
    uint32_t carry = dec;
    for(int i = 0; i < 8; i++)
    {        
        uint32_t carry_next = (~base[i]) & carry;
        base[i] = (base[i] ^ carry);
        carry = carry_next;
    }

    if(carry > 0)
    {
        for(int i = 0; i < 8; i++)
            base[i] &= ~carry;
    }
}


int main() {


    const int n_ta = 32;
    const int n_bits = 8;
    const int mem_size = n_ta;
    const int vector_size = 4;

    
    uint8_t in_data[n_ta] = {0};
    uint8_t data[mem_size] = {0};
    uint8_t data_out[n_ta] = {0};


    for (int i = 0; i < n_ta; ++i)
        in_data[i] = i % 256;

    in_data[0] = 0xFF;
    in_data[1] = 15;

    for (int i = 0; i < n_ta; ++i)
        std::cout << (uint32_t)in_data[i] << " ";
    std::cout << std::endl;
    

    for(int ta_i = 0; ta_i < n_ta; ta_i++)
    {
        for(int bit_j = 0; bit_j < n_bits; bit_j++)
        {            
            uint8_t bit_ij = (in_data[ta_i] >> bit_j) & 1;    
            const int out_block_i = ta_i / 8;
            const int out_bit_i = ta_i % 8;
            const int out_location = (bit_j * vector_size) + out_block_i;
            data[out_location] |= bit_ij << out_bit_i;
        }
    }

    for(int i = 0; i < 1000;i++)
        dec((uint32_t*)data, (uint32_t)-1);

    for(int i = 0; i < 1000;i++)
        inc((uint32_t*)data, (uint32_t)-1);


    for(int bit_i = 0; bit_i < n_bits; bit_i++)
    {
        for(int ta_j = 0; ta_j < n_ta; ta_j++)
        {
            const int block_k = ta_j / 8;
            const int block_bit_k = ta_j % 8;
            
            int in_location = (bit_i*vector_size) + block_k;
            uint8_t bit_ij = (data[in_location] >> block_bit_k) & 1;
            data_out[ta_j] |= bit_ij << bit_i;
        }
    }

    for (size_t i = 0; i < n_bits; ++i) {
        for(int j = 0; j < 4; ++j)
        {
            std::cout << std::bitset<8>(data[(i*4)+j]) << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < n_ta; ++i)
        std::cout << (uint32_t)data_out[i] << " ";

    std::cout << std::endl;

    return 0;
}








