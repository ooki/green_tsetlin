#ifndef _GT_COMMON_HPP_
#define _GT_COMMON_HPP_

#include <stdlib.h>
#include <stdint.h>
#include <string>

namespace green_tsetlin
{
    typedef int16_t  WeightInt;
    typedef uint8_t  ClauseOutputUint;


    void* safe_aligned_alloc(int align_to, int mem)
    {
        int new_mem = mem;
        int r = mem % align_to;

        if(mem < align_to)
            new_mem = align_to;
            
        else if(r > 0)
            new_mem += (align_to - r);

        #ifdef _MSC_VER
            void* p = _aligned_malloc(new_mem, align_to);
        #else 
            void* p = std::aligned_alloc(align_to, new_mem);
        #endif 
        if(p == nullptr)
        {
            std::string msg = "Failed to allocate mem: " + std::to_string(new_mem) + " aligned to: " + std::to_string(align_to) + " req:" + std::to_string(mem);
            throw std::runtime_error(msg);
        }
        return p;
    }

    void safe_aligned_free(void* p)
    {
        #ifdef _MSC_VER
            _aligned_free(p);
        #else
            std::free(p);
        #endif
    }
    
}; // namespace green_tsetlin


#endif // #ifndef _GT_COMMON_HPP_
