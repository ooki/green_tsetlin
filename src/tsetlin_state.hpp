#ifndef _TSETLIN_STATE_HPP_
#define _TSETLIN_STATE_HPP_

#include <random>
#include <vector>

#include <gt_common.hpp>

namespace green_tsetin
{
    class TsetlinState
    {
        public:
            double s = -42.0;
            int num_clauses = 0;
            int num_classes = 0;
            int num_literals = 0;
            int num_literals_mem = 0; 

            std::default_random_engine rng;
    };
}; // namespace green_tsetin






#endif // #ifndef _TSETLIN_STATE_HPP_