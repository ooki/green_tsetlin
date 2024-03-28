
#include <inttypes.h>
#include <string.h>

/* 

def get_test_rules():
    return [[0], [0,1], [2,3]]


def get_test_weights():
    return [[-1, 2], [-3, 4], [7, -8]]


*/


#define NUM_CLASSES 2
#define DEFAULT_OUTPUT 0

int predict_tm(uint8_t* x)
{    
    static int16_t votes[NUM_CLASSES] = {0};
    memset(votes, sizeof(votes), 0);

    if(x[0] > 0) {
        votes[0] += -1; votes[1] += 2;
    }

    if(x[0] > 0 && x[1] > 0) {
        votes[0] += -3; votes[1] += 4;
    }

    if(x[0] == 0 && x[1] == 0) {
        votes[0] += 7; votes[1] += -8;
    }

    int output_class = DEFAULT_OUTPUT;
    int16_t most_votes = votes[DEFAULT_OUTPUT];
    for(int i = 0; i < NUM_CLASSES; ++i)
    {
        if(votes[i] > most_votes)
        {
            most_votes = votes[i];
            output_class = i;
        }
    }
    return output_class;
}


