#include <inttypes.h>
#include <string.h>

#define NUM_CLASSES 10
#define DEFAULT_OUTPUT 0
int predict_tm(uint8_t* x)
{
    static int16_t votes[NUM_CLASSES] = {0};
    memset(votes, sizeof(votes), 0);


if(x[859] == 0 && x[1046] == 0 && x[1076] == 0 && x[1098] == 0 && x[1127] == 0 && x[1155] == 0){
votes[0] += -186; votes[1] += 354; votes[2] += 126; votes[3] += 132; votes[4] += -74; votes[5] += -134; votes[6] += -102; votes[7] += -52; votes[8] += -115; votes[9] += -175;
}
if(x[154] > 0 && x[918] == 0 && x[1095] == 0 && x[1230] == 0 && x[1314] == 0 && x[1519] == 0){
votes[0] += 106; votes[1] += -52; votes[2] += 151; votes[3] += 97; votes[4] += -202; votes[5] += -52; votes[6] += 31; votes[7] += -185; votes[8] += 72; votes[9] += -122;
}
int output_class = DEFAULT_OUTPUT;
int16_t most_votes = votes[DEFAULT_OUTPUT];
for(int i = 1; i < NUM_CLASSES; ++i)
{
    if(votes[i] > most_votes)
    {
        most_votes = votes[i];
        output_class = i;
    }
}
    return output_class;
}
