#include <inttypes.h>
#include <string.h>

#define NUM_CLASSES 10
#define DEFAULT_OUTPUT 0
int predict_tm(uint8_t* x)
{
    static int16_t votes[NUM_CLASSES] = {0};
    memset(votes, sizeof(votes), 0);


if(x[377] > 0 && x[884] == 0 && x[920] == 0 && x[1239] == 0 && x[1547] == 0){
votes[0] += -360; votes[1] += 76; votes[2] += -307; votes[3] += 278; votes[4] += -221; votes[5] += 116; votes[6] += -198; votes[7] += -329; votes[8] += 212; votes[9] += -65;
}
if(x[932] == 0 && x[1135] == 0 && x[1162] == 0 && x[1190] == 0 && x[1245] == 0 && x[1272] == 0){
votes[0] += 484; votes[1] += -456; votes[2] += -185; votes[3] += -165; votes[4] += -316; votes[5] += -105; votes[6] += -132; votes[7] += 357; votes[8] += -288; votes[9] += -306;
}
if(x[437] > 0 && x[936] == 0 && x[1325] == 0 && x[1354] == 0 && x[1377] == 0 && x[1407] == 0){
votes[0] += -324; votes[1] += -157; votes[2] += -298; votes[3] += -166; votes[4] += 180; votes[5] += -109; votes[6] += -275; votes[7] += 122; votes[8] += -178; votes[9] += 274;
}
if(x[996] == 0 && x[1023] == 0 && x[1079] == 0 && x[1134] == 0 && x[1161] == 0 && x[1500] == 0){
votes[0] += -125; votes[1] += -290; votes[2] += 95; votes[3] += -219; votes[4] += 206; votes[5] += -135; votes[6] += 182; votes[7] += -156; votes[8] += -132; votes[9] += -178;
}
if(x[987] == 0 && x[1109] == 0 && x[1138] == 0 && x[1193] == 0 && x[1249] == 0 && x[1525] == 0){
votes[0] += 244; votes[1] += 428; votes[2] += -222; votes[3] += -397; votes[4] += -244; votes[5] += -2; votes[6] += -209; votes[7] += -292; votes[8] += -203; votes[9] += -254;
}
if(x[458] > 0 && x[805] == 0 && x[1004] == 0 && x[1437] == 0 && x[1441] == 0){
votes[0] += -212; votes[1] += -281; votes[2] += 122; votes[3] += -288; votes[4] += 61; votes[5] += -217; votes[6] += 442; votes[7] += -382; votes[8] += -183; votes[9] += -18;
}
if(x[409] > 0 && x[911] == 0 && x[936] == 0 && x[1326] == 0 && x[1353] == 0 && x[1407] == 0){
votes[0] += -290; votes[1] += -193; votes[2] += -321; votes[3] += -179; votes[4] += 196; votes[5] += -139; votes[6] += -273; votes[7] += 145; votes[8] += -171; votes[9] += 240;
}
if(x[1104] == 0 && x[1130] == 0 && x[1131] == 0 && x[1153] == 0 && x[1156] == 0){
votes[0] += -313; votes[1] += 231; votes[2] += 403; votes[3] += 19; votes[4] += -319; votes[5] += -339; votes[6] += -296; votes[7] += 144; votes[8] += -257; votes[9] += -321;
}
if(x[1246] == 0 && x[1248] == 0 && x[1302] == 0 && x[1447] == 0 && x[1449] == 0){
votes[0] += 167; votes[1] += -136; votes[2] += -206; votes[3] += 140; votes[4] += -224; votes[5] += 54; votes[6] += 40; votes[7] += -319; votes[8] += -90; votes[9] += -249;
}
if(x[154] > 0 && x[1047] == 0 && x[1099] == 0 && x[1130] == 0 && x[1487] == 0){
votes[0] += -258; votes[1] += 53; votes[2] += 84; votes[3] += 125; votes[4] += -226; votes[5] += -191; votes[6] += -123; votes[7] += -196; votes[8] += 25; votes[9] += -169;
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
    return output_class;
}
}
