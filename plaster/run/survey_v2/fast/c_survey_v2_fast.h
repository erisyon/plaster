#ifndef SURVEY_V2_H
#define SURVEY_V2_H

#include "c_common.h"
#include "flann.h"


typedef struct {
    Table dyemat;
    Table dyepeps;
} Context;


void context_start(Context *ctx);


#endif
