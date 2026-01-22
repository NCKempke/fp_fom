#pragma once

#include "gpu_data.h"

class EvolutionSearch {
public:
    GpuModel model;

    /* TODO: GPU solution pool. */

    EvolutionSearch(GpuModel model_) : model(model_) {};

    void run();
};