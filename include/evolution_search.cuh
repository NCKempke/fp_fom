#pragma once

#include "gpu_data.h"

class EvolutionSearch {
public:
    GpuModel model;
    int n_random_moves = 100000;
    int n_rounds = 10;

    /* TODO: GPU solution pool. */

    EvolutionSearch(GpuModel model_) : model(model_) {};

    void run();
};