#pragma once

class GpuModel;

class EvolutionSearch {
public:
    const GpuModel& model;
    int n_random_moves = 100000;
    int n_rounds = 10;

    /* TODO: GPU solution pool. */

    EvolutionSearch(const GpuModel& model_) : model(model_) {};

    void run();
};