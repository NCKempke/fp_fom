#pragma once

class MIPInstance;
class GpuModel;

class EvolutionSearch {
public:
    const MIPInstance& model_host;
    const GpuModel& model_device;
    int n_random_moves = 100000;
    int n_rounds = 1000;
    int tabu_tenure = 10;

    /* TODO: GPU solution pool. */

    EvolutionSearch(const MIPInstance& model_host_, const GpuModel& model_device_) : model_host(model_host_), model_device(model_device_) {};

    void run();
};