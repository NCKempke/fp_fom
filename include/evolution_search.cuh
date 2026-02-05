#pragma once

class GpuModel;
class MIPInstance;
class SolutionPool;

class EvolutionSearch {
public:
    const MIPInstance& model_host;
    const GpuModel& model_device;
    /* Solution pool for storing partial (infeasible) solutions we then try to repair with FPR. This pool is threadsafe. */
    SolutionPool& partials;
    int n_random_moves = 100000;
    int n_rounds = 10000;
    int tabu_tenure = 10;

    /* TODO: GPU solution pool. */

    EvolutionSearch(const MIPInstance& model_host_, const GpuModel& model_device_, SolutionPool& partials_) : model_host(model_host_), model_device(model_device_), partials(partials_) {};

    void run();
};