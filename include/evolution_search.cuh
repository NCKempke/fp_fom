#pragma once
#include "mip.h"

class GpuModel;
class MIPInstance;
class SolutionPool;
class TabuSearchDataDevice;
class TabuSearchKernelArgs;

class EvolutionSearch {
public:
    const MIPInstance& model_host;
    const GpuModel& model_device;
    int n_random_moves = 100000;
    int n_rounds = 10000;
    int tabu_tenure = 10;


    EvolutionSearch(const MIPInstance& model_host_, const GpuModel& model_device_) : model_host(model_host_), model_device(model_device_) {};

    void run(MIPData &data);

private:
    void try_store_partial_solution_for_fpr(MIPData& data, const TabuSearchDataDevice& data_device, const TabuSearchKernelArgs& args_device, int sol_idx);
};