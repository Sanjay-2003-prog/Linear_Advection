#ifndef LINEARSOLVER_H_
#define LINEARSOLVER_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <chrono>
#include <fstream>
#include <map>

using namespace amrex;

class LinearSolver
{
public:
    LinearSolver();
    virtual ~LinearSolver();
    
    void ReadParameters();
    void InitialData();
    void Evolve();
    void WritePlotFile();
    
    //Cuda it requires to be here
    void LinearConvection();
    
    void Finalize();  
    
private:
    void BoundaryConditions();
    void WriteTimingSummary();  
    
    // Timing helpers
    void StartTimer(const std::string& name);
    void StopTimer(const std::string& name);
    
    // 2D upwind flux calculation
    static AMREX_GPU_DEVICE Real UpwindFluxX(Real C, Real ul, Real ur)
    {
        return 0.5 * (std::abs(C) * (ur + ul) - C * (ur - ul));
    }

    static AMREX_GPU_DEVICE Real UpwindFluxY(Real C, Real ub, Real ut)
    {
        return 0.5 * (std::abs(C) * (ut + ub) - C * (ut - ub));
    }

    // Timing variables
    struct TimerData {
        double total_time = 0.0;
        int count = 0;
        std::chrono::high_resolution_clock::time_point start;
    };
    
    std::map<std::string, TimerData> timers;
    std::ofstream timing_log;
    
    // GPU-specific timing
    double total_gpu_kernel_time = 0.0;
    double total_gpu_memory_time = 0.0;
    double total_host_time = 0.0;
    
    // Existing variables (keep your original variables)
    int curr_iter = 0;
    int maxStep = 100;
    int SaveEvery = 10;
    Real Curr_time = 0.0;
    Real dt = 1.0e-4;
    Real finalTime = 10.0;

    int n_cell_x = 100;
    int n_cell_y = 100;
    int n_max = 25;
    int n_ghost = 3;

    Real Cx = 1.0;
    Real Cy = 0.5;

    Real hx, hy;

    Real x_min = -1.0;
    Real x_max = 1.0;
    Real y_min = -1.0;
    Real y_max = 1.0;

    MultiFab u, u_old;
    MultiFab flux_x, flux_y;
    Geometry geom;
};

#endif
