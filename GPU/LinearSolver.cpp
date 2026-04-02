#include <LinearSolver.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace amrex;

LinearSolver::LinearSolver()
{
    if (ParallelDescriptor::IOProcessor()) {
        timing_log.open("gpu_timing.log", std::ios::out | std::ios::app);
        timing_log << "=== GPU VERSION STARTED ===" << std::endl;
        timing_log << std::fixed << std::setprecision(6);
    }
}

LinearSolver::~LinearSolver()
{
    // Write timing summary in destructor
    WriteTimingSummary();
    if (timing_log.is_open()) {
        timing_log.close();
    }
}

void LinearSolver::StartTimer(const std::string& name)
{
    timers[name].start = std::chrono::high_resolution_clock::now();
}

void LinearSolver::StopTimer(const std::string& name)
{
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - timers[name].start).count();
    timers[name].total_time += duration;
    timers[name].count++;
}

void LinearSolver::WriteTimingSummary()
{
    // Check if we're on IOProcessor and if log is open
    if (!ParallelDescriptor::IOProcessor() || !timing_log.is_open()) {
        return;
    }
    
    // Don't write if no steps were completed
    if (curr_iter == 0) {
        timing_log << "\nNo steps completed." << std::endl;
        return;
    }
    
    timing_log << "\n=== FINAL GPU TIMING SUMMARY ===" << std::endl;
    timing_log << "Total steps: " << curr_iter << std::endl;
    timing_log << "Final time: " << Curr_time << std::endl;
    
    timing_log << "Grid size: " << n_cell_x << " " << n_cell_y << std::endl;
    
    timing_log << "\nBreakdown per step (average):" << std::endl;
    
    // Output in SAME ORDER as CPU for easy comparison
    const std::vector<std::string> categories = {
        "BoundaryConditions",
        "Flux_Computation", 
        "InitialData_Computation",
        "InitialData_Total",
        "LinearConvection_Total",
        "Memory_Copy",
        "WritePlotFile"
    };
    
    for (const auto& category : categories) {
        if (timers.find(category) != timers.end() && timers[category].count > 0) {
            double avg = timers[category].total_time / timers[category].count;
            timing_log << "  " << category << ": " 
                      << avg * 1000 << " ms"
                      << " (" << timers[category].count << " calls)" << std::endl;
        } else {
            timing_log << "  " << category << ": 0.000000 ms (0 calls)" << std::endl;
        }
    }
    
    // GPU-specific breakdown (additional info)
    timing_log << "\n=== GPU-SPECIFIC BREAKDOWN ===" << std::endl;
    timing_log << "Total GPU Kernel time: " << total_gpu_kernel_time * 1000 << " ms" << std::endl;
    timing_log << "Total GPU Memory transfer: " << total_gpu_memory_time * 1000 << " ms" << std::endl;
    timing_log << "Total Host CPU time: " << total_host_time * 1000 << " ms" << std::endl;
    
    double total_time = total_gpu_kernel_time + total_gpu_memory_time + total_host_time;
    
    if (total_time > 0) {
        timing_log << "\nPercentage distribution:" << std::endl;
        timing_log << "  GPU Kernels: " << (total_gpu_kernel_time/total_time) * 100 << "%" << std::endl;
        timing_log << "  GPU Memory Transfer: " << (total_gpu_memory_time/total_time) * 100 << "%" << std::endl;
        timing_log << "  Host CPU: " << (total_host_time/total_time) * 100 << "%" << std::endl;
        
        // Compute efficiency
        double compute_efficiency = (total_gpu_kernel_time/total_time) * 100;
        timing_log << "\nGPU Compute Efficiency: " << compute_efficiency << "%" << std::endl;
        if (compute_efficiency < 30) {
            timing_log << "WARNING: Low GPU efficiency - memory bound" << std::endl;
        }
    }
    
    timing_log << "\n=== END OF RUN ===" << std::endl;
    timing_log.flush();
    
    // Also print to console
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "\nGPU Timing summary written to gpu_timing.log" << std::endl;
    }
}

void LinearSolver::ReadParameters()
{
    ParmParse pp;

    pp.get("maxStep", maxStep);
    pp.get("SaveEvery", SaveEvery);
    pp.get("dt", dt);
    pp.get("finalTime", finalTime);

    pp.get("n_cell_x", n_cell_x);
    pp.get("n_cell_y", n_cell_y);
    pp.get("n_max", n_max);
    pp.get("n_ghost", n_ghost);

    pp.get("x_min", x_min);
    pp.get("x_max", x_max);
    pp.get("y_min", y_min);
    pp.get("y_max", y_max);

    pp.get("Cx", Cx);
    pp.get("Cy", Cy);
}

void LinearSolver::InitialData()
{
    StartTimer("InitialData_Total");
    
    hx = (x_max - x_min) / n_cell_x;
    hy = (y_max - y_min) / n_cell_y;

    Box domain(IntVect(0, 0), IntVect(n_cell_x - 1, n_cell_y - 1));
    RealBox real_box({x_min, y_min}, {x_max, y_max});
    Array<int, AMREX_SPACEDIM> is_periodic{1, 1};

    Geometry::Setup(&real_box, 0, is_periodic.data());
    geom.define(domain, &real_box, CoordSys::cartesian, is_periodic.data());

    BoxArray ba(domain);
    ba.maxSize(n_max);

    DistributionMapping dm(ba);

    u.define(ba, dm, 1, n_ghost);
    u_old.define(ba, dm, 1, 0);

    BoxArray ba_flux_x = ba;
    ba_flux_x.surroundingNodes(0);
    flux_x.define(ba_flux_x, dm, 1, 0);

    BoxArray ba_flux_y = ba;
    ba_flux_y.surroundingNodes(1);
    flux_y.define(ba_flux_y, dm, 1, 0);

    Real dx = geom.CellSize(0);
    Real dy = geom.CellSize(1);
    Real x_min_local = x_min;
    Real y_min_local = y_min;

    {
        StartTimer("InitialData_Computation");
        for (MFIter mfi(u); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.validbox();
            Array4<Real> const &u_arr = u.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                        // Cell center coordinates
                        Real x = x_min_local + (i + 0.5) * dx;
                        Real y = y_min_local + (j + 0.5) * dy;

                        // 2D initial condition options:
                        // Option 1: 2D sine wave
                        // u_arr(i, j, k, 0) = sin(M_PI * x) * sin(M_PI * y);

                        // Option 2: Gaussian pulse
                         Real r2 = (x - 0.0)*(x - 0.0) + (y - 0.0)*(y - 0.0);
                         u_arr(i, j, k, 0) = exp(-50.0 * r2);

                        // Option 3: Moving 2D Gaussian pulse
                        // Real r2 = (x - 0.0) * (x - 0.0) + (y - 0.0) * (y - 0.0);
                        // u_arr(i, j, k, 0) = exp(-50.0 * r2);

                        // Option 4: Checkerboard pattern
                        // u_arr(i, j, k, 0) = sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y); 
                    
            });
        }
        Gpu::synchronize();
        StopTimer("InitialData_Computation");
    }

    BoundaryConditions();
    StopTimer("InitialData_Total");
}

void LinearSolver::BoundaryConditions()
{
    StartTimer("BoundaryConditions");
    u.FillBoundary(geom.periodicity());
    Gpu::synchronize();
    StopTimer("BoundaryConditions");
}

void LinearSolver::LinearConvection()
{
    StartTimer("LinearConvection_Total");
    
    // Memory copy (same as CPU)
    {
        StartTimer("Memory_Copy");
        MultiFab::Copy(u_old, u, 0, 0, 1, 0);
        StopTimer("Memory_Copy");
        
        // Track memory transfer time
        if (timers["Memory_Copy"].count > 0) {
            double mem_time = timers["Memory_Copy"].total_time / timers["Memory_Copy"].count;
            total_gpu_memory_time += mem_time;
        }
    }
    
    BoundaryConditions();
    
    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);
    const Real inv_dx = 1.0 / dx;
    const Real inv_dy = 1.0 / dy;
    
    Real dt_local = dt;
    Real Cx_local = Cx;
    Real Cy_local = Cy;
    
    // Flux computation (GPU version)
    {
        StartTimer("Flux_Computation");
        auto flux_start = std::chrono::high_resolution_clock::now();
        
        // Flux X kernels
        for (MFIter mfi(u); mfi.isValid(); ++mfi)
        {
            const Box &flux_x_bx = flux_x[mfi].box();
            Array4<Real> const &u_arr = u.array(mfi);
            Array4<Real> const &flux_x_arr = flux_x.array(mfi);

            ParallelFor(flux_x_bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                Real ul = u_arr(i-1, j, k, 0);
                Real ur = u_arr(i, j, k, 0);
                flux_x_arr(i, j, k, 0) = UpwindFluxX(Cx_local, ul, ur);
            });
        }
        
        // Flux Y kernels
        for (MFIter mfi(u); mfi.isValid(); ++mfi)
        {
            const Box &flux_y_bx = flux_y[mfi].box();
            Array4<Real> const &u_arr = u.array(mfi);
            Array4<Real> const &flux_y_arr = flux_y.array(mfi);

            ParallelFor(flux_y_bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                Real ub = u_arr(i, j-1, k, 0);
                Real ut = u_arr(i, j, k, 0);
                flux_y_arr(i, j, k, 0) = UpwindFluxY(Cy_local, ub, ut);
            });
        }
        
        // Update kernels
        for (MFIter mfi(u); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.validbox();
            Array4<Real> const &flux_x_arr = flux_x.array(mfi);
            Array4<Real> const &flux_y_arr = flux_y.array(mfi);
            Array4<Real> const &u_arr = u.array(mfi);
            Array4<Real> const &u_old_arr = u_old.array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        { 
                Real divF = inv_dx * (flux_x_arr(i+1, j, k, 0) - flux_x_arr(i, j, k, 0))
                          + inv_dy * (flux_y_arr(i, j+1, k, 0) - flux_y_arr(i, j, k, 0));
                u_arr(i, j, k, 0) = u_old_arr(i, j, k, 0) - dt_local * divF;
            });
        }
        
        Gpu::synchronize();
        auto flux_end = std::chrono::high_resolution_clock::now();
        StopTimer("Flux_Computation");
        
        double kernel_time = std::chrono::duration<double>(flux_end - flux_start).count();
        total_gpu_kernel_time += kernel_time;
    }
    
    // Host operations
    {
        auto host_start = std::chrono::high_resolution_clock::now();
        Curr_time += dt;
        curr_iter++;
        auto host_end = std::chrono::high_resolution_clock::now();
        
        double host_time = std::chrono::duration<double>(host_end - host_start).count();
        total_host_time += host_time;
    }
    
    StopTimer("LinearConvection_Total");
    
    // Periodic logging (similar to CPU)
    if (curr_iter % 100 == 0 && ParallelDescriptor::IOProcessor()) {
        double step_time = timers["LinearConvection_Total"].total_time / curr_iter;
        
        timing_log << "Step " << curr_iter 
                  << " | Time: " << std::fixed << std::setprecision(6) << Curr_time
                  << " | Avg step time: " << step_time * 1000 << " ms" << std::endl;
    }
}

void LinearSolver::Evolve()
{
    std::ofstream log("gpu_run.log");
    log << "Initial time: " << Curr_time
        << ", Final time: " << finalTime
        << ", dt: " << dt << std::endl;

    while (Curr_time < finalTime && curr_iter < maxStep)
    {
        Real CFL_x = Cx * dt / hx;
        Real CFL_y = Cy * dt / hy;

        if (CFL_x > 1.0 || CFL_y > 1.0)
        {
            amrex::Print() << "Warning: CFL condition violated! CFL_x = "
                           << CFL_x << ", CFL_y = " << CFL_y << std::endl;
        }

        LinearConvection();

        log << "Iteration: " << curr_iter
            << ", Time: " << Curr_time
            << ", Max value: " << u.max(0)
            << ", Min value: " << u.min(0) << std::endl;

        if (curr_iter % SaveEvery == 0)
        {
            StartTimer("WritePlotFile");
            WritePlotFile();
            StopTimer("WritePlotFile");
        }
    }
    
    log.close();
    
    // Force write timing summary at the end of Evolve
    WriteTimingSummary();
}

void LinearSolver::WritePlotFile()
{
    std::string plotfilename = amrex::Concatenate("Output/plt", curr_iter, 5);

    Vector<std::string> varnames;
    varnames.push_back("u");

    amrex::WriteSingleLevelPlotfile(plotfilename, u, varnames, geom, Curr_time, curr_iter);
}
