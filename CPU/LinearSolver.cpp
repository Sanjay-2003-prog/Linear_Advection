#include <LinearSolver.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <iostream>

using namespace amrex;

LinearSolver::LinearSolver()
{
    timing_log.open("cpu_timing.log", std::ios::out | std::ios::app);
    timing_log << "=== CPU VERSION STARTED ===" << std::endl;
}

LinearSolver::~LinearSolver()
{
    WriteTimingSummary();
    timing_log.close();
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
    if (ParallelDescriptor::IOProcessor()) {
        timing_log << "\n=== FINAL TIMING SUMMARY ===" << std::endl;
        timing_log << "Total steps: " << curr_iter << std::endl;
        timing_log << "Final time: " << Curr_time << std::endl;
	timing_log << "Grid Size: " << n_cell_x << " x  " << n_cell_y << std::endl;
        
        timing_log << "\nBreakdown per step (average):" << std::endl;
        timing_log << std::fixed << std::setprecision(6);
        
        for (auto& timer : timers) {
            if (timer.second.count > 0) {
                double avg = timer.second.total_time / timer.second.count;
                timing_log << "  " << timer.first << ": " 
                          << avg * 1000 << " ms"
                          << " (" << timer.second.count << " calls)" << std::endl;
            }
        }
        
        // Log to console as well
        std::cout << "\nCPU Timing summary written to cpu_timing.log" << std::endl;
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

            // CPU loop (not ParallelFor)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    Real x = x_min_local + (i + 0.5) * dx;
                    Real y = y_min_local + (j + 0.5) * dy;
                    Real r2 = (x - 0.0)*(x - 0.0) + (y - 0.0)*(y - 0.0);
                    u_arr(i, j, 0, 0) = exp(-50.0 * r2);
                }
            }
        }
        StopTimer("InitialData_Computation");
    }

    BoundaryConditions();
    StopTimer("InitialData_Total");
}

void LinearSolver::BoundaryConditions()
{
    StartTimer("BoundaryConditions");
    u.FillBoundary(geom.periodicity());
    StopTimer("BoundaryConditions");
}

void LinearSolver::LinearConvection()
{
    StartTimer("LinearConvection_Total");
    
    // Memory operations
    {
        StartTimer("Memory_Copy");
        MultiFab::Copy(u_old, u, 0, 0, 1, 0);
        StopTimer("Memory_Copy");
    }
    
    BoundaryConditions();
    
    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);
    const Real inv_dx = 1.0 / dx;
    const Real inv_dy = 1.0 / dy;

    // Computation loops (CPU version)
    {
        StartTimer("Flux_Computation");
        for (MFIter mfi(u); mfi.isValid(); ++mfi)
        {
            const Box &bx = mfi.validbox();
            const Box &flux_x_bx = flux_x[mfi].box();
            const Box &flux_y_bx = flux_y[mfi].box();

            Array4<Real> const &u_arr = u.array(mfi);
            Array4<Real> const &flux_x_arr = flux_x.array(mfi);
            Array4<Real> const &flux_y_arr = flux_y.array(mfi);
            Array4<Real> const &u_old_arr = u_old.array(mfi);

            // Flux X computation - CPU loop
            for (int j = flux_x_bx.smallEnd(1); j <= flux_x_bx.bigEnd(1); ++j) {
                for (int i = flux_x_bx.smallEnd(0); i <= flux_x_bx.bigEnd(0); ++i) {
                    Real ul = u_arr(i-1, j, 0, 0);
                    Real ur = u_arr(i, j, 0, 0);
                    flux_x_arr(i, j, 0, 0) = UpwindFluxX(Cx, ul, ur);
                }
            }

            // Flux Y computation - CPU loop
            for (int j = flux_y_bx.smallEnd(1); j <= flux_y_bx.bigEnd(1); ++j) {
                for (int i = flux_y_bx.smallEnd(0); i <= flux_y_bx.bigEnd(0); ++i) {
                    Real ub = u_arr(i, j-1, 0, 0);
                    Real ut = u_arr(i, j, 0, 0);
                    flux_y_arr(i, j, 0, 0) = UpwindFluxY(Cy, ub, ut);
                }
            }

            // Update computation - CPU loop
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    Real divF = inv_dx * (flux_x_arr(i+1, j, 0, 0) - flux_x_arr(i, j, 0, 0))
                              + inv_dy * (flux_y_arr(i, j+1, 0, 0) - flux_y_arr(i, j, 0, 0));
                    u_arr(i, j, 0, 0) = u_old_arr(i, j, 0, 0) - dt * divF;
                }
            }
        }
        StopTimer("Flux_Computation");
    }

    Curr_time += dt;
    curr_iter++;
    
    StopTimer("LinearConvection_Total");
    
    // Log periodic timing
    if (curr_iter % 100 == 0 && ParallelDescriptor::IOProcessor()) {
        timing_log << "Step " << curr_iter 
                  << " | Time: " << Curr_time
                  << " | Avg step time: " 
                  << timers["LinearConvection_Total"].total_time / curr_iter * 1000 
                  << " ms" << std::endl;
    }
}

void LinearSolver::Evolve()
{
    total_start_time = std::chrono::high_resolution_clock::now();
    
    std::ofstream log("cpu_run.log");
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
}

void LinearSolver::WritePlotFile()
{
    std::string plotfilename = amrex::Concatenate("Output/plt", curr_iter, 5);

    Vector<std::string> varnames;
    varnames.push_back("u");

    amrex::WriteSingleLevelPlotfile(plotfilename, u, varnames, geom, Curr_time, curr_iter);
}
