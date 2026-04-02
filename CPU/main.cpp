#include <LinearSolver.H>
#include <AMReX.H>

using namespace amrex;

int main(int argc, char **argv)
{

    Initialize(argc, argv);

    Real start_time_total = ParallelDescriptor::second();

    LinearSolver solver;
    solver.ReadParameters();
    solver.InitialData();
    solver.Evolve();

    Real end_time_total = ParallelDescriptor::second();
    Real total_time = end_time_total - start_time_total;

    amrex::PrintToFile("log") << "Total execution time: " << total_time * 1000 << "ms" << std::endl;
    
    //this part is required in cuda
    //#ifdef AMREX_USE_CUDA
    //    amrex::Gpu::synchronize();
    //   cudaDeviceSynchronize();
    //    cudaDeviceReset();  // This can help with cleanup
    //#endif

    Finalize();
    return 0;
}
