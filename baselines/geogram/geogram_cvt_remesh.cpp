// geogram_cvt_remesh.cpp
// Geogram RVD-CVT baseline for GPU_CVT comparison.
//
// Usage:
//   geogram_cvt_remesh <input.obj> <output.obj> <nb_points> [nb_lloyd] [nb_newton]
//
// Default: 5 Lloyd + 30 Newton iterations (geogram defaults).
// Prints machine-readable timing to stdout: TIME_MS=<value>

#include <geogram/basic/common.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>

#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_remesh.h>
#include <geogram/mesh/mesh_geometry.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace GEO;

    GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
    Logger::instance()->set_quiet(false);

    CmdLine::import_arg_group("standard");
    CmdLine::import_arg_group("algo");
    CmdLine::import_arg_group("remesh");
    CmdLine::import_arg_group("mesh");

    if(argc < 4) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " <input.obj> <output.obj> <nb_points>"
               " [nb_lloyd=5] [nb_newton=30]\n";
        return 1;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];

    char* endptr = nullptr;
    const index_t nb_points = index_t(std::strtoll(argv[3], &endptr, 10));
    if(endptr == argv[3] || nb_points == 0) {
        std::cerr << "Error: nb_points must be a positive integer.\n";
        return 1;
    }

    index_t nb_lloyd  = 5;
    index_t nb_newton = 30;
    if(argc >= 5) nb_lloyd  = index_t(std::strtoll(argv[4], nullptr, 10));
    if(argc >= 6) nb_newton = index_t(std::strtoll(argv[5], nullptr, 10));

    try {
        Mesh mesh_in;
        if(!mesh_load(in_path, mesh_in)) {
            std::cerr << "Error: failed to load mesh: " << in_path << "\n";
            return 1;
        }
        std::cerr << "Input: " << mesh_in.vertices.nb() << " vertices, "
                  << mesh_in.facets.nb() << " faces\n";

        Mesh mesh_out;

        // --- timed section: CVT remeshing only ---
        auto t0 = std::chrono::high_resolution_clock::now();

        GEO::remesh_smooth(
            mesh_in, mesh_out, nb_points,
            /*dim=*/0,
            /*nb_Lloyd_iter=*/nb_lloyd,
            /*nb_Newton_iter=*/nb_newton
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cerr << "Output: " << mesh_out.vertices.nb() << " vertices, "
                  << mesh_out.facets.nb() << " faces\n";
        std::cerr << "Remesh time: " << ms << " ms\n";

        // Machine-readable line for scripts
        std::cout << "TIME_MS=" << ms << "\n";
        std::cout << "NV_OUT=" << mesh_out.vertices.nb() << "\n";
        std::cout << "NF_OUT=" << mesh_out.facets.nb() << "\n";

        if(!mesh_save(mesh_out, out_path)) {
            std::cerr << "Error: failed to save mesh: " << out_path << "\n";
            return 1;
        }

        std::cerr << "Wrote: " << out_path << "\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    } catch(...) {
        std::cerr << "Unknown exception.\n";
        return 1;
    }
}
