#include <geogram/basic/common.h>
#include <geogram/mesh/mesh.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh_AABB.h>

#include <Eigen/Core>

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <limits>

#define TRACEF(...) \
    do { \
        std::fprintf(stderr, "[TRACE] %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        std::fflush(stderr); \
    } while(0)

static void validate_mesh_inputs_or_die(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& C
) {
    TRACEF("validate: V=%d x %d, F=%d x %d, C=%d x %d",
           int(V.rows()), int(V.cols()),
           int(F.rows()), int(F.cols()),
           int(C.rows()), int(C.cols()));

    if(V.cols() != 3) throw std::runtime_error("V must be (#V,3)");
    if(F.cols() != 3) throw std::runtime_error("F must be (#F,3)");
    if(C.cols() != 3) throw std::runtime_error("C must be (#sites,3)");
    if(V.rows() == 0 || F.rows() == 0 || C.rows() == 0)
        throw std::runtime_error("Empty input");

    int vmin = std::numeric_limits<int>::max();
    int vmax = std::numeric_limits<int>::min();
    for(int i=0;i<F.rows();++i)
        for(int k=0;k<3;++k){
            vmin = std::min(vmin, F(i,k));
            vmax = std::max(vmax, F(i,k));
        }

    if(vmin < 0 || vmax >= V.rows())
        throw std::runtime_error("F has out-of-range indices");

    auto bad = [](double x){ return !std::isfinite(x); };
    for(int i=0;i<V.rows();++i)
        for(int k=0;k<3;++k)
            if(bad(V(i,k))) throw std::runtime_error("NaN in V");
    for(int i=0;i<C.rows();++i)
        for(int k=0;k<3;++k)
            if(bad(C(i,k))) throw std::runtime_error("NaN in C");
}

static void validate_seeds_or_die(const Eigen::MatrixXd& C) {
    if(C.rows() < 4)
        throw std::runtime_error("Need >= 4 seeds");

    Eigen::Vector3d mn = C.colwise().minCoeff();
    Eigen::Vector3d mx = C.colwise().maxCoeff();
    Eigen::Vector3d ext = mx - mn;

    TRACEF("Seeds bbox extents: [%g %g %g]", ext[0], ext[1], ext[2]);

    double maxe = ext.maxCoeff();
    double mine = ext.minCoeff();
    if(maxe > 0 && mine / maxe < 1e-9)
        throw std::runtime_error("Nearly degenerate seed bbox");
}


static void eigen_to_geo_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    GEO::Mesh& M
) {
    M.clear();
    M.vertices.set_dimension(3);
    M.vertices.create_vertices(GEO::index_t(V.rows()));

    for(GEO::index_t i=0;i<GEO::index_t(V.rows());++i){
        double* p = M.vertices.point_ptr(i);
        p[0]=V(i,0); p[1]=V(i,1); p[2]=V(i,2);
    }

    M.facets.create_triangles(GEO::index_t(F.rows()));
    for(GEO::index_t f=0;f<GEO::index_t(F.rows());++f){
        M.facets.set_vertex(f,0,GEO::index_t(F(f,0)));
        M.facets.set_vertex(f,1,GEO::index_t(F(f,1)));
        M.facets.set_vertex(f,2,GEO::index_t(F(f,2)));
    }

    M.facets.connect();
}

struct QuantizedVec3 {
    int64_t x,y,z;
    bool operator==(const QuantizedVec3& o) const {
        return x==o.x && y==o.y && z==o.z;
    }
};

struct QuantizedVec3Hash {
    size_t operator()(const QuantizedVec3& v) const {
        size_t h = 1469598103934665603ULL;
        h ^= v.x; h *= 1099511628211ULL;
        h ^= v.y; h *= 1099511628211ULL;
        h ^= v.z; h *= 1099511628211ULL;
        return h;
    }
};

static Eigen::MatrixXd filter_duplicate_seeds(
    const Eigen::MatrixXd& C,
    std::vector<int>& old2new
) {
    old2new.assign(C.rows(), -1);

    Eigen::Vector3d bb_min = C.colwise().minCoeff();
    Eigen::Vector3d bb_max = C.colwise().maxCoeff();
    double scale = (bb_max - bb_min).maxCoeff();
    double eps = std::max(1.0, scale) * 1e-15;

    std::unordered_map<QuantizedVec3,int,QuantizedVec3Hash> map;
    std::vector<Eigen::Vector3d> kept;

    for(int i=0;i<C.rows();++i){
        Eigen::Vector3d p = C.row(i).transpose() - bb_min;
        QuantizedVec3 q{
            (int64_t)std::llround(p.x()/eps),
            (int64_t)std::llround(p.y()/eps),
            (int64_t)std::llround(p.z()/eps)
        };

        auto it = map.find(q);
        if(it == map.end()){
            int id = (int)kept.size();
            map[q] = id;
            old2new[i] = id;
            kept.push_back(C.row(i));
        } else {
            old2new[i] = it->second;
        }
    }

    Eigen::MatrixXd C2(kept.size(),3);
    for(int i=0;i<C2.rows();++i)
        C2.row(i) = kept[i].transpose();

    return C2;
}

static void write_obj(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
) {
    std::ofstream out(filename);
    if(!out) throw std::runtime_error("Cannot open OBJ");

    out << std::setprecision(17);
    for(int i=0;i<V.rows();++i)
        out<<"v "<<V(i,0)<<" "<<V(i,1)<<" "<<V(i,2)<<"\n";

    for(int i=0;i<F.rows();++i)
        out<<"f "<<F(i,0)+1<<" "<<F(i,1)+1<<" "<<F(i,2)+1<<"\n";
}

void compute_surface_rdt(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& C_in,
    const std::string& out_obj
) {
    static bool geogram_inited = false;
    if(!geogram_inited){
        GEO::initialize();
        GEO::CmdLine::import_arg_group("standard");
        GEO::CmdLine::import_arg_group("algo");
        geogram_inited = true;
    }

    std::vector<int> old2new;
    Eigen::MatrixXd C = filter_duplicate_seeds(C_in, old2new);

    TRACEF("Seeds: %d -> %d", int(C_in.rows()), int(C.rows()));

    validate_mesh_inputs_or_die(V,F,C);
    validate_seeds_or_die(C);

    GEO::Mesh surface;
    eigen_to_geo_mesh(V,F,surface);

    std::vector<double> seed_xyz(C.rows()*3);
    for(int i=0;i<C.rows();++i){
        seed_xyz[3*i+0]=C(i,0);
        seed_xyz[3*i+1]=C(i,1);
        seed_xyz[3*i+2]=C(i,2);
    }

    GEO::SmartPointer<GEO::PeriodicDelaunay3d> delaunay =
        new GEO::PeriodicDelaunay3d(false /*periodic*/, 1.0 /*period*/);

    delaunay->set_stores_neighbors(true);

    delaunay->set_vertices(GEO::index_t(C.rows()), seed_xyz.data());
    delaunay->compute();  // valid for PeriodicDelaunay3d


    GEO::RestrictedVoronoiDiagram_var rvd =
        GEO::RestrictedVoronoiDiagram::create(delaunay.get(), &surface);

    rvd->set_volumetric(false);

    GEO::vector<GEO::index_t> simplices;
    GEO::vector<double> embedding;

    GEO::MeshFacetsAABB facets_aabb(surface);
    GEO::vector<bool> facet_ok(surface.facets.nb(), true);

    const GEO::RestrictedVoronoiDiagram::RDTMode mode =
        static_cast<GEO::RestrictedVoronoiDiagram::RDTMode>(
            GEO::RestrictedVoronoiDiagram::RDT_PREFER_SEEDS |
            GEO::RestrictedVoronoiDiagram::RDT_SEEDS_ALWAYS
        );

    rvd->compute_RDT(simplices, embedding, mode, facet_ok, &facets_aabb);

    if(simplices.size() % 3 != 0)
        throw std::runtime_error("Non-triangle RDT");

    if(embedding.size() % 3 != 0) {
        throw std::runtime_error("RDT embedding is not 3D");
    }

    Eigen::MatrixXd Vout(embedding.size() / 3, 3);
    for(int i = 0; i < Vout.rows(); ++i) {
        Vout(i,0) = embedding[3*i + 0];
        Vout(i,1) = embedding[3*i + 1];
        Vout(i,2) = embedding[3*i + 2];
    }

    // --- Build triangle connectivity ---
    if(simplices.size() % 3 != 0) {
        throw std::runtime_error("RDT simplices are not triangles");
    }

    Eigen::MatrixXi Fout(simplices.size() / 3, 3);
    for(int i = 0; i < Fout.rows(); ++i) {
        Fout(i,0) = int(simplices[3*i + 0]);
        Fout(i,1) = int(simplices[3*i + 1]);
        Fout(i,2) = int(simplices[3*i + 2]);
    }

    // --- Write RDT mesh ---
    write_obj(out_obj, Vout, Fout);

}
