#ifndef SOLVER_H
#define SOLVER_H

#include "../../../libs/eigen/Eigen/Dense"

#include <cusolverDn.h>

#include <QObject>

#include "Model.h"

//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_euclidean_traits_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
//typedef CGAL::Triangulation_vertex_base_with_info_3<int, Kernel> Vb_3;
//typedef CGAL::Triangulation_data_structure_3<Vb_3> Tds_3;
//typedef CGAL::Delaunay_triangulation_3<Kernel, Tds_3> Delaunay;
//typedef Delaunay::Point Point_3;
typedef CGAL::Regular_triangulation_euclidean_traits_3<Kernel>  Traits;
typedef Traits::RT                                          Weight;
typedef Traits::Bare_point                                  Point_3;
typedef Traits::Weighted_point                              Weighted_point;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Traits> Vb_3;
typedef CGAL::Triangulation_data_structure_3<Vb_3, CGAL::Regular_triangulation_cell_base_3<Traits> > Tds_3;
typedef CGAL::Regular_triangulation_3<Traits, Tds_3>               Rt;
typedef Rt::Vertex_iterator                                 Vertex_iterator;
typedef Rt::Vertex_handle                                   Vertex_handle;


class Solver : public QObject
{
    Q_OBJECT

public:
    enum MATERIAL_MODEL
    {
        MATERIAL_MODEL_NEO_HOOKEAN
    };

    typedef struct Particle
    {
        float _Mass; // Kg.
        Model::Vertex* _pVertex;
        Eigen::Vector3f _EstimatedPos; // Estimated position.
        Eigen::Vector3f _PrevPos; // Previous position.
        Eigen::Vector3f _Velocity;
        Eigen::Vector3f _Force;

        Particle()
        {
            _Mass = 0.0;
            _pVertex = NULL;
            _EstimatedPos.setZero();
            _PrevPos.setZero();
            _Velocity.setZero();
            _Force.setZero();
        }
    } Particle;

    // Lame coefficients.
    const float k = 60e4; // (Pascal) Young's modulus.
    const float v = 0.49f; // Poisson ratio.
    const float MU = k / (2.0f * (1.0f + v));
    const float LAMBDA = (k * v) / ((1.0f + v) * (1.0f - 2.0f * v));
    const float HALF_MU = 0.5f * MU;
    const float HALF_LAMBDA = 0.5f * LAMBDA;
    const float OCTET_LAMBDA = 0.125f * LAMBDA;

    Solver();

    void Initialise(Model& AModel);
    void InitialiseCUDA(Model& AModel);

    void GroupDisconnectedCells(void);
    std::vector<std::vector<int> >& DisconnectedCellGroups(void) { return m_DisconnectedCellGroups; }

    void Solve(void);
    void SolveCUDA(void);
    void SetAllParticlesUnMoved();

//    Rt& Triangulation(void) { return m_Triangulation; }

    void SetNumOfInnerIterations(int Num) { m_NumOfInnerIterations = Num; }
    void SetNumOfOuterIterations(int Num) { m_NumOfOuterIterations = Num; }

    Eigen::MatrixXf& X1() { return m_X1; }
    Eigen::MatrixXf& X2() { return m_X2; }
    Eigen::MatrixXf& X3() { return m_X3; }
    Eigen::MatrixXf& X4() { return m_X4; }

signals:
    void UpdateGL(void);

private:
    MATERIAL_MODEL m_MaterialModel;
    Model m_Model;
    Rt m_Triangulation;
    std::vector<Particle> m_Particles;
    int m_NumOfParticles;
    unsigned int m_NumOfCells;
    std::vector<std::vector<int> > m_VerticesInCells;
    std::vector<std::vector<int> > m_DisconnectedCellGroups;
    unsigned int m_NumOfDisconnectedCellGroups;
    std::vector<bool> m_OutsideMeshCells;

    float m_TotalMass; // Kg.
    float m_ParticleMass;
    float m_w;
    Eigen::MatrixXf m_X1, m_X2, m_X3, m_X4;
    std::vector<Eigen::MatrixXf> m_InvDm;
    std::vector<float> m_V;

    // Gravity;
    float m_Lambda;
    Eigen::Vector3f m_fg;

    //********** For CUDA. **********//

    // For host.
    float* m_pInvDm;
    int* m_pVertexIndicesPerCell;
    bool* m_pVertexMoved;
    float* m_pCellVolumes;
    float* m_pParticleMasses;
    float* m_pParticlePositions;
    float* m_pParticleEstimatedPositions;
    float* m_pParticleIntermediateEstimatedPositions;
    float* m_pLambdaX; // Compliance.
    unsigned int* m_pVertexIndexCounts; // Counts for each vertex in cells.
    int* m_pVertexIndexCurrentCounts;
    float* m_pParticleVelocities;
    int* m_pDisconnectedCellGroups;
    int* m_pDisconnectedCellGroupCounts;

    // For device.
    float* m_pInvDmDevice;
    int* m_pVertexIndicesPerCellDevice;
    bool* m_pVertexMovedDevice;
    float* m_pCellVolumesDevice;
    float* m_pParticleMassesDevice;
    float* m_pParticlePositionsDevice;
    float* m_pParticleEstimatedPositionsDevice;
    float* m_pParticleIntermediateEstimatedPositionsDevice;
    float* m_pLambdaXDevice; // Compliance.
    unsigned int* m_pVertexIndexCountsDevice; // Counts for each vertex in cells.
    int* m_pVertexIndexCurrentCountsDevice;
    float* m_pParticleVelocitiesDevice;
    int* m_pDisconnectedCellGroupsDevice;
    int* m_pDisconnectedCellGroupCountsDevice;

    // For SVD.
    cusolverDnHandle_t m_cuSOLVERHandle;
    int m_WorkSize;
    float* m_pWork;

    //********************//

    int m_NumOfInnerIterations;
    int m_NumOfOuterIterations;
};

#endif // SOLVER_H
