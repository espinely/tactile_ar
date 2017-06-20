#include "Solver.h"
#include "SolverCUDA.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "./cuda_batched_solver/inverse.h"
#include "./cuda_batched_solver/solve.h"


// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define A(row,col)     A[(col)*N+(row)]
#define C(row,col)     C[(col)*N+(row)]
#define T(row,col)     T[(col)*N+(row)]
#define TC(row,col)    TC[(col)*N+(row)]
#define Ainv(row,col)  Ainv[(col)*N+(row)]
#define Cinv(row,col)  Cinv[(col)*N+(row)]
#define Tinv(row,col)  Tinv[(col)*N+(row)]
#define TCinv(row,col) TCinv[(col)*N+(row)]

#define N 3
#define BATCH 3


Solver::Solver()
    : m_MaterialModel(MATERIAL_MODEL_NEO_HOOKEAN),
      m_NumOfParticles(0),
      m_TotalMass(1.5), // Approximate weight of a human liver.
      m_ParticleMass(1.0),
      m_w(0.0),
      m_Lambda(0.0),
      m_NumOfCells(0),
      m_pInvDm(NULL),
      m_pVertexIndicesPerCell(NULL),
      m_pParticleMasses(NULL),
      m_pParticlePositions(NULL),
      m_pParticleEstimatedPositions(NULL),
      m_pParticleVelocities(NULL),
      m_pVertexIndicesPerCellDevice(NULL),
      m_pParticleMassesDevice(NULL),
      m_pParticlePositionsDevice(NULL),
      m_pParticleEstimatedPositionsDevice(NULL),
      m_pParticleVelocitiesDevice(NULL),
      m_pLambdaX(NULL),
      m_pLambdaXDevice(NULL),
      m_NumOfInnerIterations(3),
      m_NumOfOuterIterations(500)
{
    m_fg.setZero();
}

void Solver::Initialise(Model& AModel)
{
    m_Model = AModel;

    std::vector<std::pair<Weighted_point, int> > points;
    int index = 0;
    m_OutsideMeshCells.clear();

    m_NumOfParticles = m_Model.Vertices().size();
    m_ParticleMass = m_TotalMass / m_NumOfParticles;

    for (Model::Vertex*& vertex : m_Model.Vertices())
    {
        Point_3 point(vertex->_Pos[0], vertex->_Pos[1], vertex->_Pos[2]);

        // Set all the weights to 0 in order to make the triangulation Delaunay, not regular triangulation.
        points.push_back(std::make_pair(Weighted_point(point, 0.0), index));

        Particle particle;
        particle._Mass = m_ParticleMass;
        particle._pVertex = vertex;
        particle._EstimatedPos = vertex->_Pos;
        particle._PrevPos = vertex->_Pos;

        m_Particles.push_back(particle);

        ++index;
    }

    m_w = 1.0 / m_ParticleMass;

    // Compute Delaunay triangulation.
    m_Triangulation.insert(points.begin(), points.end());

    // Initialise the Neo-Hookean model.
    m_X1.resize(3, 1); //m_Triangulation.number_of_finite_cells());
    m_X2.resize(3, 1); //m_Triangulation.number_of_finite_cells());
    m_X3.resize(3, 1); //m_Triangulation.number_of_finite_cells());
    m_X4.resize(3, 1); //m_Triangulation.number_of_finite_cells());

    index = 0;
    std::vector<Model::Vertex*> vertices = AModel.Vertices();
    QVector3D a, b, c, centroid;
    float t = 0.0f;

    for (Rt::Finite_cells_iterator it = m_Triangulation.finite_cells_begin(); it != m_Triangulation.finite_cells_end(); ++it)
    {

#if 0 // Not using now.

        int intersectionCount = 0;

        centroid.setX((it->vertex(0)->point().x() + it->vertex(1)->point().x() + it->vertex(2)->point().x() + it->vertex(3)->point().x()) * 0.25);
        centroid.setY((it->vertex(0)->point().y() + it->vertex(1)->point().y() + it->vertex(2)->point().y() + it->vertex(3)->point().y()) * 0.25);
        centroid.setZ((it->vertex(0)->point().z() + it->vertex(1)->point().z() + it->vertex(2)->point().z() + it->vertex(3)->point().z()) * 0.25);

        // Filter out the cells outside the mesh. Using a constrained Delaulay triangulation will produce an exact result (TODO).
        for (Model::Face* pFace : AModel.Faces())
        {
            std::vector<int>::iterator itIndex = pFace->_VertexIndices.begin();
            a = QVector3D(vertices[*itIndex]->_Pos[0], vertices[*itIndex]->_Pos[1], vertices[*itIndex]->_Pos[2]);
            b = QVector3D(vertices[*(itIndex + 1)]->_Pos[0], vertices[*(itIndex + 1)]->_Pos[1], vertices[*(itIndex + 1)]->_Pos[2]);
            c = QVector3D(vertices[*(itIndex + 2)]->_Pos[0], vertices[*(itIndex + 2)]->_Pos[1], vertices[*(itIndex + 2)]->_Pos[2]);

            if (Utils::TriangleRayIntersection(a, b, c, centroid, QVector3D(1.0, 1.0, 1.0), &t) == 1)
            {
                // Intersection.
                ++intersectionCount;
            }
        }

#endif

//        if (intersectionCount % 2 == 1)
        {
            // The centroid of the cell is inside the mesh.

            if (index > 0)
            {
                m_X1.conservativeResize(Eigen::NoChange, m_X1.cols()+1);
                m_X2.conservativeResize(Eigen::NoChange, m_X2.cols()+1);
                m_X3.conservativeResize(Eigen::NoChange, m_X3.cols()+1);
                m_X4.conservativeResize(Eigen::NoChange, m_X4.cols()+1);
            }

            m_X1(0, index) = it->vertex(0)->point().x();
            m_X1(1, index) = it->vertex(0)->point().y();
            m_X1(2, index) = it->vertex(0)->point().z();

            m_X2(0, index) = it->vertex(1)->point().x();
            m_X2(1, index) = it->vertex(1)->point().y();
            m_X2(2, index) = it->vertex(1)->point().z();

            m_X3(0, index) = it->vertex(2)->point().x();
            m_X3(1, index) = it->vertex(2)->point().y();
            m_X3(2, index) = it->vertex(2)->point().z();

            m_X4(0, index) = it->vertex(3)->point().x();
            m_X4(1, index) = it->vertex(3)->point().y();
            m_X4(2, index) = it->vertex(3)->point().z();

            ++index;

            m_OutsideMeshCells.push_back(false);
        }
//        else
//        {
//            // The centroid of the cell is outside the mesh.
//            m_OutsideMeshCells.push_back(true);
//        }
    }

    Eigen::MatrixXf X14 = m_X1 - m_X4;
    Eigen::MatrixXf X24 = m_X2 - m_X4;
    Eigen::MatrixXf X34 = m_X3 - m_X4;
    float detDm = 0.0f;

    for (unsigned int i = 0; i < index /*m_Triangulation.number_of_finite_cells()*/; ++i)
    {
        Eigen::Matrix3f Dm;
        Dm.col(0) = X14.col(i);
        Dm.col(1) = X24.col(i);
        Dm.col(2) = X34.col(i);

        detDm = Dm.determinant();
        m_InvDm.push_back(Dm.inverse());

        m_V.push_back(0.166666 * fabs(detDm));
    }

    // Gravity.
    m_Lambda = 0.0; // 0.002 * m_ParticleMass;
    m_fg = m_Lambda * 9.806 * Eigen::Vector3f(0.0, -1.0, 0.0);
}

void Solver::InitialiseCUDA(Model& AModel)
{
    float *A_d;
    float *Ainv_d;
    float *T;

    CUDA_SAFE_CALL (cudaMalloc ((void**)&A_d, BATCH*N*N*sizeof(A_d[0])));
    CUDA_SAFE_CALL (cudaMalloc ((void**)&Ainv_d, BATCH*N*N*sizeof(Ainv_d[0])));


    m_Model = AModel;

    std::vector<std::pair<Weighted_point, int> > points;
    int index = 0;

    m_NumOfParticles = m_Model.Vertices().size();
    m_ParticleMass = m_TotalMass / m_NumOfParticles;

    // For CUDA.
    m_pParticleMasses = new float[m_NumOfParticles];
    m_pParticlePositions = new float[m_NumOfParticles * 3];
    m_pParticleEstimatedPositions  = new float[m_NumOfParticles * 3];
    m_pParticleVelocities  = new float[m_NumOfParticles * 3];


    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pParticleMassesDevice, m_NumOfParticles * sizeof(m_pParticleMassesDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pParticlePositionsDevice, m_NumOfParticles * 3 * sizeof(m_pParticlePositionsDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pParticleEstimatedPositionsDevice, m_NumOfParticles * 3 * sizeof(m_pParticleEstimatedPositionsDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pParticleVelocitiesDevice, m_NumOfParticles * 3 * sizeof(m_pParticleVelocitiesDevice[0])));


    for (Model::Vertex*& vertex : m_Model.Vertices())
    {
        Point_3 point(vertex->_Pos[0], vertex->_Pos[1], vertex->_Pos[2]);

        // Set all the weights to 0 in order to make the triangulation Delaunay, not regular triangulation.
        points.push_back(std::make_pair(Weighted_point(point, 0.0), index));

        Particle particle;
        particle._Mass = m_ParticleMass;
        particle._pVertex = vertex;
        particle._EstimatedPos = vertex->_Pos;
        particle._PrevPos = vertex->_Pos;

        m_Particles.push_back(particle);

        // For CUDA.
        m_pParticleMasses[index] = m_ParticleMass;

        float* pPtr = m_pParticlePositions + index * 3;
        pPtr[0] = vertex->_Pos[0];
        pPtr[1] = vertex->_Pos[1];
        pPtr[2] = vertex->_Pos[2];

        pPtr = m_pParticleEstimatedPositions + index * 3;
        pPtr[0] = vertex->_Pos[0];
        pPtr[1] = vertex->_Pos[1];
        pPtr[2] = vertex->_Pos[2];

        pPtr = m_pParticleVelocities + index * 3;
        pPtr[0] = 0.0;
        pPtr[1] = 0.0;
        pPtr[2] = 0.0;

        ++index;
    }

    CUDA_SAFE_CALL(cudaMemcpy(m_pParticleMassesDevice, m_pParticleMasses, m_NumOfParticles * sizeof(m_pParticleMassesDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pParticlePositionsDevice, m_pParticlePositions, m_NumOfParticles * 3 * sizeof(m_pParticlePositionsDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pParticleEstimatedPositionsDevice, m_pParticleEstimatedPositions, m_NumOfParticles * 3 * sizeof(m_pParticleEstimatedPositionsDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pParticleVelocitiesDevice, m_pParticleVelocities, m_NumOfParticles * 3 * sizeof(m_pParticleVelocitiesDevice[0]), cudaMemcpyHostToDevice));


    m_w = 1.0 / m_ParticleMass;

    // Compute Delaunay triangulation.
    m_Triangulation.insert(points.begin(), points.end());

    // Initialise the Neo-Hookean model.
    m_X1.resize(3, 1); //m_Triangulation.number_of_finite_cells());
    m_X2.resize(3, 1); //m_Triangulation.number_of_finite_cells());
    m_X3.resize(3, 1); //m_Triangulation.number_of_finite_cells());
    m_X4.resize(3, 1); //m_Triangulation.number_of_finite_cells());

    index = 0;
    std::vector<Model::Vertex*> vertices = AModel.Vertices();
    QVector3D a, b, c, centroid;
    float t = 0.0f;
    m_OutsideMeshCells.clear();

    for (Rt::Finite_cells_iterator it = m_Triangulation.finite_cells_begin(); it != m_Triangulation.finite_cells_end(); ++it)
    {

#if 0 // Not using now.

        int intersectionCount = 0;

        centroid.setX((it->vertex(0)->point().x() + it->vertex(1)->point().x() + it->vertex(2)->point().x() + it->vertex(3)->point().x()) * 0.25);
        centroid.setY((it->vertex(0)->point().y() + it->vertex(1)->point().y() + it->vertex(2)->point().y() + it->vertex(3)->point().y()) * 0.25);
        centroid.setZ((it->vertex(0)->point().z() + it->vertex(1)->point().z() + it->vertex(2)->point().z() + it->vertex(3)->point().z()) * 0.25);

        // Filter out the cells outside the mesh. Using a constrained Delaulay triangulation will produce an exact result (TODO).

        for (Model::Face* pFace : AModel.Faces())
        {
            std::vector<int>::iterator itIndex = pFace->_VertexIndices.begin();
            a = QVector3D(vertices[*itIndex]->_Pos[0], vertices[*itIndex]->_Pos[1], vertices[*itIndex]->_Pos[2]);
            b = QVector3D(vertices[*(itIndex + 1)]->_Pos[0], vertices[*(itIndex + 1)]->_Pos[1], vertices[*(itIndex + 1)]->_Pos[2]);
            c = QVector3D(vertices[*(itIndex + 2)]->_Pos[0], vertices[*(itIndex + 2)]->_Pos[1], vertices[*(itIndex + 2)]->_Pos[2]);

            if (Utils::TriangleRayIntersection(a, b, c, centroid, QVector3D(1.0, 1.0, 1.0), &t) == 1)
            {
                // Intersection.
                ++intersectionCount;
            }
        }

#endif

//        if (intersectionCount % 2 == 1)
        {
            // The centroid of the cell is inside the mesh.

            if (index > 0)
            {
                m_X1.conservativeResize(Eigen::NoChange, m_X1.cols()+1);
                m_X2.conservativeResize(Eigen::NoChange, m_X2.cols()+1);
                m_X3.conservativeResize(Eigen::NoChange, m_X3.cols()+1);
                m_X4.conservativeResize(Eigen::NoChange, m_X4.cols()+1);
            }

            m_X1(0, index) = it->vertex(0)->point().x();
            m_X1(1, index) = it->vertex(0)->point().y();
            m_X1(2, index) = it->vertex(0)->point().z();

            m_X2(0, index) = it->vertex(1)->point().x();
            m_X2(1, index) = it->vertex(1)->point().y();
            m_X2(2, index) = it->vertex(1)->point().z();

            m_X3(0, index) = it->vertex(2)->point().x();
            m_X3(1, index) = it->vertex(2)->point().y();
            m_X3(2, index) = it->vertex(2)->point().z();

            m_X4(0, index) = it->vertex(3)->point().x();
            m_X4(1, index) = it->vertex(3)->point().y();
            m_X4(2, index) = it->vertex(3)->point().z();

            ++index;

            m_OutsideMeshCells.push_back(false);
        }
//        else
//        {
//            // The centroid of the cell is outside the mesh.
//            m_OutsideMeshCells.push_back(true);
//        }      
    }

    Eigen::MatrixXf X14 = m_X1 - m_X4;
    Eigen::MatrixXf X24 = m_X2 - m_X4;
    Eigen::MatrixXf X34 = m_X3 - m_X4;
    float detDm = 0.0f;

    // For CUDA.
    m_NumOfCells = index; //m_Triangulation.number_of_finite_cells();
    float* pA = new float[m_NumOfCells * 3 * 3]; // [BATCH*N*N];
//    m_pInvDm = new float[m_NumOfCells * 3 * 3]; // [BATCH*N*N];
    m_pVertexIndicesPerCell = new int[m_NumOfCells * 4];
    m_pVertexMoved = new bool[m_NumOfParticles];
//    m_pCellVolumes = new float[m_NumOfCells];
//    m_pVertexIndexCounts = new unsigned int[m_NumOfParticles]();
//    m_pVertexIndexCurrentCounts = new int[m_NumOfParticles]();

    // Compliance.
    m_pLambdaX = new float[m_NumOfCells];


    CUDA_SAFE_CALL(cudaMalloc((void**)&A_d, m_NumOfCells * 3 * 3 * sizeof(A_d[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pInvDmDevice, m_NumOfCells * 3 * 3 * sizeof(m_pInvDmDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pVertexIndicesPerCellDevice, m_NumOfCells * 4 * sizeof(m_pVertexIndicesPerCellDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pVertexMovedDevice, m_NumOfParticles * sizeof(m_pVertexMovedDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pCellVolumesDevice, m_NumOfCells * sizeof(m_pCellVolumesDevice[0])));
//    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pVertexIndexCountsDevice, m_NumOfParticles * sizeof(m_pVertexIndexCountsDevice[0])));
//    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pVertexIndexCurrentCountsDevice, m_NumOfParticles * sizeof(m_pVertexIndexCurrentCountsDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pLambdaXDevice, m_NumOfCells * sizeof(m_pLambdaXDevice[0])));

    Rt::Finite_cells_iterator it = m_Triangulation.finite_cells_begin();
    index = 0;

    for (unsigned int i = 0; i < m_Triangulation.number_of_finite_cells(); ++i)
    {
//        if (m_OutsideMeshCells[i])
//        {
//            ++it;
//            continue;
//        }

        Eigen::Matrix3f Dm;
        Dm.col(0) = X14.col(index);
        Dm.col(1) = X24.col(index);
        Dm.col(2) = X34.col(index);

//        detDm = Dm.determinant();
//        m_V.push_back(0.166666 * fabs(detDm));

        T = pA + index * 3 * 3;

        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                T(k, j) = Dm(k, j);
            }
        }

        m_pVertexIndicesPerCell[index * 4 + 0] = it->vertex(0)->info();
        m_pVertexIndicesPerCell[index * 4 + 1] = it->vertex(1)->info();
        m_pVertexIndicesPerCell[index * 4 + 2] = it->vertex(2)->info();
        m_pVertexIndicesPerCell[index * 4 + 3] = it->vertex(3)->info();

//        ++m_pVertexIndexCounts[it->vertex(0)->info()];
//        ++m_pVertexIndexCounts[it->vertex(1)->info()];
//        ++m_pVertexIndexCounts[it->vertex(2)->info()];
//        ++m_pVertexIndexCounts[it->vertex(3)->info()];

        std::vector<int> vertices;
        vertices.push_back(it->vertex(0)->info());
        vertices.push_back(it->vertex(1)->info());
        vertices.push_back(it->vertex(2)->info());
        vertices.push_back(it->vertex(3)->info());
        m_VerticesInCells.push_back(vertices);

        m_pLambdaX[index] = 0.0f;

        ++it;
        ++index;
    }


//    int numOfParticleIntermediatePositions = 0;

//    for (int i = 0; i < m_NumOfParticles; ++i)
//    {
//        numOfParticleIntermediatePositions += m_pVertexIndexCounts[i];
//    }

//    m_pParticleIntermediateEstimatedPositions = new float[numOfParticleIntermediatePositions * 3]();
//    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pParticleIntermediateEstimatedPositionsDevice, numOfParticleIntermediatePositions * 3 * sizeof(m_pParticleIntermediateEstimatedPositionsDevice[0])));
//    CUDA_SAFE_CALL(cudaMemcpy(m_pParticleIntermediateEstimatedPositionsDevice, m_pParticleIntermediateEstimatedPositions, numOfParticleIntermediatePositions * 3 * sizeof(m_pParticleIntermediateEstimatedPositionsDevice[0]), cudaMemcpyHostToDevice));



    CUDA_SAFE_CALL(cudaMemcpy(m_pVertexIndicesPerCellDevice, m_pVertexIndicesPerCell, m_NumOfCells * 4 * sizeof(m_pVertexIndicesPerCellDevice[0]), cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(m_pVertexIndexCountsDevice, m_pVertexIndexCounts, m_NumOfParticles * sizeof(m_pVertexIndexCountsDevice[0]), cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(m_pVertexIndexCurrentCountsDevice, m_pVertexIndexCurrentCounts, m_NumOfParticles * sizeof(m_pVertexIndexCurrentCountsDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(A_d, pA, m_NumOfCells * 3 * 3 * sizeof(A_d[0]), cudaMemcpyHostToDevice));

    smatinv_batch(A_d, m_pInvDmDevice, 3, m_NumOfCells);
    Determinant3X3MatrixInBatch(A_d, m_pCellVolumesDevice, m_NumOfCells);

//    CUDA_SAFE_CALL(cudaMemcpy(m_pInvDm, m_pInvDmDevice, m_NumOfCells * 3 * 3 * sizeof(m_pInvDmDevice[0]), cudaMemcpyDeviceToHost));
//    CUDA_SAFE_CALL(cudaMemcpy(m_pCellVolumes, m_pCellVolumesDevice, m_NumOfCells * sizeof(m_pCellVolumesDevice[0]), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(m_pLambdaXDevice, m_pLambdaX, m_NumOfCells * sizeof(m_pLambdaXDevice[0]), cudaMemcpyHostToDevice));


    // TODO: Temp for generating dataset for CNNs as we use the non-deformed model for now. Uncomment later.
#if 1


    // TODO: Temp - implement checking if the file exists and run the function if not.
//    GroupDisconnectedCells();


    m_NumOfDisconnectedCellGroups = m_DisconnectedCellGroups.size();

    m_pDisconnectedCellGroups = new int[m_NumOfCells]();
    m_pDisconnectedCellGroupCounts = new int[m_NumOfDisconnectedCellGroups]();

    index = 0;

    for (unsigned int i = 0; i < m_NumOfDisconnectedCellGroups; ++i)
    {
        for (unsigned int j = 0; j < m_DisconnectedCellGroups[i].size(); ++j)
        {
            m_pDisconnectedCellGroups[index] = (m_DisconnectedCellGroups[i])[j];

            ++index;
        }

        m_pDisconnectedCellGroupCounts[i] = m_DisconnectedCellGroups[i].size();
    }

    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pDisconnectedCellGroupsDevice, m_NumOfCells * sizeof(m_pDisconnectedCellGroupsDevice[0])));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_pDisconnectedCellGroupCountsDevice, m_NumOfDisconnectedCellGroups * sizeof(m_pDisconnectedCellGroupCountsDevice[0])));
    CUDA_SAFE_CALL(cudaMemcpy(m_pDisconnectedCellGroupsDevice, m_pDisconnectedCellGroups, m_NumOfCells * sizeof(m_pDisconnectedCellGroupsDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pDisconnectedCellGroupCountsDevice, m_pDisconnectedCellGroupCounts, m_NumOfDisconnectedCellGroups * sizeof(m_pDisconnectedCellGroupCountsDevice[0]), cudaMemcpyHostToDevice));

#endif


    // Gravity.
    m_Lambda = 0.0; // 0.002 * m_ParticleMass;
    m_fg = m_Lambda * 9.806 * Eigen::Vector3f(0.0, -1.0, 0.0);

    // cuSOLVER for SVD.
    cusolverDnCreate(&m_cuSOLVERHandle);
    cusolverDnDgesvd_bufferSize(m_cuSOLVERHandle, 3, 3, &m_WorkSize);
    cudaMalloc(&m_pWork, m_WorkSize * sizeof(float));
}

void Solver::GroupDisconnectedCells(void)
{
    int numOfUngroupedCells = (int)m_NumOfCells;
    bool initial = true;
    bool finished = false;

    std::vector<int> group;

    while (numOfUngroupedCells > 0)
    {
        if (!initial)
        {
            group.clear();

            bool grouped = false;

            for (unsigned int k = 0; k < m_NumOfCells; ++k)
            {
                grouped = false;

                for (unsigned int i = 0; i < m_DisconnectedCellGroups.size(); ++i)
                {
                    std::vector<int> members = m_DisconnectedCellGroups[i];

                    if (std::find(members.begin(), members.end(), k) == members.end())
                    {
                    }
                    else
                    {
                        // This cell is already grouped.
                        grouped = true;

                        break;
                    }
                }

                if (!grouped)
                {
                    group.push_back(k);

//                    printf("numOfUngroupedCells: %d\n", numOfUngroupedCells);
                    --numOfUngroupedCells;

                    if (numOfUngroupedCells <= 0)
                    {
                        m_DisconnectedCellGroups.push_back(group);

                        finished = true;

                        break;
                    }

                    break;
                }
            }

            if (finished)
            {
                break;
            }
        }
        else
        {
            group.push_back(0);

            initial = false;

//            printf("numOfUngroupedCells: %d\n", numOfUngroupedCells);
            --numOfUngroupedCells;

            if (numOfUngroupedCells <= 0)
            {
                m_DisconnectedCellGroups.push_back(group);

                break;
            }
        }

//        Rt::Finite_cells_iterator itSub = m_Triangulation.finite_cells_begin();
        int index = 0;

        for (unsigned int i = 0; i < m_NumOfCells; ++i)
        {
            bool grouped = false;

            for (unsigned int j = 0; j < m_DisconnectedCellGroups.size(); ++j)
            {
                std::vector<int> members = m_DisconnectedCellGroups[j];

                if (std::find(members.begin(), members.end(), i) != members.end())
                {
                    // This cell is already grouped.
                    grouped = true;

                    break;
                }
            }

            if (grouped)
            {
//                ++itSub;
                ++index;

                continue;
            }

            if (std::find(group.begin(), group.end(), i) != group.end())
            {
                // This cell is already grouped.
            }
            else
            {
                bool notConnected = true;

                for (unsigned int j = 0; j < group.size(); ++j)
                {
                    std::vector<int> vertices = m_VerticesInCells[group[j]];

//                    if ((std::find(vertices.begin(), vertices.end(), itSub->vertex(0)->info()) == vertices.end())
//                     && (std::find(vertices.begin(), vertices.end(), itSub->vertex(1)->info()) == vertices.end())
//                     && (std::find(vertices.begin(), vertices.end(), itSub->vertex(2)->info()) == vertices.end())
//                     && (std::find(vertices.begin(), vertices.end(), itSub->vertex(3)->info()) == vertices.end()))
                    if ((std::find(vertices.begin(), vertices.end(), m_VerticesInCells[index][0]) == vertices.end())
                     && (std::find(vertices.begin(), vertices.end(), m_VerticesInCells[index][1]) == vertices.end())
                     && (std::find(vertices.begin(), vertices.end(), m_VerticesInCells[index][2]) == vertices.end())
                     && (std::find(vertices.begin(), vertices.end(), m_VerticesInCells[index][3]) == vertices.end()))
                    {
                    }
                    else
                    {
                        notConnected = false;

                        break;
                    }
                }

                if (notConnected)
                {
                    // This cell is not connected to those in this group.
                    group.push_back(i);

//                    printf("numOfUngroupedCells: %d\n", numOfUngroupedCells);
                    --numOfUngroupedCells;

                    if (numOfUngroupedCells <= 0)
                    {
                        break;
                    }
                }
            }

//            ++itSub;
            ++index;
        }

        m_DisconnectedCellGroups.push_back(group);
    }

    int count = 0;

    for (unsigned int i = 0; i < m_DisconnectedCellGroups.size(); ++i)
    {
        std::cout << i << ": " << m_DisconnectedCellGroups[i].size() << std::endl;

        count += m_DisconnectedCellGroups[i].size();
    }

    std::cout << "Num: " << count << std::endl;
}

void Solver::Solve(void)
{
    clock_t time = clock();

    float dt = 1.0f;
    float kDamping = (1.0f / 24.0f) * (1.0f / 24.0f) * 2.2f * sqrt(k * m_ParticleMass);
    int numOfIterations = 0;

    // For checking the convergence.
    // Root-mean-square-error (RMSE).
    float rmse = 0.0f;

    // Update particle estimated positions because some vertices should have been moved.
    for (Particle& particle : m_Particles)
    {
        if (particle._pVertex->_Moved)
        {
            particle._EstimatedPos = particle._pVertex->_Pos;
        }
    }

    do
    {
        std::cout << "#iterations: " << numOfIterations << std::endl;

        for (Particle& particle : m_Particles)
        {
            if (!particle._pVertex->_Moved)
            {
                // External force.
                particle._Velocity += dt * m_fg * 1.0;

                // Damp velocity.
                particle._Velocity -= kDamping * particle._Velocity;

                // Estimate new position.
                particle._EstimatedPos = particle._pVertex->_Pos + dt * particle._Velocity;
            }
        }

        for (int i = 0; i < 2; ++i)
        {
            // Solve constraints.
            Eigen::Matrix3f Ds, F;
            std::vector<int> indices;

//            for (Rt::Finite_cells_iterator it = m_Triangulation.finite_cells_begin(); it != m_Triangulation.finite_cells_end(); ++it)
            for (int j = 0; j < m_VerticesInCells.size(); ++j)
            {
                indices = m_VerticesInCells[j];
                int i1 = indices[0]; //it->vertex(0)->info();
                int i2 = indices[1]; //it->vertex(1)->info();
                int i3 = indices[2]; //it->vertex(2)->info();
                int i4 = indices[3]; //it->vertex(3)->info();

                Ds.col(0) = m_Particles[i1]._EstimatedPos - m_Particles[i4]._EstimatedPos;
                Ds.col(1) = m_Particles[i2]._EstimatedPos - m_Particles[i4]._EstimatedPos;
                Ds.col(2) = m_Particles[i3]._EstimatedPos - m_Particles[i4]._EstimatedPos;

                F = Ds * m_InvDm[j];

//                if (fabs(F.determinant()) < 1e-6)
//                {
//                    // TODO: Approximate inverse.
//                }

                Eigen::Matrix3f F2 = F.transpose() * F;
                float invariant1 = F2.trace();
                float detF2 = F2.determinant();

                float logI3 = log(detF2);
                float E = m_V[j] * (HALF_MU * (invariant1 - logI3 - 3.0) + OCTET_LAMBDA * logI3 * logI3);

                Eigen::Matrix3f invF = F.inverse();
                Eigen::Matrix3f P = MU * F + (HALF_LAMBDA * logI3 - MU) * invF.transpose();

                Eigen::Matrix3f d123E = m_V[j] * P * m_InvDm[j].transpose();
                Eigen::Vector3f d1E = d123E.col(0);
                Eigen::Vector3f d2E = d123E.col(1);
                Eigen::Vector3f d3E = d123E.col(2);
                Eigen::Vector3f d4E = -(d1E + d2E + d3E);
                float diE2 = d1E.dot(d1E) + d2E.dot(d2E) + d3E.dot(d3E) + d4E.dot(d4E);
                float smallLambda = 0.0f;

                if (fabs(diE2) < 1e-6)
                {
                    smallLambda = 0.0f;
                }
                else
                {
                    smallLambda = - E / diE2;
                }

                // Update.
                Eigen::Vector3f dx1 = smallLambda * d1E;
                Eigen::Vector3f dx2 = smallLambda * d2E;
                Eigen::Vector3f dx3 = smallLambda * d3E;
                Eigen::Vector3f dx4 = smallLambda * d4E;

                m_Particles[i1]._EstimatedPos += dx1;
                m_Particles[i2]._EstimatedPos += dx2;
                m_Particles[i3]._EstimatedPos += dx3;
                m_Particles[i4]._EstimatedPos += dx4;
            }
        }

        // Update particle states.
        int count = 0;

        for (Particle& particle : m_Particles)
        {
            if (!particle._pVertex->_Moved)
            {
                Eigen::Vector3f diff = particle._EstimatedPos - particle._pVertex->_Pos;
                rmse += diff.dot(diff);

                particle._Velocity = diff / dt;
                particle._pVertex->_Pos = particle._EstimatedPos;

                ++count;
            }
        }

        emit UpdateGL();

        rmse = sqrt(rmse / count);

        std::cout << "RMSE: " << rmse << std::endl;

        if (numOfIterations >= 100)
        {
            break;
        }

        ++numOfIterations;
    }
    while ((numOfIterations < 25 && rmse < 0.002) || rmse >= 0.002); // Check the convergence.

    for (Particle& particle : m_Particles)
    {
        particle._pVertex->_Moved = false;
    }

    m_Model.ComputeFaceNormals();
    emit UpdateGL();

    time = clock() - time;

    std::cout << ((float)time) / CLOCKS_PER_SEC << " seconds." << std::endl;
}

void Solver::SolveCUDA(void)
{    
    clock_t time = clock();

//    float dt = 1.0f;
//    float kDamping = (1.0f / 24.0f) * (1.0f / 24.0f) * 2.2f * sqrt(k * m_ParticleMass);
//    int numOfIterations = 0;
    int index = 0;

    // For checking the convergence.
    // Root-mean-square-error (RMSE).
    float rmse = 0.0f;

    // Update particle estimated positions because some vertices should have been moved.
    for (Particle& particle : m_Particles)
    {
        m_pParticlePositions[index * 3 + 0] = particle._pVertex->_Pos[0];
        m_pParticlePositions[index * 3 + 1] = particle._pVertex->_Pos[1];
        m_pParticlePositions[index * 3 + 2] = particle._pVertex->_Pos[2];

        if (particle._pVertex->_Moved)
        {
            particle._EstimatedPos = particle._pVertex->_Pos;

            m_pParticleEstimatedPositions[index * 3 + 0] = particle._pVertex->_Pos[0];
            m_pParticleEstimatedPositions[index * 3 + 1] = particle._pVertex->_Pos[1];
            m_pParticleEstimatedPositions[index * 3 + 2] = particle._pVertex->_Pos[2];

            m_pVertexMoved[index] = true;
        }
        else
        {
            m_pVertexMoved[index] = false;
        }

        m_pParticleVelocities[index * 3 + 0] = 0.0;
        m_pParticleVelocities[index * 3 + 1] = 0.0;
        m_pParticleVelocities[index * 3 + 2] = 0.0;

        ++index;
    }

    CUDA_SAFE_CALL(cudaMemcpy(m_pParticlePositionsDevice, m_pParticlePositions, m_NumOfParticles * 3 * sizeof(m_pParticlePositionsDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pParticleEstimatedPositionsDevice, m_pParticleEstimatedPositions, m_NumOfParticles * 3 * sizeof(m_pParticleEstimatedPositionsDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pParticleVelocitiesDevice, m_pParticleVelocities, m_NumOfParticles * 3 * sizeof(m_pParticleVelocitiesDevice[0]), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_pVertexMovedDevice, m_pVertexMoved, m_NumOfParticles * sizeof(m_pVertexMovedDevice[0]), cudaMemcpyHostToDevice));

    memset(m_pLambdaX, 0, m_NumOfCells * sizeof(m_pLambdaX[0]));
    CUDA_SAFE_CALL(cudaMemcpy(m_pLambdaXDevice, m_pLambdaX, m_NumOfCells * sizeof(m_pLambdaXDevice[0]), cudaMemcpyHostToDevice));

    SolveConstraintsInBatch(
                m_pInvDmDevice,
                m_pVertexIndicesPerCellDevice,
                m_pVertexMovedDevice,
                m_pCellVolumesDevice,
                m_pParticleMassesDevice,
                m_pParticlePositionsDevice,
                m_pParticleEstimatedPositionsDevice,
                m_pParticleIntermediateEstimatedPositionsDevice,
                m_pVertexIndexCountsDevice,
                m_pVertexIndexCurrentCountsDevice,
                m_pParticleVelocitiesDevice,
                m_pLambdaXDevice,
                m_pDisconnectedCellGroupsDevice,
                m_pDisconnectedCellGroupCountsDevice,
                m_DisconnectedCellGroups,
                m_NumOfParticles,
                m_NumOfCells,
                m_NumOfDisconnectedCellGroups,
                m_cuSOLVERHandle,
                m_WorkSize,
                m_pWork,
                m_NumOfInnerIterations,
                m_NumOfOuterIterations);

#if 0

    do
    {
        std::cout << "#iterations: " << numOfIterations << std::endl;

        for (Particle& particle : m_Particles)
        {
            if (!particle._pVertex->_Selected)
            {
                // External force.
                particle._Velocity += dt * m_fg * 1.0;

                // Damp velocity.
                particle._Velocity -= kDamping * particle._Velocity;

                // Estimate new position.
                particle._EstimatedPos = particle._pVertex->_Pos + dt * particle._Velocity;
            }
        }

        for (int i = 0; i < 2; ++i)
        {
            // Solve constraints.
            Eigen::Matrix3f Ds, invDm, F;
            int index = 0;

            for (Rt::Finite_cells_iterator it = m_Triangulation.finite_cells_begin(); it != m_Triangulation.finite_cells_end(); ++it)
            {
                int i1 = it->vertex(0)->info();
                int i2 = it->vertex(1)->info();
                int i3 = it->vertex(2)->info();
                int i4 = it->vertex(3)->info();

                Ds.col(0) = m_Particles[i1]._EstimatedPos - m_Particles[i4]._EstimatedPos;
                Ds.col(1) = m_Particles[i2]._EstimatedPos - m_Particles[i4]._EstimatedPos;
                Ds.col(2) = m_Particles[i3]._EstimatedPos - m_Particles[i4]._EstimatedPos;

                // Set the inverse of Dm.
                float* Tinv = m_pInvDm + index * N * N;

                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        invDm(k, j) = Tinv(k, j);
                    }
                }

                F = Ds * invDm; // m_InvDm[index];

//                if (fabs(F.determinant()) < 1e-6)
//                {
//                    // TODO: Approximate inverse.
//                }

                Eigen::Matrix3f F2 = F.transpose() * F;
                float invariant1 = F2.trace();
                float detF2 = F2.determinant();

                float logI3 = log(detF2);
                float E = m_V[index] * (HALF_MU * (invariant1 - logI3 - 3.0) + OCTET_LAMBDA * logI3 * logI3);

                Eigen::Matrix3f invF = F.inverse();
                Eigen::Matrix3f P = MU * F + (HALF_LAMBDA * logI3 - MU) * invF.transpose();

                Eigen::Matrix3f d123E = m_V[index] * P * invDm.transpose(); //m_InvDm[index].transpose();
                Eigen::Vector3f d1E = d123E.col(0);
                Eigen::Vector3f d2E = d123E.col(1);
                Eigen::Vector3f d3E = d123E.col(2);
                Eigen::Vector3f d4E = -(d1E + d2E + d3E);
                float diE2 = d1E.dot(d1E) + d2E.dot(d2E) + d3E.dot(d3E) + d4E.dot(d4E);
                float smallLambda = 0.0f;

                if (fabs(diE2) < 1e-6)
                {
                    smallLambda = 0.0f;
                }
                else
                {
                    smallLambda = - E / diE2;
                }

                // Update.
                Eigen::Vector3f dx1 = smallLambda * d1E;
                Eigen::Vector3f dx2 = smallLambda * d2E;
                Eigen::Vector3f dx3 = smallLambda * d3E;
                Eigen::Vector3f dx4 = smallLambda * d4E;

                m_Particles[i1]._EstimatedPos += dx1;
                m_Particles[i2]._EstimatedPos += dx2;
                m_Particles[i3]._EstimatedPos += dx3;
                m_Particles[i4]._EstimatedPos += dx4;

                ++index;
            }
        }

        // Update particle states.
        // TODO: move only not selected vertices for now.
        int count = 0;

        for (Particle& particle : m_Particles)
        {
            if (!particle._pVertex->_Selected)
            {
                Eigen::Vector3f diff = particle._EstimatedPos - particle._pVertex->_Pos;
                rmse += diff.dot(diff);

                particle._Velocity = diff / dt;
                particle._pVertex->_Pos = particle._EstimatedPos;

                ++count;
            }
        }

        emit UpdateGL();

        rmse = sqrt(rmse / count);

        std::cout << "RMSE: " << rmse << std::endl;

        if (numOfIterations >= 100)
        {
            break;
        }

        ++numOfIterations;
    }
    while ((numOfIterations < 25 && rmse < 0.002) || rmse >= 0.002); // Check the convergence.

#endif

//    cudaThreadSynchronize();

    CUDA_SAFE_CALL(cudaMemcpy(m_pParticlePositions, m_pParticlePositionsDevice, m_NumOfParticles * 3 * sizeof(m_pParticlePositionsDevice[0]), cudaMemcpyDeviceToHost));

    index = 0;

    for (Particle& particle : m_Particles)
    {
        if (!particle._pVertex->_Moved)
        {
            particle._pVertex->_Pos[0] = m_pParticlePositions[index * 3 + 0];
            particle._pVertex->_Pos[1] = m_pParticlePositions[index * 3 + 1];
            particle._pVertex->_Pos[2] = m_pParticlePositions[index * 3 + 2];

            m_pParticleEstimatedPositions[index * 3 + 0] = m_pParticlePositions[index * 3 + 0];
            m_pParticleEstimatedPositions[index * 3 + 1] = m_pParticlePositions[index * 3 + 1];
            m_pParticleEstimatedPositions[index * 3 + 2] = m_pParticlePositions[index * 3 + 2];
        }
        else
        {
//            particle._pVertex->_Moved = false;
        }

        ++index;
    }

    time = clock() - time;

//    std::cout << ((float)time) / CLOCKS_PER_SEC << " seconds." << std::endl;
}

void Solver::SetAllParticlesUnMoved()
{
    for (Particle& particle : m_Particles)
    {
        particle._pVertex->_Moved = false;
    }
}
