#if !defined(SOLVERCUDA_H)
#define SOLVERCUDA_H

#include <cusolverDn.h>

#ifdef __cplusplus
extern "C" {
#endif


void Determinant3X3MatrixInBatch(const float* pA, float* pDeterminant, unsigned int NumOfBatches);
void SolveConstraintsInBatch(
        float* pInvDmDevice,
        int* pVertexIndicesPerCellDevice,
        bool* pVertexMovedDevice,
        float* pCellVolumesDevice,
        float* pParticleMassesDevice,
        float* pParticlePositionsDevice,
        float* pParticleEstimatedPositionsDevice,
        float* pParticleIntermediateEstimatedPositionsDevice,
        unsigned int* pVertexIndexCountsDevice,
        int* pVertexIndexCurrentCountsDevice,
        float* pParticleVelocitiesDevice,
        float* pLambdaXDevice,
        int* pDisconnectedCellGroupsDevice,
        int* pDisconnectedCellGroupCountsDevice,
        std::vector<std::vector<int> >& DisconnectedCellGroups,
        unsigned int NumOfParticles,
        unsigned int NumOfCells,
        unsigned int NumOfDisconnectedCellGroups,
        cusolverDnHandle_t cuSOLVERHandle,
        int WorkSize,
        float* pWork,
        int NumOfInnerIterations,
        int NumOfOuterIterations);


#ifdef __cplusplus
}
#endif

#endif /* SOLVERCUDA_H */
