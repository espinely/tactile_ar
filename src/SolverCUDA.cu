#include <stdio.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "SolverCUDA.h"
#include "./svd3/svd3_cuda/svd3_cuda.h"


#define GRID_DIM_LIMIT  (65520)


/***** Compute the determinants of 3 by 3 matrices in batch. *****/

__global__ void Determinant3X3MatrixPerThread(const float* pA, float* pDeterminant, unsigned int NumOfBatches)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;
    float A00, A01, A02;
    float A10, A11, A12;
    float A20, A21, A22;

    pA += thrdNum * N * N;
    pDeterminant += thrdNum;

    if (thrdNum < NumOfBatches)
    {
        A00 = pA[0];
        A10 = pA[1];
        A20 = pA[2];
        A01 = pA[3];
        A11 = pA[4];
        A21 = pA[5];
        A02 = pA[6];
        A12 = pA[7];
        A22 = pA[8];

//        *pDeterminant = A00 * A11 * A22 + A01 * A12 * A20 + A02 * A10 * A21
//                      - A02 * A11 * A20 - A01 * A10 * A22 - A00 * A12 * A21;

        *pDeterminant = A00 * (A11 * A22 - A12 * A21)
                      + A01 * (A12 * A20 - A10 * A22)
                      + A02 * (A10 * A21 - A11 * A20);
    }
}

void Determinant3X3MatrixInBatch(const float* pA, float* pDeterminant, unsigned int NumOfBatches)
{
    cudaError_t err;
    dim3 dimBlock(128);
    dim3 dimGrid;
    int numBlocks;

    numBlocks = (NumOfBatches + dimBlock.x - 1) / dimBlock.x;
    if (numBlocks <= GRID_DIM_LIMIT) {
        dimGrid.x = numBlocks;
        dimGrid.y = 1;
        dimGrid.z = 1;
    } else {
        dimGrid.x = GRID_DIM_LIMIT;
        dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
        dimGrid.z = 1;
    }

    Determinant3X3MatrixPerThread<<<dimGrid,dimBlock>>>(pA, pDeterminant, NumOfBatches);

    /* Check synchronous errors, i.e. pre-launch */
    err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        std::cout << "Determinant3X3MatrixInBatch(): CUDA error occured." << std::endl;
    }
}

/**********/


// Matrix multiplication.
__device__ void Multiply3X3Matrix(const float* pA, const float* pB, float* pResult)
{
    for (int i = 0; i < 3 * 3; ++i)
    {
        pResult[i] = 0.0f;
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                pResult[i * 3 + j] += pA[j + k * 3] * pB[k + i * 3];
            }
        }
    }
}

// Multiply a 3 by 3 matrix by a scalar.
__device__ void MultiplyScalar3X3Matrix(const float* pA, const float Scalar, float* pResult)
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            pResult[i * 3 + j] = pA[i * 3 + j] * Scalar;
        }
    }
}

// Add two 3 by 3 matrices.
__device__ void Add3X3Matrix(const float* pA, const float* pB, float* pResult)
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            pResult[i * 3 + j] = pA[i * 3 + j] + pB[i * 3 + j];
        }
    }
}

// Matrix transpose.
__device__ void Transpose3X3Matrix(const float* pA, float* pResult)
{
    pResult[0 * 3 + 0] = pA[0 * 3 + 0];
    pResult[0 * 3 + 1] = pA[1 * 3 + 0];
    pResult[0 * 3 + 2] = pA[2 * 3 + 0];

    pResult[1 * 3 + 0] = pA[0 * 3 + 1];
    pResult[1 * 3 + 1] = pA[1 * 3 + 1];
    pResult[1 * 3 + 2] = pA[2 * 3 + 1];

    pResult[2 * 3 + 0] = pA[0 * 3 + 2];
    pResult[2 * 3 + 1] = pA[1 * 3 + 2];
    pResult[2 * 3 + 2] = pA[2 * 3 + 2];
}

// Compute the determinant of a 3 by 3 matrix.
__device__ float Determinant3X3Matrix(const float* pA)
{
    float A00, A01, A02;
    float A10, A11, A12;
    float A20, A21, A22;

    A00 = pA[0];
    A10 = pA[1];
    A20 = pA[2];
    A01 = pA[3];
    A11 = pA[4];
    A21 = pA[5];
    A02 = pA[6];
    A12 = pA[7];
    A22 = pA[8];

//    return (A00 * A11 * A22 + A01 * A12 * A20 + A02 * A10 * A21
//          - A02 * A11 * A20 - A01 * A10 * A22 - A00 * A12 * A21);

    return (A00 * (A11 * A22 - A12 * A21)
          + A01 * (A12 * A20 - A10 * A22)
          + A02 * (A10 * A21 - A11 * A20));
}

// Compute the inverse of a 3 by 3 matrix.
__device__ bool Inverse3X3Matrix(const float* pA, float* pResult)
{
    float det = Determinant3X3Matrix(pA);

    if (fabs(det) < 1e-6)
    {
        // The matrix is singular.
        printf("The matrix is singular.\n");

        return false;
    }

    float A00, A01, A02;
    float A10, A11, A12;
    float A20, A21, A22;

    A00 = pA[0];
    A10 = pA[1];
    A20 = pA[2];
    A01 = pA[3];
    A11 = pA[4];
    A21 = pA[5];
    A02 = pA[6];
    A12 = pA[7];
    A22 = pA[8];

    det = 1.0 / det;

    pResult[0 * 3 + 0] = det * (A11 * A22 - A21 * A12);
    pResult[0 * 3 + 1] = det * (A12 * A20 - A22 * A10);
    pResult[0 * 3 + 2] = det * (A10 * A21 - A20 * A11);

    pResult[1 * 3 + 0] = det * (A02 * A21 - A22 * A01);
    pResult[1 * 3 + 1] = det * (A00 * A22 - A20 * A02);
    pResult[1 * 3 + 2] = det * (A01 * A20 - A21 * A00);

    pResult[2 * 3 + 0] = det * (A01 * A12 - A11 * A02);
    pResult[2 * 3 + 1] = det * (A02 * A10 - A12 * A00);
    pResult[2 * 3 + 2] = det * (A00 * A11 - A10 * A01);

    return true;
}

// Compute the trace of a 3 by 3 matrix.
__device__ float Trace3X3Matrix(const float* pA)
{
    return (pA[0] + pA[4] + pA[8]);
}

/**********/


/***** For mutex. *****/

__device__ void LockMutex(int* pMutex)
{
    while (atomicCAS(pMutex, 0, 1) != 0)
    {
        // Do nothing.
    }
}

__device__ void UnlockMutex(int* pMutex)
{
    atomicExch(pMutex, 0);
}

/**********/


/***** Solve Neo-Hookean material model in batch. *****/

// Lame coefficients.
const float k = 60e3; // (Pascal) Young's modulus.
const float v = 0.49f; // Poisson ratio.
const float MU = k / (2.0f * (1.0f + v));
const float LAMBDA = (k * v) / ((1.0f + v) * (1.0f - 2.0f * v));
const float HALF_MU = 0.5f * MU;
const float HALF_LAMBDA = 0.5f * LAMBDA;
const float OCTET_LAMBDA = 0.125f * LAMBDA;

__global__ void EstimateNewParticlePositionPerThread(bool* pVertexMovedDevice,
                                            float* pParticleMassesDevice,
                                            float* pParticlePositionsDevice,
                                            float* pParticleEstimatedPositionsDevice,
                                            float* pParticleVelocitiesDevice,
                                            unsigned int NumOfParticles)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;

    pVertexMovedDevice += thrdNum;
    pParticleVelocitiesDevice += thrdNum * N;
    pParticlePositionsDevice += thrdNum * N;
    pParticleEstimatedPositionsDevice += thrdNum * N;
    pParticleMassesDevice += thrdNum;

    float dt = 1.0f;
    float kDamping = (1.0f / 24.0f) * (1.0f / 24.0f) * 2.2f * sqrt(k * (*pParticleMassesDevice));

    if (thrdNum < NumOfParticles)
    {
        // Damp velocity.
        pParticleVelocitiesDevice[0] -= kDamping *  pParticleVelocitiesDevice[0];
        pParticleVelocitiesDevice[1] -= kDamping *  pParticleVelocitiesDevice[1];
        pParticleVelocitiesDevice[2] -= kDamping *  pParticleVelocitiesDevice[2];

        // Estimate new position.
        pParticleEstimatedPositionsDevice[0] = pParticlePositionsDevice[0] + dt * pParticleVelocitiesDevice[0];
        pParticleEstimatedPositionsDevice[1] = pParticlePositionsDevice[1] + dt * pParticleVelocitiesDevice[1];
        pParticleEstimatedPositionsDevice[2] = pParticlePositionsDevice[2] + dt * pParticleVelocitiesDevice[2];
    }
}

__global__ void SolveConstraintsPerThread(
        float* pInvDmDevice,
        int* pVertexIndicesPerCellDevice,
        bool* pVertexMovedDevice,
        float* pCellVolumesDevice,
        float* pParticleMassesDevice,
        float* pParticlePositionsDevice,
        float* pParticleEstimatedPositionsDevice,
        float* pParticleVelocitiesDevice,
        float* pLambdaXDevice,
        int* pDisconnectedCellGroupsDevice,
        unsigned int NumOfParticles,
        unsigned int NumOfCells,
        unsigned int NumOfDisconnectedCells,
        int Idx,
        cusolverDnHandle_t cuSOLVERHandle,
        int WorkSize,
        float* pWork)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;

//    // Use only 1 thread.
//    thrdNum = Idx;
//    if (thrdNum < NumOfCells)
    if (thrdNum < NumOfDisconnectedCells)
    {
        float Ds[3 * 3] = {0.0f, };
        float F[3 * 3] = {0.0f, };
        float FTransposed[3 * 3] = {0.0f, };
        float F2[3 * 3] = {0.0f, };

//        pParticleMassesDevice += thrdNum;
//        pInvDmDevice += thrdNum * N * N;
//        pVertexIndicesPerCellDevice += thrdNum * 4;
//        pCellVolumesDevice += thrdNum;

        pDisconnectedCellGroupsDevice += Idx;
        pInvDmDevice += pDisconnectedCellGroupsDevice[thrdNum] * N * N;
        pVertexIndicesPerCellDevice += pDisconnectedCellGroupsDevice[thrdNum] * 4;
        pCellVolumesDevice += pDisconnectedCellGroupsDevice[thrdNum];
        pLambdaXDevice += pDisconnectedCellGroupsDevice[thrdNum];

        int i1 = pVertexIndicesPerCellDevice[0];
        int i2 = pVertexIndicesPerCellDevice[1];
        int i3 = pVertexIndicesPerCellDevice[2];
        int i4 = pVertexIndicesPerCellDevice[3];

        Ds[0 * 3 + 0] = pParticleEstimatedPositionsDevice[i1 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[0 * 3 + 1] = pParticleEstimatedPositionsDevice[i1 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[0 * 3 + 2] = pParticleEstimatedPositionsDevice[i1 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Ds[1 * 3 + 0] = pParticleEstimatedPositionsDevice[i2 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[1 * 3 + 1] = pParticleEstimatedPositionsDevice[i2 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[1 * 3 + 2] = pParticleEstimatedPositionsDevice[i2 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Ds[2 * 3 + 0] = pParticleEstimatedPositionsDevice[i3 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[2 * 3 + 1] = pParticleEstimatedPositionsDevice[i3 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[2 * 3 + 2] = pParticleEstimatedPositionsDevice[i3 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Multiply3X3Matrix(Ds, pInvDmDevice, F);

        // Handle tetrahedron inversion.
        float detF = Determinant3X3Matrix(F);

        if (detF < 0.0f)
        {
            float S[3 * 3];
            float U[3 * 3];
            float V[3 * 3];

            float u11, u12, u13, u21, u22, u23, u31, u32, u33;
            float s11, s12, s13, s21, s22, s23, s31, s32, s33;
            float v11, v12, v13, v21, v22, v23, v31, v32, v33;

            svd(F[0 + 0 * 3], F[0 + 1 * 3], F[0 + 2 * 3],
                F[1 + 0 * 3], F[1 + 1 * 3], F[1 + 2 * 3],
                F[2 + 0 * 3], F[2 + 1 * 3], F[2 + 2 * 3],
                u11, u12, u13, u21, u22, u23, u31, u32, u33,
                s11, s12, s13, s21, s22, s23, s31, s32, s33,
                v11, v12, v13, v21, v22, v23, v31, v32, v33);

            U[0 + 0 * 3] = u11; U[0 + 1 * 3] = u12; U[0 + 2 * 3] = u13;
            U[1 + 0 * 3] = u21; U[1 + 1 * 3] = u22; U[1 + 2 * 3] = u23;
            U[2 + 0 * 3] = u31; U[2 + 1 * 3] = u32; U[2 + 2 * 3] = u33;

            S[0 + 0 * 3] = s11; S[0 + 1 * 3] = s12; S[0 + 2 * 3] = s13;
            S[1 + 0 * 3] = s21; S[1 + 1 * 3] = s22; S[1 + 2 * 3] = s23;
            S[2 + 0 * 3] = s31; S[2 + 1 * 3] = s32; S[2 + 2 * 3] = s33;

            V[0 + 0 * 3] = v11; V[0 + 1 * 3] = v12; V[0 + 2 * 3] = v13;
            V[1 + 0 * 3] = v21; V[1 + 1 * 3] = v22; V[1 + 2 * 3] = v23;
            V[2 + 0 * 3] = v31; V[2 + 1 * 3] = v32; V[2 + 2 * 3] = v33;

            // Negate the smallest value in S.
            S[2 + 2 * 3] *= -1.0f;

            // Recompute F.
            float temp0[3 * 3] = {0.0f, };
            float temp1[3 * 3] = {0.0f, };

            Multiply3X3Matrix(U, S, temp0);
            Transpose3X3Matrix(V, temp1);

            Multiply3X3Matrix(temp0, temp1, F);

            detF = Determinant3X3Matrix(F);
        }

        Transpose3X3Matrix(F, FTransposed);

        Multiply3X3Matrix(FTransposed, F, F2);
        float invariant1 = Trace3X3Matrix(F2);

        float detF2 = Determinant3X3Matrix(F2);
        float detFT = Determinant3X3Matrix(FTransposed);

        if (detF2 > 0.0f)
        {
            // Good.
        }
//        else if (fabs(detF2) < 1e-6)
//        {
//            // The determinant is singular.
//            printf("F2 is singular.\n");

//            return;
//        }
        else
        {
            // TODO: When there is an overflow, compute it by squaring detF. An overflow might not happen on a better graphics card, which supports double floating point.
            detF2 = detF * detF;

//            printf("Ds: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", Ds[0], Ds[1], Ds[2], Ds[3], Ds[4], Ds[5], Ds[6], Ds[7], Ds[8]);
//            printf("pInvDmDevice: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", pInvDmDevice[0], pInvDmDevice[1], pInvDmDevice[2], pInvDmDevice[3], pInvDmDevice[4], pInvDmDevice[5], pInvDmDevice[6], pInvDmDevice[7], pInvDmDevice[8]);
//            printf("F: %.30f %.30f %.30f\n %.30f %.30f %.30f\n %.30f %.30f %.30f\n", F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8]);
//            printf("F2: %.30f %.30f %.30f\n %.30f %.30f %.30f\n %.30f %.30f %.30f\n", F2[0], F2[3], F2[6], F2[1], F2[4], F2[7], F2[2], F2[5], F2[8]);
////            printf("F: %.20f %.20f %.20f\n %.20f %.20f %.20f\n %.20f %.20f %.20f\n", FTransposed[0], FTransposed[3], FTransposed[6], FTransposed[1], FTransposed[4], FTransposed[7], FTransposed[2], FTransposed[5], FTransposed[8]);
//            printf("detF: %f, detF2: %f, detF * detFT: %f, detFT: %f\n", detF, detF2, detF * detFT, detFT);
        }

        float logI3 = log(detF2);

        // Compute the cell volume here.
        float cellVolume = fabs(*pCellVolumesDevice) * 0.16666f;

        // Compute the energy of a cell to be minimised.
        float E = cellVolume * (HALF_MU * (invariant1 - logI3 - 3.0f) + OCTET_LAMBDA * logI3 * logI3);

        float invF[3 * 3] = {0.0f, };
        Inverse3X3Matrix(F, invF);

        float P[3 * 3] = {0.0f, };
        float temp0[3 * 3] = {0.0f, };
        float temp1[3 * 3] = {0.0f, };
        float invFTransposed[3 * 3] = {0.0f, };

        MultiplyScalar3X3Matrix(F, MU, temp0);
        Transpose3X3Matrix(invF, invFTransposed);
        MultiplyScalar3X3Matrix(invFTransposed, HALF_LAMBDA * logI3 - MU, temp1);
        Add3X3Matrix(temp0, temp1, P);

        float d123E[3 * 3] = {0.0f, };
        memset(temp0, 0, 3 * 3 * sizeof(temp0[0]));
        memset(temp1, 0, 3 * 3 * sizeof(temp1[0]));

        MultiplyScalar3X3Matrix(P, cellVolume, temp0);
        Transpose3X3Matrix(pInvDmDevice, temp1);

        Multiply3X3Matrix(temp0, temp1, d123E);

        float d1E[3] = {d123E[0 * 0 + 0], d123E[0 * 0 + 1], d123E[0 * 0 + 2]};
        float d2E[3] = {d123E[1 * 3 + 0], d123E[1 * 3 + 1], d123E[1 * 3 + 2]};
        float d3E[3] = {d123E[2 * 3 + 0], d123E[2 * 3 + 1], d123E[2 * 3 + 2]};
        float d4E[3] = {0.0f, };

        // TODO: Implement vector addition.
        d4E[0]= -(d1E[0] + d2E[0] + d3E[0]);
        d4E[1]= -(d1E[1] + d2E[1] + d3E[1]);
        d4E[2]= -(d1E[2] + d2E[2] + d3E[2]);

        // TODO: Implement dot product.
        float diE2 = (d1E[0] * d1E[0] + d1E[1] * d1E[1] + d1E[2] * d1E[2])
                    + (d2E[0] * d2E[0] + d2E[1] * d2E[1] + d2E[2] * d2E[2])
                    + (d3E[0] * d3E[0] + d3E[1] * d3E[1] + d3E[2] * d3E[2])
                    + (d4E[0] * d4E[0] + d4E[1] * d4E[1] + d4E[2] * d4E[2]);

        float alpha = 1.0f;

        float delta_lambda = (-E - alpha * *pLambdaXDevice) / (diE2 + alpha);
        *pLambdaXDevice = *pLambdaXDevice + delta_lambda;

//        float smallLambda = 0.0f;

//        if (fabs(diE2) < 1e-6)
//        {
//            smallLambda = 0.0f;
//        }
//        else
//        {
//            smallLambda = - E / diE2;
//        }

        // Update.
        float dx1[3] = {delta_lambda * d1E[0], delta_lambda * d1E[1], delta_lambda * d1E[2]};
        float dx2[3] = {delta_lambda * d2E[0], delta_lambda * d2E[1], delta_lambda * d2E[2]};
        float dx3[3] = {delta_lambda * d3E[0], delta_lambda * d3E[1], delta_lambda * d3E[2]};
        float dx4[3] = {delta_lambda * d4E[0], delta_lambda * d4E[1], delta_lambda * d4E[2]};

        pParticleEstimatedPositionsDevice[i1 * 3 + 0] += dx1[0];
        pParticleEstimatedPositionsDevice[i1 * 3 + 1] += dx1[1];
        pParticleEstimatedPositionsDevice[i1 * 3 + 2] += dx1[2];

        pParticleEstimatedPositionsDevice[i2 * 3 + 0] += dx2[0];
        pParticleEstimatedPositionsDevice[i2 * 3 + 1] += dx2[1];
        pParticleEstimatedPositionsDevice[i2 * 3 + 2] += dx2[2];

        pParticleEstimatedPositionsDevice[i3 * 3 + 0] += dx3[0];
        pParticleEstimatedPositionsDevice[i3 * 3 + 1] += dx3[1];
        pParticleEstimatedPositionsDevice[i3 * 3 + 2] += dx3[2];

        pParticleEstimatedPositionsDevice[i4 * 3 + 0] += dx4[0];
        pParticleEstimatedPositionsDevice[i4 * 3 + 1] += dx4[1];
        pParticleEstimatedPositionsDevice[i4 * 3 + 2] += dx4[2];
    }
}

// This version of the function uses multiple threads.
// Update particle positions in parallel and average the positions after.
__global__ void SolveConstraintsInParallelPerThread(
        float* pInvDmDevice,
        int* pVertexIndicesPerCellDevice,
        bool* pVertexMovedDevice,
        float* pCellVolumesDevice,
        float* pParticleMassesDevice,
        float* pParticlePositionsDevice,
        float* pParticleEstimatedPositionsDevice,
        float* pParticleIntermediateEstimatedPositionsDevice,
        unsigned int* pVertexIndexCounts,
        int* pVertexIndexCurrentCounts,
        float* pParticleVelocitiesDevice,
        unsigned int NumOfParticles,
        unsigned int NumOfCells
)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;

    if (thrdNum < NumOfCells)
    {
        float Ds[3 * 3] = {0.0f, };
        float F[3 * 3] = {0.0f, };
        float FTransposed[3 * 3] = {0.0f, };
        float F2[3 * 3] = {0.0f, };

        pParticleMassesDevice += thrdNum;
        pInvDmDevice += thrdNum * N * N;
        pVertexIndicesPerCellDevice += thrdNum * 4;
        pCellVolumesDevice += thrdNum;

        int i1 = pVertexIndicesPerCellDevice[0];
        int i2 = pVertexIndicesPerCellDevice[1];
        int i3 = pVertexIndicesPerCellDevice[2];
        int i4 = pVertexIndicesPerCellDevice[3];

        Ds[0 * 3 + 0] = pParticleEstimatedPositionsDevice[i1 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[0 * 3 + 1] = pParticleEstimatedPositionsDevice[i1 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[0 * 3 + 2] = pParticleEstimatedPositionsDevice[i1 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Ds[1 * 3 + 0] = pParticleEstimatedPositionsDevice[i2 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[1 * 3 + 1] = pParticleEstimatedPositionsDevice[i2 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[1 * 3 + 2] = pParticleEstimatedPositionsDevice[i2 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Ds[2 * 3 + 0] = pParticleEstimatedPositionsDevice[i3 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[2 * 3 + 1] = pParticleEstimatedPositionsDevice[i3 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[2 * 3 + 2] = pParticleEstimatedPositionsDevice[i3 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Multiply3X3Matrix(Ds, pInvDmDevice, F);

        Transpose3X3Matrix(F, FTransposed);

        Multiply3X3Matrix(FTransposed, F, F2);
        float invariant1 = Trace3X3Matrix(F2);
        float detF2 = Determinant3X3Matrix(F2);

        float logI3 = log(detF2);

        // Compute the cell volume here.
        float cellVolume = fabs(*pCellVolumesDevice) * 0.16666f;

        float E = cellVolume * (HALF_MU * (invariant1 - logI3 - 3.0f) + OCTET_LAMBDA * logI3 * logI3);

        float invF[3 * 3] = {0.0f, };
        Inverse3X3Matrix(F, invF);

        float P[3 * 3] = {0.0f, };
        float temp0[3 * 3] = {0.0f, };
        float temp1[3 * 3] = {0.0f, };
        float invFTransposed[3 * 3] = {0.0f, };

        MultiplyScalar3X3Matrix(F, MU, temp0);
        Transpose3X3Matrix(invF, invFTransposed);
        MultiplyScalar3X3Matrix(invFTransposed, HALF_LAMBDA * logI3 - MU, temp1);
        Add3X3Matrix(temp0, temp1, P);

        float d123E[3 * 3] = {0.0f, };
        memset(temp0, 0, 3 * 3 * sizeof(temp0[0]));
        memset(temp1, 0, 3 * 3 * sizeof(temp1[0]));

        MultiplyScalar3X3Matrix(P, cellVolume, temp0);
        Transpose3X3Matrix(pInvDmDevice, temp1);

        Multiply3X3Matrix(temp0, temp1, d123E);

        float d1E[3] = {d123E[0 * 0 + 0], d123E[0 * 0 + 1], d123E[0 * 0 + 2]};
        float d2E[3] = {d123E[1 * 3 + 0], d123E[1 * 3 + 1], d123E[1 * 3 + 2]};
        float d3E[3] = {d123E[2 * 3 + 0], d123E[2 * 3 + 1], d123E[2 * 3 + 2]};
        float d4E[3] = {0.0f, };

        // TODO: Implement vector addition.
        d4E[0]= -(d1E[0] + d2E[0] + d3E[0]);
        d4E[1]= -(d1E[1] + d2E[1] + d3E[1]);
        d4E[2]= -(d1E[2] + d2E[2] + d3E[2]);

        // TODO: Implement dot product.
        float diE2 = (d1E[0] * d1E[0] + d1E[1] * d1E[1] + d1E[2] * d1E[2])
                    + (d2E[0] * d2E[0] + d2E[1] * d2E[1] + d2E[2] * d2E[2])
                    + (d3E[0] * d3E[0] + d3E[1] * d3E[1] + d3E[2] * d3E[2])
                    + (d4E[0] * d4E[0] + d4E[1] * d4E[1] + d4E[2] * d4E[2]);

        float smallLambda = 0.0f;

        if (fabs(diE2) < 1e-16)
        {
            smallLambda = 0.0f;
        }
        else
        {
            smallLambda = - E / diE2;
        }

        // Update.
        float dx1[3] = {smallLambda * d1E[0], smallLambda * d1E[1], smallLambda * d1E[2]};
        float dx2[3] = {smallLambda * d2E[0], smallLambda * d2E[1], smallLambda * d2E[2]};
        float dx3[3] = {smallLambda * d3E[0], smallLambda * d3E[1], smallLambda * d3E[2]};
        float dx4[3] = {smallLambda * d4E[0], smallLambda * d4E[1], smallLambda * d4E[2]};


        atomicAdd(&pVertexIndexCurrentCounts[i1], 1);

        float* pPtr = pParticleIntermediateEstimatedPositionsDevice;

        for (int i = 0; i < i1; ++i)
        {
            pPtr += pVertexIndexCounts[i] * 3;
        }

        pPtr[pVertexIndexCurrentCounts[i1] * 3 + 0] = dx1[0];
        pPtr[pVertexIndexCurrentCounts[i1] * 3 + 1] = dx1[1];
        pPtr[pVertexIndexCurrentCounts[i1] * 3 + 2] = dx1[2];


        atomicAdd(&pVertexIndexCurrentCounts[i2], 1);

        pPtr = pParticleIntermediateEstimatedPositionsDevice;

        for (int i = 0; i < i2; ++i)
        {
            pPtr += pVertexIndexCounts[i] * 3;
        }

        pPtr[pVertexIndexCurrentCounts[i2] * 3 + 0] = dx2[0];
        pPtr[pVertexIndexCurrentCounts[i2] * 3 + 1] = dx2[1];
        pPtr[pVertexIndexCurrentCounts[i2] * 3 + 2] = dx2[2];


        atomicAdd(&pVertexIndexCurrentCounts[i3], 1);

        pPtr = pParticleIntermediateEstimatedPositionsDevice;

        for (int i = 0; i < i3; ++i)
        {
            pPtr += pVertexIndexCounts[i] * 3;
        }

        pPtr[pVertexIndexCurrentCounts[i3] * 3 + 0] = dx3[0];
        pPtr[pVertexIndexCurrentCounts[i3] * 3 + 1] = dx3[1];
        pPtr[pVertexIndexCurrentCounts[i3] * 3 + 2] = dx3[2];


        atomicAdd(&pVertexIndexCurrentCounts[i4], 1);

        pPtr = pParticleIntermediateEstimatedPositionsDevice;

        for (int i = 0; i < i4; ++i)
        {
            pPtr += pVertexIndexCounts[i] * 3;
        }

        pPtr[pVertexIndexCurrentCounts[i4] * 3 + 0] = dx4[0];
        pPtr[pVertexIndexCurrentCounts[i4] * 3 + 1] = dx4[1];
        pPtr[pVertexIndexCurrentCounts[i4] * 3 + 2] = dx4[2];
    }
}

__global__ void UpdateParticleStatePerThread(bool* pVertexMovedDevice,
                                             float* pParticlePositionsDevice,
                                             float* pParticleEstimatedPositionsDevice,
                                             float* pParticleVelocitiesDevice,
                                             unsigned int NumOfParticles,
                                             float* pRMSEDevice)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;

    pVertexMovedDevice += thrdNum;
    pParticleVelocitiesDevice += thrdNum * N;
    pParticlePositionsDevice += thrdNum * N;
    pParticleEstimatedPositionsDevice += thrdNum * N;

    float dt = 1.0f;

    // Update particle states.
    if (thrdNum < NumOfParticles)
    {
        if (!(*pVertexMovedDevice))
        {
            float diff[3] = {
                pParticleEstimatedPositionsDevice[0] - pParticlePositionsDevice[0],
                pParticleEstimatedPositionsDevice[1] - pParticlePositionsDevice[1],
                pParticleEstimatedPositionsDevice[2] - pParticlePositionsDevice[2]
            };

            atomicAdd(pRMSEDevice, (float)(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]));
//             atomicAdd(pRMSEDevice, (float)(sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) / (pParticleEstimatedPositionsDevice[0] * pParticleEstimatedPositionsDevice[0] + pParticleEstimatedPositionsDevice[1] * pParticleEstimatedPositionsDevice[1] + pParticleEstimatedPositionsDevice[2] * pParticleEstimatedPositionsDevice[2]))));

            pParticleVelocitiesDevice[0] = diff[0] / dt;
            pParticleVelocitiesDevice[1] = diff[1] / dt;
            pParticleVelocitiesDevice[2] = diff[2] / dt;

            pParticlePositionsDevice[0] = pParticleEstimatedPositionsDevice[0];
            pParticlePositionsDevice[1] = pParticleEstimatedPositionsDevice[1];
            pParticlePositionsDevice[2] = pParticleEstimatedPositionsDevice[2];
        }
    }
}

__global__ void UpdateAverageParticleStatePerThread(bool* pVertexMovedDevice,
                                             float* pParticlePositionsDevice,
                                             float* pParticleEstimatedPositionsDevice,
                                             float* pParticleIntermediateEstimatedPositionsDevice,
                                             unsigned int* pVertexIndexCounts,
                                             float* pParticleVelocitiesDevice,
                                             unsigned int NumOfParticles,
                                             bool Intermediate)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;

    pVertexMovedDevice += thrdNum;
    pParticleVelocitiesDevice += thrdNum * N;
    pParticlePositionsDevice += thrdNum * N;
    pParticleEstimatedPositionsDevice += thrdNum * N;

    float dt = 1.0f;

    // Update particle states.
    if (thrdNum < NumOfParticles)
    {
        if (Intermediate)
        {
            for (int i = 0; i < thrdNum; ++i)
            {
                pParticleIntermediateEstimatedPositionsDevice += pVertexIndexCounts[i] * 3;
            }

            pVertexIndexCounts += thrdNum;

            // Compute the average vertex position.
//            float x = pParticleEstimatedPositionsDevice[0];
//            float y = pParticleEstimatedPositionsDevice[1];
//            float z = pParticleEstimatedPositionsDevice[2];
//            pParticleEstimatedPositionsDevice[0] = 0.0;
//            pParticleEstimatedPositionsDevice[1] = 0.0;
//            pParticleEstimatedPositionsDevice[2] = 0.0;
            float dx = 0.0f;
            float dy = 0.0f;
            float dz = 0.0f;

            for (unsigned int i = 0; i < *pVertexIndexCounts; ++i)
            {
                dx += pParticleIntermediateEstimatedPositionsDevice[i * 3 + 0];
                dy += pParticleIntermediateEstimatedPositionsDevice[i * 3 + 1];
                dz += pParticleIntermediateEstimatedPositionsDevice[i * 3 + 2];
            }

            dx /= (float)(*pVertexIndexCounts);
            dy /= (float)(*pVertexIndexCounts);
            dz /= (float)(*pVertexIndexCounts);
//            dx *= 0.005;
//            dy *= 0.005;
//            dz *= 0.005;

            pParticleEstimatedPositionsDevice[0] += dx;
            pParticleEstimatedPositionsDevice[1] += dy;
            pParticleEstimatedPositionsDevice[2] += dz;
        }
        else if (!(*pVertexMovedDevice))
        {
//            for (int i = 0; i < thrdNum; ++i)
//            {
//                pParticleIntermediateEstimatedPositionsDevice += pVertexIndexCounts[i] * 3;
//            }

//            pVertexIndexCounts += thrdNum;

//            // Compute the average vertex position.
//            float x = pParticleEstimatedPositionsDevice[0];
//            float y = pParticleEstimatedPositionsDevice[1];
//            float z = pParticleEstimatedPositionsDevice[2];
//            pParticleEstimatedPositionsDevice[0] = 0.0;
//            pParticleEstimatedPositionsDevice[1] = 0.0;
//            pParticleEstimatedPositionsDevice[2] = 0.0;

//            for (unsigned int i = 0; i < *pVertexIndexCounts; ++i)
//            {
//                pParticleEstimatedPositionsDevice[0] += x + pParticleIntermediateEstimatedPositionsDevice[i * 3 + 0];
//                pParticleEstimatedPositionsDevice[1] += y + pParticleIntermediateEstimatedPositionsDevice[i * 3 + 1];
//                pParticleEstimatedPositionsDevice[2] += z + pParticleIntermediateEstimatedPositionsDevice[i * 3 + 2];
//            }

//            pParticleEstimatedPositionsDevice[0] /= (float)(*pVertexIndexCounts);
//            pParticleEstimatedPositionsDevice[1] /= (float)(*pVertexIndexCounts);
//            pParticleEstimatedPositionsDevice[2] /= (float)(*pVertexIndexCounts);

            float diff[3] = {
                pParticleEstimatedPositionsDevice[0] - pParticlePositionsDevice[0],
                pParticleEstimatedPositionsDevice[1] - pParticlePositionsDevice[1],
                pParticleEstimatedPositionsDevice[2] - pParticlePositionsDevice[2]
            };

            //            rmse += diff.dot(diff);

            pParticleVelocitiesDevice[0] = diff[0] / dt;
            pParticleVelocitiesDevice[1] = diff[1] / dt;
            pParticleVelocitiesDevice[2] = diff[2] / dt;

            pParticlePositionsDevice[0] = pParticleEstimatedPositionsDevice[0];
            pParticlePositionsDevice[1] = pParticleEstimatedPositionsDevice[1];
            pParticlePositionsDevice[2] = pParticleEstimatedPositionsDevice[2];
        }
    }
}


// Compute the energy of a cell.
__global__ void ComputeCellEnergyPerThread(
        float* pInvDmDevice,
        int* pVertexIndicesPerCellDevice,
        float* pCellVolumesDevice,
        float* pParticleMassesDevice,
        float* pParticleEstimatedPositionsDevice,
        float* pEnergyDevice,
        unsigned int NumOfCells)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
    const int N = 3;

    if (thrdNum < NumOfCells)
    {
        float Ds[3 * 3] = {0.0f, };
        float F[3 * 3] = {0.0f, };
        float FTransposed[3 * 3] = {0.0f, };
        float F2[3 * 3] = {0.0f, };

//        pParticleMassesDevice += thrdNum;
        pInvDmDevice += thrdNum * N * N;
        pVertexIndicesPerCellDevice += thrdNum * 4;
        pCellVolumesDevice += thrdNum;
        pEnergyDevice += thrdNum;

        int i1 = pVertexIndicesPerCellDevice[0];
        int i2 = pVertexIndicesPerCellDevice[1];
        int i3 = pVertexIndicesPerCellDevice[2];
        int i4 = pVertexIndicesPerCellDevice[3];

        Ds[0 * 3 + 0] = pParticleEstimatedPositionsDevice[i1 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[0 * 3 + 1] = pParticleEstimatedPositionsDevice[i1 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[0 * 3 + 2] = pParticleEstimatedPositionsDevice[i1 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Ds[1 * 3 + 0] = pParticleEstimatedPositionsDevice[i2 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[1 * 3 + 1] = pParticleEstimatedPositionsDevice[i2 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[1 * 3 + 2] = pParticleEstimatedPositionsDevice[i2 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Ds[2 * 3 + 0] = pParticleEstimatedPositionsDevice[i3 * 3 + 0] - pParticleEstimatedPositionsDevice[i4 * 3 + 0];
        Ds[2 * 3 + 1] = pParticleEstimatedPositionsDevice[i3 * 3 + 1] - pParticleEstimatedPositionsDevice[i4 * 3 + 1];
        Ds[2 * 3 + 2] = pParticleEstimatedPositionsDevice[i3 * 3 + 2] - pParticleEstimatedPositionsDevice[i4 * 3 + 2];

        Multiply3X3Matrix(Ds, pInvDmDevice, F);

        Transpose3X3Matrix(F, FTransposed);

        Multiply3X3Matrix(FTransposed, F, F2);
        float invariant1 = Trace3X3Matrix(F2);
        float detF2 = Determinant3X3Matrix(F2);

        float logI3 = log(detF2);

        // Compute the cell volume here.
        float cellVolume = fabs(*pCellVolumesDevice) * 0.16666f;

        // Compute the energy of a cell.
        float E = cellVolume * (HALF_MU * (invariant1 - logI3 - 3.0f) + OCTET_LAMBDA * logI3 * logI3);
        *pEnergyDevice = E;
    }
}

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
        int NumOfOuterIterations)
{
    cudaError_t err;
    dim3 dimBlock(128);
    dim3 dimGrid;
    int numBlocks;

    float prevRMSE = 1e3;
    float* pRMSE = new float;
    *pRMSE = 0.0f;

    float* pRMSEDevice;
    cudaMalloc((void**)&pRMSEDevice, sizeof(pRMSEDevice[0]));
    cudaMemcpy(pRMSEDevice, pRMSE, sizeof(pRMSEDevice[0]), cudaMemcpyHostToDevice);

    float *pEnergy = new float[NumOfCells]();
    float* pEnergyDevice;
    cudaMalloc((void**)&pEnergyDevice, NumOfCells * sizeof(pEnergyDevice[0]));
    cudaMemcpy(pEnergyDevice, pEnergy, NumOfCells * sizeof(pEnergyDevice[0]), cudaMemcpyHostToDevice);


#ifdef PARALLEL

    int vertexIndexCurrentCounts[NumOfParticles];

#endif

    int numOfIterations = 0;
    float prevEnergy = 0.0f;
    int smallEnergyDiffCount = 0;

    do
    {
//        std::cout << "#iterations: " << numOfIterations << std::endl;

        float energy = 0.0f;

        // Estimate new particle positions.
        numBlocks = (NumOfParticles + dimBlock.x - 1) / dimBlock.x;
        if (numBlocks <= GRID_DIM_LIMIT) {
            dimGrid.x = numBlocks;
            dimGrid.y = 1;
            dimGrid.z = 1;
        } else {
            dimGrid.x = GRID_DIM_LIMIT;
            dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
            dimGrid.z = 1;
        }

        EstimateNewParticlePositionPerThread<<<dimGrid,dimBlock>>>(pVertexMovedDevice,
                                                                   pParticleMassesDevice,
                                                                   pParticlePositionsDevice,
                                                                   pParticleEstimatedPositionsDevice,
                                                                   pParticleVelocitiesDevice,
                                                                   NumOfParticles);

//        cudaThreadSynchronize();

        /* Check synchronous errors, i.e. pre-launch */
        err = cudaGetLastError();

        if (cudaSuccess != err)
        {
            std::cout << "EstimateNewParticlePositionPerThread(): CUDA error occured." << std::endl;
        }

        for (int j = 0; j < NumOfInnerIterations; ++j)
        {
            // Solve constraints.
//            numBlocks = (NumOfCells + dimBlock.x - 1) / dimBlock.x;
//            if (numBlocks <= GRID_DIM_LIMIT) {
//                dimGrid.x = numBlocks;
//                dimGrid.y = 1;
//                dimGrid.z = 1;
//            } else {
//                dimGrid.x = GRID_DIM_LIMIT;
//                dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
//                dimGrid.z = 1;
//            }

#ifdef PARALLEL

            cudaMemset(pVertexIndexCurrentCountsDevice, -1, NumOfParticles * sizeof(pVertexIndexCurrentCountsDevice[0]));

            SolveConstraintsInParallelPerThread<<<dimGrid, dimBlock>>>(pInvDmDevice,
                                                                pVertexIndicesPerCellDevice,
                                                                pVertexMovedDevice,
                                                                pCellVolumesDevice,
                                                                pParticleMassesDevice,
                                                                pParticlePositionsDevice,
                                                                pParticleEstimatedPositionsDevice,
                                                                pParticleIntermediateEstimatedPositionsDevice,
                                                                pVertexIndexCountsDevice,
                                                                pVertexIndexCurrentCountsDevice,
                                                                pParticleVelocitiesDevice,
                                                                NumOfParticles,
                                                                NumOfCells
                                                                );
#else

            // Partly parallel: Group the cells into groups of disconnected cells,
            // then run the optimisation in parallel for a cell.
            // Each group is run sequentially.
            int offset = 0;

            for (int k = 0; k < DisconnectedCellGroups.size(); ++k)
            {
                numBlocks = (DisconnectedCellGroups[k].size() + dimBlock.x - 1) / dimBlock.x;
                if (numBlocks <= GRID_DIM_LIMIT) {
                    dimGrid.x = numBlocks;
                    dimGrid.y = 1;
                    dimGrid.z = 1;
                } else {
                    dimGrid.x = GRID_DIM_LIMIT;
                    dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
                    dimGrid.z = 1;
                }

                SolveConstraintsPerThread<<<dimGrid, dimBlock>>>(pInvDmDevice,
                                                                 pVertexIndicesPerCellDevice,
                                                                 pVertexMovedDevice,
                                                                 pCellVolumesDevice,
                                                                 pParticleMassesDevice,
                                                                 pParticlePositionsDevice,
                                                                 pParticleEstimatedPositionsDevice,
                                                                 pParticleVelocitiesDevice,
                                                                 pLambdaXDevice,
                                                                 pDisconnectedCellGroupsDevice,
                                                                 NumOfParticles,
                                                                 NumOfCells,
                                                                 DisconnectedCellGroups[k].size(),
                                                                 offset,
                                                                 cuSOLVERHandle,
                                                                 WorkSize,
                                                                 pWork);

                /* Check synchronous errors, i.e. pre-launch */
                err = cudaGetLastError();

                if (cudaSuccess != err)
                {
                    std::cout << "SolveConstraintsPerThread(): CUDA error occured." << std::endl;
                }

                offset += DisconnectedCellGroups[k].size();
            }

//        // Run with 1 thread because this function cannot be parallelised.
//        for (int k = 0; k < NumOfCells; ++k)
//        {
//            SolveConstraintsPerThread<<<1, 1>>>(pInvDmDevice,
//                                                            pVertexIndicesPerCellDevice,
//                                                            pVertexMovedDevice,
//                                                            pCellVolumesDevice,
//                                                            pParticleMassesDevice,
//                                                            pParticlePositionsDevice,
//                                                            pParticleEstimatedPositionsDevice,
//                                                            pParticleVelocitiesDevice,
//                                                            NumOfParticles,
//                                                            NumOfCells,
//                                                            k);
//        }



#endif


#ifdef PARALLEL

//            cudaThreadSynchronize();

            /* Check synchronous errors, i.e. pre-launch */
            err = cudaGetLastError();

            if (cudaSuccess != err)
            {
                std::cout << "SolveConstraintsPerThread(): CUDA error occured." << std::endl;
            }

            // Update intermediate particle states.
            numBlocks = (NumOfParticles + dimBlock.x - 1) / dimBlock.x;
            if (numBlocks <= GRID_DIM_LIMIT) {
                dimGrid.x = numBlocks;
                dimGrid.y = 1;
                dimGrid.z = 1;
            } else {
                dimGrid.x = GRID_DIM_LIMIT;
                dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
                dimGrid.z = 1;
            }

            UpdateAverageParticleStatePerThread<<<dimGrid,dimBlock>>>(pVertexMovedDevice,
                                                               pParticlePositionsDevice,
                                                               pParticleEstimatedPositionsDevice,
                                                               pParticleIntermediateEstimatedPositionsDevice,
                                                               pVertexIndexCountsDevice,
                                                               pParticleVelocitiesDevice,
                                                               NumOfParticles,
                                                               true);

//            cudaThreadSynchronize();

#endif

        }


        // Update particle states.
        numBlocks = (NumOfParticles + dimBlock.x - 1) / dimBlock.x;
        if (numBlocks <= GRID_DIM_LIMIT) {
            dimGrid.x = numBlocks;
            dimGrid.y = 1;
            dimGrid.z = 1;
        } else {
            dimGrid.x = GRID_DIM_LIMIT;
            dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
            dimGrid.z = 1;
        }

#ifdef PARALLEL

        UpdateAverageParticleStatePerThread<<<dimGrid,dimBlock>>>(pVertexMovedDevice,
                                                           pParticlePositionsDevice,
                                                           pParticleEstimatedPositionsDevice,
                                                           pParticleIntermediateEstimatedPositionsDevice,
                                                           pVertexIndexCountsDevice,
                                                           pParticleVelocitiesDevice,
                                                           NumOfParticles,
                                                           false);

#endif

        UpdateParticleStatePerThread<<<dimGrid,dimBlock>>>(pVertexMovedDevice,
                                                           pParticlePositionsDevice,
                                                           pParticleEstimatedPositionsDevice,
                                                           pParticleVelocitiesDevice,
                                                           NumOfParticles,
                                                           pRMSEDevice);

//        cudaThreadSynchronize();

        /* Check synchronous errors, i.e. pre-launch */
        err = cudaGetLastError();

        if (cudaSuccess != err)
        {
            std::cout << "UpdateParticleStatePerThread(): CUDA error occured." << std::endl;
        }


#if 1
        // Compute the energy of the cells.
        numBlocks = (NumOfCells + dimBlock.x - 1) / dimBlock.x;
        if (numBlocks <= GRID_DIM_LIMIT) {
            dimGrid.x = numBlocks;
            dimGrid.y = 1;
            dimGrid.z = 1;
        } else {
            dimGrid.x = GRID_DIM_LIMIT;
            dimGrid.y = (numBlocks + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
            dimGrid.z = 1;
        }

        ComputeCellEnergyPerThread<<<dimGrid,dimBlock>>>(
                pInvDmDevice,
                pVertexIndicesPerCellDevice,
                pCellVolumesDevice,
                pParticleMassesDevice,
                pParticleEstimatedPositionsDevice,
                pEnergyDevice,
                NumOfCells);

        cudaMemcpy(pEnergy, pEnergyDevice, NumOfCells * sizeof(pEnergyDevice[0]), cudaMemcpyDeviceToHost);

        for (unsigned int i = 0; i < NumOfCells; ++i)
        {
            energy += pEnergy[i];
        }

//        std::cout << "Energy: " << energy << std::endl;



//        prevDiffSumsOfEnergy[diffSumsOfEnergyIndex] = fabs(sumOfEnergy - prevSumOfEnergy);

//        // Check if the difference in the energy does not change for a few iterations.
//        bool converged = false;

//        if (fabs(energy - prevEnergy) < 1e-2)
//        {
//            ++smallEnergyDiffCount;
//        }
//        else
//        {
//            smallEnergyDiffCount = 0;
//        }

//        if (smallEnergyDiffCount == 5)
//        {
//            converged = true;
//        }

//        prevEnergy = energy;


#endif

        // Compute RMSE for convergence.
        cudaMemcpy(pRMSE, pRMSEDevice, sizeof(pRMSEDevice[0]), cudaMemcpyDeviceToHost);

        *pRMSE = sqrt(*pRMSE / NumOfParticles);

//        std::cout << "RMSE (deformation simulation): " << *pRMSE << std::endl;

//        if (fabs(*pRMSE - prevRMSE) < 1e-7/* && fabs(*pRMSE) < 1e-4*/)
        if (*pRMSE < 1e-6)
        {
            break;
        }

        // TODO: Temp.
//        prevRMSE = *pRMSE;
        *pRMSE = 0.0f;
        cudaMemcpy(pRMSEDevice, pRMSE, sizeof(pRMSEDevice[0]), cudaMemcpyHostToDevice);

        ++numOfIterations;
    }
    while (numOfIterations < NumOfOuterIterations);
}

/**********/
