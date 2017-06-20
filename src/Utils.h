#ifndef UTILS_H
#define UTILS_H

#include <QOpenGLWidget>
#include <QMatrix4x4>

#include "../../../libs/eigen/Eigen/Dense"

#define EPSILON 1e-6


class Utils
{
public:
    Utils();

    static int Unproject2DPointOnto3D(float x2D, float y2D, float z2D, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport, QVector3D& Point3D);
    static int TriangleRayIntersection(const QVector3D& V1, // Triangle vertices.
                                       const QVector3D& V2,
                                       const QVector3D& V3,
                                       const QVector3D& O, // Ray origin.
                                       const QVector3D& D, // Ray direction.
                                       float* out);
    static float TriangleArea(const Eigen::Vector3f& A, const Eigen::Vector3f& B, const Eigen::Vector3f& C);
    static bool PointOnLineSegment(const Eigen::Vector2f& Point, const Eigen::Vector2f& LineSegmentStart, const Eigen::Vector2f& LineSegmentEnd);
    static float PointLineDistance(const Eigen::Vector3f& PointAOnLine, const Eigen::Vector3f& PointBOnLine, const Eigen::Vector3f& Point, bool LineSegment = true);

    static void ConvertRGBToXYZ(QImage& Image);

    static float MedianAbsoluteDeviation(std::vector<float>& Data);
    static float MeanOfMinDistances(const std::vector<Eigen::Vector3f>& A, const std::vector<Eigen::Vector3f>& B);
    static float ModifiedHausdorffDistance(const std::vector<Eigen::Vector3f>& A, const std::vector<Eigen::Vector3f>& B);

    static void polyfit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order);
    static std::vector<double> polyval(const std::vector<double>& oCoeff, const std::vector<double>& oX);
};

#endif // UTILS_H
