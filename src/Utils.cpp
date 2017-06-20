#include "Utils.h"

#include <iostream>

Utils::Utils()
{

}

int Utils::Unproject2DPointOnto3D(float x2D, float y2D, float z2D, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport, QVector3D& Point3D)
{
    QMatrix4x4 transform = ProjectionMatrix * ModelViewMatrix;
    bool invertible = true;
    transform = transform.inverted(&invertible);

    if (!invertible)
    {
        // The matrix is not invertible.
        return 0;
    }

    // Normalise the coordinates between -1 and 1.
    QVector4D point;
    point.setX((x2D - (float)pViewport[0]) / (float)pViewport[2] * 2.0f - 1.0f);
    point.setY((y2D - (float)pViewport[1]) / (float)pViewport[3] * 2.0f - 1.0f);
    point.setZ(2.0f * z2D - 1.0f);
    point.setW(1.0f);

    point = transform * point;

    if (point.w() == 0.0f)
    {
        return 0;
    }

    point /= point.w();

    Point3D.setX(point.x());
    Point3D.setY(point.y());
    Point3D.setZ(point.z());

    return 1;
}

// Find an intersection between a triangle and ray in 3D (see https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm).
int Utils::TriangleRayIntersection(const QVector3D& V1, // Triangle vertices.
                                   const QVector3D& V2,
                                   const QVector3D& V3,
                                   const QVector3D& O, // Ray origin.
                                   const QVector3D& D, // Ray direction.
                                   float* out)
{
    QVector3D e1, e2;  // Edge1, Edge2.
    QVector3D P, Q, T;
    float det, inv_det, u, v;
    float t;

    // Find vectors for two edges sharing V1
    e1 = V2 - V1;
    e2 = V3 - V1;

    // Begin calculating determinant - also used to calculate u parameter.
    P = QVector3D::crossProduct(D, e2);

    // If determinant is near zero, ray lies in plane of triangle or ray is parallel to plane of triangle
    det = QVector3D::dotProduct(e1, P);

    // NOT CULLING.
    if (det > -EPSILON && det < EPSILON)
    {
        return 0;
    }

    inv_det = 1.f / det;

    // Calculate distance from V1 to ray origin.
    T = O - V1;

    // Calculate u parameter and test bound.
    u = QVector3D::dotProduct(T, P) * inv_det;

    // The intersection lies outside of the triangle.
    if (u < 0.f || u > 1.f)
    {
        return 0;
    }

    // Prepare to test v parameter.
    Q = QVector3D::crossProduct(T, e1);

    // Calculate V parameter and test bound.
    v = QVector3D::dotProduct(D, Q) * inv_det;

    // The intersection lies outside of the triangle.
    if (v < 0.f || u + v  > 1.f)
    {
        return 0;
    }

    t = QVector3D::dotProduct(e2, Q) * inv_det;

    // Ray intersection.
    if (t > EPSILON)
    {
        *out = t;

        return 1;
    }

    // No hit, no win.
    return 0;
}

float Utils::TriangleArea(const Eigen::Vector3f& A, const Eigen::Vector3f& B, const Eigen::Vector3f& C)
{
    float a = (A - B).norm();
    float b = (B - C).norm();
    float c = (C - A).norm();
    float p = (a + b + c) * 0.5f;

    return (sqrt(p * (p - a) * (p - b) * (p - c)));
}

bool Utils::PointOnLineSegment(const Eigen::Vector2f& Point, const Eigen::Vector2f& LineSegmentStart, const Eigen::Vector2f& LineSegmentEnd)
{
    float distPointToStart = (LineSegmentStart - Point).norm();
    float distPointToEnd = (LineSegmentEnd - Point).norm();
    float distStartToEnd = (LineSegmentEnd - LineSegmentStart).norm();

    if (fabs(distPointToStart + distPointToEnd - distStartToEnd) < 0.1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

float Utils::PointLineDistance(const Eigen::Vector3f& PointAOnLine, const Eigen::Vector3f& PointBOnLine, const Eigen::Vector3f& Point, bool LineSegment)
{
    Eigen::Vector3f v = PointBOnLine - PointAOnLine;
    Eigen::Vector3f w = Point - PointAOnLine;

    float c1 = w.dot(v);

    if (LineSegment && c1 <= 0.0f)
    {
        return w.norm();
    }

    float c2 = v.dot(v);

    if (LineSegment && c2 <= c1)
    {
        return (Point - PointBOnLine).norm();
    }

    float b = c1 / c2;
    Eigen::Vector3f point = PointAOnLine + b * v;

    return (Point - point).norm();
//    return ((Point - PointAOnLine).cross(Point - PointBOnLine)).norm() / (PointBOnLine - PointAOnLine).norm();
}

// Reference: http://www.easyrgb.com/index.php?X=MATH&H=02#text2
void Utils::ConvertRGBToXYZ(QImage& Image)
{
    int width = Image.width();
    int height = Image.height();

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            QRgb colour = Image.pixel(col, row);
            float r = (float)qRed(colour) / 255.0f;
            float g = (float)qGreen(colour) / 255.0f;
            float b = (float)qBlue(colour) / 255.0f;

            if (r > 0.04045f)
            {
                r = pow((r + 0.055f) / 1.055f, 2.4f);
            }
            else
            {
                r = r / 12.92f;
            }

            if (g > 0.04045f)
            {
                g = pow((g + 0.055f) / 1.055f, 2.4f);
            }
            else
            {
                g = g / 12.92f;
            }

            if (b > 0.04045f)
            {
                b = pow((b + 0.055f) / 1.055f, 2.4f);
            }
            else
            {
                b = b / 12.92f;
            }

            r = r * 100.0f;
            g = g * 100.0f;
            b = b * 100.0f;

            //Observer. = 2°, Illuminant = D65
            float X = r * 0.4124f + g * 0.3576f + b * 0.1805f;
            float Y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
            float Z = r * 0.0193f + g * 0.1192f + b * 0.9505f;

            // TODO: Temp for making it brighter.
            Y *= 5.5f;

            if (Y > 255.0f)
            {
                Y = 255.0f;
            }

            QColor XYZ(Y, Y, Y);
            Image.setPixel(col, row, XYZ.rgb());
        }
    }
}

// Compute Median Absoute Deviation (MAD).
float Utils::MedianAbsoluteDeviation(std::vector<float>& Data)
{
    std::nth_element(Data.begin(), Data.begin() + Data.size() / 2, Data.end());
    float median = Data[Data.size() / 2];

    std::vector<float> deviations;

    for (float x : Data)
    {
        deviations.push_back(fabs(x - median));
    }

    std::nth_element(deviations.begin(), deviations.begin() + deviations.size() / 2, deviations.end());
    return deviations[deviations.size() / 2];
}

float Utils::MeanOfMinDistances(const std::vector<Eigen::Vector3f>& A, const std::vector<Eigen::Vector3f>& B)
{
    float dist = 0.0f, minDist = 0.0f;
    std::vector<float> distances;

    for (const Eigen::Vector3f& a : A)
    {
        minDist = std::numeric_limits<float>::max();

        for (const Eigen::Vector3f& b : B)
        {
            dist = (a - b).norm();

            if (dist < minDist)
            {
                minDist = dist;
            }
        }

        distances.push_back(minDist);
    }

//    float maxDist = std::numeric_limits<float>::min();

//    for (float distance : distances)
//    {
//        if (distance > maxDist)
//        {
//            maxDist = distance;
//        }
//    }

//    return maxDist;

    float mean = 0.0f;

    for (float d : distances)
    {
        std::cout << d << std::endl;

        mean += d;
    }

    return mean / (float)distances.size();
}

float Utils::ModifiedHausdorffDistance(const std::vector<Eigen::Vector3f>& A, const std::vector<Eigen::Vector3f>& B)
{
    return std::max(MeanOfMinDistances(A, B), MeanOfMinDistances(B, A));
}

// Polynomial curve fitting to points. Reference: http://svn.clifford.at/handicraft/2014/polyfit/polyfit.cc
void Utils::polyfit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order)
{
    Eigen::MatrixXd A(xv.size(), order+1);
    Eigen::VectorXd yv_mapped = Eigen::VectorXd::Map(&yv.front(), yv.size());
    Eigen::VectorXd result;

    assert(xv.size() == yv.size());
    assert(xv.size() >= order+1);

    // create matrix
    for (size_t i = 0; i < xv.size(); i++)
    for (size_t j = 0; j < order+1; j++)
        A(i, j) = pow(xv.at(i), j);

    // solve for linear least squares fit
    result = A.householderQr().solve(yv_mapped);

    coeff.resize(order+1);
    for (size_t i = 0; i < order+1; i++)
        coeff[i] = result[i];
}

// Polynomial evaluation. Reference: http://vilipetek.com/2013/10/07/polynomial-fitting-in-c-using-boost/
/*
    Calculates the value of a polynomial of degree n evaluated at x. The input
    argument pCoeff is a vector of length n+1 whose elements are the coefficients
    in incremental powers of the polynomial to be evaluated.

    param:
        oCoeff			polynomial coefficients generated by polyfit() function
        oX				x axis values

    return:
        Fitted Y values. C++0x-compatible compilers make returning locally
        created vectors very efficient.
*/
std::vector<double> Utils::polyval(const std::vector<double>& oCoeff, const std::vector<double>& oX)
{
    size_t nCount =  oX.size();
    size_t nDegree = oCoeff.size();
    std::vector<double>	oY( nCount );

    for ( size_t i = 0; i < nCount; i++ )
    {
        double nY = 0;
        double nXT = 1;
        double nX = oX[i];

        for ( size_t j = 0; j < nDegree; j++ )
        {
            // multiply current x by a coefficient
            nY += oCoeff[j] * nXT;
            // power up the X
            nXT *= nX;
        }
        oY[i] = nY;
    }

    return oY;
}
