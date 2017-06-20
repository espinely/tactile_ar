#include "Model.h"
#include "Utils.h"

#include <iostream>
#include <fstream>
#include <tr1/functional>
#include <random>

#include <GL/glu.h>

#include <CGAL/ch_graham_andrew.h> // For computing the convex hull.
//#include <CGAL/IO/Polyhedron_iostream.h>

#define RENDERING_FACE_CENTROID 1
#define RENDERING_SAMPLED_POINTS 0

Model::Model()
    : m_Centroid(0.0f, 0.0f, 0.0f),
      m_HasTextures(false),
      m_HasVertexNormals(false),
      m_HasMaterials(false),
      m_HasSelectedVertices(false),
      m_pAlphaShape(NULL),
      m_pOStream(&std::cout)
{
    m_Colour << 1.0f, 1.0f, 1.0f, 1.0f;

//    using namespace std::tr1::placeholders;

    m_ObjParser.geometric_vertex_callback(std::tr1::bind(&Model::GeometricVertexCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2, std::tr1::placeholders::_3));
    m_ObjParser.texture_vertex_callback(std::tr1::bind(&Model::TextureVertexCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2));
    m_ObjParser.vertex_normal_callback(std::tr1::bind(&Model::VertexNormalCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2, std::tr1::placeholders::_3));
    m_ObjParser.face_callbacks(
                std::tr1::bind(&Model::TriangularFaceGeometricVerticesCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2, std::tr1::placeholders::_3),
                std::tr1::bind(&Model::TriangularFaceGeometricVerticesTextureVerticesCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2, std::tr1::placeholders::_3),
                std::tr1::bind(&Model::TriangularFaceGeometricVerticesVertexNormalsCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2, std::tr1::placeholders::_3),
                std::tr1::bind(&Model::TriangularFaceGeometricVerticesTextureVerticesVertexNormalsCallback, this, std::tr1::placeholders::_1, std::tr1::placeholders::_2, std::tr1::placeholders::_3),
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
    );
    m_ObjParser.material_library_callback(std::tr1::bind(&Model::MaterialLibraryCallback, this, std::tr1::placeholders::_1));
    m_ObjParser.material_name_callback(std::tr1::bind(&Model::MaterialNameCallback, this, std::tr1::placeholders::_1));
    m_ObjParser.comment_callback(std::tr1::bind(&Model::CommentCallback, this, std::tr1::placeholders::_1));
}

void Model::CleanUp()
{
    for (Vertex*& pVertex : m_Vertices)
    {
        if (pVertex)
        {
            delete pVertex;
        }
    }

    m_Vertices.clear();

    for (Face*& pFace : m_Faces)
    {
        if (pFace)
        {
            delete pFace;
        }
    }

    m_Faces.clear();

    for (Halfedge*& pHalfedge : m_Halfedges)
    {
        if (pHalfedge)
        {
            delete pHalfedge;
        }
    }

    m_Halfedges.clear();
    m_VertexNormals.clear();
    m_TexCoords.clear();
    m_VertexData.clear();
}

void Model::Load(const QString& FileName)
{
    if(!FileName.isEmpty())
    {
        m_ObjParser.parse(FileName.toUtf8().constData());

//        // Load an off file into a CGAL mesh.
//        std::ifstream in(FileName.toUtf8().constData());
//        in >> m_Polyhedron;

        // TODO: Temp - uncomment later.
//        ComputeCentroid();

        int index = 0;

        for (Vertex*& pVertex : m_Vertices)
        {
            pVertex->_Index = index;

            // Set the vertex normal.
            pVertex->_Normal = m_VertexNormals[index];

            // TODO: Temp - uncomment later.
//             Translate the model to the origin.
//            pVertex->_Pos -= m_Centroid;

            // Convert the metric to metres.
            pVertex->_Pos *= 0.001f;

            ++index;
        }

        // TODO: Temp - uncomment later.
//        m_Centroid.setZero();
        ComputeCentroid();

        ComputeFaceNormals();
        ComputeFaceCentroids();
        ComputeVertexNormals();

        BuildHalfedges();
    }
}

// Build vertex data for vertex buffer object.
void Model::BuildVertexData()
{
    m_VertexData.clear();

    for (Face*& pFace : m_Faces)
    {
        for (int i : {0, 1, 2})
        {
            // Add a vertex.
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Pos[0]);
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Pos[1]);
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Pos[2]);

#if NOT_FOR_CNN

            // Add the face normal in place of the vertex normal to match it to our formulation.
            // Use the vertex normal if the better rendering is needed.
            m_VertexData.push_back(pFace->_Normal[0]);
            m_VertexData.push_back(pFace->_Normal[1]);
            m_VertexData.push_back(pFace->_Normal[2]);

#else

            // Add a vertex normal.
            m_VertexData.push_back(m_VertexNormals[pFace->_VertexNormalIndices[i]][0]);
            m_VertexData.push_back(m_VertexNormals[pFace->_VertexNormalIndices[i]][1]);
            m_VertexData.push_back(m_VertexNormals[pFace->_VertexNormalIndices[i]][2]);

#endif

            // Add a texture coordinate.
            m_VertexData.push_back((m_TexCoords[pFace->_TexCoordIndices[i]])._u);
            m_VertexData.push_back((m_TexCoords[pFace->_TexCoordIndices[i]])._v);

            // Add a variable for vertex selection.
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Selected);

            // Temp for computing reflectance.
            m_VertexData.push_back(pFace->_Centroid[0]);
            m_VertexData.push_back(pFace->_Centroid[1]);
            m_VertexData.push_back(pFace->_Centroid[2]);

            // Temp for vertex colour.
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Colour[0]);
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Colour[1]);
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Colour[2]);
            m_VertexData.push_back(m_Vertices[pFace->_VertexIndices[i]]->_Colour[3]);
        }
    }
}

void Model::Render(void)
{
//        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

//#if (!NOT_FOR_CNN)

    // Scale the texture randomly.
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0, 1);

    static float scale = 1.0f;
    scale = dist(gen);

    float sign = dist(gen);

    if (sign < 0.5f)
    {
        sign = -1.0f;
    }
    else
    {
        sign = 1.0f;
    }

    scale *= sign;

//            static float shiftU0 = 0.0f;
//            static float shiftV0 = 0.0f;
//            static float shiftU1 = 0.0f;
//            static float shiftV1 = 0.0f;
//            shiftU0 = dist(gen);
//            shiftV0 = dist(gen);
//            shiftU1 = dist(gen);
//            shiftV1 = dist(gen);

//#endif

    if (m_HasTextures)
    {
        glEnable(GL_TEXTURE_2D);
//            glBindTexture(GL_TEXTURE_2D, m_Texture.getTexture());
        m_pTexture->bind();
    }

    glBegin(GL_TRIANGLES);

    for (Face*& pFace : m_Faces)
    {
        if (!m_HasVertexNormals)
        {
            glNormal3f(pFace->_Normal[0], pFace->_Normal[1], pFace->_Normal[2]);
        }

        for (int i : {0, 1, 2})
        {
            if (m_HasVertexNormals)
            {
                glNormal3f(m_VertexNormals[pFace->_VertexNormalIndices[i]][0], m_VertexNormals[pFace->_VertexNormalIndices[i]][1], m_VertexNormals[pFace->_VertexNormalIndices[i]][2]);
            }

            if (m_HasTextures)
            {

//#if NOT_FOR_CNN

                glTexCoord2f((m_TexCoords[pFace->_TexCoordIndices[i]])._u * scale, (m_TexCoords[pFace->_TexCoordIndices[i]])._v * scale);

//#else

//                    glTexCoord2f((m_TexCoords[pFace->_TexCoordIndices[i]])._u + shiftU, (m_TexCoords[pFace->_TexCoordIndices[i]])._v + shiftV);

//#endif
            }

            // Render in another colour if the vertex is selected.
            if (m_Vertices[pFace->_VertexIndices[i]]->_Selected)
            {
                // Selected.
                glColor4f(1.0f, 1.0f, 1.0f, m_Colour[3]);
            }
            else
            {
                // Not selected.
                glColor4f(m_Colour[0], m_Colour[1], m_Colour[2], m_Colour[3]);
            }

            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 120.0);

            glVertex3f(m_Vertices[pFace->_VertexIndices[i]]->_Pos[0], m_Vertices[pFace->_VertexIndices[i]]->_Pos[1], m_Vertices[pFace->_VertexIndices[i]]->_Pos[2]);
        }
    }

    glEnd();

    if (m_HasTextures)
    {
        glDisable(GL_TEXTURE_2D);
    }
}

void Model::RenderFixedPipeline()
{
    {
        std::vector<int>::iterator it;

//        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

//#if (!NOT_FOR_CNN)

            // Scale the texture randomly.
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<> dist(0, 1);

            static float scale = 1.0f;
            scale = dist(gen);

            float sign = dist(gen);

            if (sign < 0.5f)
            {
                sign = -1.0f;
            }
            else
            {
                sign = 1.0f;
            }

            scale *= sign;

//            static float shiftU0 = 0.0f;
//            static float shiftV0 = 0.0f;
//            static float shiftU1 = 0.0f;
//            static float shiftV1 = 0.0f;
//            shiftU0 = dist(gen);
//            shiftV0 = dist(gen);
//            shiftU1 = dist(gen);
//            shiftV1 = dist(gen);

//#endif

        if (m_HasTextures)
        {
            glEnable(GL_TEXTURE_2D);
//            glBindTexture(GL_TEXTURE_2D, m_Texture.getTexture());
            m_pTexture->bind();
        }

        glBegin(GL_TRIANGLES);

        for (Face*& pFace : m_Faces)
        {
            if (!m_HasVertexNormals)
            {
                glNormal3f(pFace->_Normal[0], pFace->_Normal[1], pFace->_Normal[2]);
            }

            for (int i : {0, 1, 2})
            {
                if (m_HasVertexNormals)
                {
                    glNormal3f(m_VertexNormals[pFace->_VertexNormalIndices[i]][0], m_VertexNormals[pFace->_VertexNormalIndices[i]][1], m_VertexNormals[pFace->_VertexNormalIndices[i]][2]);
                }

                if (m_HasTextures)
                {

//#if NOT_FOR_CNN

                    glTexCoord2f((m_TexCoords[pFace->_TexCoordIndices[i]])._u * scale, (m_TexCoords[pFace->_TexCoordIndices[i]])._v * scale);

//#else

//                    glTexCoord2f((m_TexCoords[pFace->_TexCoordIndices[i]])._u + shiftU, (m_TexCoords[pFace->_TexCoordIndices[i]])._v + shiftV);

//#endif
                }

                // Render in another colour if the vertex is selected.
                if (m_Vertices[pFace->_VertexIndices[i]]->_Selected)
                {
                    // Selected.
                    glColor4f(1.0f, 1.0f, 1.0f, m_Colour[3]);
                }
                else
                {
                    // Not selected.
                    glColor4f(m_Colour[0], m_Colour[1], m_Colour[2], m_Colour[3]);
                }

                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 120.0);

                glVertex3f(m_Vertices[pFace->_VertexIndices[i]]->_Pos[0], m_Vertices[pFace->_VertexIndices[i]]->_Pos[1], m_Vertices[pFace->_VertexIndices[i]]->_Pos[2]);
            }
        }

        glEnd();

        if (m_HasTextures)
        {
            glDisable(GL_TEXTURE_2D);
        }
    }
}

void Model::RenderContour(void)
{

#ifdef USE_ALPHA_SHAPE

    if (m_Contour.size() == 0)
    {
        return;
    }

#endif

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    glLineWidth(4.0f);


#ifdef USE_ALPHA_SHAPE

    glBegin(GL_LINES);

    for (Segment& segment : m_Contour)
    {
        glVertex2f(segment.source().x(), segment.source().y());
        glVertex2f(segment.target().x(), segment.target().y());

    }

#else

    glBegin(GL_LINE_STRIP);

    for (Point& point : m_ConvexHull)
    {
        glVertex2f(point.x(), point.y());
    }

    glVertex2f(m_ConvexHull.front().x(), m_ConvexHull.front().y());

#endif

    glEnd();

    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

bool Model::FaceFrontFacing(const Face* pFace, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    QVector4D normal, eyeVector;
    QMatrix3x3 normalMat;

    normal.setX(pFace->_Normal[0]);
    normal.setY(pFace->_Normal[1]);
    normal.setZ(pFace->_Normal[2]);
    normal.setW(1.0);

    normalMat = ModelViewMatrix.normalMatrix();
    QMatrix4x4 mat(normalMat(0, 0), normalMat(0, 1), normalMat(0, 2), 0.0,
                   normalMat(1, 0), normalMat(1, 1), normalMat(1, 2), 0.0,
                   normalMat(2, 0), normalMat(2, 1), normalMat(2, 2), 0.0,
                   0.0            , 0.0            , 0.0            , 1.0);

    normal = mat * normal;
    normal /= normal.w();
    QVector3D normal3D(normal.x(), normal.y(), normal.z());
    normal3D = normal3D.normalized();

    eyeVector.setX(pFace->_Centroid[0]);
    eyeVector.setY(pFace->_Centroid[1]);
    eyeVector.setZ(pFace->_Centroid[2]);
    eyeVector.setW(1.0);
    eyeVector = ModelViewMatrix * eyeVector;
    eyeVector /= eyeVector.w();
    QVector3D eyeVector3D(-eyeVector.x(), -eyeVector.y(), -eyeVector.z());
    eyeVector3D = eyeVector3D.normalized();

    if (QVector3D::dotProduct(eyeVector3D, normal3D) <= 0.0)
    {
        // The face is back-facing.
        return false;
    }
    else
    {
        // The face is front-facing.
        return true;
    }
}

// Points2D/3D are the projected 2D points and original 3D points of the face.
std::vector<QVector3D> Model::SamplePointsOnFace(const std::vector<QVector3D>& Points2D, const std::vector<QVector3D>& Points3D, std::vector<QVector3D>& VisiblePixelCoords, const GLfloat* pDepthData, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Check if the face is obstructed by others.
    // And sample visible pixles on the face.

    // Sample points on the face using barycentric coordinate.
    // The number of the samples depends on the area of the triangle.

    // Assumes that the face is triangular.
//    Eigen::Vector2f a(Points2D[0].x(), Points2D[0].y());
//    Eigen::Vector2f b(Points2D[1].x(), Points2D[1].y());
//    Eigen::Vector2f c(Points2D[2].x(), Points2D[2].y());
    QVector3D a = Points2D[0];
    QVector3D b = Points2D[1];
    QVector3D c = Points2D[2];

    float area = fabs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) * 0.5f;
//    std::cout << "Area: " << area << std::endl;

    // Random uniform sampling on a triangle (see http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle).
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0, 1);

    std::vector<QVector3D> sampledPoints3D;
    QVector3D pixelCoord;

    // Sample up to 30 points on a face.
    int numOfSamples = 30;

    if ((int)area > numOfSamples)
    {
        area = numOfSamples;
    }

    for (int i = 0; i < (int)area; ++i)
    {
        int numOfIterations = 0;

        // Resample the point if it is out of the viewport.
        do
        {
            if (numOfIterations == 10)
            {
                break;
            }

            float r1 = dist(gen);
            float r2 = dist(gen);
            float sqrtR1 = sqrt(r1);

            pixelCoord = (1.0f - sqrtR1) * a + (sqrtR1 * (1.0f - r2)) * b + (r2 * sqrtR1) * c;

            ++numOfIterations;
        }
        while (pixelCoord.x() < 0.0 || pixelCoord.x() >= pViewport[2] || pixelCoord.y() < 0.0 || pixelCoord.y() >= pViewport[3]);

        if (numOfIterations == 10)
        {
            // Give up the sampling.
            continue;
        }

        // Check if the pixel is covered by others
        // by casting a ray to the face and comparing the z value to the one that are read from the frame buffer.
        QVector3D near, far;
        Utils::Unproject2DPointOnto3D(pixelCoord.x(), pixelCoord.y(), 0.0f, ModelViewMatrix, ProjectionMatrix, pViewport, near);
        Utils::Unproject2DPointOnto3D(pixelCoord.x(), pixelCoord.y(), 1.0f, ModelViewMatrix, ProjectionMatrix, pViewport, far);

        QVector3D direction = far - near;
        QVector3D intersection;
        direction.normalize();
        float t = 0.0f;

        if (Utils::TriangleRayIntersection(Points3D[0], Points3D[1], Points3D[2], near, direction, &t) == 1)
        {
            // There is an intersection.
            intersection = near + t * direction;

            QVector3D point = ProjectPointOnto2D(intersection, ModelViewMatrix, ProjectionMatrix, pViewport);

//            std::cout << "pixel depth: " << pDepthData[(int)pixelCoord.x() + (int)pixelCoord.y() * pViewport[2]] << ", face depth: " << point.z() << std::endl;

            if (fabs(pDepthData[(int)pixelCoord.x() + (int)pixelCoord.y() * pViewport[2]] - point.z()) >= 1e-5 / (point.z() * point.z()))
            {
                // The pixel is covered.
            }
            else
            {
                VisiblePixelCoords.push_back(pixelCoord);
                sampledPoints3D.push_back(intersection);
            }
        }
    }

    return sampledPoints3D;
}

bool Model::VertexVisible(const Vertex* pVertex, const GLfloat* pDepthData, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Check if the vertex is obstructed by faces.

    // Project the vertex onto 2D.
    QVector3D point3D(pVertex->_Pos[0], pVertex->_Pos[1], pVertex->_Pos[2]);
    QVector3D point2D = ProjectPointOnto2D(point3D, ModelViewMatrix, ProjectionMatrix, pViewport);

    // Check if the pixel is covered by faces
    // by comparing the z value to the one that are read from the frame buffer.
    if (fabs(pDepthData[(int)point2D.x() + (int)point2D.y() * pViewport[2]] - point2D.z()) >= 1e-5)// / (point2D.z() * point2D.z()))
    {
        // The pixel is covered.
        return false;
    }
    else
    {
        return true;
    }
}

void Model::RenderFaces(const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport, bool RenderingVisibleFaceOnly)
{
    std::vector<QVector3D> points2D, points3D, visiblePixelCoords;
    QVector3D point2D, point3D;
    GLfloat pDepthData[pViewport[2] * pViewport[3]];

    glReadPixels(0, 0, pViewport[2], pViewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, pDepthData);

    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, pViewport[2], 0, pViewport[3], -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    for (Face*& pFace : m_Faces)
    {
        pFace->_Sampled = false;

        if (RenderingVisibleFaceOnly)
        {
            if (!FaceFrontFacing(pFace, ModelViewMatrix, ProjectionMatrix, pViewport))
            {
                continue;
            }

            for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
            {
                point3D = QVector3D(m_Vertices[*it]->_Pos[0], m_Vertices[*it]->_Pos[1], m_Vertices[*it]->_Pos[2]);
                point2D = ProjectPointOnto2D(point3D, ModelViewMatrix, ProjectionMatrix, pViewport);

                points3D.push_back(point3D);
                points2D.push_back(point2D);
            }

            // Check if the face is completely out of the viewport.
            bool outOfViewport = true;

            for (QVector3D& point : points2D)
            {
                if (point.x() >= 0.0 && point.x() < pViewport[2] && point.y() >= 0.0 && point.y() < pViewport[3])
                {
                    outOfViewport = false;

                    break;
                }
            }

            if (outOfViewport)
            {
                points2D.clear();
                points3D.clear();

                continue;
            }

            SamplePointsOnFace(points2D, points3D, visiblePixelCoords, pDepthData, ModelViewMatrix, ProjectionMatrix, pViewport);

            if (visiblePixelCoords.size() == 0)
            {
                points2D.clear();
                points3D.clear();

                continue;
            }

            pFace->_Sampled = true;
        }
        else
        {
            for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
            {
                point3D = QVector3D(m_Vertices[*it]->_Pos[0], m_Vertices[*it]->_Pos[1], m_Vertices[*it]->_Pos[2]);
                point2D = ProjectPointOnto2D(point3D, ModelViewMatrix, ProjectionMatrix, pViewport);

                points3D.push_back(point3D);
                points2D.push_back(point2D);
            }

            // Check if the face is completely out of the viewport.
            bool outOfViewport = true;

            for (QVector3D& point : points2D)
            {
                if (point.x() >= 0.0 || point.x() < pViewport[2] || point.y() >= 0.0 || point.y() < pViewport[3])
                {
                    outOfViewport = false;

                    break;
                }
            }

            if (outOfViewport)
            {
                points2D.clear();
                points3D.clear();

                continue;
            }
        }

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glBegin(GL_LINE_STRIP);

        for (QVector3D& point : points2D)
        {
            glVertex2f(point.x(), point.y());
        }

        glVertex2f(points2D.front().x(), points2D.front().y());
        glEnd();

        // Render the face centroid.
#if RENDERING_FACE_CENTROID

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glBegin(GL_POINTS);

        Vertex vertex(pFace->_Centroid[0], pFace->_Centroid[1], pFace->_Centroid[2]);
        Point point = ProjectVertexOnto2D(&vertex, ModelViewMatrix, ProjectionMatrix, pViewport);

        glVertex2f(point.x(), point.y());
        glEnd();

#endif

        // Render the sampled points.
#if RENDERING_SAMPLED_POINTS

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glPointSize(2.0);

        glBegin(GL_POINTS);

        for (QVector3D& point : visiblePixelCoords)
        {
            glVertex2f(point.x(), point.y());
        }

        glEnd();
        glPointSize(1.0);

#endif

        points2D.clear();
        points3D.clear();
        visiblePixelCoords.clear();
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();
}

void Model::BuildHalfedges(void)
{
    m_Halfedges.clear();

    // For getting the opposite halfedges.
    std::map<std::pair<unsigned int, unsigned int>, Halfedge*> edges;

    for (Face*& pFace : m_Faces)
    {
        int numOfHalfedges = 0;

        for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
        {
            Halfedge* pHalfedge = new Halfedge;

            int u = 0, v = 0;

            if (*it == pFace->_VertexIndices.back())
            {
                pHalfedge->_pVertex = m_Vertices[*pFace->_VertexIndices.begin()];

                v = *pFace->_VertexIndices.begin();
            }
            else
            {
                pHalfedge->_pVertex = m_Vertices[*(it + 1)];

                v = *(it + 1);
            }

            pHalfedge->_pFace = pFace;
            m_Vertices[*it]->_pHalfedge = pHalfedge;
            u = *it;

            m_Halfedges.push_back(pHalfedge);

//            if (*it == pFace->_VertexIndices.front())
//            {
//                m_Vertices[pFace->_VertexIndices.back()]->_pHalfedge = m_Halfedges.back();

//                u = pFace->_VertexIndices.back();
//            }
//            else
//            {
//                m_Vertices[*(it - 1)]->_pHalfedge = m_Halfedges.back();

//                u = *(it - 1);
//            }

            edges[std::pair<unsigned int, unsigned int>(u, v)] = pHalfedge;

            ++numOfHalfedges;
        }

        for (std::vector<Halfedge*>::iterator it = m_Halfedges.end() - numOfHalfedges; it != m_Halfedges.end(); ++it)
        {
            if (it == m_Halfedges.end() - 1)
            {
                (*it)->_pPrev = *(it - 1);
                (*it)->_pNext = *(m_Halfedges.end() - numOfHalfedges);
            }
            else if (it == m_Halfedges.end() - numOfHalfedges)
            {
                (*it)->_pPrev = *(m_Halfedges.end() - 1);
                (*it)->_pNext = *(it + 1);
            }
            else
            {
                (*it)->_pPrev = *(it - 1);
                (*it)->_pNext = *(it + 1);
            }
        }

        pFace->_pHalfedge = *(m_Halfedges.end() - numOfHalfedges);
    }

    // Get the opposite halfedges.
    for (Face*& pFace : m_Faces)
    {
        for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
        {
            int u = 0, v = 0;

            if (*it == pFace->_VertexIndices.back())
            {
                v = *pFace->_VertexIndices.begin();
            }
            else
            {
                v = *(it + 1);
            }

            u = *it;

//            if (*it == pFace->_VertexIndices.front())
//            {
//                u = pFace->_VertexIndices.back();
//            }
//            else
//            {
//                u = *(it - 1);
//            }

            if (edges.find(std::pair<unsigned int, unsigned int>(v, u)) != edges.end())
            {
               edges[std::pair<unsigned int, unsigned int>(u, v)]->_pOpposite = edges[std::pair<unsigned int, unsigned int>(v, u)];
               edges[std::pair<unsigned int, unsigned int>(v, u)]->_pOpposite = edges[std::pair<unsigned int, unsigned int>(u, v)];
            }
        }
    }

//    int i = 0;

//    for (Halfedge*& pHalfedge : m_Halfedges)
//    {
//        if (!pHalfedge->_pOpposite)
//        {
//            std::cout << i << std::endl;
//        }

//        ++i;
//    }
}

void Model::ComputeCentroid(void)
{
    m_Centroid.setZero();

    for (Vertex*& pVertex : m_Vertices)
    {
        m_Centroid += pVertex->_Pos;
    }

    m_Centroid /= m_Vertices.size();

//    std::cout << "Centroid of the model: " << m_Centroid << std::endl;
}

void Model::ComputeFaceNormals(void)
{
    for (Face*& pFace : m_Faces)
    {
        pFace->_Normal = (m_Vertices[pFace->_VertexIndices[1]]->_Pos - m_Vertices[pFace->_VertexIndices[0]]->_Pos).cross(m_Vertices[pFace->_VertexIndices[2]]->_Pos - m_Vertices[pFace->_VertexIndices[0]]->_Pos);
        pFace->_Normal.normalize();
    }
}

// Compute vertex normals by a simple average of the normals of the incident faces to each vertex.
void Model::ComputeVertexNormals(void)
{
    m_VertexNormals.clear();

    for (Vertex*& pVertex : m_Vertices)
    {
        pVertex->_Normal.setZero();
    }

    for (Face*& pFace : m_Faces)
    {
        for (int i : { 0, 1, 2 })
        {
            m_Vertices[pFace->_VertexIndices[i]]->_Normal += pFace->_Normal;
        }
    }

    for (Vertex*& pVertex : m_Vertices)
    {
        pVertex->_Normal.normalize();

        m_VertexNormals.push_back(pVertex->_Normal);
    }
}

void Model::ComputeFaceCentroids(void)
{
    for (Face*& pFace : m_Faces)
    {
        pFace->_Centroid = (m_Vertices[pFace->_VertexIndices[0]]->_Pos + m_Vertices[pFace->_VertexIndices[1]]->_Pos + m_Vertices[pFace->_VertexIndices[2]]->_Pos) / 3.0f;
    }
}

int Model::ClosestVertexIndexToPoint(const Eigen::Vector2f& APoint, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, GLint* pViewport, float& Distance)
{
    int index = -1;
    int i = 0;
    float minDist = std::numeric_limits<float>::max();
    Eigen::Vector2f windowPos;
    Point point2D;

    for (Vertex*& pVertex : m_Vertices)
    {
        point2D = ProjectVertexOnto2D(pVertex, ModelViewMatrix, ProjectionMatrix, pViewport);
        windowPos << point2D.x(), point2D.y();

        Eigen::Vector2f diff = windowPos - APoint;
        float dist = diff.norm();

        if (dist < minDist)
        {
            minDist = dist;
            index = i;
        }

        ++i;
    }

    Distance = minDist;

    return index;
}

int Model::ClosestVertexIndexToPointFixedPipeline(const Eigen::Vector2f& APoint, GLdouble* pModelViewMatrix, GLdouble* pProjectionMatrix, GLint* pViewport)
{
    int index = -1;
    int i = 0;
    float minDist = std::numeric_limits<float>::max();
    GLdouble x = 0.0, y = 0.0, z = 0.0;
    Eigen::Vector2f windowPos;

    for (Vertex*& pVertex : m_Vertices)
    {
        gluProject((GLdouble)pVertex->_Pos[0], (GLdouble)pVertex->_Pos[1], (GLdouble)pVertex->_Pos[2], pModelViewMatrix, pProjectionMatrix, pViewport, &x, &y, &z);
        windowPos << x, y;

        Eigen::Vector2f diff = windowPos - APoint;
        float dist = diff.norm();

        if (dist < minDist)
        {
            minDist = dist;
            index = i;
        }

        ++i;
    }

    return index;
}

void Model::TranslateSelectedVertices(const Eigen::Vector3f& Vector)
{
    for (Vertex*& pVertex : m_Vertices)
    {
        if (pVertex->_Selected)
        {
            pVertex->_Pos += Vector;
        }
    }
}

void Model::RotateSelectedVertices(const Eigen::Vector3f& Vector, const Eigen::Vector3f& VectorPerpendicular)
{
    // Compute the centroid of the selected vertices.
    Eigen::Vector3f centroid(0.0, 0.0, 0.0);
    int count = 0;

    for (Vertex*& pVertex : m_Vertices)
    {
        if (pVertex->_Selected)
        {
            centroid += pVertex->_Pos;

            ++count;
        }
    }

    centroid /= count;

    // Rotate the vertices.
    for (Vertex*& pVertex : m_Vertices)
    {
        if (pVertex->_Selected)
        {
            Eigen::Vector3f axis = (Vector.normalized()).cross(VectorPerpendicular);
            axis.normalize();

            Eigen::AngleAxisf angleAxis(Vector.norm(), axis);
            Eigen::Quaternionf quaternion(angleAxis);

            Eigen::Affine3f transform;
            transform = quaternion;

            Eigen::Vector3f pos = pVertex->_Pos;
            pos -= centroid;
            pos = transform * pos;
            pVertex->_Pos = pos + centroid;
        }
    }
}

void Model::UnselectAllVertices(void)
{
    for (Vertex*& pVertex : m_Vertices)
    {
        pVertex->_Selected = false;
    }

    m_HasSelectedVertices = false;
}

void Model::UnsampleAllFaces(void)
{
    for (Face* pFace : m_Faces)
    {
        pFace->_Sampled = false;
    }
}

void Model::OneRingNeighbours(Vertex* pVertex, std::vector<Vertex*>& Neighbours)
{
    Halfedge* pCurrentHalfedge = pVertex->_pHalfedge;

    do
    {
        Neighbours.push_back(pVertex);

        do
        {
            Neighbours.push_back(pCurrentHalfedge->_pVertex);
            pCurrentHalfedge = pCurrentHalfedge->_pNext;
        }
        while (pCurrentHalfedge && pCurrentHalfedge->_pVertex != pVertex);

        pCurrentHalfedge = pCurrentHalfedge->_pOpposite;
    }
    while (pCurrentHalfedge && pCurrentHalfedge != pVertex->_pHalfedge);
}

void Model::SelectVertexWithOneRingNeighbours(Vertex* pVertex)
{
    Halfedge* pCurrentHalfedge = pVertex->_pHalfedge;
    pVertex->_Selected = true;

    do
    {
        do
        {
            pCurrentHalfedge->_pVertex->_Selected = true;
            pCurrentHalfedge = pCurrentHalfedge->_pNext;
        }
        while (pCurrentHalfedge->_pVertex != pVertex);

        pCurrentHalfedge = pCurrentHalfedge->_pOpposite;
    }
    while (pCurrentHalfedge != pVertex->_pHalfedge);
}

void Model::OneRingNeighbourFaces(Vertex* pVertex, std::vector<Face*>& Faces)
{
    Halfedge* pCurrentHalfedge = pVertex->_pHalfedge;

    do
    {
        Faces.push_back(pCurrentHalfedge->_pFace);

        do
        {
            pCurrentHalfedge = pCurrentHalfedge->_pNext;
        }
        while (pCurrentHalfedge->_pVertex != pVertex);

        pCurrentHalfedge = pCurrentHalfedge->_pOpposite;
    }
    while (pCurrentHalfedge != pVertex->_pHalfedge);
}

QVector3D Model::ProjectPointOnto2D(const QVector3D& Pos, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    QVector4D vertex;
    GLdouble x, y, z;
    QMatrix4x4 transform = ProjectionMatrix * ModelViewMatrix;

    vertex[0] = Pos.x();
    vertex[1] = Pos.y();
    vertex[2] = Pos.z();
    vertex[3] = 1.0;

    vertex = transform * vertex;

    if (fabs(vertex[3]) >= 1e-12)
    {
        vertex /= vertex[3];
    }

    x = vertex[0] * 0.5 + 0.5;
    y = vertex[1] * 0.5 + 0.5;
    z = vertex[2] * 0.5 + 0.5;

    x = x * pViewport[2] + pViewport[0];
    y = y * pViewport[3] + pViewport[1];

//        std::cout << x << ", " << y << ", " << z << std::endl;

    return QVector3D(x, y, z);
}

Point Model::ProjectVertexOnto2D(const Vertex* pVertex, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    QVector4D vertex;
    GLdouble x, y, z;
    QMatrix4x4 transform = ProjectionMatrix * ModelViewMatrix;

    vertex[0] = pVertex->_Pos[0];
    vertex[1] = pVertex->_Pos[1];
    vertex[2] = pVertex->_Pos[2];
    vertex[3] = 1.0;

    vertex = transform * vertex;

    if (fabs(vertex[3]) >= 1e-12)
    {
        vertex /= vertex[3];
    }

    x = vertex[0] * 0.5 + 0.5;
    y = vertex[1] * 0.5 + 0.5;
    z = vertex[2] * 0.5 + 0.5;

    x = x * pViewport[2] + pViewport[0];
    y = y * pViewport[3] + pViewport[1];

//        std::cout << x << ", " << y << ", " << z << std::endl;

    return Point(x, y);
}

// Project the vertices of the model in 3D onto 2D.
void Model::ProjectVerticesOnto2D(std::vector<Point>& Points2D, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    Point point;

    for (Vertex*& pVertex : m_Vertices)
    {
        point = ProjectVertexOnto2D(pVertex, ModelViewMatrix, ProjectionMatrix, pViewport);

//        // Add a point only if it is within the viewport.
//        if (point.x() >= 0 && point.x() < pViewport[2] && point.y() >= 0 && point.y() < pViewport[3])
        {
            Points2D.push_back(point);
        }
    }
}

// Reference: https://people.eecs.berkeley.edu/~jrs/meshpapers/MeyerDesbrunSchroderBarr.pdf
void Model::ComputeGaussianCurvature(void)
{
    // Compute the Gaussian curvature of the mesh.
    Eigen::Vector3f a, b, c;
    std::vector<Vertex*> neighbours;
    std::vector<float> angles; // Vector of the two inner angles for each triangle.

    m_GaussianCurvatures.clear();

//    Halfedge* pHalfedge = NULL;

    int count = 0;

    std::cout << "#verts: " << m_Vertices.size() << std::endl;

    for (Vertex* pVertex : m_Vertices)
    {
        OneRingNeighbours(pVertex, neighbours);

        float area = 0.0f, sumOfAngles = 0.0f;
        Eigen::Vector3f sum;

        for (unsigned int i = 0; i < neighbours.size(); i += 3)
        {
//            std::cout << neighbours[i]->_Pos[0] << ", " << neighbours[i]->_Pos[1] << ", " << neighbours[i]->_Pos[2] << std::endl;
//            std::cout << neighbours[i + 1]->_Pos[0] << ", " << neighbours[i + 1]->_Pos[1] << ", " << neighbours[i + 1]->_Pos[2] << std::endl;
//            std::cout << neighbours[i + 2]->_Pos[0] << ", " << neighbours[i + 2]->_Pos[1] << ", " << neighbours[i + 2]->_Pos[2] << std::endl;
//            std::cout << neighbours[i]->_Index << ", " << neighbours[i + 1]->_Index << ", " << neighbours[i + 2]->_Index << std::endl;

            a = neighbours[i]->_Pos;
            b = neighbours[i + 1]->_Pos;
            c = neighbours[i + 2]->_Pos;

            sumOfAngles += acos(((b - a).normalized()).dot((c - a).normalized()));


            Eigen::Vector3f vec0 = a - b;
            Eigen::Vector3f vec1 = c - b;
            vec0.normalize();
            vec1.normalize();
            float angle = acos(vec0.dot(vec1));
            angles.push_back(angle);

            vec0 = a - c;
            vec1 = b - c;
            vec0.normalize();
            vec1.normalize();

            angle = acos(vec0.dot(vec1));
            angles.push_back(angle);
        }

        int angleIndex = 0;
        float alpha = 0.0f, beta = 0.0f, betaForArea = 0.0f, cotAlpha = 0.0f, cotBeta = 0.0f, cotBetaForArea = 0.0f;

        for (unsigned int i = 0; i < neighbours.size(); i += 3)
        {
            a = neighbours[i]->_Pos;
            b = neighbours[i + 1]->_Pos;
            c = neighbours[i + 2]->_Pos;

            alpha = angles[angleIndex];

            if (i == neighbours.size() - 3)
            {
                beta = angles[1];
            }
            else
            {
                beta = angles[angleIndex + 3];
            }

            betaForArea = angles[angleIndex + 1];

            cotAlpha = cos(alpha) / sin(alpha); //1.0f / tan(alpha);
            cotBeta = cos(beta) / sin(beta); //1.0f / tan(beta);
            cotBetaForArea = cos(betaForArea) / sin(betaForArea); //1.0f / tan(betaForArea);

//            sum += (cotAlpha + cotBeta) * (a - c);

            if (angles[angleIndex] < M_PI * 0.5f && angles[angleIndex + 1] < M_PI * 0.5f && M_PI - (angles[angleIndex] + angles[angleIndex + 1]) < M_PI * 0.5f)
            {
                // The triangle is non-obtuse.
                area += (cotAlpha * (c - a).squaredNorm() + cotBetaForArea * (b - a).squaredNorm()) / 8.0f;
            }
            else
            {
                if (angles[angleIndex] + angles[angleIndex + 1] < M_PI * 0.5f)
                {
                    // The triangle is obtuse and the angle at the current vertex is obtuse.
                    area += Utils::TriangleArea(a, b, c) * 0.5f;
                }
                else
                {
                    // The triangle is obtuse and the angle at a neighbouring vertex is obtuse.
                    area += Utils::TriangleArea(a, b, c) * 0.25f;
                }
            }

//                            area += ((b - a).cross(c - a)).norm() * 0.5f;

            angleIndex += 2;
        }

//        std::cout << std::endl << std::endl;

        sumOfAngles = 2.0f * M_PI - sumOfAngles;
        sumOfAngles /= area;

        sumOfAngles = fabs(sumOfAngles);

//        float lb = ((1.0f / (2.0f * area)) * sum).norm() * 0.5f;



//        if (lb > 7.0f)
//        {
//            lb = 7.0f;
//        }

        m_GaussianCurvatures.push_back(sumOfAngles);
//          m_GaussianCurvatures.push_back(lb);

        neighbours.clear();
        angles.clear();
    }

    // Filter out outliers.
    // Compute the Median Absolute Deviation.
    std::vector<float> curvatures = m_GaussianCurvatures;
    float MAD = Utils::MedianAbsoluteDeviation(curvatures);
//    std::cout << "MAD: " << MAD << std::endl;

    MAD *= 1.4826f;

    curvatures = m_GaussianCurvatures;
    std::nth_element(curvatures.begin(), curvatures.begin() + curvatures.size() / 2, curvatures.end());
    float median = curvatures[curvatures.size() / 2];

    for (float& curvature : m_GaussianCurvatures)
    {
        if (curvature > 2.0f * MAD + median)
        {
            curvature = 2.0f * MAD + median;

//            std::cout << curvature << std::endl;
        }
    }

    // For displaying the Gaussian curvature.
    std::vector<float>::iterator maxIt = std::max_element(m_GaussianCurvatures.begin(), m_GaussianCurvatures.end());
    float max = *maxIt;

//    std::cout << "Max Gaussian curvature: " << max << std::endl;

    int i = 0;

    for (Vertex* pVertex : m_Vertices)
    {
        Eigen::Vector4f colour;
        float gaussianCurvature = m_GaussianCurvatures[i] / max;

        colour << 1.0f - gaussianCurvature, 0.0f, gaussianCurvature, 1.0f;

        if (gaussianCurvature <= 0.5f)
        {
            colour[1] = 2.0f * colour[2];
        }
        else if (gaussianCurvature > 0.5f)
        {
            colour[1] = 2.0f * colour[0];
        }

#if 0 // Not change the colour for now.

        pVertex->_Colour = colour;

#endif

        if (gaussianCurvature >= 0.7f)
        {
            pVertex->_HighCurvature = true;
        }
        else
        {
            pVertex->_HighCurvature = false;
        }

        ++i;
    }
}

void Model::ExtractVerticesOnContour(const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Project vertices of the model on 2D and extract vertices on the contour using the alpha shape or convex hull.
    // First, project vertices in 3D on 2D.
    std::vector<Point> points;
    ProjectVerticesOnto2D(points, ModelViewMatrix, ProjectionMatrix, pViewport);

#ifdef USE_ALPHA_SHAPE

    // Create the alpha shape.
    if (m_pAlphaShape)
    {
        delete m_pAlphaShape;
        m_pAlphaShape = NULL;
    }

    m_pAlphaShape = new Alpha_shape_2(points.begin(), points.end(), FT(100), Alpha_shape_2::REGULARIZED);

//    int i = 0;

//    for (Alpha_shape_2::Vertex_iterator it = A.vertices_begin(); it != A.vertices_end(); ++it)
//    {
//        if (i <= 1)
//        {
//            std::cout << it->point().x() << ", " << it->point().y() << std::endl;
//        }

//        ++i;
//    }

//    std::cout << i << std::endl;

    std::cout << "Alpha Shape computed" << std::endl;
    Alpha_shape_2::Alpha_iterator optimalAlpha = m_pAlphaShape->find_optimal_alpha(1);
    std::cout << "Optimal alpha: " << *optimalAlpha <<std::endl;

    m_pAlphaShape->set_alpha((*optimalAlpha)/* * 2*/);

    m_Contour.clear();

    alpha_edges(*m_pAlphaShape, std::back_inserter(m_Contour));
    std::cout << m_Contour.size() << " alpha shape edges" << std::endl;

    // TODO: retain only the longest segment in the contour because there can be some segements corresponding to holes.

#else

    // Compute the convex hull.
    m_ConvexHull.clear();
    CGAL::ch_graham_andrew(points.begin(), points.end(), std::inserter(m_ConvexHull, m_ConvexHull.begin()));

#endif

}

void Model::ExtractVerticesOnContourFixedPipeline(GLdouble* pModelViewMatrix, GLdouble* pProjectionMatrix, GLint* pViewport)
{
    // Project vertices of the model on 2D and extract vertices on the contour using the alpha shape.
    // First, project vertices in 3D on 2D.
    std::vector<Eigen::Vector2f> points2D;
    std::vector<Point> points;

    for (Vertex*& pVertex : m_Vertices)
    {
        Eigen::Vector2f windowPos;
        GLdouble x, y, z;

        gluProject((GLdouble)pVertex->_Pos[0], (GLdouble)pVertex->_Pos[1], (GLdouble)pVertex->_Pos[2], pModelViewMatrix, pProjectionMatrix, pViewport, &x, &y, &z);
        windowPos << x, y;
        points2D.push_back(windowPos);

        points.push_back(Point(x, y));
    }

    // Create the alpha shape.
    if (m_pAlphaShape)
    {
        delete m_pAlphaShape;
        m_pAlphaShape = NULL;
    }

    m_pAlphaShape = new Alpha_shape_2(points.begin(), points.end(), FT(100), Alpha_shape_2::REGULARIZED);

//    int i = 0;

//    for (Alpha_shape_2::Vertex_iterator it = A.vertices_begin(); it != A.vertices_end(); ++it)
//    {
//        if (i <= 1)
//        {
//            std::cout << it->point().x() << ", " << it->point().y() << std::endl;
//        }

//        ++i;
//    }

//    std::cout << i << std::endl;

    std::cout << "Alpha Shape computed" << std::endl;
    Alpha_shape_2::Alpha_iterator optimalAlpha = m_pAlphaShape->find_optimal_alpha(1);
    std::cout << "Optimal alpha: " << *optimalAlpha <<std::endl;

    m_pAlphaShape->set_alpha((*optimalAlpha) * 2);

    m_Contour.clear();

    alpha_edges(*m_pAlphaShape, std::back_inserter(m_Contour));
    std::cout << m_Contour.size() << " alpha shape edges" << std::endl;

    // TODO: retain only the longest segment in the contour because there can be some segements corresponding to holes.


#if 0

    // Get the camera position.
    GLfloat x, y, z;
    gluUnProject((m_Viewport[2] - m_Viewport[0]) * 0.5 , (m_Viewport[3] - m_Viewport[1]) * 0.5, 0.0, m_ModelViewMatrix, m_ProjectionMatrix, m_Viewport, &x, &y, &z);
    Eigen::Vector3f cameraPos(x, y, z);

    // A vertex is on the contour if one of its face is facing the camera and another is not facing the camera.
    for (Halfedge*& pHalfedge : m_Halfedges)
    {
        Eigen::Vector3f vec = cameraPos - pHalfedge->_pVertex->_Pos;
        vec.normalize();
        float dotA = (pHalfedge->_pFace->_Normal).dot(vec);
        float dotB = (pHalfedge->_pOpposite->_pFace->_Normal).dot(vec);

        if ((dotA < 0.0f && dotB >= 0.0f) || (dotA >= 0.0f && dotB < 0.0f))
        {
            // Todo: temp.
            // Select the vertex and its one-ring neighbours.
            SelectVertexWithOneRingNeighbours(pHalfedge->_pVertex);
        }
    }

#endif

}

void Model::SelectVerticesOnContour(const std::vector<Eigen::Vector2f>& Contour, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Project vertices of the model on 2D.
    std::vector<Point> points2D;
    ProjectVerticesOnto2D(points2D, ModelViewMatrix, ProjectionMatrix, pViewport);

    // Choose vertices that are closest to the points on the contour.
    for (std::vector<Eigen::Vector2f>::const_iterator itPoint = Contour.begin(); itPoint != Contour.end(); ++itPoint)
    {
        int index = 0;
        int minIndex = 0;
        Eigen::Vector2f p;
        float minDist = std::numeric_limits<float>::max();

        for (Point& point2D : points2D)
        {
            p << point2D.x(), point2D.y();
            float dist = (*itPoint - p).norm();

            if (dist < minDist)
            {
                minDist = dist;
                minIndex = index;
            }

            ++index;
        }

        //                        SelectVertexWithOneRingNeighbours(m_Vertices[index]);
        m_Vertices[minIndex]->_Selected = true;
    }
}

void Model::SelectVerticesOnContour(const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Project vertices of the model on 2D.
    std::vector<Point> points2D;
    ProjectVerticesOnto2D(points2D, ModelViewMatrix, ProjectionMatrix, pViewport);

    // Select vertices close to the convex hull.
    Point p0, p1;
    Eigen::Vector2f vec, point;
    float length = 0.0f;

    // Sample points on the convex hull and choose vertices that are closest to the points.
    for (std::vector<Point>::iterator itPoint = m_ConvexHull.begin(); itPoint != m_ConvexHull.end(); ++itPoint)
    {
        p0 = *itPoint;

        if (itPoint < m_ConvexHull.end() - 1)
        {
            p1 = *(itPoint + 1);
        }
        else
        {
            p1 = *m_ConvexHull.begin();
        }

        point << p0.x(), p0.y();
        vec << p1.x() - p0.x(), p1.y() - p0.y();
        length = vec.norm();

        if (length < 1.0f)
        {
            length = 1.0f;
        }

        for (int i = 0; i < (int)floor(length); ++i)
        {
            if (i > 0)
            {
                point += vec / length;
            }

            int index = 0;
            int minIndex = 0;
            Eigen::Vector2f p;
            float minDist = std::numeric_limits<float>::max();

            for (Point& point2D : points2D)
            {
                p << point2D.x(), point2D.y();
                float dist = (point - p).norm();

                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = index;
                }

                ++index;
            }

            //                        SelectVertexWithOneRingNeighbours(m_Vertices[index]);
            m_Vertices[minIndex]->_Selected = true;
        }
    }
}

void Model::SelectVerticesOnContour(const std::vector<Eigen::Vector2f>& Contour, const QRectF& VertexSelectionRect, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Project vertices of the model on 2D.
    std::vector<Point> points2D;
    ProjectVerticesOnto2D(points2D, ModelViewMatrix, ProjectionMatrix, pViewport);

    // Choose vertices that are closest to the points on the contour only within the vertex selection rect.
    for (std::vector<Eigen::Vector2f>::const_iterator itPoint = Contour.begin(); itPoint != Contour.end(); ++itPoint)
    {
        if (itPoint->x() < VertexSelectionRect.center().x() - fabs(VertexSelectionRect.width()) * 0.5f || itPoint->x() > VertexSelectionRect.center().x() + fabs(VertexSelectionRect.width()) * 0.5f
         || itPoint->y() < VertexSelectionRect.center().y() - fabs(VertexSelectionRect.height()) * 0.5f || itPoint->y() > VertexSelectionRect.center().y() + fabs(VertexSelectionRect.height()) * 0.5f)
        {
            // The vertex is outside the rect.
            continue;
        }
        else
        {
            int index = 0;
            int minIndex = 0;
            Eigen::Vector2f p;
            float minDist = std::numeric_limits<float>::max();

            for (Point& point2D : points2D)
            {
                p << point2D.x(), point2D.y();
                float dist = (*itPoint - p).norm();

                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = index;
                }

                ++index;
            }

            //                        SelectVertexWithOneRingNeighbours(m_Vertices[index]);
            m_Vertices[minIndex]->_Selected = true;
        }
    }
}

void Model::SelectVerticesOnContour(const QRectF& VertexSelectionRect, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport)
{
    // Project vertices of the model on 2D.
    std::vector<Point> points2D;
    ProjectVerticesOnto2D(points2D, ModelViewMatrix, ProjectionMatrix, pViewport);

    // Select vertices close to the convex hull, only within the vertex selection rect.
    Point p0, p1;
    Eigen::Vector2f vec, point;
    float length = 0.0f;

    // Sample points on the convex hull and choose vertices that are closest to the points.
    for (std::vector<Point>::iterator itPoint = m_ConvexHull.begin(); itPoint != m_ConvexHull.end(); ++itPoint)
    {
        p0 = *itPoint;

        if (itPoint < m_ConvexHull.end() - 1)
        {
            p1 = *(itPoint + 1);
        }
        else
        {
            p1 = *m_ConvexHull.begin();
        }

        point << p0.x(), p0.y();
        vec << p1.x() - p0.x(), p1.y() - p0.y();
        length = vec.norm();

        if (length < 1.0f)
        {
            length = 1.0f;
        }

        for (int i = 0; i < (int)floor(length); ++i)
        {
            if (i > 0)
            {
                point += vec / length;
            }

            if (point.x() < VertexSelectionRect.center().x() - fabs(VertexSelectionRect.width()) * 0.5f || point.x() > VertexSelectionRect.center().x() + fabs(VertexSelectionRect.width()) * 0.5f
             || point.y() < VertexSelectionRect.center().y() - fabs(VertexSelectionRect.height()) * 0.5f || point.y() > VertexSelectionRect.center().y() + fabs(VertexSelectionRect.height()) * 0.5f)
            {
                // The vertex is outside the rect.
                continue;
            }
            else
            {
                int index = 0;
                int minIndex = 0;
                Eigen::Vector2f p;
                float minDist = std::numeric_limits<float>::max();

                for (Point& point2D : points2D)
                {
                    p << point2D.x(), point2D.y();
                    float dist = (point - p).norm();

                    if (dist < minDist)
                    {
                        minDist = dist;
                        minIndex = index;
                    }

                    ++index;
                }

                //                        SelectVertexWithOneRingNeighbours(m_Vertices[index]);
                m_Vertices[minIndex]->_Selected = true;
            }
        }
    }


//    for (Alpha_shape_2::Alpha_shape_vertices_iterator it = m_pAlphaShape->alpha_shape_vertices_begin(); it != m_pAlphaShape->alpha_shape_vertices_end(); ++it)
//    {
//        if ((*it)->point().x() < VertexSelectionRect.center().x() - fabs(VertexSelectionRect.width()) * 0.5f || (*it)->point().x() > VertexSelectionRect.center().x() + fabs(VertexSelectionRect.width()) * 0.5f
//         || (*it)->point().y() < VertexSelectionRect.center().y() - fabs(VertexSelectionRect.height()) * 0.5f || (*it)->point().y() > VertexSelectionRect.center().y() + fabs(VertexSelectionRect.height()) * 0.5f)
//        {
//            // The vertex is outside the rect.
//            continue;
//        }
//        else
//        {
//            pointOnAlphaShape << (*it)->point().x(), (*it)->point().y();
//            int index = 0;

//            for (Eigen::Vector2f& point : points2D)
//            {
//                if (!m_Vertices[index]->_Selected && ((pointOnAlphaShape - point).norm() < 5.0f))
//                {
//                    SelectVertexWithOneRingNeighbours(m_Vertices[index]);
//    //                m_Vertices[index]->_Selected = true;

//                    break;
//                }

//                ++index;
//            }
//        }
//    }
}

void Model::SelectVerticesOnContourFixedPipeline(const QRectF& VertexSelectionRect, GLdouble* pModelViewMatrix, GLdouble* pProjectionMatrix, GLint* pViewport)
{
    // Project vertices of the model on 2D.
    std::vector<Eigen::Vector2f> points2D;

    for (Vertex*& pVertex : m_Vertices)
    {
        Eigen::Vector2f windowPos;
        GLdouble x, y, z;

        gluProject((GLdouble)pVertex->_Pos[0], (GLdouble)pVertex->_Pos[1], (GLdouble)pVertex->_Pos[2], pModelViewMatrix, pProjectionMatrix, pViewport, &x, &y, &z);
        windowPos << x, y;
        points2D.push_back(windowPos);
    }

    // Select vertices on the contour only within the vertex selection rect.
    Eigen::Vector2f pointOnAlphaShape;

    for (Alpha_shape_2::Alpha_shape_vertices_iterator it = m_pAlphaShape->alpha_shape_vertices_begin(); it != m_pAlphaShape->alpha_shape_vertices_end(); ++it)
    {
        if ((*it)->point().x() < VertexSelectionRect.center().x() - fabs(VertexSelectionRect.width()) * 0.5f || (*it)->point().x() > VertexSelectionRect.center().x() + fabs(VertexSelectionRect.width()) * 0.5f
         || (*it)->point().y() < VertexSelectionRect.center().y() - fabs(VertexSelectionRect.height()) * 0.5f || (*it)->point().y() > VertexSelectionRect.center().y() + fabs(VertexSelectionRect.height()) * 0.5f)
        {
            // The vertex is outside the rect.
            continue;
        }
        else
        {
            pointOnAlphaShape << (*it)->point().x(), (*it)->point().y();
            int index = 0;

            for (Eigen::Vector2f& point : points2D)
            {
                if (!m_Vertices[index]->_Selected && ((pointOnAlphaShape - point).norm() < 5.0f))
                {
                    SelectVertexWithOneRingNeighbours(m_Vertices[index]);
    //                m_Vertices[index]->_Selected = true;

                    break;
                }

                ++index;
            }
        }
    }
}

/************** For obj file loading. **************/

void Model::GeometricVertexCallback(obj::float_type x, obj::float_type y, obj::float_type z)
{
    m_Vertices.push_back(new Vertex(x, y, z));

    (*m_pOStream) << "v " << x << " " << y << " " << z << "\n";
}

void Model::TextureVertexCallback(obj::float_type u, obj::float_type v)
{
    m_TexCoords.push_back(TexCoord(u, v));
//    m_HasTextures = true;

    (*m_pOStream) << "vt " << u << " " << v << "\n";
}

void Model::VertexNormalCallback(obj::float_type x, obj::float_type y, obj::float_type z)
{
    Eigen::Vector3f normal(x, y, z);
    normal.normalize();

    m_VertexNormals.push_back(normal);
    m_HasVertexNormals = true;

    (*m_pOStream) << "vn " << normal[0] << " " << normal[1] << " " << normal[2] << "\n";
}

void Model::TriangularFaceGeometricVerticesCallback(obj::index_type v1, obj::index_type v2, obj::index_type v3)
{
    Face* pFace = new Face;
    pFace->_VertexIndices.push_back(v1 - 1);
    pFace->_VertexIndices.push_back(v2 - 1);
    pFace->_VertexIndices.push_back(v3 - 1);
    m_Faces.push_back(pFace);

    (*m_pOStream) << "f " << v1 << " " << v2 << " " << v3 << "\n";
}

void Model::TriangularFaceGeometricVerticesTextureVerticesCallback(const obj::index_2_tuple_type& v1_vt1, const obj::index_2_tuple_type& v2_vt2, const obj::index_2_tuple_type& v3_vt3)
{
    Face* pFace = new Face;
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v1_vt1) - 1);
    pFace->_TexCoordIndices.push_back(std::tr1::get<1>(v1_vt1) - 1);
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v2_vt2) - 1);
    pFace->_TexCoordIndices.push_back(std::tr1::get<1>(v2_vt2) - 1);
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v3_vt3) - 1);
    pFace->_TexCoordIndices.push_back(std::tr1::get<1>(v3_vt3) - 1);
    m_Faces.push_back(pFace);

    (*m_pOStream) << "f " << std::tr1::get<0>(v1_vt1) << "/" << std::tr1::get<1>(v1_vt1) << " " << std::tr1::get<0>(v2_vt2) << "/" << std::tr1::get<1>(v2_vt2) << " " << std::tr1::get<0>(v3_vt3) << "/" << std::tr1::get<1>(v3_vt3) << "\n";
}

void Model::TriangularFaceGeometricVerticesVertexNormalsCallback(const obj::index_2_tuple_type& v1_vn1, const obj::index_2_tuple_type& v2_vn2, const obj::index_2_tuple_type& v3_vn3)
{
    Face* pFace = new Face;
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v1_vn1) - 1);
    pFace->_VertexNormalIndices.push_back(std::tr1::get<1>(v1_vn1) - 1);
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v2_vn2) - 1);
    pFace->_VertexNormalIndices.push_back(std::tr1::get<1>(v2_vn2) - 1);
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v3_vn3) - 1);
    pFace->_VertexNormalIndices.push_back(std::tr1::get<1>(v3_vn3) - 1);
    m_Faces.push_back(pFace);

    (*m_pOStream) << "f " << std::tr1::get<0>(v1_vn1) << "//" << std::tr1::get<1>(v1_vn1) << " " << std::tr1::get<0>(v2_vn2) << "//" << std::tr1::get<1>(v2_vn2) << " " << std::tr1::get<0>(v3_vn3) << "//" << std::tr1::get<1>(v3_vn3) << "\n";
}

void Model::TriangularFaceGeometricVerticesTextureVerticesVertexNormalsCallback(const obj::index_3_tuple_type& v1_vt1_vn1, const obj::index_3_tuple_type& v2_vt2_vn2, const obj::index_3_tuple_type& v3_vt3_vn3)
{
    Face* pFace = new Face;
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v1_vt1_vn1) - 1);
    pFace->_TexCoordIndices.push_back(std::tr1::get<1>(v1_vt1_vn1) - 1);
    pFace->_VertexNormalIndices.push_back(std::tr1::get<2>(v1_vt1_vn1) - 1);
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v2_vt2_vn2) - 1);
    pFace->_TexCoordIndices.push_back(std::tr1::get<1>(v2_vt2_vn2) - 1);
    pFace->_VertexNormalIndices.push_back(std::tr1::get<2>(v2_vt2_vn2) - 1);
    pFace->_VertexIndices.push_back(std::tr1::get<0>(v3_vt3_vn3) - 1);
    pFace->_TexCoordIndices.push_back(std::tr1::get<1>(v3_vt3_vn3) - 1);
    pFace->_VertexNormalIndices.push_back(std::tr1::get<2>(v3_vt3_vn3) - 1);
    m_Faces.push_back(pFace);

    (*m_pOStream) << "f " << std::tr1::get<0>(v1_vt1_vn1) << "/" << std::tr1::get<1>(v1_vt1_vn1) << "/" << std::tr1::get<2>(v1_vt1_vn1) << " " << std::tr1::get<0>(v2_vt2_vn2) << "/" << std::tr1::get<1>(v2_vt2_vn2) << "/" << std::tr1::get<2>(v2_vt2_vn2) << " " << std::tr1::get<0>(v3_vt3_vn3) << "/" << std::tr1::get<1>(v3_vt3_vn3) << "/" << std::tr1::get<2>(v3_vt3_vn3) << "\n";
}

void Model::MaterialLibraryCallback(const std::string& FileName)
{
    m_HasMaterials = true;
    m_MaterialLibraryFileNames.push_back(FileName);

    (*m_pOStream) << "mtllib " << FileName << "\n";
}

void Model::MaterialNameCallback(const std::string& MaterialName)
{
    m_MaterialNames.push_back(MaterialName);

    (*m_pOStream) << "usemtl " << MaterialName << "\n";
}

void Model::CommentCallback(const std::string& Comment)
{
    (*m_pOStream) << Comment << "\n";
}
