#ifndef MODEL_H
#define MODEL_H

#include <QTextStream>
#include <QOpenGLWidget>
#include <QOpenGLTexture>
#include <QMatrix4x4>

//#include <CGAL/Simple_cartesian.h>
//#include <CGAL/Polyhedron_3.h>

// libobj - reference: https://people.cs.kuleuven.be/~ares.lagae/libobj/index.html
#include <obj.hpp>

#include "../../../libs/eigen/Eigen/Dense"

//typedef CGAL::Simple_cartesian<double> Kernel;
//typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

/************** For alpha shape. **************/
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>

// For generating training set and testing for CNNs
#define NOT_FOR_CNN 1


template <class Gt, class Vb = CGAL::Alpha_shape_vertex_base_2<Gt> >
class My_vertex_base
  : public  Vb
{
  typedef Vb                              Base;

public:
  typedef typename Vb::Vertex_handle      Vertex_handle;
  typedef typename Vb::Face_handle        Face_handle;
  typedef typename Vb::Point              Point;
  template < typename TDS2 >
  struct Rebind_TDS {
    typedef typename Vb::template Rebind_TDS<TDS2>::Other    Vb2;
    typedef My_vertex_base<Gt,Vb2>                           Other;
  };

private:
  Vertex_handle  va_;

public:
  My_vertex_base() : Base() {}
  My_vertex_base(const Point & p) : Base(p) {}
  My_vertex_base(const Point & p, Face_handle f) : Base(f,p) {}
  My_vertex_base(Face_handle f) : Base(f) {}

  void set_associated_vertex(Vertex_handle va) { va_ = va;}
  Vertex_handle get_associated_vertex() {return va_ ; }
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef K::Point_2  Point;
typedef K::Segment_2  Segment;
typedef My_vertex_base<K> Vb;
//typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
typedef CGAL::Alpha_shape_face_base_2<K>  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds> Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2>  Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator Alpha_shape_edges_iterator;


template <class OutputIterator>
void alpha_edges( const Alpha_shape_2&  A,
         OutputIterator out)
{
    for(Alpha_shape_edges_iterator it = A.alpha_shape_edges_begin(); it != A.alpha_shape_edges_end(); ++it)
    {
        *out++ = A.segment(*it);
    }
}

template <class OutputIterator>
bool file_input(OutputIterator out)
{
  std::ifstream is("./data/fin", std::ios::in);
  if(is.fail()){
    std::cerr << "unable to open file for input" << std::endl;
    return false;
  }
  int n;
  is >> n;
  std::cout << "Reading " << n << " points from file" << std::endl;
  CGAL::cpp11::copy_n(std::istream_iterator<Point>(is), n, out);
  return true;
}

/****************************/


class Model
{
public:

    struct Halfedge;

    typedef struct Vertex
    {
        int _Index;
        Eigen::Vector3f _Pos;
        Eigen::Vector3f _Normal;
        Eigen::Vector4f _Colour;
        Halfedge* _pHalfedge;
        bool _Selected;
        bool _Moved;
        bool _HighCurvature;

        Vertex()
        {
            _Index = -1;
            _Pos.setZero();
            _Normal.setZero();
            _Colour.setOnes();
            _pHalfedge = NULL;
            _Selected = false;
            _Moved = false;
            _HighCurvature = false;
        }

        Vertex(float x, float y, float z)
        {
            _Index = -1;
            _Pos << x, y, z;
            _Normal.setZero();
            _Colour.setOnes();
            _pHalfedge = NULL;
            _Selected = false;
            _Moved = false;
            _HighCurvature = false;
        }
    } Vertex;

    typedef struct TexCoord
    {
        float _u;
        float _v;

        TexCoord()
        {
            _u = 0.0f;
            _v = 0.0f;
        }

        TexCoord(float u, float v)
        {
            _u = u;
            _v = v;
        }
    } TexCoord;

    typedef struct Face
    {
        std::vector<int> _VertexIndices;
        std::vector<int> _TexCoordIndices;
        std::vector<int> _VertexNormalIndices;
        Eigen::Vector3f _Normal;
        Eigen::Vector3f _Centroid;
        Halfedge* _pHalfedge;
        bool _Sampled;

        Face()
        {
            _Normal.setZero();
            _Centroid.setZero();
            _pHalfedge = NULL;
            _Sampled = false;
        }
    } Face;

    typedef struct Halfedge
    {
        Vertex* _pVertex;
        Face* _pFace;
        Halfedge* _pPrev;
        Halfedge* _pNext;
        Halfedge* _pOpposite;

        Halfedge()
        {
            _pVertex = NULL;
            _pFace = NULL;
            _pPrev = NULL;
            _pNext = NULL;
            _pOpposite = NULL;
        }
    } Halfedge;


    Model();

    void CleanUp();
    void Load(const QString& FileName);
    void BuildVertexData();
    std::vector<GLfloat>& VertexData() { return m_VertexData; }

    void Render();
    void RenderFixedPipeline();
    void RenderContour(void);
    void RenderFaces(const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport, bool RenderingVisibleFaceOnly = true);

    static bool FaceFrontFacing(const Face* pFace, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    static std::vector<QVector3D> SamplePointsOnFace(const std::vector<QVector3D>& Points2D, const std::vector<QVector3D>& Points3D, std::vector<QVector3D>& VisiblePixelCoords, const GLfloat* pDepthData, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    static bool VertexVisible(const Vertex* pVertex, const GLfloat* pDepthData, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);

    void BuildHalfedges(void);

    void ComputeCentroid(void);
    Eigen::Vector3f& Centroid(void) { return m_Centroid; }

    void ComputeFaceNormals(void);
    void ComputeVertexNormals(void);
    void ComputeFaceCentroids(void);
    void ComputeGaussianCurvature(void);

    std::vector<Vertex*>& Vertices(void) { return m_Vertices; }

    std::vector<Eigen::Vector3f>& VertexNormals(void) { return m_VertexNormals; }
    void SetVertexNormals(std::vector<Eigen::Vector3f>& VertexNormals) { m_VertexNormals = VertexNormals; }

    std::vector<TexCoord>& TexCoords(void) { return m_TexCoords; }
    void SetTexCoords(std::vector<TexCoord>& TexCoords) { m_TexCoords = TexCoords; }

    std::vector<Face*>& Faces(void) { return m_Faces; }

    bool HasTextures(void) { return m_HasTextures; }
    bool HasVertexNormals(void) { return m_HasVertexNormals; }
    bool HasMaterials(void) { return m_HasMaterials; }

    Eigen::Vector4f& Colour(void) { return m_Colour; }
    void SetColour(Eigen::Vector4f& Colour)
    {
        m_Colour = Colour;

        for (Vertex* pVertex : m_Vertices)
        {
            if (m_HasSelectedVertices)
            {
                if (pVertex->_Selected)
                {
                    pVertex->_Colour = Colour;
                }
            }
            else
            {
                pVertex->_Colour = Colour;
            }
        }
    }

    void SetTexture(QOpenGLTexture* pTexture)
    {
        m_HasTextures = true;
        m_pTexture = pTexture;
    }

    bool HasSelectedVertices() { return m_HasSelectedVertices; }
    void SetHasSelectedVertices(bool HasSelectedVertices) { m_HasSelectedVertices = HasSelectedVertices; }

    int ClosestVertexIndexToPoint(const Eigen::Vector2f& APoint, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, GLint* pViewport, float& Distance);
    int ClosestVertexIndexToPointFixedPipeline(const Eigen::Vector2f& APoint, GLdouble* pModelViewMatrix, GLdouble* pProjectionMatrix, GLint* pViewport);

//    std::vector<int>& SelectedVertexIndices(void) { return m_SelectedVertexIndices; }

    Eigen::Affine3f& Transform(void) { return m_Transform; }

    void TranslateSelectedVertices(const Eigen::Vector3f& Vector);
    void RotateSelectedVertices(const Eigen::Vector3f& Vector, const Eigen::Vector3f& VectorPerpendicular);
    void UnselectAllVertices(void);
    void UnsampleAllFaces(void);
    void OneRingNeighbours(Vertex* pVertex, std::vector<Vertex*>& Neighbours);
    void SelectVertexWithOneRingNeighbours(Vertex* pVertex);
    void OneRingNeighbourFaces(Vertex* pVertex, std::vector<Face*>& Faces);

    static QVector3D ProjectPointOnto2D(const QVector3D& Pos, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    static Point ProjectVertexOnto2D(const Vertex* pVertex, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void ProjectVerticesOnto2D(std::vector<Point>& Points2D, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void ExtractVerticesOnContour(const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void ExtractVerticesOnContourFixedPipeline(GLdouble* pModelViewMatrix, GLdouble* pProjectionMatrix, GLint* pViewport);
    void SelectVerticesOnContour(const std::vector<Eigen::Vector2f>& Contour, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void SelectVerticesOnContour(const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void SelectVerticesOnContour(const std::vector<Eigen::Vector2f>& Contour, const QRectF& VertexSelectionRect, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void SelectVerticesOnContour(const QRectF& VertexSelectionRect, const QMatrix4x4& ModelViewMatrix, const QMatrix4x4& ProjectionMatrix, const GLint* pViewport);
    void SelectVerticesOnContourFixedPipeline(const QRectF& VertexSelectionRect, GLdouble* pModelViewMatrix, GLdouble* pProjectionMatrix, GLint* pViewport);
    void RemoveContour(void) { m_Contour.clear(); }
    std::vector<Segment>& Contour() { return m_Contour; }

    std::vector<float>& GaussianCurvatures() { return m_GaussianCurvatures; }

private:
    std::vector<Vertex*> m_Vertices;
    std::vector<Eigen::Vector3f> m_VertexNormals;
    std::vector<TexCoord> m_TexCoords;
    std::vector<Face*> m_Faces;
    std::vector<Halfedge*> m_Halfedges;
    std::vector<GLfloat> m_VertexData;
    std::vector<float> m_GaussianCurvatures;

//    Polyhedron m_Polyhedron;

    Eigen::Vector3f m_Centroid;
    std::vector<std::string> m_MaterialLibraryFileNames;
    std::vector<std::string> m_MaterialNames;
    bool m_HasTextures;
    bool m_HasVertexNormals;
    bool m_HasMaterials;
    Eigen::Vector4f m_Colour;
    QOpenGLTexture* m_pTexture;
    bool m_HasSelectedVertices;

//    std::vector<int> m_SelectedVertexIndices;
    Eigen::Affine3f m_Transform;
    std::vector<int> m_VertexIndicesOnContour;
    std::vector<Segment> m_Contour;
    Alpha_shape_2* m_pAlphaShape;
    std::vector<Point> m_ConvexHull;

    /************** For obj file loading. **************/

    obj::obj_parser m_ObjParser;
    std::ostream* m_pOStream;

    void GeometricVertexCallback(obj::float_type x, obj::float_type y, obj::float_type z);
    void TextureVertexCallback(obj::float_type u, obj::float_type v);
    void VertexNormalCallback(obj::float_type x, obj::float_type y, obj::float_type z);
    void TriangularFaceGeometricVerticesCallback(obj::index_type v1, obj::index_type v2, obj::index_type v3);
    void TriangularFaceGeometricVerticesTextureVerticesCallback(const obj::index_2_tuple_type& v1_vt1, const obj::index_2_tuple_type& v2_vt2, const obj::index_2_tuple_type& v3_vt3);
    void TriangularFaceGeometricVerticesVertexNormalsCallback(const obj::index_2_tuple_type& v1_vn1, const obj::index_2_tuple_type& v2_vn2, const obj::index_2_tuple_type& v3_vn3);
    void TriangularFaceGeometricVerticesTextureVerticesVertexNormalsCallback(const obj::index_3_tuple_type& v1_vt1_vn1, const obj::index_3_tuple_type& v2_vt2_vn2, const obj::index_3_tuple_type& v3_vt3_vn3);
    void MaterialLibraryCallback(const std::string& FileName);
    void MaterialNameCallback(const std::string& MaterialName);
    void CommentCallback(const std::string& Comment);
};

#endif // MODEL_H
