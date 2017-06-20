#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <GL/glu.h>
#include <QApplication>
#include <QMessageBox>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>
#include <QOpenGLTexture>
#include <QTimer>

#include "trackBall.h"
//#include "GLtexture.h"
#include "GLmodel.h"

#include "Model.h"
#include "Solver.h"

#include "../../../libs/tetgen1.5.1-beta1/tetgen.h"

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)


class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();

  // GETTERS
    GLfloat getFramePictureRatio();
    qreal getCameraSettings(GLint settingNumber);
    GLfloat getSensibility(), getSensibilityPlus();
    GLfloat getRotationSpeed();
    GLfloat getTagsRadius();

    bool RenderingModelFaces(void) { return m_RenderingModelFaces; }

    QStringList getModelsList();
    QStringList GetModelContoursList();
    QStringList GetImageContoursList();

  // SETTERS
    void setCameraSettings(GLint settingNumber, qreal newValue);
    void setFramePictureRatio(GLfloat new_frame_picture_Ratio);
    void setSensibility(GLfloat newValue), setSensibilityPlus(GLfloat newValue);
    void setRotationSpeed(GLfloat newValue);
    void scaleSliderState(bool newState);
    void setTagsRadius(GLfloat newValue);
    void SetRenderingModelFaces(bool Rendering);
    void FOVScaleSliderState(bool newState);
    void SetFOVScale(float Scale);
    void UpdateSelectedHighCurvatureVertices(void);
    float HighCurvatureVertexSearchAreaRadius(void) { m_HighCurvatureVertexSearchAreaRadius; }
    int HighCurvatureVerticesPolynomialOrder(void) { return m_HighCurvatureVerticesPolynomialOrder; }

    void ComputeModelCentroid();
    void RemoveModel(GLint modelNumber);

    void Render2DImage(void);
    void Render2DImageFixedPipeline(void);

    void GetContour(std::vector<Eigen::Vector2f>& Contour);
    void GetContourFromImage(const QImage& Image, std::vector<Eigen::Vector2f>& Contour);
    void RenderContour();
    bool InsideFOV(Eigen::Vector2f& Point2D);
    void FilterInvisibleFacesInModel();
    void MoveClosestVerticesOnMeshToContour(bool Using2DSearch);
    void PreCameraCalibration();
    void OptimiseMeshWithShading(bool UpdatingDepth = false);
    float ComputeRMSEForFineRegistration(const std::vector<Eigen::Vector3f>& Vertices);
    void SaveModelData(QString& FileName, bool IsGroundTruth);
    void LoadModelData(QString& FileName, bool IsGroundTruth);
    void SaveContourToFile(std::vector<Eigen::Vector2f>& Contour);
    void SaveDisconnectedCellGroupsForSimulation(void);
    void LoadDisconnectedCellGroupsForSimulation(void);

    void ComputeVertexErrors(const std::vector<Model::Vertex*>& GroundTruthVertices, std::vector<Model::Vertex*>& ModelVertices);
    void UpdateModel(bool FaceNormals = true, bool VertexNormals = true, bool FaceCentroids = true);

    /************** For generating training set for CNNs. **************/

    // Returns the latitude and longitude for a point (of a given index) out of n evenly distributed points on a sphere using a spiral method.
    // (Reference: https://gist.github.com/ironwallaby/7121695).
    void DistributePointsOnSphere(const unsigned int NumOfPoints, const unsigned int Index, float& Latitude, float& Longitude);
    void GenerateTrainingSet(void);
    void GenerateTestSet(void);
    void GenerateDeformedModelTrainingSet(void);

    /****************************/

private:
//    QOpenGLVertexArrayObject m_ModelVAO;
//    QOpenGLVertexArrayObject m_GroundTruthModelVAO;
    std::vector<QOpenGLVertexArrayObject*> m_ModelVAOs;
    std::vector<QOpenGLVertexArrayObject*> m_GroundTruthModelVAOs;
    QOpenGLVertexArrayObject m_QuadVAO;
    QOpenGLVertexArrayObject m_ContourVAO;
//    QOpenGLBuffer m_ModelVBO;
//    QOpenGLBuffer m_GroundTruthModelVBO;
    std::vector<QOpenGLBuffer*> m_ModelVBOs;
    std::vector<QOpenGLBuffer*> m_GroundTruthModelVBOs;
    QOpenGLBuffer m_QuadVBO;
    QOpenGLBuffer m_ContourVBO;
    QOpenGLShaderProgram* m_pShaderProgramModel;
    QOpenGLShaderProgram* m_pShaderProgramGroundTruthModel;
    QOpenGLShaderProgram* m_pShaderProgramQuad;
    QOpenGLShaderProgram* m_pShaderProgramContour;

    TrackBall trackball;
//    GLtexture texture[4];
//    GLmodel model;
    QOpenGLTexture* texture[4];
    QOpenGLTexture* m_pBackgroundTexture;

    std::vector<Model> m_Models;
    std::vector<Model> m_GroundTruthModels;
    Model m_ModelForSimulation;
    Model m_ModelForSegmentation;
    std::vector<tetgenio> m_TetGenIns, m_TetGenOuts;

    bool m_EditingModel;
    bool m_TranslatingVertices;
    bool m_RotatingVertices;
    bool m_SelectingMultipleVertices;
    QRectF m_VertexSelectionRect;
    bool m_SelectingContour;
    bool m_RenderingModelFaces;

    enum MODEL_CONTOUR_TYPE
    {
        MODEL_CONTOUR_TYPE_NULL = -1,
        MODEL_CONTOUR_TYPE_FRONTIER,
        MODEL_CONTOUR_TYPE_OCCLUDING,
        MODEL_CONTOUR_TYPE_LIGAMENT
    };

    MODEL_CONTOUR_TYPE m_SelectingModelContour;

    Eigen::Vector2f m_PrevMousePos;
    GLdouble m_ModelViewMatrix[16];
    GLdouble m_ProjectionMatrix[16];
    GLint m_Viewport[4];
    GLint m_FOVViewport[4];
    float m_FOVScale;
    float m_WindowScale;
    float m_PrevWindowScaleValue;
    Eigen::Vector2f m_FOVPosOffset;
    std::vector<Eigen::Vector2f> m_ContourOffsetVectors;

    int m_projMatrixLoc;
    int m_mvMatrixLoc;
    int m_normalMatrixLoc;
    int m_lightPosLoc;
    int m_ModelTexScaleLoc;
    int m_ModelCLoc;
    int m_ModelVertexSelectedLoc;
    int m_IsGroundTruthModelLoc;

    int m_QuadScaleLoc;
    int m_ContourScaleLoc;

    QMatrix4x4 m_proj;
    QMatrix4x4 m_camera;
    QMatrix4x4 m_world;
    QMatrix4x4 m_ModelView;

    QMatrix4x4 m_GroundTruthModelModelView;

    // Material model solver.
    Solver m_MaterialModelSolver;
    bool m_SimulationInitialised;

    GLfloat opacity;
    GLfloat frame_picture_Ratio;
    GLfloat scaleFactor;
    GLfloat sensibility, sensibilityPlus;
    GLfloat rotationSpeed;
    float m_RotationAngle;
    QQuaternion m_RotationQuat;
    QVector3D m_RotationAxis;
    QTimer m_RotationTimer;
    bool m_RotatingCamera;
    QMatrix4x4 m_PrevCameraRotation;

    QTimer m_FineRegistrationTimer;
    int m_FineRegistrationCountStage1;
    int m_FineRegistrationCountStage2;
    bool m_OptimisingShading;
    int m_FineRegistrationStage;
    std::vector<Eigen::Vector3f> m_PrevModelVertices;
    std::vector<Eigen::Vector3f> m_DiffModelVertices;

    QFile* m_pExperimentResultsFile;
    QTextStream* m_pExperimentResultsFileStream;
    bool m_ModelDataLoaded;
    QMatrix4x4 m_LoadedModelView;

    QTimer m_DataGenerationTimer;
    QFile* m_pDataGenerationFile;
    QTextStream* m_pDataGenerationFileStream;

    Eigen::Vector3f m_ModelCentroid; // Centroid of all models.
    Eigen::Vector3f m_PrevModelCentroid;
    Eigen::Vector3f m_ModelCentroidDiff;
    Eigen::Vector3f coordModels;         // Models translations
    QVector3D coordTumor;                   // Tumor translations
    QVector3D surfaceCoordinates, distanceCoordinates1; // Coordinates displayed when model clicked
    GLfloat distanceBetweenTags;
    qreal cameraParameters[8];
/*
 *  cameraParameters[0] = alphaX, focal (px)
 *  cameraParameters[1] = alphaY, focal (px)
 *  cameraParameters[2] = skewness
 *  cameraParameters[3] = u, image center abscissa (px)
 *  cameraParameters[4] = v, image center ordinate (px)
 *  cameraParameters[5] = near, distance to the nearer depth clipping plane (m)
 *  cameraParameters[6] = far, distance to the farther depth clipping plane (m)
 *  cameraParameters[7] = scale of the perspective view
 */

    bool m_GeneratingTrainingSet;
    unsigned int m_NumOfTrainingImages;
    unsigned int m_TrainingImageIndex;
    unsigned int m_FrameCount;
    int m_PointsOnSphereIndex;
    Eigen::Vector3f m_CameraPosition;
    Eigen::AngleAxisf m_CameraRotation;
    float m_CameraRadius;
    Eigen::Vector3f m_CameraPositionNoise;
    Eigen::Vector3f m_CameraLookAtNoise;
    float m_CameraRollNoise;

//    Model m_GroundTruthModel;
    std::vector<Eigen::Vector2f> m_ModelContour;
    std::vector<Eigen::Vector2f> m_GroundTruthModelContour;
    std::vector<Eigen::Vector2f> m_ContourSelectionPoints;
    bool m_GroundTruthModelExisting;
    bool m_RenderingGroundTruthModel;
    int m_RenderingGroundTruthModelContour; // 0: not rendering, 1: rendering all, 2: rendering only inside FOV.
//    std::vector<int> m_FreeVertexIndicesInModel;
    std::vector<Eigen::Vector3f> m_ShadingOptimisationVertices;

    GLfloat* m_pDepthData;
    GLfloat* m_pGroundTruthDepthData;

    QImage m_InputImage;
    QImage m_InputImageMedianFilteredY;
    QImage m_SavingImage;
    float m_c; // c = l (light intensity) * k (camera response) * a (albedo) for Lambertian lighting model.
    float m_cForRendering;

    bool m_UsingShadingOptimisation;

    std::vector<float> m_VertexErrors;
    std::vector<Eigen::Vector4f> m_VertexErrorColours;

    // TODO: Temp.
    std::vector<std::vector<QVector3D> > m_allSampledPixelCoords, m_allSampledPoints3D;

    bool scaleSliderPressed, m_FOVScaleSliderPressed, tumorMode, distanceMode;
    GLuint tumor, picture, crosshair, tags; //DisplayLists
    GLfloat tumorRadius, tagsRadius;        // Radius of the tumor, tags (m)
    QStringList modelsList;                 // List of actives modelse->key()==Qt::Key_Left ||
    QStringList m_ModelContoursList;
    std::vector<std::vector<std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> > > m_ModelContours; // Tuple <vertex index, model index, contour type>.
    QStringList m_ImageContoursList;
    std::vector<std::vector<Eigen::Vector2f > > m_ImageContours;
    std::map<Model::Vertex*, Eigen::Vector2f> m_FrontierContourFixedPoints;

    QVector<unsigned int> m_CheckedModels;
    QVector<unsigned int> m_CheckedModelContours;
    QVector<unsigned int> m_CheckedImageContours;
    QString m_SelectedModel;
    QString m_SelectedModelContour;
    QString m_SelectedImageContour;
    unsigned int m_FrontierContourCount;
    unsigned int m_OccludingContourCount;
    unsigned int m_LigamentContourCount;
    unsigned int m_ImageContourCount;
    unsigned int m_SelectedImageContourIndex;
    std::map<QString, QString> m_ModelToImageContourMap;
    bool m_IsContourSelectionOn;

    bool m_SelectingShadingOptimisationRegion;
    Eigen::Vector2f m_ShadingOptimisationRegionCentre;
    float m_ShadingOptimisationRegionRadius;

    bool m_RenderingBackgroundImage;

    std::vector<Point> m_HighCurvaturePoints;
    std::vector<int> m_HighCurvatureVertexIndices;
    std::vector<QVector3D> m_HighCurvatureFittedCurve;
    int m_HighCurvatureStartPosition;
    int m_HighCurvatureEndPosition;
    bool m_SelectingHighCurvatureVertices;
    float m_HighCurvatureVertexSearchAreaRadius;
    int m_HighCurvatureVerticesPolynomialOrder;
    bool m_HighCurvatureRangeReversed;

    void initializeGL();
    void paintGL();
    void paintGLFixedPipeline();
    void camera();
    void cameraFixedPipeline();
    void resetCameraSettings();
    void LoadCameraSettings();

    void resizeWidget();
    void resetTransformations();
    void resetTumor();
//    void UpdateTumorTransform(void);
    void createCrosshair(QPointF screenCoordinates);
    void createTags(QPointF screenCoordinates);

    void mouseMoveEvent(QMouseEvent *e);
    void mousePressEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void wheelEvent(QWheelEvent *e);
    void keyPressEvent(QKeyEvent *e);
    void keyReleaseEvent(QKeyEvent *e);

    QPointF pixelPosToViewPos(const QPointF &p);
    void multMatrix(const QMatrix4x4 &m);
    QVector3D screenToModelPixel(const QPointF& screenCoordinates, bool* pModelPicked = NULL);

    void RasteriseLine(int x0, int y0, int x1, int y1, std::vector<Eigen::Vector2f>& Pixels);

    void SelectModelContour(unsigned int VertexIndex, unsigned int ModelIndex, MODEL_CONTOUR_TYPE ModelContourType);

signals:
    void pictureChanged(int newWidth, int newHeight);
    void tumorModeIsON(bool tumorModeON);
    void distanceModeIsON(bool distanceModeON);
    void modelsChanged();
    void ModelContoursChanged();
    void ImageContoursChanged();
    void OpenContourSelectionTool();
    void FOVScaleChanged(float Scale);
    void BackgroundImageChanged();

public slots:
    void cleanup();
    void UpdateModelVBO(int Index);
    void UpdateGroundTruthModelVBO(int Index);
//    void UpdateModelVBO();
//    void UpdateGroundTruthModelVBO();
    void InitialiseSimulation();
    void ComputeModelSegmentation();//Model& OriginalModel, Model& SegmentationModel);
    void ComputeGaussianCurvature();
    void SelectHighCurvatureVertices(void);

    void Rotate();
    void FineRegistration();

    void ContourSelection(bool Enabled, bool Finalised);
    void ResetContourSelection();
    void FinaliseContourSelection();

    void RunOptimisation(bool UsingContourAndShading);

  // BUTTONS
    void setTexturePath();
    void addModel();
    void saveObj();
    void createTumor(bool buttonChecked);
    void CentreModel(void);
    void setDistanceMode(bool buttonChecked);
    void rotateX();
    void rotateY();

  // SLIDER
    void setOpacity(int sliderValue);

    void SetCheckedModels(QVector<unsigned int>& CheckedModels);
    void SetSelectedModel(QString& SelectedModel);
    void RemoveModels();

    void SetCheckedModelContours(QVector<unsigned int>& CheckedModelContours);
    void SetSelectedModelContour(QString& SelectedModelContour);
    void AddFrontierContour();
    void AddOccludingContour();
    void AddLigamentContour();
    void RemoveModelContours();
    void ShowModelContour(unsigned int Index);

    void SetCheckedImageContours(QVector<unsigned int>& CheckedImageContours);
    void SetSelectedImageContour(QString& SelectedImageContour);
    void AddImageContour();
    void RemoveImageContours();
    void ShowImageContour(unsigned int Index);
    void LinkModelToImageContour();
    void SetHighCurvatureStartPosition(float Position);
    void SetHighCurvatureEndPosition(float Position);
    void SetHighCurvatureRangeReversed(bool Reversed);
    void SetHighCurvatureVertexSearchAreaRadius(float Radius);
    void SetHighCurvatureVerticesPolynomialOrder(int Order);


    void ChangeColour(QColor& Colour);

    void RandomlyDeformModel(void);
};

#endif // GLWIDGET_H
