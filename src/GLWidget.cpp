#include "GLWidget.h"
#include "Utils.h"

#include <QOpenGLShaderProgram>

#include <iostream>
#include <random>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

#define USING_SYNTHETIC_INPUT_IMAGE 0
#define USING_PHANTOM_INPUT_IMAGE 0

/* ============================ INITIALIZATION ============================ */
GLWidget::GLWidget(QWidget *parent) : QOpenGLWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);             // Keyboard ON
    setCursor(Qt::PointingHandCursor);           // Cursor shape

    m_EditingModel = false;
    m_TranslatingVertices = false;
    m_RotatingVertices = false;
    m_SelectingMultipleVertices = false;
    m_RenderingModelFaces = false;
    m_VertexSelectionRect = QRectF(0.0f, 0.0f, 0.0f, 0.0f);
    m_SelectingContour = false;
    m_SelectingModelContour = MODEL_CONTOUR_TYPE_NULL;
    m_PrevMousePos.setZero();
    m_ModelCentroid.setZero();
    m_PrevModelCentroid.setZero();
    m_ModelCentroidDiff.setZero();
    coordModels.setZero();

    m_FrontierContourCount = 0;
    m_OccludingContourCount = 0;
    m_LigamentContourCount = 0;
    m_ImageContourCount = 0;
    m_SelectedImageContourIndex = 0;

    opacity = 1;                                 // Initialization of modele's opacity
//    scaleFactor = 0.001;                         // Millimeters to meters conversion
    scaleFactor = 1.0f;
    frame_picture_Ratio = 0.0f; //0.1;                   // Size of the black frame relative to image size
    sensibility = 0.01, sensibilityPlus = 0.001; // Sensibility of the translation (m)
    rotationSpeed = 0.3;
    m_RotationAngle = 0.0f;
    m_RotatingCamera = false;

    tumorMode = false;                           // "Add/Move Tumor" mode OFF
    tumor = 0;                                   // Tumor OFF
    tumorRadius = 0.01;                          // Radius of the tumor (m)

    crosshair = tags = 0;                        // Crosshair, tags OFF
    tagsRadius = 0.001;                          // Tags radius

    distanceMode = false;                        // Distance mode OFF
    distanceCoordinates1 = QVector3D(0,0,0);
    distanceBetweenTags = 0;

    scaleSliderPressed = false;                  // Scale Slider OFF
    m_FOVScaleSliderPressed = false;
    m_FOVScale = 1.0f;
    m_WindowScale = 1.0f;
    m_PrevWindowScaleValue = 1.0f;
    cameraParameters[7] = 1.0; //0.85;                  // Relative scale of the perspective view

    m_GeneratingTrainingSet = false;
    m_NumOfTrainingImages = 1000; // Training set: 10000, test set: 1000.
    m_TrainingImageIndex = 0;
    m_FrameCount = 0;
    m_PointsOnSphereIndex = 0;
    m_CameraRadius = 0.35f; // Closest: 0.15f;    
    m_CameraPositionNoise.setZero();
    m_CameraLookAtNoise.setZero();
    m_CameraRollNoise = 0.0f;

    m_GroundTruthModelExisting = false;
    m_RenderingGroundTruthModel = false;
    m_RenderingGroundTruthModelContour = 0;
    m_pDepthData = NULL;
    m_pGroundTruthDepthData = NULL;

    // TODO: here - normalise the brightness of the liver textures, check the rendered image brightness, centre the massicot model. Then, generate data.
    m_c = 0.02f; //0.015f;
    m_cForRendering = m_c;

    m_UsingShadingOptimisation = true;
    m_OptimisingShading = false;
    m_FineRegistrationStage = 0;

    m_pExperimentResultsFile = NULL;
    m_pExperimentResultsFileStream = NULL;
    m_ModelDataLoaded = false;
    m_LoadedModelView.setToIdentity();

    m_FOVPosOffset.setZero();

    m_VertexErrors.clear();
    m_VertexErrorColours.clear();

    m_IsContourSelectionOn = false;

    m_SelectingShadingOptimisationRegion = false;
    m_ShadingOptimisationRegionCentre.setZero();
    m_ShadingOptimisationRegionRadius = 0.0f;

    // TODO: Temp.
//    m_c = 0.02f;

    m_pBackgroundTexture = NULL;
    m_RenderingBackgroundImage = true;

    m_SimulationInitialised = false;

    m_pDataGenerationFile = NULL;
    m_pDataGenerationFileStream = NULL;

    m_HighCurvatureStartPosition = 0;
    m_HighCurvatureEndPosition = 0;
    m_SelectingHighCurvatureVertices = false;
    m_HighCurvatureVertexSearchAreaRadius = 0.005f;
    m_HighCurvatureVerticesPolynomialOrder = 20;
    m_HighCurvatureRangeReversed = false;

#if USING_PHANTOM_INPUT_IMAGE

    m_InputImage.load(QString("../tensorflow/liver_data/fine_registration/phantom_full_view/phantom.png"));

    // Load the input image, convert it to XYZ colour space and choose Y channel for illumination.
    // N.B. Currently does not do the median filtering.
    m_InputImageMedianFilteredY.load(QString("../tensorflow/liver_data/fine_registration/phantom_full_view/phantom.png"));
//    m_InputImageMedianFilteredY.load(QString("../tensorflow/liver_data/fine_registration/phantom_full_view/phantom_median_filtered_y.png"));

    Utils::ConvertRGBToXYZ(m_InputImageMedianFilteredY);

//    QImage mask;
//    mask.load(QString("../tensorflow/liver_data/fine_registration/phantom_full_view/phantom_mask.png"));
//    GetContourFromImage(mask, m_GroundTruthModelContour);
//    m_RenderingGroundTruthModelContour = 2;

#else

    // Using laparoscopy images.
    m_InputImage.load(QString("../tensorflow/liver_data/fine_registration/3d_models/massicot_20170125/videos/liver_images/image0232.png"));

    // Load the input image, convert it to XYZ colour space and choose Y channel for illumination.
    // N.B. Currently does not do the median filtering.
    m_InputImageMedianFilteredY.load(QString("../tensorflow/liver_data/fine_registration/3d_models/massicot_20170125/videos/liver_images/image0232.png"));
//    m_InputImageMedianFilteredY.load(QString("../tensorflow/liver_data/fine_registration/laparoscopy/laparoscopy_median_filtered_y.png"));

//    Utils::ConvertRGBToXYZ(m_InputImageMedianFilteredY);
    m_InputImageMedianFilteredY = m_InputImage.convertToFormat(QImage::Format_Grayscale8);

//    QImage mask;
//    mask.load(QString("../tensorflow/liver_data/fine_registration/laparoscopy/laparoscopy_mask.png"));
//    GetContourFromImage(mask, m_GroundTruthModelContour);
//    m_RenderingGroundTruthModelContour = 2;

#endif

    // Load a liver model for segmentation.
//    m_ModelForSegmentation.Load(QString("../tensorflow/liver_data/fine_registration/3d_models/massicot_20170125/liver_segmentation.obj"));
    m_ModelForSegmentation.Load(QString("../tensorflow/liver_data/fine_registration/3d_models/bologna/cleaned/liver_segmentation.obj"));
}

GLWidget::~GLWidget()
{
    cleanup();
}

void GLWidget::cleanup()
{
    makeCurrent();

    while (m_Models.size() > 0)
    {
        RemoveModel(0);
    }

//    for (int i = 0; i < 4; ++i)
//    {
//        if (texture[i])
//        {
//            delete texture[i];
//        }
//    }

//    if (m_pBackgroundTexture)
//    {
//        delete m_pBackgroundTexture;
//    }

    delete m_pShaderProgramModel;
    delete m_pShaderProgramQuad;
    m_pShaderProgramModel = 0;
    m_pShaderProgramQuad = 0;

    if (m_pGroundTruthDepthData)
    {
        delete m_pGroundTruthDepthData;
    }

    doneCurrent();
}

static const char *vertexShaderSourceCore =
    "#version 150\n"
    "in vec4 vertex;\n"
    "in vec3 normal;\n"
    "out vec3 vert;\n"
    "out vec3 vertNormal;\n"
    "uniform mat4 projMatrix;\n"
    "uniform mat4 mvMatrix;\n"
    "uniform mat3 normalMatrix;\n"
    "void main() {\n"
    "   vert = vertex.xyz;\n"
    "   vertNormal = normalMatrix * normal;\n"
    "   gl_Position = projMatrix * mvMatrix * vertex;\n"
    "}\n";

static const char *fragmentShaderSourceCore =
    "#version 150\n"
    "in highp vec3 vert;\n"
    "in highp vec3 vertNormal;\n"
    "out highp vec4 fragColor;\n"
    "uniform highp vec3 lightPos;\n"
    "void main() {\n"
    "   highp vec3 L = normalize(lightPos - vert);\n"
    "   highp float NL = max(dot(normalize(vertNormal), L), 0.0);\n"
    "   highp vec3 color = vec3(0.39, 1.0, 0.0);\n"
    "   highp vec3 col = clamp(color * 0.2 + color * 0.8 * NL, 0.0, 1.0);\n"
    "   fragColor = vec4(col, 1.0);\n"
    "}\n";

static const char *vertexShaderSource =
    "attribute highp vec4 vertex;\n"
    "attribute highp vec3 normal;\n"
    "attribute mediump vec4 texCoord;\n"
    "varying highp vec3 vert;\n"
    "varying highp vec3 vertNormal;\n"
    "varying mediump vec4 texc;\n"
    "uniform mat4 projMatrix;\n"
    "uniform mat4 mvMatrix;\n"
    "uniform mat3 normalMatrix;\n"
    "void main() {\n"
    "   vert = vertex.xyz;\n"
    "   vertNormal = normalMatrix * normal;\n"
    "   gl_Position = projMatrix * mvMatrix * vertex;\n"
    "    texc = texCoord;\n"
    "}\n";

static const char *fragmentShaderSource =
    "varying highp vec3 vert;\n"
    "varying highp vec3 vertNormal;\n"
    "uniform sampler2D texture;\n"
    "varying mediump vec4 texc;\n"
    "uniform highp vec3 lightPos;\n"
    "void main() {\n"
    "   highp vec3 L = normalize(lightPos - vert);\n"
    "   highp float NL = max(dot(normalize(vertNormal), L), 0.0);\n"
    "   highp vec3 color = vec3(1.0, 1.0, 1.0);\n"
    "   highp vec3 col = clamp(color * 0.2 + color * 0.8 * NL, 0.0, 1.0);\n"
    //"   gl_FragColor = texture2D(texture, texc.st) * vec4(col, 1.0);\n"
    "   gl_FragColor = vec4(col, 1.0);\n"
    "}\n";

void GLWidget::initializeGL()   // OPENGL SPACE INITIALIZATION
{
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &GLWidget::cleanup);

    initializeOpenGLFunctions();
    glClearColor(0, 0, 0, 1);

    glEnable(GL_DEPTH_TEST);                                // Depth buffer ON
    glEnable(GL_CULL_FACE);

    // Shader program for the model.
    m_pShaderProgramModel = new QOpenGLShaderProgram;
//    m_pShaderProgramModel->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderPhongBRDFSource); //m_core ? vertexShaderSourceCore : vertexShaderSource);
//    m_pShaderProgramModel->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderPhongBRDFSource); // m_core ? fragmentShaderSourceCore : fragmentShaderSource);
    m_pShaderProgramModel->addShaderFromSourceFile(QOpenGLShader::Vertex, "../src/shaders/phongBRDF.vert"); //m_core ? vertexShaderSourceCore : vertexShaderSource);
    m_pShaderProgramModel->addShaderFromSourceFile(QOpenGLShader::Fragment, "../src/shaders/phongBRDF.frag"); // m_core ? fragmentShaderSourceCore : fragmentShaderSource);
    m_pShaderProgramModel->bindAttributeLocation("vertex", 0);
    m_pShaderProgramModel->bindAttributeLocation("normal", 1);
    m_pShaderProgramModel->bindAttributeLocation("texCoord", 2);
    m_pShaderProgramModel->bindAttributeLocation("selected", 3);
    m_pShaderProgramModel->bindAttributeLocation("faceCentre", 4);

    // TODO: Temp for rendering vertex error colour.
    m_pShaderProgramModel->bindAttributeLocation("vertexErrorColour", 5);

    m_pShaderProgramModel->link();

    m_pShaderProgramModel->bind();
    m_projMatrixLoc = m_pShaderProgramModel->uniformLocation("projMatrix");
    m_mvMatrixLoc = m_pShaderProgramModel->uniformLocation("mvMatrix");
    m_normalMatrixLoc = m_pShaderProgramModel->uniformLocation("normalMatrix");
    m_lightPosLoc = m_pShaderProgramModel->uniformLocation("lightPos");
    m_ModelTexScaleLoc = m_pShaderProgramModel->uniformLocation("texScale");
    m_ModelCLoc = m_pShaderProgramModel->uniformLocation("c");
    m_ModelVertexSelectedLoc = m_pShaderProgramModel->uniformLocation("vertexSelected");

    m_pShaderProgramModel->setUniformValue("texture", 0);

    // Default 2D image initialisation.
    texture[0] = new QOpenGLTexture(QImage("../data/liver_texture_1.jpg").mirrored());
    texture[1] = new QOpenGLTexture(QImage("../data/liver_texture_2.jpg").mirrored());
//    texture[1] = new QOpenGLTexture(QImage("../data/deformed_02.JPG").mirrored());
    texture[2] = new QOpenGLTexture(QImage("../data/liver_texture_3.jpg").mirrored());
    texture[3] = new QOpenGLTexture(QImage("../data/liver_texture_4.jpg").mirrored());

    // Laparoscopy image.
    m_pBackgroundTexture = new QOpenGLTexture(m_InputImage.mirrored());

    for (int i = 0; i < 4; ++i)
    {
        texture[i]->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
        texture[i]->setMagnificationFilter(QOpenGLTexture::Linear);
    }

    m_pBackgroundTexture->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    m_pBackgroundTexture->setMagnificationFilter(QOpenGLTexture::Linear);

//    texture[0].setTexture(QString("../data/liver_texture_1.jpg"));
//    texture[1].setTexture(QString("../data/liver_texture_2.jpg"));
//    texture[2].setTexture(QString("../data/liver_texture_3.jpg"));
//    texture[3].setTexture(QString("../data/liver_texture_4.jpg"));
//    texture.setTexture(QString("../data/deformed_02.JPG"));
//    texture.setTexture(QString("../data/img/standardPicture1.jpg"));

#if LOADING_DEFAULT_3D_MODEL

    // Default model initialisation.
    Model model;
//    model.Load("../data/sphere.obj");
//    model.Load("../data/human_liver_cleaned.obj"); // For generating training/test set for CNNs.

    // For simulation.
       model.Load("../data/human_liver_simplified.obj");
//    model.Load("../data/human_liver_simplified_5000faces.obj");

//    model.Load("../data/SimpleLiver.obj");
//    model.Load("../data/img/modelRéduit50%.obj");

//    model.LoadTexture("../data/liver_texture_1.jpg");
    model.BuildVertexData();
    model.SetTexture(texture[0]);
    m_Models.push_back(model);

    // Create a vertex array object. In OpenGL ES 2.0 and OpenGL 2.x
    // implementations this is optional and support may not be present
    // at all. Nonetheless the below code works in all cases and makes
    // sure there is a VAO when one is needed.
    m_ModelVAO.create();
    QOpenGLVertexArrayObject::Binder modelVAOBinder(&m_ModelVAO);

    // Setup vertex buffer object.
    m_ModelVBO.create();
    m_ModelVBO.bind();
    m_ModelVBO.allocate(model.VertexData().data(), model.VertexData().size() * sizeof(GLfloat));

    // Store the vertex attribute bindings for the program.
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glEnableVertexAttribArray(0);
    f->glEnableVertexAttribArray(1);
    f->glEnableVertexAttribArray(2);
    f->glEnableVertexAttribArray(3);
    f->glEnableVertexAttribArray(4);
    f->glEnableVertexAttribArray(5);
    f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), 0);
    f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(3 * sizeof(GLfloat)));
    f->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(6 * sizeof(GLfloat)));
    f->glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(8 * sizeof(GLfloat)));
    f->glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(9 * sizeof(GLfloat)));
    f->glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(12 * sizeof(GLfloat)));

    m_ModelVBO.release();

    m_GroundTruthModelVAO.create();
    QOpenGLVertexArrayObject::Binder groundTruthModelVAOBinder(&m_GroundTruthModelVAO);

    m_GroundTruthModelVBO.create();
    m_GroundTruthModelVBO.bind();
    m_GroundTruthModelVBO.allocate(model.VertexData().data(), model.VertexData().size() * sizeof(GLfloat));

    f = QOpenGLContext::currentContext()->functions();
    f->glEnableVertexAttribArray(0);
    f->glEnableVertexAttribArray(1);
    f->glEnableVertexAttribArray(2);
    f->glEnableVertexAttribArray(3);
    f->glEnableVertexAttribArray(4);
    f->glEnableVertexAttribArray(5);
    f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), 0);
    f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(3 * sizeof(GLfloat)));
    f->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(6 * sizeof(GLfloat)));
    f->glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(8 * sizeof(GLfloat)));
    f->glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(9 * sizeof(GLfloat)));
    f->glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(12 * sizeof(GLfloat)));

    m_GroundTruthModelVBO.release();

#endif

    m_camera.setToIdentity();

    m_pShaderProgramModel->release();

    // Shader program for quad.
    m_pShaderProgramQuad = new QOpenGLShaderProgram;
    m_pShaderProgramQuad->addShaderFromSourceFile(QOpenGLShader::Vertex, "../src/shaders/quad.vert");
    m_pShaderProgramQuad->addShaderFromSourceFile(QOpenGLShader::Fragment, "../src/shaders/quad.frag");
    m_pShaderProgramQuad->bindAttributeLocation("vertex", 0);
//    m_pShaderProgramQuad->bindAttributeLocation("texCoord", 1);
    m_pShaderProgramQuad->link();

    m_pShaderProgramQuad->bind();
    m_QuadScaleLoc = m_pShaderProgramQuad->uniformLocation("scale");
//    m_projMatrixLoc = m_pShaderProgramModel->uniformLocation("projMatrix");
//    m_mvMatrixLoc = m_pShaderProgramModel->uniformLocation("mvMatrix");
//    m_normalMatrixLoc = m_pShaderProgramModel->uniformLocation("normalMatrix");
//    m_lightPosLoc = m_pShaderProgramModel->uniformLocation("lightPos");
    m_pShaderProgramQuad->setUniformValue("texture", 0);

    static const GLfloat quadVertexData[] =
    {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f
    };

    m_QuadVAO.create();
    QOpenGLVertexArrayObject::Binder quadVAOBinder(&m_QuadVAO);

    // Setup vertex buffer object.
    m_QuadVBO.create();
    m_QuadVBO.bind();
    m_QuadVBO.allocate(quadVertexData, 12 * sizeof(GLfloat));

    m_QuadVBO.bind();
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glEnableVertexAttribArray(0);
//    f->glEnableVertexAttribArray(1);
    f->glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
//    f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), reinterpret_cast<void *>(3 * sizeof(GLfloat)));
//    f->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), reinterpret_cast<void *>(6 * sizeof(GLfloat)));
    m_QuadVBO.release();

    m_pShaderProgramQuad->release();

    glEnable(GL_BLEND);                                 // Opacity ON
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Opacity parameters


//    // Shader program for contour.
//    m_pShaderProgramContour = new QOpenGLShaderProgram;
//    m_pShaderProgramContour->addShaderFromSourceFile(QOpenGLShader::Vertex, "../src/shaders/contour.vert");
//    m_pShaderProgramContour->addShaderFromSourceFile(QOpenGLShader::Fragment, "../src/shaders/contour.frag");
//    m_pShaderProgramContour->bindAttributeLocation("vertex", 0);
////    m_pShaderProgramQuad->bindAttributeLocation("texCoord", 1);
//    m_pShaderProgramContour->link();

//    m_pShaderProgramContour->bind();
//    m_ContourScaleLoc = m_pShaderProgramContour->uniformLocation("scale");

//    m_ContourVAO.create();
//    QOpenGLVertexArrayObject::Binder contourVAOBinder(&m_ContourVAO);

//    m_ContourVBO.create();
//    m_ContourVBO.bind();
//    f = QOpenGLContext::currentContext()->functions();
//    f->glEnableVertexAttribArray(0);
//    f->glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);
//    m_ContourVBO.release();

//    m_pShaderProgramContour->release();


//    resetTransformations();
    resetCameraSettings();
//    CentreModel();

    LoadCameraSettings();

    resizeWidget();

    QObject::connect(&m_MaterialModelSolver, SIGNAL(UpdateGL()), this, SLOT(update())); //SLOT(updateGL()));

    m_RotationTimer.setParent(this);
    connect(&m_RotationTimer, SIGNAL(timeout()), this, SLOT(Rotate()));

    m_FineRegistrationTimer.setParent(this);
    connect(&m_FineRegistrationTimer, SIGNAL(timeout()), this, SLOT(FineRegistration()));
    m_FineRegistrationCountStage1 = 0;
    m_FineRegistrationCountStage2 = 0;

    m_DataGenerationTimer.setParent(this);
    connect(&m_DataGenerationTimer, SIGNAL(timeout()), this, SLOT(RandomlyDeformModel()));
}

void GLWidget::LoadCameraSettings()
{
    // Load camera settings from a file.
    QFile file("./../tensorflow/liver_data/fine_registration/camera_settings.txt");
    file.open(QIODevice::ReadOnly);
    QTextStream stream(&file);

    QString line = stream.readLine();
    std::cout << line.toStdString() << std::endl << std::endl;

    line = stream.readLine();

    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    // Focal length x.
    line = stream.readLine();
    float val = line.toFloat();
    cameraParameters[0] = val;
    std::cout << val << std::endl;

    // Focal length y.
    line = stream.readLine();
    val = line.toFloat();
    cameraParameters[1] = val;
    std::cout << val << std::endl;

    // Skewness.
    line = stream.readLine();
    val = line.toFloat();
    cameraParameters[2] = val;
    std::cout << val << std::endl;

    // Optical centre x.
    line = stream.readLine();
    val = line.toFloat();
    cameraParameters[3] = val;
    std::cout << val << std::endl;

    // Optical centre y.
    line = stream.readLine();
    val = line.toFloat();
    cameraParameters[4] = val;
    std::cout << val << std::endl;

    // Near.
    line = stream.readLine();
    val = line.toFloat();
    cameraParameters[5] = val;
    std::cout << val << std::endl;

    // Far.
    line = stream.readLine();
    val = line.toFloat();
    cameraParameters[6] = val;
    std::cout << val << std::endl;

    file.close();
}

void GLWidget::InitialiseSimulation()
{
    // While there are several 3D models for the liver, tumor, veins, etc. for visualisation,
    // they should be put into one mesh for deformation.
    std::vector<Model::Vertex*>& vertices = m_ModelForSimulation.Vertices();
    std::vector<Model::Face*>& faces = m_ModelForSimulation.Faces();
    vertices.clear();
    faces.clear();

    int offset = 0;

    for (Model& model : m_Models)
    {
        for (Model::Vertex* pVertex : model.Vertices())
        {
            vertices.push_back(pVertex);
        }

        for (const Model::Face* pFace : model.Faces())
        {
            Model::Face* pNewFace = new Model::Face;

            for (int index : pFace->_VertexIndices)
            {
                pNewFace->_VertexIndices.push_back(index + offset);
            }

            faces.push_back(pNewFace);
        }

        offset += model.Vertices().size();
    }

    m_MaterialModelSolver.InitialiseCUDA(m_ModelForSimulation);
//    m_MaterialModelSolver.Initialise(m_ModelForSimulation);
    m_SimulationInitialised = true;
}

void GLWidget::UpdateModelVBO(int Index)
{
    makeCurrent();

    m_ModelVBOs[Index]->bind();

    glBufferData(GL_ARRAY_BUFFER, m_Models[Index].VertexData().size() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_Models[Index].VertexData().size() * sizeof(GLfloat), m_Models[Index].VertexData().data());

    m_ModelVBOs[Index]->release();

    doneCurrent();
}

void GLWidget::UpdateGroundTruthModelVBO(int Index)
{
    makeCurrent();

    m_GroundTruthModelVBOs[Index]->bind();

    glBufferData(GL_ARRAY_BUFFER, m_GroundTruthModels[Index].VertexData().size() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, m_GroundTruthModels[Index].VertexData().size() * sizeof(GLfloat), m_GroundTruthModels[Index].VertexData().data());

    m_GroundTruthModelVBOs[Index]->release();

    doneCurrent();
}

//void GLWidget::UpdateModelVBO()
//{
//    makeCurrent();

//    m_ModelVBO.bind();

//    glBufferData(GL_ARRAY_BUFFER, m_Models[0].VertexData().size() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
//    glBufferSubData(GL_ARRAY_BUFFER, 0, m_Models[0].VertexData().size() * sizeof(GLfloat), m_Models[0].VertexData().data());

//    m_ModelVBO.release();

//    doneCurrent();
//}

//void GLWidget::UpdateGroundTruthModelVBO()
//{
//    makeCurrent();

//    m_GroundTruthModelVBO.bind();

//    glBufferData(GL_ARRAY_BUFFER, m_GroundTruthModel.VertexData().size() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
//    glBufferSubData(GL_ARRAY_BUFFER, 0, m_GroundTruthModel.VertexData().size() * sizeof(GLfloat), m_GroundTruthModel.VertexData().data());

//    m_GroundTruthModelVBO.release();

//    doneCurrent();
//}

void GLWidget::Render2DImage(void)
{
//    glPushMatrix();

//    glMatrixMode(GL_MODELVIEW);                             // Set model matrix as current matrix
//    glLoadIdentity();
//    glEnable(GL_TEXTURE_2D);                                // Texture ON

    // Get a texture and scale it randomly.
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0, 4);
    int index = floor(dist(gen));
    float scale = dist(gen) * 0.25f;
    float sign = dist(gen) * 0.25f;

    if (sign < 0.5f)
    {
        sign = -1.0f;
    }
    else
    {
        sign = 1.0;
    }

    scale *= sign;

    // TODO: remove later!
    scale = 1;

    QOpenGLVertexArrayObject::Binder quadVAOBinder(&m_QuadVAO);
    m_pShaderProgramQuad->bind();

    m_pShaderProgramQuad->setUniformValue(m_QuadScaleLoc, scale);

    m_pBackgroundTexture->bind();

    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_pShaderProgramQuad->release();


//    glBegin(GL_QUADS);
////    glTexCoord2f(0 + shiftU, 0 + shiftV); glVertex2i(-1, -1);
////    glTexCoord2f(0 + shiftU, 1 + shiftV); glVertex2i(-1, 1);
////    glTexCoord2f(1 + shiftU, 1 + shiftV); glVertex2i(1, 1);
////    glTexCoord2f(1 + shiftU, 0 + shiftV); glVertex2i(1, -1);
//    glTexCoord2f(0, 0); glVertex2i(-1, -1);
//    glTexCoord2f(0, 1 * scale); glVertex2i(-1, 1);
//    glTexCoord2f(1 * scale, 1 * scale); glVertex2i(1, 1);
//    glTexCoord2f(1 * scale, 0); glVertex2i(1, -1);
//    glEnd();

//    glDisable(GL_TEXTURE_2D);                               // Texture OFF (enables color)
//    glDisable(GL_LIGHT1);
//    glDisable(GL_LIGHTING);

//    glPopMatrix();
}

void GLWidget::Render2DImageFixedPipeline(void)
{
    glPushMatrix();

    glMatrixMode(GL_MODELVIEW);                             // Set model matrix as current matrix
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);                                // Texture ON

    // Get a texture and scale it randomly.
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0, 4);
    int index = floor(dist(gen));
    float scale = dist(gen) / 4.0f;

    int sign = floor(dist(gen));

    if (sign <= 1)
    {
        sign = -1;
    }
    else
    {
        sign = 1;
    }

    scale *= (float)sign;

    //glBindTexture(GL_TEXTURE_2D, texture[index].getTexture()); // Texture load
    texture[index]->bind();

    // TODO: Temp.
    static const GLfloat dir_light[4] = { 0.0f, 0.0f, -1.0f, 1.0f };    // Light vector
    GLfloat lightDiffuseColour[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat lightAmbientColour[4] = { 0.2f, 0.2f, 0.2f, 1.0f };

    glLightfv(GL_LIGHT1, GL_POSITION, dir_light);       // Setting light
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightDiffuseColour);
    glLightfv(GL_LIGHT1, GL_AMBIENT, lightAmbientColour);
    glEnable(GL_COLOR_MATERIAL);                        // Model color ON
    glEnable(GL_LIGHT1);                                // LIGHT0 ON
    glEnable(GL_LIGHTING);                              // Shadows ON

    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);


    glBegin(GL_QUADS);
//    glTexCoord2f(0 + shiftU, 0 + shiftV); glVertex2i(-1, -1);
//    glTexCoord2f(0 + shiftU, 1 + shiftV); glVertex2i(-1, 1);
//    glTexCoord2f(1 + shiftU, 1 + shiftV); glVertex2i(1, 1);
//    glTexCoord2f(1 + shiftU, 0 + shiftV); glVertex2i(1, -1);
    glTexCoord2f(0, 0); glVertex2i(-1, -1);
    glTexCoord2f(0, 1 * scale); glVertex2i(-1, 1);
    glTexCoord2f(1 * scale, 1 * scale); glVertex2i(1, 1);
    glTexCoord2f(1 * scale, 0); glVertex2i(1, -1);
    glEnd();

    glDisable(GL_TEXTURE_2D);                               // Texture OFF (enables color)
    glDisable(GL_LIGHT1);
    glDisable(GL_LIGHTING);

    glPopMatrix();
}

void GLWidget::RenderContour()
{
    QOpenGLVertexArrayObject::Binder contourVAOBinder(&m_ContourVAO);
    m_pShaderProgramContour->bind();

    glDrawArrays(GL_LINES, 0, m_Models[0].Contour().size() * 2);

    m_pShaderProgramContour->release();
}

/* ============================ DRAWING LOOP ============================ */
void GLWidget::paintGL()
{
    QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Buffers cleared

//    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);   // Light only applied on the exposed sight
//    glEnable(GL_NORMALIZE);                             // Normalization vectors ON
//    glShadeModel(GL_SMOOTH);
//    glCullFace(GL_BACK);


//    glMatrixMode(GL_PROJECTION);                        // Set projection matrix as current matrix
//    glLoadIdentity();
//    glMatrixMode(GL_MODELVIEW);                             // Set model matrix as current matrix
//    glLoadIdentity();

    GLfloat scale = frame_picture_Ratio * cameraParameters[7];

    if (m_RenderingBackgroundImage)
    {
        // Render the background image.
        glDepthMask(GL_FALSE);
        glDisable(GL_DEPTH_TEST);

        glViewport(m_InputImage.width() * scale, m_InputImage.height() * scale, // Drawable area in the widget
                   (GLint)width() - (2 * m_InputImage.width() * scale),
                   (GLint)height() - (2 * m_InputImage.height() * scale));

        Render2DImage();
    }

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    // PROJECTION CAMERA
    glViewport(0, 0, (GLint)width(), (GLint)height());

    camera();

//    glGetDoublev(GL_MODELVIEW_MATRIX, m_ModelViewMatrix);
//    glGetDoublev(GL_PROJECTION_MATRIX, m_ProjectionMatrix);
//    glGetIntegerv(GL_VIEWPORT, m_Viewport);

//// TAGS
//    glPushMatrix();
//        glColor3f(1.f, 1.f, 1.f);
//        glCallList(tags);
//    glPopMatrix();


//// TUMOR
//    glPushMatrix();
//        glColor3f(0.f, 0.f, 1.f);
//        glTranslatef(coordTumor.x(),coordTumor.y(),coordTumor.z());
//        glCallList(tumor);
//    glPopMatrix();

// MODELS

    // TODO: Temp - render points distributed on a hemisphere.
    float latitude = 0.0f, longitude = 0.0f;
    unsigned int numOfPoints = 100; // Training set: 1000, test set: 100.
    Eigen::Vector3f cameraPos(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f cameraUp(0.0f, 1.0f, 0.0f);

//    static bool f = true;
//    static int count = 0;
//    static unsigned int i = numOfPoints * 2 - 1;

    if (m_GeneratingTrainingSet)
    {
//        GLUquadric* params = gluNewQuadric();

//            for (unsigned int i = numOfPoints * 2 - 1; i >= 0; --i) // Double the number of points because the half of the original points are not drawn.
        {
            DistributePointsOnSphere(numOfPoints * 2, m_PointsOnSphereIndex, latitude, longitude);

//            if (f)
            {
                std::cout << "Index: " << (int)numOfPoints * 2 - 1 - m_PointsOnSphereIndex << ", lat: " << latitude << ", lon: " << longitude << std::endl;
            }

            if (latitude <= (M_PI * 0.5f))
            {
                cameraPos[0] = m_CameraRadius * cos(longitude) * sin(latitude);
                cameraPos[2] = m_CameraRadius * sin(longitude) * sin(latitude);
                cameraPos[1] = m_CameraRadius * cos(latitude);

                cameraPos += m_CameraPositionNoise;
//                cameraUp[0] += m_CameraRollNoise;
//                cameraUp.normalize();

//                glPushMatrix();


                // Compute the camera rotation w.r.t. (0.0f, 0.0f, -1.0f).
                Eigen::Vector3f vec = m_CameraLookAtNoise - cameraPos;
                vec.normalize();
                float angle = acos(vec.dot(Eigen::Vector3f(0.0f, 0.0f, -1.0f)));
                Eigen::Vector3f axis = vec.cross(Eigen::Vector3f(0.0f, 0.0f, -1.0f));
                axis.normalize();

                // Roll the camera by the noise.
                Eigen::Vector3f vecLeft = vec.cross(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
                vecLeft.normalize();

                Eigen::Vector3f vecUp = vecLeft.cross(vec);
                cameraUp = vecUp;

//                Eigen::AngleAxisf roll = Eigen::AngleAxisf(m_CameraRollNoise, vec);
//                cameraUp = roll * cameraUp;

//                gluLookAt(cameraPos[0], cameraPos[1], cameraPos[2], m_CameraLookAtNoise[0], m_CameraLookAtNoise[1], m_CameraLookAtNoise[2], cameraUp[0], cameraUp[1], cameraUp[2]);

//                glLoadIdentity();

                Eigen::Matrix4f matRotation;
                matRotation << vecLeft[0], vecLeft[1], vecLeft[2], 0.0f, vecUp[0], vecUp[1], vecUp[2], 0.0f, -vec[0], -vec[1], -vec[2], 0.0f, 0.0f, 0.0f, 0.0f, 1.0f;

                QMatrix4x4 mat(vecLeft[0], vecLeft[1], vecLeft[2], 0.0f, vecUp[0], vecUp[1], vecUp[2], 0.0f, -vec[0], -vec[1], -vec[2], 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

//                QMatrix4x4 mat(vecLeft[0], vecUp[0], -vec[0], 0.0f, vecLeft[1], vecUp[1], -vec[1], 0.0f, vecLeft[2], vecUp[2], -vec[2], 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

//                glMultMatrixf(matRotation.data());
//                glTranslatef(-cameraPos[0], -cameraPos[1], -cameraPos[2]);

//                m_camera.setToIdentity();
                m_camera = mat;
                m_camera.translate(-cameraPos[0], -cameraPos[1], -cameraPos[2]);

                m_CameraPosition = cameraPos;

//                float rollAngle = acos(cameraUp.dot(Eigen::Vector3f(0.0f, 1.0f, 0.0f)));
                //Eigen::Vector3f rollAxis = cameraUp.cross(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
                //m_CameraRotation = /*roll * */ Eigen::AngleAxisf(angle, axis);
                m_CameraRotation.fromRotationMatrix(matRotation.block<3, 3>(0, 0));

//                glLoadIdentity();
////                gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0);
//                glRotatef(m_CameraRotation.angle(), m_CameraRotation.axis()[0], m_CameraRotation.axis()[1], m_CameraRotation.axis()[2]);
//                glTranslatef(m_CameraPosition[0], m_CameraPosition[1], m_CameraPosition[2]);



//                glTranslatef(cameraPos[0], cameraPos[1], cameraPos[2]);
//                gluSphere(params, 0.001f, 20, 20);

//                glPopMatrix();

//                ++m_FrameCount;

//                if (m_FrameCount >= 1)
//                {
//                    m_FrameCount = 0;
//                    --m_PointsOnSphereIndex;

//                    if (m_PointsOnSphereIndex < 0)
//                    {
//                        // Finish.
//                        m_GeneratingTrainingSet = false;
//                    }
//                }
            }
            else
            {
//                f = false;
            }
        }
    }


    Eigen::Vector4f liverColour(1.0f, 1.0f, 1.0f, opacity);

    // The light source is on the camera 'in the eye space', which is (0, 0, 0).
    QVector3D lightPos(0, 0, 0);

    m_world.setToIdentity();
//        m_world.rotate(180.0f - (m_xRot / 16.0f), 1, 0, 0);
//        m_world.rotate(m_yRot / 16.0f, 0, 1, 0);
//        m_world.rotate(m_zRot / 16.0f, 0, 0, 1);

#if 1

//        m_camera.scale(scaleFactor);
    m_camera.setToIdentity();
    m_camera.translate(coordModels[0], coordModels[1], coordModels[2]); // Keyboard and wheel translation

    QMatrix4x4 m;
    m.setToIdentity();

    if (m_RotatingCamera)
    {
        m = m_PrevCameraRotation;
        m.translate(m_ModelCentroid[0], m_ModelCentroid[1], m_ModelCentroid[2]);
        m.rotate(trackball.rotation());
        m.translate(-m_ModelCentroid[0], -m_ModelCentroid[1], -m_ModelCentroid[2]);
    }
    else
    {
        m.rotate(trackball.rotation());
    }

    m_camera *= m;

#endif

    for (unsigned int i = 0; i < m_Models.size(); ++i)
    {
        Model& model = m_Models[i];

#if NOT_FOR_CNN

        liverColour << 0.7f, 0.0f, 0.7f, opacity;

#else

        liverColour << 0.1f, 0.1f, 0.1f, 1.0f;

#endif

//        model.SetColour(liverColour);

        QOpenGLVertexArrayObject::Binder modelVAOBinder(m_ModelVAOs[i]);
        m_pShaderProgramModel->bind();
        m_pShaderProgramModel->setUniformValue(m_projMatrixLoc, m_proj);

//        if (!m_ModelDataLoaded)
        {
            m_ModelView = m_camera * m_world;
        }
//        else
//        {
//            m_ModelView = m_camera * m_world * m_LoadedModelView;
//            m_ModelDataLoaded = false;
//        }

        // For CNN prediction test.
#if 0

        Eigen::Quaternionf quat(0.21625039,  0.02245756,  1.00799429,  0.11998872);
        quat.normalize();
        Eigen::Matrix3f mat = quat.matrix();
        m_ModelView = QMatrix4x4(mat(0), mat(3), mat(6), 0.0f, mat(1), mat(4), mat(7), 0.0f, mat(2), mat(5), mat(8), 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
        QVector3D translation(-0.08156733,
                              0.05464094, -0.15159416);
        m_ModelView.translate(-translation);

//        modelViewMatrix = m_camera * modelViewMatrix;

//        Eigen::Quaternionf quat(0.25866744,  0.05707699,  1.03581452,  0.00339718);
//        quat.normalize();
//        Eigen::Matrix3f mat = quat.matrix();
//        GLfloat matGL[16] = { mat(0), mat(1), mat(2), 0.0f, mat(3), mat(4), mat(5), 0.0f, mat(6), mat(7), mat(8),0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
//        glMultMatrixf(matGL);

//        glTranslatef(0.15116364,
//                     -0.07748976, 0.20997973);

#endif

        m_pShaderProgramModel->setUniformValue(m_mvMatrixLoc, m_ModelView);

        QMatrix3x3 normalMatrix = m_ModelView.normalMatrix();
        m_pShaderProgramModel->setUniformValue(m_normalMatrixLoc, normalMatrix);
        m_pShaderProgramModel->setUniformValue(m_lightPosLoc, lightPos);

        // TODO: Temp - get a random texture.
        // Get a random texture.
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dist(0, 4);
        int index = floor(dist(gen));

        // Scale the texture randomly.
        float scale = dist(gen) * 0.25f;

        float sign = dist(gen) * 0.25f;

        if (sign < 0.5f)
        {
            sign = -1.0f;
        }
        else
        {
            sign = 1.0f;
        }

        scale *= sign;

        // TODO: remove later!
//        scale = 1;

        m_pShaderProgramModel->setUniformValue(m_ModelTexScaleLoc, scale);

        texture[index]->bind();

        // TODO: Temp.
//        m_pShaderProgramModel->setUniformValue(m_ModelCLoc, m_c);
//        m_pShaderProgramModel->setUniformValue(m_ModelCLoc, 0.015f);
        m_pShaderProgramModel->setUniformValue(m_ModelCLoc, m_cForRendering);

        glDrawArrays(GL_TRIANGLES, 0, model.VertexData().size() / 15);

        m_pShaderProgramModel->release();
    }

    if (m_RenderingModelFaces)
    {
        glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, m_pDepthData);

        for (Model& model : m_Models)
        {
            model.RenderFaces(m_ModelView, m_proj, m_Viewport);
        }
    }

    // Render the ground truth model.
    if (m_RenderingGroundTruthModel)
    {
        for (int i = 0; i < m_GroundTruthModels.size(); ++i)
        {
            Model& model = m_GroundTruthModels[i];

            QOpenGLVertexArrayObject::Binder groundTruthModelVAOBinder(m_GroundTruthModelVAOs[i]);
            m_pShaderProgramModel->bind();
            m_pShaderProgramModel->setUniformValue(m_projMatrixLoc, m_proj);

            m_pShaderProgramModel->setUniformValue(m_mvMatrixLoc, m_GroundTruthModelModelView);

            QMatrix3x3 normalMatrix = m_GroundTruthModelModelView.normalMatrix();
            m_pShaderProgramModel->setUniformValue(m_normalMatrixLoc, normalMatrix);

            m_pShaderProgramModel->setUniformValue(m_lightPosLoc, lightPos);

            glDrawArrays(GL_TRIANGLES, 0, m_GroundTruthModels[i].VertexData().size() / 15);

            m_pShaderProgramModel->release();
        }
    }

    // Rendering the vertex errors.
    if (0) // m_VertexErrors.size() > 0)
    {
        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glPointSize(4);

        glBegin(GL_POINTS);

        QVector3D point2D;

        for (Model& model : m_Models)
        {
            int index = 0;

            for (const Model::Vertex* pVertex : model.Vertices())
            {
                glColor4f(m_VertexErrorColours[index][0], m_VertexErrorColours[index][1], m_VertexErrorColours[index][2], m_VertexErrorColours[index][3]);

                point2D = model.ProjectPointOnto2D(QVector3D(pVertex->_Pos[0], pVertex->_Pos[1], pVertex->_Pos[2]), m_ModelView, m_proj, m_Viewport);

                glVertex2f(point2D.x(), point2D.y());

                ++index;
            }
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

//    //TODO: temp.
//    {


//        glPushMatrix();

//        glMatrixMode(GL_PROJECTION);
//        glLoadIdentity();
//        glOrtho(0, width(), 0, height(), -1, 1);
//        glMatrixMode(GL_MODELVIEW);
//        glLoadIdentity();

//        glDisable(GL_DEPTH_TEST);
//        glDisable(GL_LIGHTING);
//        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);

//        glBegin(GL_POINTS);

//        for (std::vector<QVector3D>& points : m_allSampledPoints3D)
//        {
//            QVector3D p = Model::ProjectPointOnto2D(points[0], m_ModelView, m_proj, m_Viewport);

//            glVertex2f(p.x(), p.y());
//        }

//        for (std::vector<QVector3D>& points : m_allSampledPixelCoords)
//        {
//            QVector3D p = points[0]; //Model::ProjectPointOnto2D(points[0], m_ModelView, m_proj, m_Viewport);

//            glVertex2f(p.x(), p.y());
//        }



//        glEnd();

//        glEnable(GL_DEPTH_TEST);
//        glEnable(GL_LIGHTING);

//        glPopMatrix();

//    }

    // Render high curvature vertices.
    if (m_SelectingHighCurvatureVertices && m_HighCurvatureVertexIndices.size() > 0)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glPointSize(6);
        glLineWidth(6.0f);

//        glBegin(GL_LINES);
        glBegin(GL_LINE_STRIP);

        Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);

        Eigen::Vector2f point2D;

//        for (Point& point : m_HighCurvaturePoints)
//        {
//            point2D << point.x(), point.y();
//            point2D *= cameraParameters[7];
//            point2D += scale * dimension;

//            glVertex2f(point2D[0], point2D[1]);
//        }

//        for (int i = 0; i < m_HighCurvatureVertexIndices.size() - 1; ++i)
//        {
//            int index0 = m_HighCurvatureVertexIndices[i];
//            int index1 = m_HighCurvatureVertexIndices[i + 1];
//            Point point0 = Model::ProjectVertexOnto2D(m_Models[0].Vertices()[index0], m_ModelView, m_proj, m_Viewport);
//            Point point1 = Model::ProjectVertexOnto2D(m_Models[0].Vertices()[index1], m_ModelView, m_proj, m_Viewport);
//            glVertex2f(point0.x(), point0.y());
//            glVertex2f(point1.x(), point1.y());
//        }

        for (QVector3D point : m_HighCurvatureFittedCurve)
        {
            QVector3D point2D = Model::ProjectPointOnto2D(point, m_ModelView, m_proj, m_Viewport);
            glVertex2f(point2D.x(), point2D.y());
        }

        glEnd();


        glBegin(GL_POINTS);

        glColor4f(0.0f, 0.0f, 1.0f, 1.0f);

        for (int index : m_HighCurvatureVertexIndices)
        {
            Point point = Model::ProjectVertexOnto2D(m_Models.back().Vertices()[index], m_ModelView, m_proj, m_Viewport);
            glVertex2f(point.x(), point.y());
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    // Render the ground truth model contour for fine registration.
    if (m_RenderingGroundTruthModelContour > 0)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glPointSize(6);

        glBegin(GL_POINTS);

        Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

        unsigned int index = 0;

//        for (Eigen::Vector2f point : m_GroundTruthModelContour)
        for (std::vector<Eigen::Vector2f> contour : m_ImageContours)
        {
            if (index == m_SelectedImageContourIndex)
            {
                glColor4f(1.0f, 0.0f, 0.2f, 1.0f);
            }
            else
            {
                glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
            }

            // TODO: Temp for colouring ligament contours differently.
            if (index == 2)
            {
                glColor4f(0.0f, 0.2f, 1.0f, 1.0f);
            }

            for (Eigen::Vector2f point : contour)
            {
                point *= cameraParameters[7];
                point += scale * dimension;

                if (m_RenderingGroundTruthModelContour == 2)
                {
                    // Skip those outside FOV.
                    if (!InsideFOV(point))
                    {
                        continue;
                    }
                }

                glVertex2f(point[0], point[1]);
            }

            ++index;
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    // Render the model contour.
    if (m_SelectingContour)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        GetContour(m_ModelContour);

        // TODO: Rendering the contour with shader does not work now.
        //        RenderContour();

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glPointSize(6);

        glBegin(GL_POINTS);

        for (std::vector<Eigen::Vector2f>::iterator itPoint = m_ModelContour.begin(); itPoint != m_ModelContour.end(); ++itPoint)
        {
            glVertex2f((*itPoint)[0], (*itPoint)[1]);
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    // Render the tetrahedra of the model after triangulation.
    if (0)//m_SimulationInitialised)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);

        glBegin(GL_LINES);

        Eigen::MatrixXf& x1 = m_MaterialModelSolver.X1();
        Eigen::MatrixXf& x2 = m_MaterialModelSolver.X2();
        Eigen::MatrixXf& x3 = m_MaterialModelSolver.X3();
        Eigen::MatrixXf& x4 = m_MaterialModelSolver.X4();

//        for (Rt::Finite_cells_iterator it = m_MaterialModelSolver.Triangulation().finite_cells_begin(); it != m_MaterialModelSolver.Triangulation().finite_cells_end(); ++it)
        for (int i = 0; i < x1.cols(); ++i)
        {
            QVector3D a(x1(0, i), x1(1, i), x1(2, i));
            QVector3D b(x2(0, i), x2(1, i), x2(2, i));
            QVector3D c(x3(0, i), x3(1, i), x3(2, i));
            QVector3D d(x4(0, i), x4(1, i), x4(2, i));
//            QVector3D a(it->vertex(0)->point().x(), it->vertex(0)->point().y(), it->vertex(0)->point().z());
//            QVector3D b(it->vertex(1)->point().x(), it->vertex(1)->point().y(), it->vertex(1)->point().z());
//            QVector3D c(it->vertex(2)->point().x(), it->vertex(2)->point().y(), it->vertex(2)->point().z());
//            QVector3D d(it->vertex(3)->point().x(), it->vertex(3)->point().y(), it->vertex(3)->point().z());

            a = Model::ProjectPointOnto2D(a, m_ModelView, m_proj, m_Viewport);
            b = Model::ProjectPointOnto2D(b, m_ModelView, m_proj, m_Viewport);
            c = Model::ProjectPointOnto2D(c, m_ModelView, m_proj, m_Viewport);
            d = Model::ProjectPointOnto2D(d, m_ModelView, m_proj, m_Viewport);

            glVertex2f(a.x(), a.y()); glVertex2f(b.x(), b.y());
            glVertex2f(a.x(), a.y()); glVertex2f(c.x(), c.y());
            glVertex2f(a.x(), a.y()); glVertex2f(d.x(), d.y());
            glVertex2f(b.x(), b.y()); glVertex2f(c.x(), c.y());
            glVertex2f(c.x(), c.y()); glVertex2f(d.x(), d.y());
            glVertex2f(d.x(), d.y()); glVertex2f(b.x(), b.y());
        }

//        for (std::vector<Eigen::Vector2f>::iterator itPoint = m_ModelContour.begin(); itPoint != m_ModelContour.end(); ++itPoint)
//        {
//            glVertex2f((*itPoint)[0], (*itPoint)[1]);
//        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }


#if 0 // For rendering colourbar.

    glViewport(0, 0, (GLint)width(), (GLint)height());

    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width(), 0, height(), -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);

    glPointSize(5);

    glBegin(GL_POINTS);

    Eigen::Vector4f colour;

    for (int i = 0; i < 256; ++i)
    {
        float error = (float)i / 255.0f;

        colour << error, 0.0f, 1.0f - error, 1.0f;

        if (error <= 0.5f)
        {
            colour[1] = 2.0f * colour[0];
        }
        else if (error > 0.5f)
        {
            colour[1] = 2.0f * colour[2];
        }

        glColor4f(colour[0], colour[1], colour[2], colour[3]);

        for (int j = 0; j < 50; ++j)
        {
            glVertex2f(50 + i * 5, -j * 5 + 500);
        }
    }

    glEnd();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();


#endif

    if (m_IsContourSelectionOn)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glLineWidth(6.0f);

        Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

        glBegin(GL_LINE_STRIP);

        for (Eigen::Vector2f point : m_ContourSelectionPoints)
        {
            point *= cameraParameters[7];
            point += scale * dimension;

            glVertex2f(point[0], point[1]);
        }

        glEnd();


        glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
        glPointSize(8);

        glBegin(GL_POINTS);

        for (Eigen::Vector2f point : m_ContourSelectionPoints)
        {
            point *= cameraParameters[7];
            point += scale * dimension;

            glVertex2f(point[0], point[1]);
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    if (m_SelectingShadingOptimisationRegion)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        glLineWidth(6.0f);

        Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());
        Eigen::Vector2f centre;

        glBegin(GL_LINE_LOOP);

        for (int i = 0; i < 50; ++i)
        {
            float theta = 2.0f * M_PI * float(i) / 50.0f;
            float x = m_ShadingOptimisationRegionRadius * cos(theta);
            float y = m_ShadingOptimisationRegionRadius * sin(theta);

            x *= cameraParameters[7];
            y *= cameraParameters[7];

            centre = m_ShadingOptimisationRegionCentre;
            centre *= cameraParameters[7];
            centre += scale * dimension;

            glVertex2f(x + centre[0], y + centre[1]);
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    // Render FOV.
    if (m_FOVScale < 1.0f)
    {
        glViewport(m_InputImage.width() * scale, m_InputImage.height() * scale, // Drawable area in the widget
                   (GLint)width() - (2 * m_InputImage.width() * scale),
                   (GLint)height() - (2 * m_InputImage.height() * scale));

//        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(6.0f);

        glBegin(GL_LINE_STRIP);

        glVertex2f(m_Viewport[0] + m_Viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0],                 m_Viewport[1] + m_Viewport[3] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[1]);
        glVertex2f(m_Viewport[0] + m_Viewport[2] - m_Viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0], m_Viewport[1] + m_Viewport[3] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[1]);
        glVertex2f(m_Viewport[0] + m_Viewport[2] - m_Viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0], m_Viewport[1] + m_Viewport[3] - m_Viewport[3] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[1]);
        glVertex2f(m_Viewport[0] + m_Viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0],                 m_Viewport[1] + m_Viewport[3] - m_Viewport[3] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[1]);
        glVertex2f(m_Viewport[0] + m_Viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0],                 m_Viewport[1] + m_Viewport[3] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[1]);

        glEnd();

        glLineWidth(1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    // Render the contour offset vectors.
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);

        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
        glLineWidth(1.0f);

        glBegin(GL_LINES);

        Eigen::Vector2f point0, point1;

        for (unsigned int i = 0; i < m_ContourOffsetVectors.size(); i += 2)
        {
            point0 = m_ContourOffsetVectors[i];
            point1 = m_ContourOffsetVectors[i + 1];

            glVertex2f(point0[0], point0[1]);
            glVertex2f(point1[0], point1[1]);
        }

        glEnd();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);

        glPopMatrix();
    }

    // Render vertex selection rect.
    if (m_SelectingMultipleVertices)
    {
        glViewport(0, 0, (GLint)width(), (GLint)height());

        glPushMatrix();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
        glPushAttrib(GL_ENABLE_BIT);

        glLineStipple(3, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.top());
        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.top());

        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.top());
        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.bottom());

        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.bottom());
        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.bottom());

        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.bottom());
        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.top());
        glEnd();

        glPopAttrib();
        glPopMatrix();
    }


#if 0

// CROSSHAIR
    glColor4f(1.f, 1.f, 1.f, 1.f);
    glDisable(GL_DEPTH_TEST);                           // Depth buffer OFF
    glDisable(GL_LIGHTING);                             // Light OFF

    glEnable(GL_COLOR_LOGIC_OP);
    glLogicOp(GL_XOR);                                  // Negative color
    glCallList(crosshair);                              // Drawing crosshair
    glDisable(GL_COLOR_LOGIC_OP);


// TEXT
    QFontMetrics fm(QFont("Lato Black", 16));
    QString depth = QString("Depth: %1 cm").arg(QString::number(-coordModels.z()/10, 'f', 2));
    renderText(width()-fm.width(depth, -1)-5, 25, depth, QFont("Lato Black", 16));

    if(distanceBetweenTags)
        renderText(10, 50, QString("Distance between tags: %1 cm").arg(QString::number(abs(distanceBetweenTags/10), 'f', 2)), QFont("Lato Black", 16));

    if(crosshair)
    {
        renderText(10, 25, QString("X: %1    Y: %2    Z: %3")
                .arg(QString::number(surfaceCoordinates.x(), 'f', 2))
                .arg(QString::number(surfaceCoordinates.y(), 'f', 2))
                .arg(QString::number(surfaceCoordinates.z()-coordModels.z(), 'f', 2)), QFont("Lato Black", 16));

        if(tumor)
        {
            QVector3D temp = m * coordTumor;

            GLfloat distanceToTumor = sqrt(pow(surfaceCoordinates.x()-temp.x(),2)
                    +pow(surfaceCoordinates.y()-temp.y(),2)
                    +pow(surfaceCoordinates.z()+temp.z(),2));

            renderText(10, height()-10, QString("Distance to tumor: %1 cm").arg(QString::number(abs((distanceToTumor/10)-(tumorRadius/scaleFactor/10)), 'f', 2)), QFont("Lato Black", 16));
        }
        crosshair = 0;
    }

#endif

}

void GLWidget::paintGLFixedPipeline()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Buffers cleared

    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);   // Light only applied on the exposed sight
    glEnable(GL_NORMALIZE);                             // Normalization vectors ON
    glShadeModel(GL_SMOOTH);
    glCullFace(GL_BACK);


    glMatrixMode(GL_PROJECTION);                        // Set projection matrix as current matrix
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);                             // Set model matrix as current matrix
    glLoadIdentity();


//#if NOT_FOR_CNN

    glDisable(GL_DEPTH_TEST);

    GLfloat scale = 1.0f; //frame_picture_Ratio*cameraParameters[7];
    glViewport(m_pBackgroundTexture->width()*scale, m_pBackgroundTexture->height()*scale, // Drawable area in the widget
               (GLint)width()-(2*m_pBackgroundTexture->width()*scale),
               (GLint)height()-(2*m_pBackgroundTexture->height()*scale));

    // Render the background image.
    Render2DImage();

//#endif



// PROJECTION CAMERA
    glViewport(0, 0, (GLint)width(), (GLint)height());
    glEnable(GL_DEPTH_TEST);                                // Depth buffer ON


#if NOT_FOR_CNN

    cameraFixedPipeline();

#else

    QMatrix4x4 perspective(cameraParameters[0] * cameraParameters[7], cameraParameters[2] * cameraParameters[7], -cameraParameters[3] * cameraParameters[7], 0,
                           0, cameraParameters[1] * cameraParameters[7], -cameraParameters[4] * cameraParameters[7], 0,
                           0, 0, cameraParameters[5] + cameraParameters[6], cameraParameters[5] * cameraParameters[6],
                           0, 0, -1, 0);

     glMatrixMode(GL_PROJECTION);
     glOrtho(0, (GLfloat)width(), 0, (GLfloat)height(), cameraParameters[5], cameraParameters[6]);        // Setting orthographic camera matrix
     multMatrix(perspective);                            // Multiplication by perspective matrixz

#endif


    glMatrixMode(GL_MODELVIEW);


#if NOT_FOR_CNN

    static const GLfloat dir_light[4] = {0.2f, 0.2f, 0.5f, 1.f};    // Light vector

    glLightfv(GL_LIGHT0, GL_POSITION, dir_light);       // Setting light

    glEnable(GL_COLOR_MATERIAL);                        // Model color ON
    glEnable(GL_LIGHT0);                                // LIGHT0 ON
    glEnable(GL_LIGHTING);                              // Shadows ON

#else

    // Set the light transform same as that of the camera.
//    Eigen::Matrix3f mat(m_ModelViewMatrix[0], m_ModelViewMatrix[1], m_ModelViewMatrix[2],
//                        m_ModelViewMatrix[4], m_ModelViewMatrix[5], m_ModelViewMatrix[6],
//                        m_ModelViewMatrix[8], m_ModelViewMatrix[9], m_ModelViewMatrix[10]);
//    mat = mat.inverse();
    GLfloat dir_light[4] = {0.0f, 0.0f, -1.0f, 1.0f};    // Light vector
    GLfloat lightPosition[4] = {0.0f, 0.0f, 0.2f, 1.0f};

    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, dir_light);       // Setting light
    glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 120);
    glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 150);
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0);
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 2);
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 5);

    glEnable(GL_COLOR_MATERIAL);                        // Model color ON
    glEnable(GL_LIGHT0);                                // LIGHT0 ON
    glEnable(GL_LIGHTING);                              // Shadows ON

#endif



#if NOT_FOR_CNN

 // TRANSFORMATIONS
    glScalef(scaleFactor,scaleFactor,scaleFactor);
    glTranslatef(coordModels.x(),coordModels.y(),coordModels.z());  // Keyboard and wheel translation

    QMatrix4x4 m;
    m.rotate(trackball.rotation());
    multMatrix(m);                             // Applies trackball rotation on current matrix

#endif

    glGetDoublev(GL_MODELVIEW_MATRIX, m_ModelViewMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, m_ProjectionMatrix);
    glGetIntegerv(GL_VIEWPORT, m_Viewport);

// TAGS
    glPushMatrix();
        glColor3f(1.f, 1.f, 1.f);
        glCallList(tags);
    glPopMatrix();


// TUMOR
    glPushMatrix();
        glColor3f(0.f, 0.f, 1.f);
        glTranslatef(coordTumor.x(),coordTumor.y(),coordTumor.z());
        glCallList(tumor);
    glPopMatrix();

// MODELS
//    // Set the model transformation.
//    QQuaternion rot = trackball.rotation();
//    QVector3D axis;
//    float angle = 0.0f;
//    rot.getAxisAndAngle(&axis, &angle);
//    axis.normalize();
//    Eigen::AngleAxisf angleAxis(angle, Eigen::Vector3f(axis.x(), axis.y(), axis.z()));
//    Eigen::Quaternionf quaternion(angleAxis);

//    Eigen::Affine3f transform;
//    transform = angleAxis; //.rotate(quaternion);
//    transform.pretranslate(coordModels);
//    transform.prescale(Eigen::Vector3f(scaleFactor, scaleFactor, scaleFactor));
//    m_Models[0].Transform() = transform;


    // TODO: Temp - render points distributed on a hemisphere.
    float latitude = 0.0f, longitude = 0.0f;
    unsigned int numOfPoints = 1000; // Training set: 1000, test set: 100.
    Eigen::Vector3f cameraPos(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f cameraUp(0.0f, 1.0f, 0.0f);

//    static bool f = true;
//    static int count = 0;
//    static unsigned int i = numOfPoints * 2 - 1;


    /******************************/
    // For CNN prediction test.

#if NOT_FOR_CNN

    glLoadIdentity();

    Eigen::Quaternionf quat(0.25866744,  0.05707699,  1.03581452,  0.00339718);
    quat.normalize();
    Eigen::Matrix3f mat = quat.matrix();
    GLfloat matGL[16] = { mat(0), mat(1), mat(2), 0.0f, mat(3), mat(4), mat(5), 0.0f, mat(6), mat(7), mat(8),0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    glMultMatrixf(matGL);

    glTranslatef(0.15116364,
                 -0.07748976, 0.20997973);

#endif

    /******************************/


    if (m_GeneratingTrainingSet)
    {
//        GLUquadric* params = gluNewQuadric();

//            for (unsigned int i = numOfPoints * 2 - 1; i >= 0; --i) // Double the number of points because the half of the original points are not drawn.
        {
            DistributePointsOnSphere(numOfPoints * 2, m_PointsOnSphereIndex, latitude, longitude);

//            if (f)
            {
                std::cout << "Index: " << (int)numOfPoints * 2 - 1 - m_PointsOnSphereIndex << ", lat: " << latitude << ", lon: " << longitude << std::endl;
            }

            if (latitude <= (M_PI * 0.5f))
            {
                cameraPos[0] = m_CameraRadius * cos(longitude) * sin(latitude);
                cameraPos[2] = m_CameraRadius * sin(longitude) * sin(latitude);
                cameraPos[1] = m_CameraRadius * cos(latitude);

                cameraPos += m_CameraPositionNoise;
//                cameraUp[0] += m_CameraRollNoise;
//                cameraUp.normalize();

//                glPushMatrix();


                // Compute the camera rotation w.r.t. (0.0f, 0.0f, -1.0f).
                Eigen::Vector3f vec = m_CameraLookAtNoise - cameraPos;
                vec.normalize();
                float angle = acos(vec.dot(Eigen::Vector3f(0.0f, 0.0f, -1.0f)));
                Eigen::Vector3f axis = vec.cross(Eigen::Vector3f(0.0f, 0.0f, -1.0f));
                axis.normalize();

                // Roll the camera by the noise.
                Eigen::Vector3f vecLeft = vec.cross(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
                vecLeft.normalize();

                Eigen::Vector3f vecUp = vecLeft.cross(vec);
                cameraUp = vecUp;

//                Eigen::AngleAxisf roll = Eigen::AngleAxisf(m_CameraRollNoise, vec);
//                cameraUp = roll * cameraUp;

//                gluLookAt(cameraPos[0], cameraPos[1], cameraPos[2], m_CameraLookAtNoise[0], m_CameraLookAtNoise[1], m_CameraLookAtNoise[2], cameraUp[0], cameraUp[1], cameraUp[2]);

                glLoadIdentity();

                Eigen::Matrix4f matRotation;
                matRotation << vecLeft[0], vecLeft[1], vecLeft[2], 0.0f, vecUp[0], vecUp[1], vecUp[2], 0.0f, -vec[0], -vec[1], -vec[2], 0.0f, 0.0f, 0.0f, 0.0f, 1.0f;

                glMultMatrixf(matRotation.data());
                glTranslatef(-cameraPos[0], -cameraPos[1], -cameraPos[2]);


                m_CameraPosition = cameraPos;

//                float rollAngle = acos(cameraUp.dot(Eigen::Vector3f(0.0f, 1.0f, 0.0f)));
                //Eigen::Vector3f rollAxis = cameraUp.cross(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
                //m_CameraRotation = /*roll * */ Eigen::AngleAxisf(angle, axis);
                m_CameraRotation.fromRotationMatrix(matRotation.block<3, 3>(0, 0));

//                glLoadIdentity();
////                gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0);
//                glRotatef(m_CameraRotation.angle(), m_CameraRotation.axis()[0], m_CameraRotation.axis()[1], m_CameraRotation.axis()[2]);
//                glTranslatef(m_CameraPosition[0], m_CameraPosition[1], m_CameraPosition[2]);



//                glTranslatef(cameraPos[0], cameraPos[1], cameraPos[2]);
//                gluSphere(params, 0.001f, 20, 20);

//                glPopMatrix();

//                ++m_FrameCount;

//                if (m_FrameCount >= 1)
//                {
//                    m_FrameCount = 0;
//                    --m_PointsOnSphereIndex;

//                    if (m_PointsOnSphereIndex < 0)
//                    {
//                        // Finish.
//                        m_GeneratingTrainingSet = false;
//                    }
//                }
            }
            else
            {
//                f = false;
            }
        }
    }


    Eigen::Vector4f liverColour(1.0f, 1.0f, 1.0f, opacity);

    for (Model& model : m_Models)
    {
//        if(model.getTextureState(modelNumber))
        if (model.HasTextures())
        {
//            glColor4f(1.f, 1.f, 1.f, opacity);

//            GLuint textureNumber = 0;
//            for(GLuint i = 0; i < modelNumber; i++)
//                if(model.getTextureState(i))
//                    textureNumber++;

//            glBindTexture(GL_TEXTURE_2D, model.getTexture(textureNumber));
//            glEnable(GL_TEXTURE_2D);
        }
        else
        {

#if NOT_FOR_CNN

            liverColour << 0.7f, 0.0f, 0.7f, opacity;

#else

            liverColour << 0.1f, 0.1f, 0.1f, 1.0f;

#endif

        }

//        model.SetColour(liverColour);

        glPushMatrix();

        // TODO: Temp - get a random texture.
        // Get a random texture.
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dist(0, 4);
        int index = floor(dist(gen));
        model.SetTexture(texture[index]);

        model.Render();

        glPopMatrix();

//        if(model.getTextureState(modelNumber))
//            glDisable(GL_TEXTURE_2D);
    }


#if NOT_FOR_CNN

    // Render model contour for fine registration.
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width(), 0, height(), -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    glLineWidth(4.0f);

    glBegin(GL_LINES);

    for (std::vector<Eigen::Vector2f>::iterator itPoint = m_GroundTruthModelContour.begin(); itPoint != m_GroundTruthModelContour.end(); ++itPoint)
    {
        glVertex2f((*itPoint)[0], (*itPoint)[1]);

        if (itPoint < m_GroundTruthModelContour.end() - 1)
        {
            glVertex2f((*(itPoint + 1))[0], (*(itPoint + 1))[1]);
        }
        else
        {
            glVertex2f((*m_GroundTruthModelContour.begin())[0], (*m_GroundTruthModelContour.begin())[1]);
        }
    }

    glEnd();

    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();

#endif


    // Render model contour.
    if (m_SelectingContour)
    {
        glPushMatrix();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        m_Models[0].RenderContour();

        glPopMatrix();
    }

    // Render vertex selection rect.
    if (m_SelectingMultipleVertices)
    {
        glPushMatrix();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
        glPushAttrib(GL_ENABLE_BIT);

        glLineStipple(3, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.top());
        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.top());

        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.top());
        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.bottom());

        glVertex2f(m_VertexSelectionRect.right(), m_VertexSelectionRect.bottom());
        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.bottom());

        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.bottom());
        glVertex2f(m_VertexSelectionRect.left(), m_VertexSelectionRect.top());
        glEnd();

        glPopAttrib();
        glPopMatrix();
    }


#if NOT_FOR_CNN

// CROSSHAIR
    glColor4f(1.f, 1.f, 1.f, 1.f);
    glDisable(GL_DEPTH_TEST);                           // Depth buffer OFF
    glDisable(GL_LIGHTING);                             // Light OFF

    glEnable(GL_COLOR_LOGIC_OP);
    glLogicOp(GL_XOR);                                  // Negative color
    glCallList(crosshair);                              // Drawing crosshair
    glDisable(GL_COLOR_LOGIC_OP);


// TEXT
    QFontMetrics fm(QFont("Lato Black", 16));
    QString depth = QString("Depth: %1 cm").arg(QString::number(-coordModels.z()/10, 'f', 2));
//    renderText(width()-fm.width(depth, -1)-5, 25, depth, QFont("Lato Black", 16));

    if(distanceBetweenTags)
//        renderText(10, 50, QString("Distance between tags: %1 cm").arg(QString::number(abs(distanceBetweenTags/10), 'f', 2)), QFont("Lato Black", 16));

    if(crosshair)
    {
//        renderText(10, 25, QString("X: %1    Y: %2    Z: %3")
//                .arg(QString::number(surfaceCoordinates.x(), 'f', 2))
//                .arg(QString::number(surfaceCoordinates.y(), 'f', 2))
//                .arg(QString::number(surfaceCoordinates.z()-coordModels.z(), 'f', 2)), QFont("Lato Black", 16));

        if(tumor)
        {
            QVector3D temp = m * coordTumor;

            GLfloat distanceToTumor = sqrt(pow(surfaceCoordinates.x()-temp.x(),2)
                    +pow(surfaceCoordinates.y()-temp.y(),2)
                    +pow(surfaceCoordinates.z()+temp.z(),2));

//            renderText(10, height()-10, QString("Distance to tumor: %1 cm").arg(QString::number(abs((distanceToTumor/10)-(tumorRadius/scaleFactor/10)), 'f', 2)), QFont("Lato Black", 16));
        }
        crosshair = 0;
    }

#endif

}

/* ============================ PROJECTION CAMERA ============================ */
void GLWidget::resetCameraSettings()
{
/*        // DEFAULT
        cameraParameters[0] = 1000;   // alphaX, focal (px)
        cameraParameters[1] = 1000;   // alphaY, focal (px)
        cameraParameters[2] = 0;      // skewness
        cameraParameters[3] = 250;    // u, image center abscissa (px)
        cameraParameters[4] = 187.5;  // v, image center ordinate (px)

        // PICTURES N°1
        cameraParameters[0] = 3260.4904041680534;  // alphaX, focal (px)
        cameraParameters[1] = 3247.8133528848193;  // alphaY, focal (px)
        cameraParameters[2] = 9.1996636861168586;  // skewness
        cameraParameters[3] = 580.40611599810813;  // u, image center abscissa (px)
        cameraParameters[4] = 572.27194900491202;  // v, image center ordinate (px)*/

        // PICTURES N°2
//        cameraParameters[0] = 3215.0213146269689;  // alphaX, focal (px)
//        cameraParameters[1] = 3227.1754390328997;  // alphaY, focal (px)
//        cameraParameters[2] = 8.7179384749909108;  // skewness
//        cameraParameters[3] = 483.46333094489626;  // u, image center abscissa (px)
//        cameraParameters[4] = 472.53980666143559;  // v, image center ordinate (px)
/*
        // PICTURES N°3
        cameraParameters[0] = 3232.528030290006;   // alphaX, focal (px)
        cameraParameters[1] = 3248.4333523027763;  // alphaY, focal (px)
        cameraParameters[2] = 14.659490990087299;  // skewness
        cameraParameters[3] = 716.19859622776949;  // u, image center abscissa (px)
        cameraParameters[4] = 443.6688114873711;   // v, image center ordinate (px)
*/
        // PICTURES SURGERY
/*        cameraParameters[0] = 2093.88374;  // alphaX, focal (px)
        cameraParameters[1] = 2093.88374;  // alphaY, focal (px)
        cameraParameters[2] = 0.00000;     // skewness
        cameraParameters[3] = 959.50000;   // u, image center abscissa (px)
        cameraParameters[4] = 539.50000;   // v, image center ordinate (px)*/


#if USING_PHANTOM_INPUT_IMAGE

    // For phantom input images. Using the intrinsic matrix from iPhone 6s.
    cameraParameters[0] = 4150/1.22 / 4.0f /*/ 21.0f*/; // alphaX, focal (px)
    cameraParameters[1] = 4150/1.22 / 4.0f /*/ 21.0f*/; // alphaY, focal (px)
    cameraParameters[2] = 0;  // skewness
    cameraParameters[3] = 2016.0f / 4.0f /*/ 21.0f*/;  // u, image center abscissa (px)
    cameraParameters[4] = 1512.0f / 4.0f /*/ 21.0f*/;  // v, image center ordinate (px)

#else

        // For the synthetic/laparoscopy input images.
        cameraParameters[0] = 4150/1.22 / 2.0f /*/ 21.0f*/; // alphaX, focal (px)
        cameraParameters[1] = 4150/1.22 / 2.0f /*/ 21.0f*/; // alphaY, focal (px)
        cameraParameters[2] = 0;  // skewness
        cameraParameters[3] = 1280.0f / 2.0f /*/ 21.0f*/;  // u, image center abscissa (px)
        cameraParameters[4] = 720.0f / 2.0f /*/ 21.0f*/;  // v, image center ordinate (px)

#endif


#if 0
        // For CNN prediction test.
        cameraParameters[0] = 4150/1.22 / 21.0f; // alphaX, focal (px)
        cameraParameters[1] = 4150/1.22 / 21.0f; // alphaY, focal (px)
        cameraParameters[2] = 0;  // skewness
        cameraParameters[3] = 2016.0f / 21.0f;  // u, image center abscissa (px)
        cameraParameters[4] = 1512.0f / 21.0f;  // v, image center ordinate (px)

#endif

        cameraParameters[5] = 0.001;       // near, distance to the nearer depth clipping plane (m)
        cameraParameters[6] = 1000;        // far, distance to the farther depth clipping plane (m)
}

void GLWidget::camera()
{
    QMatrix4x4 perspective(cameraParameters[0]*cameraParameters[7], cameraParameters[2]*cameraParameters[7], -(cameraParameters[3]+m_pBackgroundTexture->width()*frame_picture_Ratio)*cameraParameters[7], 0,
                           0, cameraParameters[1]*cameraParameters[7], -(cameraParameters[4]+m_pBackgroundTexture->height()*frame_picture_Ratio)*cameraParameters[7], 0,
                           0, 0, (cameraParameters[5] + cameraParameters[6]), (cameraParameters[5] * cameraParameters[6]),
                           0, 0, -1, 0);


    QMatrix4x4 ortho(2.0 / (float)width(), 0, 0, -1,
                     0, 2.0 / (float)height(), 0, -1,
                     0, 0, -2.0 / (cameraParameters[6] - cameraParameters[5]), -(cameraParameters[6] + cameraParameters[5]) / (cameraParameters[6] - cameraParameters[5]),
                     0, 0, 0, 1.0);

   m_proj = ortho * perspective;
}

void GLWidget::cameraFixedPipeline()
{
   QMatrix4x4 perspective(cameraParameters[0]*cameraParameters[7], cameraParameters[2]*cameraParameters[7], -(cameraParameters[3]+m_pBackgroundTexture->width()*frame_picture_Ratio)*cameraParameters[7], 0,
                          0, cameraParameters[1]*cameraParameters[7], -(cameraParameters[4]+m_pBackgroundTexture->height()*frame_picture_Ratio)*cameraParameters[7], 0,
                          0, 0, (cameraParameters[5] + cameraParameters[6]), (cameraParameters[5] * cameraParameters[6]),
                          0, 0, -1, 0);

    glMatrixMode(GL_PROJECTION);
    glOrtho(0, (GLfloat)width(), 0, (GLfloat)height(), cameraParameters[5], cameraParameters[6]);        // Setting orthographic camera matrix
    multMatrix(perspective);                            // Multiplication by perspective matrixz
}


/* ============================ PICTURE LOADER ============================ */
void GLWidget::setTexturePath()
{
    QString textureName = QFileDialog::getOpenFileName(this, "Load Background Image File", NULL, "Images (*.png *.xpm *.jpg *.bmp)");

    if(textureName!="")
    {
//        texture[0].setTexture(textureName);

        m_InputImage.load(textureName);

        if (m_pBackgroundTexture)
        {
            delete m_pBackgroundTexture;
        }

        m_pBackgroundTexture = new QOpenGLTexture(m_InputImage.mirrored());
        m_pBackgroundTexture->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
        m_pBackgroundTexture->setMagnificationFilter(QOpenGLTexture::Linear);


        // Load the input image, convert it to XYZ colour space and choose Y channel for illumination.
        // N.B. Currently does not do the median filtering.
        m_InputImageMedianFilteredY.load(textureName);

    //    Utils::ConvertRGBToXYZ(m_InputImageMedianFilteredY);
        m_InputImageMedianFilteredY = m_InputImage.convertToFormat(QImage::Format_Grayscale8);

    }

//    cameraParameters[0] = 4150/1.22 / 4.0f /*/ 21.0f*/; // alphaX, focal (px)
//    cameraParameters[1] = 4150/1.22 / 4.0f /*/ 21.0f*/; // alphaY, focal (px)
//    cameraParameters[2] = 0;  // skewness
    cameraParameters[3] = m_InputImage.width() * 0.5f; // 2016.0f / 4.0f /*/ 21.0f*/;  // u, image center abscissa (px)
    cameraParameters[4] = m_InputImage.height() * 0.5f; // 1512.0f / 4.0f /*/ 21.0f*/;  // v, image center ordinate (px)
    cameraParameters[7] = 1.0f;

    resizeWidget();

    emit BackgroundImageChanged();

//    updateGL();
    update();
}

void GLWidget::resizeWidget()
{
    GLfloat frames = (2*frame_picture_Ratio)+1;
    GLfloat scale = cameraParameters[7]*frames;

    float width = m_InputImage.width();
    float height = m_InputImage.height();
    float prevWidth = this->width();

    if(this->height() < this->width())
    {
        if(height * scale > 100)
        {
            this->setFixedHeight(height * scale);
            this->setFixedWidth(width * scale);
        }
        else
        {
            this->setFixedWidth(100 / (height / width));
            this->setFixedHeight(100);
            cameraParameters[7] = (GLfloat)100 / height / frames;
        }
    }
    else
    {
        if(width * scale > 100)
        {
            this->setFixedHeight(height * scale);
            this->setFixedWidth(width * scale);
        }
        else
        {
            this->setFixedWidth(100);
            this->setFixedHeight(100 * (height / width));
            cameraParameters[7] = (GLfloat)100 / width / frames;
        }
    }

    if (scaleSliderPressed == false)
    {
        emit pictureChanged(this->width() - 20, this->height() - 20);
    }

//    // TODO: Temp.
//    static bool isFirst = true;

//    if (isFirst)
//    {
//        isFirst = false;
//    }
//    else
//    {
//        float scaleContour = m_WindowScale; // this->width() / prevWidth;

//        for (Eigen::Vector2f& point : m_GroundTruthModelContour)
//        {
//            point *= scaleContour;
//        }

//        for (Eigen::Vector2f& point : m_ContourSelectionPoints)
//        {
//            point *= scaleContour;
//        }
//    }

    m_Viewport[0] = 0;
    m_Viewport[1] = 0;
    m_Viewport[2] = (GLint)this->width();
    m_Viewport[3] = (GLint)this->height();

    camera();

    if (m_pDepthData)
    {
        delete m_pDepthData;
    }

    int size = m_Viewport[2] * m_Viewport[3];
    m_pDepthData = new GLfloat[size];

//    if (!m_EditingModel && m_SelectingContour)
//    {
//        m_Models[0].ExtractVerticesOnContour(m_ModelView, m_proj, m_Viewport);
//    }
}

void GLWidget::ComputeModelCentroid()
{
    m_PrevModelCentroid = m_ModelCentroid;

    m_ModelCentroid.setZero();
    unsigned int numOfVertices = 0;

    for (Model& model : m_Models)
    {
        for (Model::Vertex* pVertex : model.Vertices())
        {
            m_ModelCentroid += pVertex->_Pos;
        }

        numOfVertices += model.Vertices().size();
    }

    m_ModelCentroid /= (float)numOfVertices;

    m_ModelCentroidDiff = m_ModelCentroid - m_PrevModelCentroid;
}

/* ============================ MODEL LOADER/SAVER/TRANSFORM ============================ */
void GLWidget::addModel()
{
    QString modelName;
    QStringList modelsListInit = modelsList;

    do
    {
        modelName = QFileDialog::getOpenFileName(this, tr("Load 3D Model File"), NULL, tr("3D Models (*.obj)"));

        if(!modelName.isEmpty() && !modelsListInit.contains(modelName))
        {
            modelsList << modelName;
//            model.render(modelsList);

            Model model;
            model.Load(modelName);

#if 0 // Temp for CNN prediction.

            float vertices[2002 * 3] =
            {
                1.30437881e-01,  -6.38158172e-02,  -3.59235615e-01,
                          1.24689415e-01,  -3.99982780e-02,  -3.66496593e-01,
                          1.24565966e-01,  -3.86441350e-02,  -3.62048537e-01,
                          1.27643004e-01,  -5.00708856e-02,  -3.59651774e-01,
                          1.25990614e-01,  -4.40065861e-02,  -3.61717165e-01,
                          1.23817749e-01,  -5.09956256e-02,  -3.60348523e-01,
                          1.28004909e-01,  -5.91371916e-02,  -3.58802795e-01,
                          1.30817205e-01,  -5.98571673e-02,  -3.59356344e-01,
                          1.24531217e-01,  -6.00425005e-02,  -3.59947383e-01,
                          1.30691424e-01,  -6.75447881e-02,  -3.57716650e-01,
                          1.28873095e-01,  -7.04045296e-02,  -3.55715811e-01,
                          1.26517668e-01,  -7.86665529e-02,  -3.59248936e-01,
                          1.28850967e-01,  -8.63792747e-02,  -3.59652877e-01,
                          1.27110064e-01,  -9.03320387e-02,  -3.62045348e-01,
                          1.23396344e-01,  -4.75998223e-02,  -3.61279517e-01,
                          1.28984198e-01,  -6.41660839e-02,  -3.58148992e-01,
                          1.30609393e-01,  -6.15422279e-02,  -3.61270636e-01,
                          1.26912341e-01,  -6.82194382e-02,  -3.60645831e-01,
                          1.29936248e-01,  -7.29010105e-02,  -3.60298991e-01,
                          1.30415678e-01,  -8.62430856e-02,  -3.57752085e-01,
                          1.28461435e-01,  -8.35755020e-02,  -3.58995438e-01,
                          1.28532007e-01,  -9.18605700e-02,  -3.58043194e-01,
                          1.28573418e-01,  -9.38811898e-02,  -3.59428108e-01,
                          1.19613312e-01,  -3.45820859e-02,  -3.63274723e-01,
                          1.17796928e-01,  -3.34272496e-02,  -3.65789026e-01,
                          1.24553300e-01,  -4.59176525e-02,  -3.58778626e-01,
                          1.21186197e-01,  -3.95071171e-02,  -3.65856707e-01,
                          1.29803151e-01,  -5.66391759e-02,  -3.57346296e-01,
                          1.24098212e-01,  -7.14952275e-02,  -3.59918296e-01,
                          1.27606615e-01,  -7.88639709e-02,  -3.59899253e-01,
                          1.22493729e-01,  -5.21155745e-02,  -3.63032669e-01,
                          1.28421575e-01,  -8.34291801e-02,  -3.53852361e-01,
                          1.26481637e-01,  -9.53453854e-02,  -3.60338032e-01,
                          1.21500522e-01,  -9.50072333e-02,  -3.62274915e-01,
                          1.24567404e-01,  -9.68984067e-02,  -3.59929025e-01,
                          1.23500772e-01,  -3.72188501e-02,  -3.58962655e-01,
                          1.17799968e-01,  -3.31963934e-02,  -3.61316234e-01,
                          1.25408426e-01,  -3.78299952e-02,  -3.56686503e-01,
                          1.31558970e-01,  -7.88498893e-02,  -3.55686724e-01,
                          1.28261566e-01,  -9.02478173e-02,  -3.52167904e-01,
                          1.29998878e-01,  -6.54821843e-02,  -3.53134394e-01,
                          1.22973107e-01,  -6.00406714e-02,  -3.61367196e-01,
                          1.22160181e-01,  -9.88945067e-02,  -3.60205680e-01,
                          1.26641646e-01,  -5.11600226e-02,  -3.55237633e-01,
                          1.28117010e-01,  -6.26699775e-02,  -3.53042454e-01,
                          1.24592416e-01,  -9.66136605e-02,  -3.57513964e-01,
                          1.20159172e-01,  -1.01273634e-01,  -3.60066026e-01,
                          1.14143431e-01,  -3.00545730e-02,  -3.61529887e-01,
                          1.11757502e-01,  -3.46651599e-02,  -3.62659305e-01,
                          1.13747463e-01,  -3.91696319e-02,  -3.64493042e-01,
                          1.16938859e-01,  -4.00009155e-02,  -3.67027640e-01,
                          1.20587111e-01,  -4.80209589e-02,  -3.65126967e-01,
                          1.28367096e-01,  -8.10998678e-02,  -3.49772662e-01,
                          1.24582715e-01,  -9.52058733e-02,  -3.53430539e-01,
                          1.19881764e-01,  -3.34300734e-02,  -3.58757824e-01,
                          1.23608522e-01,  -4.64832932e-02,  -3.55819821e-01,
                          1.21112451e-01,  -8.43841210e-02,  -3.57339889e-01,
                          1.18167289e-01,  -3.60593572e-02,  -3.54146749e-01,
                          1.21419020e-01,  -7.73667544e-02,  -3.61097276e-01,
                          1.19983368e-01,  -9.01255980e-02,  -3.63169700e-01,
                          1.15850829e-01,  -9.77245346e-02,  -3.59162986e-01,
                          1.20750353e-01,  -3.45278792e-02,  -3.56611013e-01,
                          1.17761619e-01,  -5.87834604e-02,  -3.60361159e-01,
                          1.25180364e-01,  -9.02038664e-02,  -3.50111991e-01,
                          1.20358706e-01,  -1.04154661e-01,  -3.56390119e-01,
                          1.24607906e-01,  -5.85174449e-02,  -3.50804150e-01,
                          1.15896054e-01,  -6.92433938e-02,  -3.59623879e-01,
                          1.22164719e-01,  -1.03032492e-01,  -3.52114528e-01,
                          1.17327124e-01,  -1.06840156e-01,  -3.53109121e-01,
                          1.11887224e-01,  -2.66340654e-02,  -3.51214141e-01,
                          1.14367545e-01,  -3.06020249e-02,  -3.57462376e-01,
                          1.05165847e-01,  -4.27493602e-02,  -3.60021174e-01,
                          1.26082316e-01,  -7.70866200e-02,  -3.46640080e-01,
                          1.13510787e-01,  -1.04897231e-01,  -3.61147791e-01,
                          1.15257107e-01,  -1.05012909e-01,  -3.57870400e-01,
                          1.10151343e-01,  -3.02581713e-02,  -3.50157470e-01,
                          1.09186523e-01,  -3.03151533e-02,  -3.59371811e-01,
                          1.26419842e-01,  -6.75057173e-02,  -3.47634584e-01,
                          1.30182996e-01,  -7.30334818e-02,  -3.51561397e-01,
                          1.07937559e-01,  -9.93971154e-02,  -3.61747831e-01,
                          1.09428428e-01,  -1.10068299e-01,  -3.58229309e-01,
                          1.10206075e-01,  -4.83867452e-02,  -3.60962033e-01,
                          1.07795194e-01,  -7.98472390e-02,  -3.58492762e-01,
                          1.10269547e-01,  -3.35221067e-02,  -3.61765981e-01,
                          1.24684885e-01,  -5.39544858e-02,  -3.50578845e-01,
                          1.13412730e-01,  -5.35715111e-02,  -3.59045327e-01,
                          1.12947948e-01,  -1.12767652e-01,  -3.52536261e-01,
                          1.15828454e-01,  -3.90463211e-02,  -3.49901080e-01,
                          1.04944289e-01,  -4.93725166e-02,  -3.57185841e-01,
                          1.12339132e-01,  -6.10020310e-02,  -3.59602302e-01,
                          1.06998421e-01,  -7.08306059e-02,  -3.57062489e-01,
                          1.13473527e-01,  -1.11688167e-01,  -3.50916475e-01,
                          1.10041991e-01,  -1.08262785e-01,  -3.58898878e-01,
                          1.05865151e-01,  -1.04075722e-01,  -3.59032691e-01,
                          1.11042909e-01,  -2.66816933e-02,  -3.53991687e-01,
                          1.05314784e-01,  -3.40701304e-02,  -3.57453823e-01,
                          1.13004215e-01,  -1.11421771e-01,  -3.57495248e-01,
                          1.05262913e-01,  -2.49700341e-02,  -3.54346752e-01,
                          1.00885369e-01,  -4.65356633e-02,  -3.56725037e-01,
                          1.27911925e-01,  -8.27716812e-02,  -3.49182874e-01,
                          1.09078258e-01,  -8.78694206e-02,  -3.58341813e-01,
                          1.20154724e-01,  -9.83552039e-02,  -3.49476963e-01,
                          1.04356989e-01,  -1.18077613e-01,  -3.55247676e-01,
                          1.06259078e-01,  -1.15891062e-01,  -3.57470512e-01,
                          1.07874818e-01,  -2.72096395e-02,  -3.55513930e-01,
                          1.14776991e-01,  -4.54668328e-02,  -3.43348712e-01,
                          1.18927643e-01,  -8.74955878e-02,  -3.40142727e-01,
                          1.01815514e-01,  -1.19959295e-01,  -3.59210640e-01,
                          1.07627161e-01,  -2.51855329e-02,  -3.48420441e-01,
                          1.20240822e-01,  -6.97361827e-02,  -3.41694146e-01,
                          1.16894878e-01,  -7.94127285e-02,  -3.32283050e-01,
                          1.14713818e-01,  -1.05507828e-01,  -3.43150526e-01,
                          1.04146250e-01,  -1.12754256e-01,  -3.60502183e-01,
                          1.03230946e-01,  -1.16161294e-01,  -3.59747857e-01,
                          9.82752368e-02,  -1.19358733e-01,  -3.62044126e-01,
                          9.69596803e-02,  -4.07277159e-02,  -3.55593711e-01,
                          9.89195257e-02,  -4.78628129e-02,  -3.55893701e-01,
                          1.01205349e-01,  -5.98434247e-02,  -3.56171161e-01,
                          1.00463703e-01,  -8.28544050e-02,  -3.54855955e-01,
                          1.13637090e-01,  -1.08840711e-01,  -3.47166270e-01,
                          1.05762050e-01,  -2.51213480e-02,  -3.54852527e-01,
                          1.01732492e-01,  -2.27637999e-02,  -3.53127271e-01,
                          1.13267854e-01,  -5.07580414e-02,  -3.37052256e-01,
                          9.87461284e-02,  -1.22082666e-01,  -3.56611878e-01,
                          1.02250345e-01,  -1.91978049e-02,  -3.49987060e-01,
                          9.91177112e-02,  -2.76825149e-02,  -3.53515029e-01,
                          1.17633328e-01,  -5.55777065e-02,  -3.38951737e-01,
                          1.16807178e-01,  -9.08494443e-02,  -3.37854773e-01,
                          9.20929611e-02,  -2.41452549e-03,  -3.59532267e-01,
                          9.34092030e-02,  -5.29404311e-03,  -3.58036846e-01,
                          9.30633843e-02,  -4.73165093e-03,  -3.59488755e-01,
                          9.45534110e-02,  -8.69374536e-03,  -3.58331442e-01,
                          9.31436718e-02,  -8.97816475e-03,  -3.58596176e-01,
                          9.10985544e-02,  -4.42788377e-03,  -3.63457143e-01,
                          1.07250161e-01,  -2.37744860e-02,  -3.49395216e-01,
                          1.00055039e-01,  -1.93049442e-02,  -3.52617085e-01,
                          9.77926478e-02,  -6.58339262e-02,  -3.55798215e-01,
                          1.12029195e-01,  -1.12335123e-01,  -3.47626269e-01,
                          8.97544175e-02,  -1.23683982e-01,  -3.61588836e-01,
                          9.59778130e-02,  -1.21859610e-01,  -3.59558791e-01,
                          1.02508865e-01,  -1.79547500e-02,  -3.50473970e-01,
                          9.99701396e-02,  -1.61330793e-02,  -3.52413535e-01,
                          1.10160470e-01,  -1.06259085e-01,  -3.36203396e-01,
                          1.10450827e-01,  -1.08600199e-01,  -3.41540545e-01,
                          9.69305634e-02,  -1.11135915e-01,  -3.59377116e-01,
                          9.50425863e-02,  -4.76118270e-03,  -3.48865062e-01,
                          9.55530554e-02,  -7.73684587e-03,  -3.54312837e-01,
                          5.57525605e-02,   2.43439265e-02,  -3.88052911e-01,
                          9.93169025e-02,  -1.38180442e-02,  -3.52765590e-01,
                          9.13626477e-02,  -4.32760594e-03,  -3.60851705e-01,
                          4.72039655e-02,   2.54938696e-02,  -3.92550111e-01,
                          6.04640804e-02,   1.38371950e-02,  -3.82221818e-01,
                          5.37907071e-02,   2.38657668e-02,  -3.87850463e-01,
                          6.03196211e-02,   1.74310710e-02,  -3.85957539e-01,
                          5.34125865e-02,   2.11332124e-02,  -3.93096983e-01,
                          5.33312522e-02,   1.69570073e-02,  -3.90116841e-01,
                          1.01476371e-01,  -1.96935553e-02,  -3.54033709e-01,
                          9.78772044e-02,  -2.00382564e-02,  -3.52034718e-01,
                          1.13518886e-01,  -6.87923282e-02,  -3.31490487e-01,
                          9.38975289e-02,  -5.50487153e-02,  -3.56751800e-01,
                          1.19150415e-01,  -7.35486969e-02,  -3.37002099e-01,
                          9.65039656e-02,  -7.78722093e-02,  -3.56787473e-01,
                          9.91396457e-02,  -9.19553712e-02,  -3.55819345e-01,
                          1.01507485e-01,  -1.60213057e-02,  -3.48190069e-01,
                          1.04362637e-01,  -2.64184400e-02,  -3.44804913e-01,
                          8.67192373e-02,  -9.00182500e-03,  -3.61866623e-01,
                          4.60209996e-02,   2.70266682e-02,  -3.96191031e-01,
                          8.97724852e-02,  -1.17724314e-02,  -3.61276507e-01,
                          9.25825164e-02,  -1.56319998e-02,  -3.55524361e-01,
                          6.06752820e-02,   1.40585164e-02,  -3.86018634e-01,
                          1.08569056e-01,  -3.17499153e-02,  -3.42047364e-01,
                          9.49896052e-02,  -4.31556143e-02,  -3.54731113e-01,
                          1.16377592e-01,  -9.78847966e-02,  -3.40918720e-01,
                          1.02890119e-01,  -1.17724642e-01,  -3.40035677e-01,
                          9.38321874e-02,  -1.04078606e-01,  -3.59048307e-01,
                          8.91096294e-02,   2.01121066e-03,  -3.59249383e-01,
                          4.63690422e-02,   3.10739763e-02,  -3.96035224e-01,
                          6.32883832e-02,   1.60999894e-02,  -3.83181065e-01,
                          4.21890989e-02,   2.76323389e-02,  -3.94240469e-01,
                          9.50401276e-02,  -1.90431960e-02,  -3.52233827e-01,
                          1.04362242e-01,  -3.65665518e-02,  -3.33646178e-01,
                          1.13378197e-01,  -5.55551425e-02,  -3.34922940e-01,
                          1.20467298e-01,  -6.07719272e-02,  -3.43223602e-01,
                          1.17004357e-01,  -6.56313971e-02,  -3.33568722e-01,
                          8.98182020e-02,  -5.87961040e-02,  -3.56975347e-01,
                          9.16991085e-02,  -8.97714272e-02,  -3.55342299e-01,
                          9.03405100e-02,   7.14274123e-04,  -3.57290089e-01,
                          8.22387934e-02,   2.29965290e-03,  -3.58552665e-01,
                          6.12000413e-02,   1.53471995e-02,  -3.78794938e-01,
                          8.58183503e-02,  -2.95136962e-03,  -3.61255616e-01,
                          1.01398252e-01,  -1.98372398e-02,  -3.41429889e-01,
                          1.09017663e-01,  -7.66076222e-02,  -3.25097531e-01,
                          9.31441560e-02,  -1.18217669e-01,  -3.61106783e-01,
                          8.78722444e-02,  -1.28702715e-01,  -3.59819382e-01,
                          4.87882644e-02,   2.95581520e-02,  -3.92340302e-01,
                          4.65871394e-02,   3.13983709e-02,  -3.89730483e-01,
                          4.58145253e-02,   3.20162177e-02,  -3.91932309e-01,
                          6.28207475e-02,   1.85552072e-02,  -3.74822259e-01,
                          5.78783154e-02,   6.86942134e-03,  -3.80481780e-01,
                          8.37800056e-02,  -1.18249012e-02,  -3.56848776e-01,
                          9.47935656e-02,  -3.59030813e-02,  -3.53827238e-01,
                          8.97374824e-02,  -4.29330543e-02,  -3.55689079e-01,
                          9.02476087e-02,  -7.76975006e-02,  -3.55974138e-01,
                          9.49624255e-02,  -1.26459077e-01,  -3.55829269e-01,
                          8.50655213e-02,  -1.25751868e-01,  -3.61727327e-01,
                          5.45109548e-02,   2.47661471e-02,  -3.84011686e-01,
                          6.11574128e-02,   2.07648836e-02,  -3.85655224e-01,
                          9.86359790e-02,  -1.10828541e-02,  -3.49775910e-01,
                          5.30805141e-02,   1.50027964e-02,  -3.88550729e-01,
                          8.62980783e-02,  -4.67441417e-02,  -3.55035961e-01,
                          1.13667369e-01,  -8.68938118e-02,  -3.30197603e-01,
                          9.03651565e-02,  -9.50180590e-02,  -3.59042108e-01,
                          8.66218880e-02,  -1.06388390e-01,  -3.56258988e-01,
                          8.70104209e-02,  -1.17529579e-01,  -3.61961871e-01,
                          8.73608142e-02,   2.47780746e-03,  -3.52307349e-01,
                          3.81508544e-02,   3.54572609e-02,  -3.93630981e-01,
                          8.36965144e-02,  -1.29404629e-03,  -3.61952424e-01,
                          4.10436206e-02,   3.57789360e-02,  -3.94851416e-01,
                          4.06861603e-02,   2.63434723e-02,  -3.95028591e-01,
                          1.03059717e-01,  -2.61896849e-02,  -3.39166015e-01,
                          6.07877746e-02,   1.01136556e-02,  -3.80640328e-01,
                          4.60441634e-02,   2.48319060e-02,  -3.94639581e-01,
                          5.71966656e-02,   1.19480342e-02,  -3.82921040e-01,
                          8.41629505e-02,  -3.58988047e-02,  -3.49527419e-01,
                          8.74706581e-02,  -5.11278138e-02,  -3.54576260e-01,
                          8.20260197e-02,  -4.91479114e-02,  -3.56116742e-01,
                          8.22923258e-02,  -9.13209543e-02,  -3.58140469e-01,
                          1.02735989e-01,  -1.22766815e-01,  -3.51339221e-01,
                          9.02205333e-02,  -1.25290766e-01,  -3.59054506e-01,
                          9.33563411e-02,  -2.12730328e-03,  -3.49003136e-01,
                          6.45620599e-02,   1.37930997e-02,  -3.73406142e-01,
                          8.19598213e-02,  -7.91505352e-03,  -3.59367460e-01,
                          6.38017282e-02,   8.56659561e-03,  -3.73648018e-01,
                          9.09512937e-02,  -2.80811060e-02,  -3.50966394e-01,
                          9.51580107e-02,  -2.94106975e-02,  -3.53232026e-01,
                          1.05403513e-01,  -1.15135223e-01,  -3.38529170e-01,
                          9.19993371e-02,  -1.27778918e-01,  -3.52227747e-01,
                          4.81330454e-02,   3.42088938e-02,  -3.83201271e-01,
                          3.24776061e-02,   4.17920835e-02,  -3.95869374e-01,
                          3.52417715e-02,   3.68516073e-02,  -3.95501941e-01,
                          4.39591333e-02,   1.60922501e-02,  -3.90434802e-01,
                          4.97858003e-02,   5.74080832e-03,  -3.80457282e-01,
                          1.06549829e-01,  -4.94834892e-02,  -3.26839775e-01,
                          8.68782848e-02,  -5.46601117e-02,  -3.58432710e-01,
                          8.23775679e-02,  -6.14734739e-02,  -3.59489590e-01,
                          8.78317654e-02,  -7.32588992e-02,  -3.58817369e-01,
                          7.81410784e-02,  -9.63331759e-02,  -3.56883585e-01,
                          8.34192187e-02,  -1.01711839e-01,  -3.57221603e-01,
                          7.96470121e-02,  -1.10314414e-01,  -3.58607769e-01,
                          8.46367031e-02,  -1.16715990e-01,  -3.59511286e-01,
                          8.14752281e-02,  -1.28884494e-01,  -3.61498326e-01,
                          7.93302506e-02,  -1.32369950e-01,  -3.58060718e-01,
                          6.29704967e-02,   1.81137472e-02,  -3.66248310e-01,
                          1.02516055e-01,  -1.62667278e-02,  -3.43255937e-01,
                          6.34849221e-02,   1.71906780e-02,  -3.72913092e-01,
                          5.41833416e-02,   5.22000622e-03,  -3.79108727e-01,
                          8.65527838e-02,  -1.98379289e-02,  -3.54028255e-01,
                          1.06706902e-01,  -1.00458182e-01,  -3.34996104e-01,
                          7.78203905e-02,  -1.24824286e-01,  -3.62192035e-01,
                          8.86137784e-02,   6.82235230e-04,  -3.41649890e-01,
                          5.97697347e-02,   2.12124996e-02,  -3.74191433e-01,
                          4.98199724e-02,   3.21774632e-02,  -3.78194779e-01,
                          3.95093523e-02,   3.88610028e-02,  -3.81628662e-01,
                          3.91810462e-02,   3.62014137e-02,  -3.90467227e-01,
                          3.43537405e-02,   4.44678590e-02,  -3.93779784e-01,
                          7.84941986e-02,   4.33906680e-04,  -3.54545414e-01,
                          3.13993916e-02,   3.73842865e-02,  -3.95431876e-01,
                          3.33397090e-02,   3.29481140e-02,  -3.97887290e-01,
                          7.89811164e-02,  -1.16348676e-02,  -3.58342648e-01,
                          1.04773603e-01,  -3.32227647e-02,  -3.35569113e-01,
                          7.79862553e-02,  -2.30805129e-02,  -3.50723743e-01,
                          8.64820182e-02,  -2.70102918e-02,  -3.50205809e-01,
                          1.12827085e-01,  -7.30476901e-02,  -3.27798337e-01,
                          7.94155672e-02,  -4.94830385e-02,  -3.55175287e-01,
                          7.90520459e-02,  -7.34545663e-02,  -3.59275997e-01,
                          7.72324726e-02,  -8.27630460e-02,  -3.57805133e-01,
                          9.85823125e-02,  -1.20967604e-01,  -3.41045111e-01,
                          9.41351354e-02,  -1.28487408e-01,  -3.50478321e-01,
                          8.48584771e-02,   3.07489978e-03,  -3.44817251e-01,
                          8.32147300e-02,   3.77795193e-03,  -3.55089992e-01,
                          9.51633006e-02,  -6.45498978e-03,  -3.43253255e-01,
                          6.14076555e-02,   1.96674354e-02,  -3.66688430e-01,
                          9.10334438e-02,  -8.34998302e-03,  -3.36108655e-01,
                          2.85849031e-02,   4.11987267e-02,  -3.96549791e-01,
                          1.10243902e-01,  -6.65115491e-02,  -3.27734172e-01,
                          7.36590698e-02,  -1.12186857e-01,  -3.60449076e-01,
                          6.81395084e-02,  -1.19801216e-01,  -3.62371385e-01,
                          5.53173311e-02,   2.44856384e-02,  -3.69710237e-01,
                          9.30248275e-02,  -1.38175127e-03,  -3.36407304e-01,
                          8.02030116e-02,   5.01821283e-03,  -3.55463713e-01,
                          3.01681254e-02,   4.48732898e-02,  -3.93704832e-01,
                          9.57855135e-02,  -1.62838940e-02,  -3.37498099e-01,
                          7.34204277e-02,  -4.98907641e-04,  -3.51617992e-01,
                          2.62007974e-02,   3.82822827e-02,  -3.95878345e-01,
                          2.98074502e-02,   3.31201851e-02,  -3.96851659e-01,
                          9.77951661e-02,  -3.43900621e-02,  -3.28731060e-01,
                          3.07884961e-02,   2.70168521e-02,  -3.95043015e-01,
                          5.80865331e-02,   5.20174019e-03,  -3.74730647e-01,
                          3.62733901e-02,   1.77394003e-02,  -3.88596147e-01,
                          7.99926370e-02,  -2.86736861e-02,  -3.49051356e-01,
                          1.09083220e-01,  -4.58438657e-02,  -3.31529528e-01,
                          8.10158849e-02,  -3.43661420e-02,  -3.48797828e-01,
                          1.10180765e-01,  -5.75351194e-02,  -3.29411864e-01,
                          7.99809247e-02,  -5.16572185e-02,  -3.57391328e-01,
                          1.09254278e-01,  -9.19948965e-02,  -3.30428094e-01,
                          7.57736117e-02,  -6.02674261e-02,  -3.60406518e-01,
                          1.09133102e-01,  -9.34217125e-02,  -3.26908022e-01,
                          7.69652501e-02,  -8.84914845e-02,  -3.58426511e-01,
                          9.94496271e-02,  -1.15170002e-01,  -3.32437575e-01,
                          9.19578448e-02,  -1.30021721e-01,  -3.43485862e-01,
                          8.39084312e-02,  -1.30573690e-01,  -3.55228513e-01,
                          7.35599399e-02,  -1.29413977e-01,  -3.63387614e-01,
                          7.42807910e-02,  -4.97665536e-03,  -3.56056601e-01,
                          1.06131479e-01,  -6.30908534e-02,  -3.20988715e-01,
                          7.54103810e-02,  -4.47103269e-02,  -3.55811566e-01,
                          9.43995342e-02,  -1.25747919e-01,  -3.35942119e-01,
                          3.38650122e-02,   4.52305190e-02,  -3.89440328e-01,
                          6.21408969e-02,   1.21952854e-02,  -3.67592841e-01,
                          4.44863550e-02,   5.55584952e-03,  -3.79881710e-01,
                          8.49973261e-02,  -2.94177718e-02,  -3.47772747e-01,
                          7.04007372e-02,  -5.14386557e-02,  -3.58357877e-01,
                          1.05739959e-01,  -8.09843764e-02,  -3.19993764e-01,
                          7.10766912e-02,  -7.94867724e-02,  -3.60618025e-01,
                          9.14939567e-02,  -1.23452798e-01,  -3.26872617e-01,
                          7.93892816e-02,  -1.36161149e-01,  -3.47216427e-01,
                          8.97860229e-02,  -1.32656589e-01,  -3.55412900e-01,
                          2.86652818e-02,   4.45100851e-02,  -3.92712563e-01,
                          7.65974522e-02,   4.55836579e-03,  -3.51093858e-01,
                          2.53120624e-02,   4.59766649e-02,  -3.93869638e-01,
                          6.24297224e-02,   1.31428577e-02,  -3.64880621e-01,
                          1.88034084e-02,   4.20315415e-02,  -3.92912358e-01,
                          5.99034950e-02,   6.37901854e-03,  -3.69182825e-01,
                          1.00558102e-01,  -2.37940028e-02,  -3.32412660e-01,
                          7.23007098e-02,  -7.61823915e-03,  -3.54254752e-01,
                          7.54239783e-02,  -1.37265753e-02,  -3.56491178e-01,
                          7.00026527e-02,  -5.60161695e-02,  -3.59573632e-01,
                          1.04300544e-01,  -1.06625743e-01,  -3.29812944e-01,
                          3.48133780e-02,   3.87521312e-02,  -3.81282270e-01,
                          2.70141978e-02,   4.61589769e-02,  -3.90397102e-01,
                          6.08403645e-02,   1.42767206e-02,  -3.57673496e-01,
                          7.36967027e-02,  -9.23721120e-03,  -3.56705904e-01,
                          5.28769679e-02,   1.46051100e-03,  -3.69500786e-01,
                          7.79310539e-02,  -4.13937829e-02,  -3.48732561e-01,
                          7.81211704e-02,  -4.46596146e-02,  -3.50591868e-01,
                          6.88991472e-02,  -6.56583533e-02,  -3.60616177e-01,
                          1.03281163e-01,  -1.11446053e-01,  -3.31583887e-01,
                          7.07906112e-02,  -9.49474722e-02,  -3.57230961e-01,
                          6.85602054e-02,  -1.27542034e-01,  -3.64449680e-01,
                          6.69321790e-02,  -1.34886175e-01,  -3.59604299e-01,
                          7.08654970e-02,  -1.31734908e-01,  -3.61497611e-01,
                          7.89988041e-02,   3.43441311e-03,  -3.44574392e-01,
                          2.28226911e-02,   4.62228544e-02,  -3.95769417e-01,
                          1.67267360e-02,   4.40741256e-02,  -3.94607693e-01,
                          6.15919456e-02,   9.53847822e-03,  -3.61214370e-01,
                          5.32146394e-02,   1.26137864e-04,  -3.65026116e-01,
                          2.62435339e-02,   2.09380556e-02,  -3.88841629e-01,
                          6.16762862e-02,  -5.99476546e-02,  -3.61553848e-01,
                          8.23928267e-02,  -1.33518577e-01,  -3.41866016e-01,
                          6.02573156e-02,   1.80088878e-02,  -3.56888920e-01,
                          7.10698217e-02,  -2.98986398e-03,  -3.54154885e-01,
                          5.52365333e-02,   4.55696508e-03,  -3.59923750e-01,
                          1.45658785e-02,   3.20695825e-02,  -3.91222090e-01,
                          6.17806353e-02,  -1.21795973e-02,  -3.54710490e-01,
                          9.81051847e-02,  -4.12465744e-02,  -3.20137113e-01,
                          3.79972421e-02,  -1.75639964e-03,  -3.77111048e-01,
                          4.40362580e-02,  -3.10459314e-03,  -3.74910593e-01,
                          7.34342486e-02,  -4.22239564e-02,  -3.51253629e-01,
                          1.02848962e-01,  -9.25258771e-02,  -3.21605891e-01,
                          1.04377344e-01,  -1.01315223e-01,  -3.27010751e-01,
                          6.18204549e-02,  -7.01342076e-02,  -3.61928880e-01,
                          6.17377758e-02,  -7.59479329e-02,  -3.59259248e-01,
                          6.21637926e-02,  -1.33956388e-01,  -3.62710714e-01,
                          8.35342631e-02,  -9.00837360e-04,  -3.36374074e-01,
                          5.63371964e-02,   2.27098018e-02,  -3.59940618e-01,
                          3.93989496e-02,   3.48354168e-02,  -3.72312427e-01,
                          2.76560709e-02,   4.46121693e-02,  -3.86136740e-01,
                          3.23049314e-02,   3.99728008e-02,  -3.77157390e-01,
                          1.60323568e-02,   4.79730628e-02,  -3.95101547e-01,
                          4.80261780e-02,  -3.28097399e-03,  -3.65657359e-01,
                          1.05588078e-01,  -4.62021045e-02,  -3.26692969e-01,
                          4.02656458e-02,   9.41409264e-03,  -3.83679241e-01,
                          4.77188677e-02,  -1.97280571e-03,  -3.75590831e-01,
                          6.38617948e-02,  -4.88760993e-02,  -3.58165205e-01,
                          6.20752834e-02,  -6.72591776e-02,  -3.60718071e-01,
                          7.14731142e-02,  -1.39166445e-01,  -3.52103561e-01,
                          7.95913860e-02,  -1.35021359e-01,  -3.51466566e-01,
                          7.51672387e-02,  -1.36514321e-01,  -3.55904013e-01,
                          2.92940903e-02,   4.59035374e-02,  -3.83095622e-01,
                          2.10480466e-02,   4.97405156e-02,  -3.93383652e-01,
                          5.98726831e-02,   1.19656408e-02,  -3.60456884e-01,
                          6.85292184e-02,   2.84667034e-03,  -3.50855559e-01,
                          1.26490332e-02,   4.43256907e-02,  -3.94647717e-01,
                          1.30911199e-02,   2.21529864e-02,  -3.87805283e-01,
                          2.61494573e-02,   1.54998209e-02,  -3.85131031e-01,
                          7.46625215e-02,  -1.77457333e-02,  -3.55282456e-01,
                          7.14315102e-02,  -4.73268963e-02,  -3.54759306e-01,
                          6.16069958e-02,  -4.87006418e-02,  -3.57280314e-01,
                          6.52480423e-02,  -9.19670835e-02,  -3.56961995e-01,
                          7.31054917e-02,  -1.01185635e-01,  -3.58824968e-01,
                          8.28286186e-02,   3.33366776e-03,  -3.40124041e-01,
                          9.06893015e-02,  -5.15568536e-03,  -3.35269570e-01,
                          1.68698467e-02,   5.14206998e-02,  -3.88241619e-01,
                          7.63508156e-02,   4.10070922e-03,  -3.46251309e-01,
                          1.17331482e-02,   5.27944528e-02,  -3.92994314e-01,
                          6.72886893e-02,  -5.08909579e-04,  -3.50064874e-01,
                          9.24242288e-02,  -2.13344190e-02,  -3.26594979e-01,
                          2.13085841e-02,   2.41654869e-02,  -3.91638398e-01,
                          1.02551028e-01,  -5.91990165e-02,  -3.19687515e-01,
                          1.00661598e-01,  -7.40237981e-02,  -3.08659136e-01,
                          6.10227175e-02,  -4.40260991e-02,  -3.52661282e-01,
                          5.60566373e-02,  -6.50301278e-02,  -3.62999260e-01,
                          6.38208911e-02,  -1.35188878e-01,  -3.61082792e-01,
                          4.83196005e-02,   3.12686935e-02,  -3.61521512e-01,
                          6.05384894e-02,   1.35778608e-02,  -3.53852600e-01,
                          7.11146817e-02,   3.75041459e-03,  -3.46315533e-01,
                          5.93377277e-02,   1.07643344e-02,  -3.53151113e-01,
                          1.14560574e-02,   4.93753701e-02,  -3.95108044e-01,
                          1.29091237e-02,   4.02361117e-02,  -3.91948193e-01,
                          4.69604842e-02,  -4.90515633e-03,  -3.68458837e-01,
                          3.65769267e-02,  -3.99959553e-03,  -3.71237487e-01,
                          7.21881837e-02,  -3.10197175e-02,  -3.46936136e-01,
                          6.99820518e-02,  -3.75445299e-02,  -3.47245157e-01,
                          1.00522734e-01,  -7.02554360e-02,  -3.13710392e-01,
                          1.08511835e-01,  -7.14754313e-02,  -3.20514709e-01,
                          6.69446960e-02,  -4.22116816e-02,  -3.49709868e-01,
                          6.11144714e-02,  -5.58898412e-02,  -3.60674322e-01,
                          5.35824634e-02,  -6.67285845e-02,  -3.64031106e-01,
                          9.65736210e-02,  -1.08469650e-01,  -3.23558986e-01,
                          6.90703914e-02,  -9.84576717e-02,  -3.56560618e-01,
                          6.97793290e-02,  -1.36072934e-01,  -3.55872542e-01,
                          6.10326193e-02,  -1.34957746e-01,  -3.60806137e-01,
                          5.79539835e-02,  -1.41906828e-01,  -3.56181294e-01,
                          8.09238777e-02,   4.23781155e-03,  -3.35724860e-01,
                          4.86544892e-02,   3.23921964e-02,  -3.72244477e-01,
                          5.75560816e-02,   1.55969486e-02,  -3.53739619e-01,
                          6.95342124e-02,   6.57289196e-03,  -3.43852699e-01,
                          9.22727287e-02,  -1.53839774e-02,  -3.27187300e-01,
                          9.45805386e-03,   5.37503511e-02,  -3.95391643e-01,
                          5.28795868e-02,   9.86849423e-04,  -3.58042091e-01,
                          4.28126529e-02,  -4.60999180e-03,  -3.64781797e-01,
                          5.61728776e-02,  -5.79045825e-02,  -3.61012399e-01,
                          9.75366160e-02,  -1.14380777e-01,  -3.21485341e-01,
                          9.85823721e-02,  -1.14027604e-01,  -3.26447457e-01,
                          9.36995596e-02,  -1.20422408e-01,  -3.19806576e-01,
                          6.49475083e-02,  -1.10893101e-01,  -3.58469784e-01,
                          6.35978580e-02,   5.05462475e-03,  -3.49652946e-01,
                          5.96276633e-02,   9.42808110e-03,  -3.55865151e-01,
                          5.48846982e-02,   6.43396657e-03,  -3.56066257e-01,
                          6.68174326e-02,  -4.63791937e-03,  -3.55362505e-01,
                          9.61711481e-02,  -4.83594872e-02,  -3.13216448e-01,
                          6.42584711e-02,  -1.88044347e-02,  -3.54068547e-01,
                          3.07040438e-02,   3.75084486e-03,  -3.77552778e-01,
                          6.77055493e-02,  -3.48737389e-02,  -3.44124794e-01,
                          5.35582006e-02,  -5.40922284e-02,  -3.55068475e-01,
                          5.74137792e-02,  -7.08595589e-02,  -3.63591701e-01,
                          5.75220510e-02,  -7.84178376e-02,  -3.63714337e-01,
                          5.06857447e-02,  -8.23086649e-02,  -3.61001641e-01,
                          6.01366572e-02,  -1.13588095e-01,  -3.58258843e-01,
                          5.48940115e-02,  -1.19316772e-01,  -3.60422373e-01,
                          5.68867549e-02,  -1.31021723e-01,  -3.62847418e-01,
                          5.35807945e-02,   2.30006687e-02,  -3.61966997e-01,
                          3.12642306e-02,   4.05268073e-02,  -3.70714962e-01,
                          2.37742551e-02,   4.46490906e-02,  -3.74012798e-01,
                          9.10610333e-03,   5.53655177e-02,  -3.93037409e-01,
                          6.12801835e-02,   5.72732743e-03,  -3.51550728e-01,
                          1.26287248e-02,   2.33074874e-02,  -3.87848347e-01,
                          4.15430777e-02,  -5.42208459e-03,  -3.69133353e-01,
                          5.84772117e-02,  -2.08425764e-02,  -3.51643473e-01,
                          1.00286625e-01,  -9.70768034e-02,  -3.15064281e-01,
                          7.38033950e-02,  -1.39147937e-01,  -3.37557852e-01,
                          7.02974573e-02,  -1.42129511e-01,  -3.43859822e-01,
                          7.29633793e-02,   8.42592493e-03,  -3.36521775e-01,
                          8.54749009e-02,  -3.26718576e-03,  -3.25546622e-01,
                          5.75675815e-02,   1.99325569e-02,  -3.49212766e-01,
                          1.55380126e-02,   5.24755716e-02,  -3.85132015e-01,
                          6.74329251e-02,   4.49600443e-03,  -3.45361024e-01,
                          6.44035414e-02,   9.95243900e-03,  -3.47013891e-01,
                          1.08446949e-03,   6.04879633e-02,  -3.94666046e-01,
                          8.49739462e-03,   5.50285876e-02,  -3.94325674e-01,
                          6.40697405e-02,   7.49324681e-03,  -3.48014176e-01,
                          6.29991665e-02,   6.95640733e-03,  -3.49122167e-01,
                          6.06047641e-03,   5.00132255e-02,  -3.94771039e-01,
                          5.81934601e-02,   2.19030748e-03,  -3.51401478e-01,
                          9.36452895e-02,  -3.48771103e-02,  -3.22204113e-01,
                          2.82984357e-02,   1.22631723e-02,  -3.83806944e-01,
                          6.98502660e-02,  -2.77775191e-02,  -3.47409755e-01,
                          1.04223281e-01,  -7.84396380e-02,  -3.14935237e-01,
                          5.09784520e-02,  -6.55063763e-02,  -3.62299770e-01,
                          4.59606051e-02,  -7.80947134e-02,  -3.63689333e-01,
                          5.79899438e-02,  -9.93299186e-02,  -3.55985105e-01,
                          5.67699857e-02,  -1.05927676e-01,  -3.54260385e-01,
                          5.96011542e-02,  -1.24004260e-01,  -3.62707973e-01,
                          6.41458929e-02,  -1.42317444e-01,  -3.55790585e-01,
                          5.43495566e-02,   2.09992211e-02,  -3.52416664e-01,
                          5.50403353e-03,   5.66608459e-02,  -3.87683392e-01,
                          8.61595571e-02,  -6.31286390e-03,  -3.18630934e-01,
                          6.92291036e-02,   9.15012881e-03,  -3.37059706e-01,
                          5.92853837e-02,   1.07091591e-02,  -3.49845380e-01,
                          8.94531608e-02,  -1.14627834e-02,  -3.20181310e-01,
                          9.18942615e-02,  -1.33222369e-02,  -3.20255637e-01,
                          5.70165627e-02,   3.40771186e-03,  -3.53019506e-01,
                          5.98771945e-02,  -2.45420430e-02,  -3.48188490e-01,
                          4.89005744e-02,  -7.03403428e-02,  -3.61972541e-01,
                          4.57976051e-02,  -1.31568193e-01,  -3.63000691e-01,
                          5.55521548e-02,  -1.37648121e-01,  -3.61360580e-01,
                          8.22732821e-02,   2.33341660e-03,  -3.26318473e-01,
                          3.88787128e-02,   3.77796665e-02,  -3.67336780e-01,
                          6.75162598e-02,   8.61933548e-03,  -3.42193425e-01,
                          6.76776841e-02,   9.38355085e-03,  -3.39934707e-01,
                         -3.56772100e-04,   5.89885563e-02,  -3.90450031e-01,
                          9.02134459e-04,   5.42560928e-02,  -3.93909842e-01,
                         -3.33118346e-03,   5.82560189e-02,  -3.94418597e-01,
                          6.13336116e-02,  -4.30326071e-03,  -3.49857748e-01,
                          7.50939082e-03,   2.86358949e-02,  -3.89699906e-01,
                          5.68915047e-02,  -1.62046272e-02,  -3.53625476e-01,
                          3.12080830e-02,  -2.11011432e-03,  -3.71049792e-01,
                          2.91850492e-02,   1.27915386e-03,  -3.76200944e-01,
                          6.31594434e-02,  -2.70646922e-02,  -3.48619401e-01,
                          5.99380732e-02,  -3.82246859e-02,  -3.46789837e-01,
                          5.69887049e-02,  -4.57867607e-02,  -3.52578044e-01,
                          9.96474475e-02,  -8.10608864e-02,  -3.08251441e-01,
                          7.62391612e-02,  -1.35255232e-01,  -3.35160077e-01,
                          5.36540039e-02,  -1.35015637e-01,  -3.63503039e-01,
                          5.99647872e-02,   1.52470507e-02,  -3.46046090e-01,
                          2.09369306e-02,   4.67927493e-02,  -3.76428425e-01,
                          1.49211343e-02,   5.05207554e-02,  -3.79202545e-01,
                         -2.00331397e-03,   5.12752943e-02,  -3.93651307e-01,
                          9.38119739e-02,  -2.56463382e-02,  -3.21367413e-01,
                          5.22015058e-02,   2.52429815e-03,  -3.52752149e-01,
                          4.13597822e-02,  -5.57325641e-03,  -3.57365280e-01,
                          3.72357443e-02,  -5.90826152e-03,  -3.58806282e-01,
                          1.30627370e-02,   1.39014190e-02,  -3.83221507e-01,
                          2.98335776e-02,  -1.18064787e-03,  -3.66629988e-01,
                          2.13177763e-02,   4.33907937e-03,  -3.77900034e-01,
                          9.64486971e-02,  -6.64950237e-02,  -3.07693303e-01,
                          6.25848472e-02,  -3.56854312e-02,  -3.46652865e-01,
                          9.72933248e-02,  -1.09234840e-01,  -3.17503929e-01,
                          5.53647801e-02,  -9.37631652e-02,  -3.56657535e-01,
                          8.07682201e-02,  -1.26079857e-01,  -3.19406271e-01,
                          6.82929382e-02,  -1.42202273e-01,  -3.48141462e-01,
                          4.89388742e-02,  -1.40176848e-01,  -3.61985326e-01,
                          6.18593246e-02,   1.64010115e-02,  -3.41284096e-01,
                          5.53861298e-02,   3.36923636e-04,  -3.50690722e-01,
                          4.03489359e-03,   3.61395963e-02,  -3.89126897e-01,
                         -4.06171707e-03,   3.99007872e-02,  -3.88111979e-01,
                          4.98896800e-02,  -1.02624111e-03,  -3.54062706e-01,
                          5.66083118e-02,  -1.33312084e-02,  -3.52162719e-01,
                          5.63655235e-02,  -2.61827232e-03,  -3.51213694e-01,
                          9.53042135e-02,  -4.50127572e-02,  -3.12547803e-01,
                          9.90429986e-03,   8.26785155e-03,  -3.81902277e-01,
                          9.77087244e-02,  -1.05836518e-01,  -3.16224933e-01,
                          4.57130633e-02,  -6.47198260e-02,  -3.62726092e-01,
                          8.84477422e-02,  -1.22936159e-01,  -3.21261704e-01,
                          5.13425991e-02,  -1.14146352e-01,  -3.58131826e-01,
                          5.96103892e-02,   2.01642942e-02,  -3.43601733e-01,
                          2.70518865e-02,   4.13915701e-02,  -3.71808648e-01,
                          1.17735472e-02,   5.48548549e-02,  -3.78397465e-01,
                          8.56246278e-02,  -1.37861697e-02,  -3.09955388e-01,
                          4.23348462e-03,   4.18967828e-02,  -3.90111774e-01,
                          9.48249549e-02,  -4.17757370e-02,  -3.14183146e-01,
                          4.51761000e-02,  -3.58257536e-03,  -3.55336875e-01,
                          4.71155718e-02,  -1.57014164e-03,  -3.59219939e-01,
                          8.03347491e-03,   1.72036979e-02,  -3.83010834e-01,
                          4.52619568e-02,  -9.26119015e-02,  -3.59215319e-01,
                          7.82510042e-02,   4.34070360e-03,  -3.26577723e-01,
                          3.06849815e-02,   3.84512693e-02,  -3.63366097e-01,
                          1.67806484e-02,   4.80845012e-02,  -3.75411838e-01,
                          8.71598199e-02,  -1.83718596e-02,  -3.11808705e-01,
                          6.65156054e-04,   5.41576333e-02,  -3.95624191e-01,
                          5.28969020e-02,  -4.77582915e-03,  -3.51857752e-01,
                         -3.01699876e-03,   3.22265401e-02,  -3.87447417e-01,
                          9.13136974e-02,  -4.88232113e-02,  -3.08117360e-01,
                          9.79841053e-02,  -5.76509684e-02,  -3.11914414e-01,
                          6.26365170e-02,  -3.18540186e-02,  -3.44015360e-01,
                          4.27286401e-02,  -7.13555664e-02,  -3.61870110e-01,
                          4.65678051e-02,  -8.57756734e-02,  -3.60863239e-01,
                          8.41244310e-02,  -1.30071759e-01,  -3.30087304e-01,
                          5.37004359e-02,  -1.06235780e-01,  -3.58897358e-01,
                          7.37500712e-02,   6.33553509e-03,  -3.18390131e-01,
                          6.68350980e-02,   9.56610590e-03,  -3.32079470e-01,
                          6.45374581e-02,   1.34312892e-02,  -3.37014198e-01,
                          1.86508484e-02,   4.75769751e-02,  -3.70759755e-01,
                          9.04783458e-02,  -2.32397188e-02,  -3.15343797e-01,
                          9.28289667e-02,  -3.39365080e-02,  -3.17803621e-01,
                          4.74031866e-02,  -1.14900870e-02,  -3.50468993e-01,
                          2.37673707e-02,  -6.18129503e-03,  -3.66325408e-01,
                          1.47663159e-02,   3.92948557e-03,  -3.78202289e-01,
                          5.44178747e-02,  -3.03261615e-02,  -3.44281673e-01,
                          4.97272201e-02,  -4.20690104e-02,  -3.46848309e-01,
                          4.84659374e-02,  -4.91631329e-02,  -3.51371497e-01,
                          5.21890223e-02,  -5.50690927e-02,  -3.56741846e-01,
                          4.34581935e-02,  -7.28063658e-02,  -3.62580180e-01,
                          4.40331586e-02,  -1.26985878e-01,  -3.61902475e-01,
                          6.02054894e-02,  -1.45651087e-01,  -3.44971478e-01,
                          5.18936068e-02,  -1.42099798e-01,  -3.59029800e-01,
                          5.15660904e-02,   2.39040200e-02,  -3.45004350e-01,
                          4.87074852e-02,   2.57224925e-02,  -3.51880372e-01,
                          4.69771028e-02,   2.52428837e-02,  -3.51781756e-01,
                         -5.07346168e-03,   6.05696142e-02,  -3.93601775e-01,
                         -1.06469896e-02,   5.45925684e-02,  -3.92513782e-01,
                          4.32002880e-02,  -8.53989739e-03,  -3.52286726e-01,
                          3.88600416e-02,  -5.61047858e-03,  -3.54837924e-01,
                          5.09709679e-02,  -1.46894976e-02,  -3.51774901e-01,
                          2.22987449e-03,   2.10604798e-02,  -3.84398907e-01,
                          5.07907979e-02,  -2.19231863e-02,  -3.49185079e-01,
                          9.25393179e-02,  -5.23069538e-02,  -3.07703793e-01,
                          2.05611810e-02,  -2.27184640e-03,  -3.70219588e-01,
                          5.34724332e-02,  -3.59102488e-02,  -3.43926728e-01,
                          5.78491911e-02,  -3.78246494e-02,  -3.45769018e-01,
                          5.05145378e-02,  -4.61077429e-02,  -3.53128612e-01,
                          9.43144560e-02,  -1.03581466e-01,  -3.11200589e-01,
                          8.99324864e-02,  -1.05999947e-01,  -3.06841850e-01,
                          3.53178345e-02,  -7.22752437e-02,  -3.58222544e-01,
                          4.39302400e-02,  -1.01029746e-01,  -3.58679295e-01,
                          6.49809167e-02,  -1.40739188e-01,  -3.33546221e-01,
                          5.70131913e-02,  -1.42436281e-01,  -3.50382924e-01,
                          7.86772743e-02,  -1.92724913e-03,  -3.17921489e-01,
                         -8.04545451e-03,   6.10250682e-02,  -3.89730364e-01,
                          9.18883681e-02,  -3.14784348e-02,  -3.14033806e-01,
                          3.50176990e-02,  -6.67544268e-03,  -3.60627413e-01,
                          5.65905012e-02,  -3.61908935e-02,  -3.41939360e-01,
                          4.43601310e-02,  -6.14894815e-02,  -3.61594528e-01,
                          9.51303542e-02,  -1.07372954e-01,  -3.13981324e-01,
                          8.89078453e-02,  -1.16358437e-01,  -3.12124252e-01,
                          3.84325720e-02,  -1.25662327e-01,  -3.60752910e-01,
                          3.88384759e-02,  -1.28303468e-01,  -3.61267269e-01,
                          4.68617976e-02,  -1.34273067e-01,  -3.61922741e-01,
                          7.11966455e-02,   6.08459674e-03,  -3.28439116e-01,
                          3.90708335e-02,   3.55281606e-02,  -3.59806806e-01,
                         -1.06024463e-02,   5.68582080e-02,  -3.94673347e-01,
                          9.30470228e-02,  -4.23747674e-02,  -3.10218900e-01,
                          9.51530561e-02,  -6.11732677e-02,  -3.01799983e-01,
                          5.37748495e-03,   8.62094481e-03,  -3.80988628e-01,
                          9.43850875e-02,  -7.48648643e-02,  -3.03246766e-01,
                          4.68199141e-02,  -4.19140980e-02,  -3.50215107e-01,
                          4.02226485e-02,  -5.29784262e-02,  -3.50374192e-01,
                          4.28501256e-02,  -1.09457850e-01,  -3.58805358e-01,
                          5.34882694e-02,  -1.44018918e-01,  -3.55329424e-01,
                          4.11041304e-02,  -1.48520932e-01,  -3.51414919e-01,
                          5.35402037e-02,   2.21357010e-02,  -3.36420238e-01,
                          3.79997715e-02,   3.54985334e-02,  -3.52824479e-01,
                         -2.19432171e-04,   6.36163503e-02,  -3.89593393e-01,
                          9.62132514e-02,  -8.84657577e-02,  -3.06593388e-01,
                          4.25656699e-02,  -7.99536332e-02,  -3.62622082e-01,
                          3.98849435e-02,  -8.75970051e-02,  -3.60735118e-01,
                          4.21581492e-02,  -9.42392498e-02,  -3.58928233e-01,
                          7.55845308e-02,  -1.31012499e-01,  -3.24814409e-01,
                          4.58610430e-02,  -1.13022842e-01,  -3.56846929e-01,
                          2.19903868e-02,   4.53411713e-02,  -3.66627187e-01,
                         -8.54813680e-03,   6.40302971e-02,  -3.80860060e-01,
                          8.96785930e-02,  -3.79801765e-02,  -3.06070179e-01,
                          3.80634628e-02,  -1.01655051e-02,  -3.52194518e-01,
                          2.91101802e-02,  -6.01824420e-03,  -3.59743088e-01,
                          1.79987513e-02,  -3.46345664e-03,  -3.65136832e-01,
                          9.61888954e-03,   2.19653593e-03,  -3.73150915e-01,
                          1.51750967e-02,   4.13922768e-04,  -3.71325970e-01,
                          9.18790177e-02,  -6.72366545e-02,  -2.99816638e-01,
                          5.29394448e-02,  -3.71130779e-02,  -3.43479872e-01,
                          4.34974022e-02,  -4.52692099e-02,  -3.50619763e-01,
                          8.94126073e-02,  -1.12388842e-01,  -3.11330587e-01,
                          8.37741345e-02,  -1.26634538e-01,  -3.15838158e-01,
                          8.14874023e-02,  -1.31533414e-01,  -3.19396287e-01,
                          6.60390854e-02,  -1.42051294e-01,  -3.41073513e-01,
                          3.56650203e-02,  -1.36079490e-01,  -3.60697925e-01,
                          4.64615226e-02,   2.77408585e-02,  -3.41186523e-01,
                          7.39685446e-02,   9.88588668e-04,  -3.14418107e-01,
                          8.12029988e-02,  -9.72780958e-03,  -3.02922070e-01,
                         -1.24791441e-02,   6.29623979e-02,  -3.90492201e-01,
                         -1.28601985e-02,   5.13012484e-02,  -3.91483337e-01,
                         -8.36220663e-03,   4.52132672e-02,  -3.91075760e-01,
                         -6.24248572e-03,   3.04195322e-02,  -3.87424827e-01,
                         -7.21625146e-03,   1.89225245e-02,  -3.83917212e-01,
                          9.29043069e-02,  -9.42005217e-02,  -3.03274989e-01,
                          3.06839310e-02,  -7.05686659e-02,  -3.54594648e-01,
                          8.40593055e-02,  -1.22085504e-01,  -3.12406451e-01,
                          3.50523032e-02,  -8.49431977e-02,  -3.59938562e-01,
                          3.81213501e-02,  -9.40776393e-02,  -3.59229833e-01,
                          3.94302122e-02,  -1.37670219e-01,  -3.61615121e-01,
                          6.18288629e-02,   1.96427684e-02,  -3.33482772e-01,
                          4.72649373e-03,   5.45125231e-02,  -3.77564400e-01,
                          8.57367888e-02,  -1.97910871e-02,  -3.06014597e-01,
                         -1.52023211e-02,   5.67018129e-02,  -3.90251994e-01,
                         -1.26752490e-02,   4.18851636e-02,  -3.88120472e-01,
                          3.37376371e-02,  -6.11134013e-03,  -3.53776634e-01,
                          3.48639339e-02,  -1.63879991e-02,  -3.53290498e-01,
                          3.05280872e-02,  -6.46881014e-03,  -3.56268376e-01,
                         -3.54677252e-03,   1.37422383e-02,  -3.81278813e-01,
                          4.50889356e-02,  -2.24544927e-02,  -3.49344403e-01,
                          1.79753061e-02,  -9.56502184e-03,  -3.65154117e-01,
                          4.94855121e-02,  -3.46208215e-02,  -3.44738603e-01,
                          4.38457914e-02,  -3.03073712e-02,  -3.46381336e-01,
                          4.56669703e-02,  -3.38346101e-02,  -3.46153289e-01,
                          4.42055836e-02,  -3.95941250e-02,  -3.47775489e-01,
                          3.97804342e-02,  -6.53330013e-02,  -3.56555849e-01,
                          8.39276090e-02,  -1.22501165e-01,  -3.08512986e-01,
                          3.47917639e-02,  -1.05735987e-01,  -3.54249239e-01,
                          3.79998386e-02,  -1.12214133e-01,  -3.59530121e-01,
                          4.08378132e-02,  -1.44043833e-01,  -3.56404483e-01,
                          2.07874179e-03,   5.47847487e-02,  -3.68909389e-01,
                         -9.44365468e-03,   6.33512139e-02,  -3.89403045e-01,
                         -2.25646812e-02,   6.13702089e-02,  -3.91721636e-01,
                          3.16499807e-02,  -1.23485122e-02,  -3.55400205e-01,
                          8.40359479e-02,  -1.02832548e-01,  -2.98990220e-01,
                          8.42728913e-02,  -1.12808809e-01,  -3.04466635e-01,
                          4.80013080e-02,  -1.48057178e-01,  -3.46060306e-01,
                          3.44221890e-02,   3.37612964e-02,  -3.47867161e-01,
                          2.15566717e-02,   4.28273790e-02,  -3.58357936e-01,
                          1.48026170e-02,   4.95241769e-02,  -3.67022157e-01,
                          3.51846144e-02,  -2.59336364e-02,  -3.52066129e-01,
                          2.12676451e-02,  -1.05010550e-02,  -3.62766862e-01,
                          1.33379195e-02,  -4.63530608e-03,  -3.67915154e-01,
                          3.13744396e-02,  -1.45248435e-02,  -3.55380297e-01,
                          2.39502788e-02,  -1.46545852e-02,  -3.61635029e-01,
                          3.75236198e-02,  -2.69027315e-02,  -3.50362182e-01,
                          4.28816751e-02,  -3.09843533e-02,  -3.46773893e-01,
                          2.55619697e-02,  -2.08014306e-02,  -3.58898818e-01,
                          4.11080457e-02,  -3.33034545e-02,  -3.46927702e-01,
                          3.35112065e-02,  -3.08364574e-02,  -3.53126645e-01,
                          3.88947539e-02,  -3.79788205e-02,  -3.50066066e-01,
                          9.17112008e-02,  -8.92136768e-02,  -2.98742026e-01,
                          8.11588988e-02,  -1.06038079e-01,  -2.98569828e-01,
                          7.15907887e-02,  -1.36878446e-01,  -3.17494631e-01,
                          3.63773927e-02,  -9.81754214e-02,  -3.58087897e-01,
                          5.40683977e-02,  -1.45866469e-01,  -3.32915038e-01,
                          3.76779065e-02,  -1.33413196e-01,  -3.60398501e-01,
                          5.73861301e-02,   1.90252084e-02,  -3.29580963e-01,
                          1.71449296e-02,   4.43728119e-02,  -3.58392537e-01,
                          8.35079327e-02,  -1.06235929e-02,  -3.02626997e-01,
                         -1.06250327e-02,   6.27023131e-02,  -3.82979900e-01,
                         -1.98038686e-02,   6.19082525e-02,  -3.88343066e-01,
                         -2.02499609e-02,   5.77903539e-02,  -3.91284645e-01,
                          8.98469761e-02,  -5.14929593e-02,  -2.97277629e-01,
                          8.94348845e-02,  -6.13800511e-02,  -2.95344472e-01,
                          4.27629286e-03,   6.28605764e-03,  -3.78003180e-01,
                          1.40055148e-02,  -1.27543267e-02,  -3.63759220e-01,
                          9.60632414e-02,  -8.35821405e-02,  -3.05268168e-01,
                          3.52934226e-02,  -4.00323458e-02,  -3.48679423e-01,
                          4.01380584e-02,  -4.27532457e-02,  -3.47810030e-01,
                          3.55595350e-02,  -4.87947837e-02,  -3.46447766e-01,
                          4.25176658e-02,  -5.79901077e-02,  -3.53898436e-01,
                          7.08405077e-02,  -1.34126380e-01,  -3.13894987e-01,
                          2.62841899e-02,  -1.32646590e-01,  -3.57374430e-01,
                          3.97538543e-02,   2.74066757e-02,  -3.39222878e-01,
                          6.85097948e-02,   8.77430104e-03,  -3.21828425e-01,
                          1.07853496e-02,   5.20841144e-02,  -3.68577391e-01,
                          8.55812877e-02,  -1.79069564e-02,  -3.05091083e-01,
                         -1.49270045e-02,   6.58017024e-02,  -3.87236118e-01,
                         -2.50150934e-02,   5.95925152e-02,  -3.90112221e-01,
                          8.93352330e-02,  -4.52662595e-02,  -3.03121835e-01,
                          1.17335794e-02,  -8.75044242e-03,  -3.63983929e-01,
                          1.50937773e-02,  -1.40149584e-02,  -3.62153292e-01,
                         -6.51474856e-03,   4.61730361e-03,  -3.77584517e-01,
                          3.40785831e-03,  -6.52894471e-03,  -3.69096071e-01,
                          2.49479655e-02,  -2.53624655e-02,  -3.58586550e-01,
                          1.79653112e-02,  -1.71755124e-02,  -3.62643123e-01,
                          2.75011566e-02,  -3.54322642e-02,  -3.54749143e-01,
                          3.39650027e-02,  -3.63453589e-02,  -3.51442397e-01,
                          1.81883313e-02,  -2.79761832e-02,  -3.63587618e-01,
                          1.97413862e-02,  -3.19386087e-02,  -3.60590786e-01,
                          7.91518539e-02,  -1.22971460e-01,  -3.06105107e-01,
                          3.55378948e-02,  -1.42918661e-01,  -3.54876876e-01,
                          6.16760030e-02,   1.42139085e-02,  -3.27186584e-01,
                          6.56064749e-02,   1.00306794e-02,  -3.23791623e-01,
                          1.28731783e-03,   5.38042411e-02,  -3.71042818e-01,
                         -6.50639366e-03,   5.90818226e-02,  -3.69100422e-01,
                          8.80298540e-02,  -2.99375877e-02,  -2.98325062e-01,
                         -2.39897054e-02,   6.00575693e-02,  -3.90382916e-01,
                         -1.97938085e-02,   5.34841083e-02,  -3.90088648e-01,
                         -1.78950839e-02,   3.36590633e-02,  -3.85484189e-01,
                         -1.55087672e-02,   2.58871969e-02,  -3.82984608e-01,
                          9.02975798e-02,  -9.71502662e-02,  -2.98008591e-01,
                          7.48853087e-02,  -1.31692395e-01,  -3.11249644e-01,
                          2.88911350e-02,  -1.17246255e-01,  -3.56688797e-01,
                          6.89774528e-02,  -1.37829185e-01,  -3.24400514e-01,
                          3.51356119e-02,  -1.17815375e-01,  -3.56018543e-01,
                          3.48605104e-02,  -1.47638142e-01,  -3.49807352e-01,
                         -1.68111213e-02,   6.50965124e-02,  -3.88294101e-01,
                         -2.00549755e-02,   4.79397140e-02,  -3.87483329e-01,
                          8.63811672e-02,  -3.45679261e-02,  -2.95785338e-01,
                         -1.65724494e-02,   3.82121690e-02,  -3.85392994e-01,
                         -4.56435792e-03,  -2.73481966e-03,  -3.76232117e-01,
                          9.08686146e-02,  -7.50575215e-02,  -2.97023743e-01,
                          1.26335518e-02,  -2.96945181e-02,  -3.61754239e-01,
                          2.37994678e-02,  -3.52015980e-02,  -3.54248643e-01,
                          8.59716535e-02,  -9.98195708e-02,  -2.96375245e-01,
                          3.08460705e-02,  -1.01578988e-01,  -3.55761766e-01,
                          5.16242832e-02,  -1.48299858e-01,  -3.37639242e-01,
                          2.73594111e-02,  -1.34403214e-01,  -3.58032674e-01,
                          3.32942680e-02,  -1.42823815e-01,  -3.57261360e-01,
                          8.17956701e-02,  -1.18700340e-02,  -3.03645641e-01,
                          8.40650275e-02,  -3.14859338e-02,  -2.93036610e-01,
                          3.21831889e-02,  -4.10814695e-02,  -3.49394947e-01,
                          3.57164368e-02,  -5.43870181e-02,  -3.48656178e-01,
                          2.65841838e-02,  -8.26480836e-02,  -3.54401946e-01,
                          6.57995865e-02,  -1.38971046e-01,  -3.17296684e-01,
                          6.85764626e-02,   3.28988605e-03,  -3.05489987e-01,
                          7.68660605e-02,  -3.83049645e-03,  -3.06390882e-01,
                          2.56276671e-02,   3.65679860e-02,  -3.49288970e-01,
                          1.86645743e-02,   4.11825068e-02,  -3.53479505e-01,
                         -2.05676202e-02,   4.33858186e-02,  -3.86046946e-01,
                         -2.01642793e-02,   1.78614128e-02,  -3.82617295e-01,
                          8.49471688e-02,  -7.71695450e-02,  -2.91118383e-01,
                          9.14246310e-03,  -1.60997901e-02,  -3.64426732e-01,
                          6.16902346e-03,  -2.00295299e-02,  -3.64649504e-01,
                          1.79404560e-02,  -3.28087248e-02,  -3.59685093e-01,
                          5.78888841e-02,  -1.41572744e-01,  -3.18999738e-01,
                          5.35809435e-02,  -1.42284527e-01,  -3.25610667e-01,
                          3.79675776e-02,   3.03750057e-02,  -3.42317790e-01,
                          8.12531635e-02,  -1.89268570e-02,  -2.96211988e-01,
                         -2.57496145e-02,   6.45304248e-02,  -3.91170949e-01,
                          8.55601355e-02,  -2.11164691e-02,  -2.96873122e-01,
                         -2.45734006e-02,   4.66849990e-02,  -3.86895716e-01,
                          8.77986029e-02,  -6.22777231e-02,  -2.91868567e-01,
                          8.93028826e-02,  -6.82945848e-02,  -2.88794488e-01,
                          2.81412750e-02,  -3.76541056e-02,  -3.51233631e-01,
                          3.05467397e-02,  -4.68028150e-02,  -3.47092301e-01,
                          1.09399864e-02,  -3.04834619e-02,  -3.60810220e-01,
                          7.88885355e-02,  -1.19276077e-01,  -3.05125266e-01,
                          7.27554783e-02,  -1.25617638e-01,  -3.01033407e-01,
                          2.97121555e-02,  -9.11057964e-02,  -3.56701493e-01,
                          2.60826610e-02,  -1.27322584e-01,  -3.55968386e-01,
                          2.21638009e-02,  -1.29838988e-01,  -3.58199120e-01,
                          4.16015610e-02,  -1.49858609e-01,  -3.39465886e-01,
                          3.15552838e-02,  -1.49158239e-01,  -3.47523987e-01,
                          5.38341850e-02,   1.85546391e-02,  -3.23334366e-01,
                          5.13777807e-02,   2.26943996e-02,  -3.31736922e-01,
                          2.24666782e-02,   3.73391211e-02,  -3.46579850e-01,
                         -1.80177875e-02,   6.62257969e-02,  -3.81129622e-01,
                         -2.72635333e-02,   6.53451160e-02,  -3.86663347e-01,
                         -2.96745896e-02,   6.11751750e-02,  -3.89569253e-01,
                          8.34725797e-02,  -3.75406630e-02,  -2.90940553e-01,
                          8.70018825e-02,  -4.50835451e-02,  -2.94888973e-01,
                          8.89248699e-02,  -8.54778215e-02,  -2.94238359e-01,
                          2.13363599e-02,  -4.20629270e-02,  -3.50238681e-01,
                          2.60520447e-02,  -4.40242514e-02,  -3.46427262e-01,
                          1.15383044e-02,  -3.56184468e-02,  -3.59993577e-01,
                          2.94354074e-02,  -5.34936562e-02,  -3.49750638e-01,
                          7.89039358e-02,  -1.13774806e-01,  -2.99303472e-01,
                          3.03166173e-02,  -6.63260221e-02,  -3.51294458e-01,
                          7.03395903e-02,  -1.30219206e-01,  -3.05892885e-01,
                          2.50734948e-02,  -1.39375538e-01,  -3.54484797e-01,
                          6.01001754e-02,   1.04727522e-02,  -3.14081639e-01,
                         -3.01890895e-02,   4.72991243e-02,  -3.85119677e-01,
                         -2.75074616e-02,   3.57144736e-02,  -3.83712590e-01,
                          8.43627825e-02,  -9.20964926e-02,  -2.91749746e-01,
                          2.71958560e-02,  -4.81569432e-02,  -3.46704036e-01,
                          2.43247803e-02,  -6.08390793e-02,  -3.44054788e-01,
                          2.36896798e-02,  -7.05086738e-02,  -3.53105694e-01,
                          2.59760041e-02,  -7.77643025e-02,  -3.53574157e-01,
                          4.72835749e-02,  -1.50390550e-01,  -3.35335255e-01,
                          4.14410196e-02,  -1.49869457e-01,  -3.44754338e-01,
                          5.16378693e-02,   2.14974638e-02,  -3.22988063e-01,
                          3.97425666e-02,   3.10100392e-02,  -3.34909469e-01,
                         -3.46777141e-02,   5.97241595e-02,  -3.90536427e-01,
                         -3.13299634e-02,   5.69945052e-02,  -3.87442976e-01,
                         -2.73857582e-02,   4.04463895e-02,  -3.85279357e-01,
                         -2.84438282e-02,   2.03781687e-02,  -3.81239802e-01,
                         -1.36452988e-02,  -5.45409974e-04,  -3.74301851e-01,
                          3.39347054e-03,  -1.77466720e-02,  -3.64764273e-01,
                          6.32699952e-03,  -2.23738812e-02,  -3.62715423e-01,
                          2.46707741e-02,  -1.00406267e-01,  -3.54999989e-01,
                         -1.66080520e-02,   6.07132167e-02,  -3.69080871e-01,
                         -3.01391482e-02,   6.63531646e-02,  -3.86409998e-01,
                         -3.06289233e-02,   6.64419159e-02,  -3.87053668e-01,
                         -3.01834550e-02,   5.20053543e-02,  -3.88602555e-01,
                          8.68497193e-02,  -5.69798686e-02,  -2.89703429e-01,
                          1.88723095e-02,  -6.24113865e-02,  -3.44839305e-01,
                          8.19510967e-02,  -1.15392223e-01,  -3.03237706e-01,
                          6.39605448e-02,  -1.33711785e-01,  -3.06277066e-01,
                          4.91911322e-02,   2.15695314e-02,  -3.19889337e-01,
                         -4.18902095e-03,   5.05478755e-02,  -3.56232882e-01,
                          7.96715319e-02,  -1.08408649e-02,  -3.00391465e-01,
                         -3.72881554e-02,   5.22787832e-02,  -3.89052957e-01,
                         -1.15107158e-02,   5.36928186e-03,  -3.79656553e-01,
                         -2.25309525e-02,   7.38534145e-04,  -3.73779148e-01,
                          1.50289759e-02,  -3.71183716e-02,  -3.54022771e-01,
                          6.02806956e-02,  -1.37905702e-01,  -3.15922022e-01,
                          1.85949933e-02,  -9.15228054e-02,  -3.51029903e-01,
                          1.83405019e-02,  -1.25849962e-01,  -3.55641007e-01,
                          3.20141129e-02,  -1.49768040e-01,  -3.45244616e-01,
                          2.21068580e-02,  -1.39839128e-01,  -3.51331502e-01,
                          5.43485656e-02,   1.32078826e-02,  -3.14540356e-01,
                          3.72739062e-02,   2.46867109e-02,  -3.24086964e-01,
                          4.46181111e-02,   2.77447831e-02,  -3.33897769e-01,
                          6.53260946e-02,   6.48407079e-03,  -3.09535950e-01,
                          3.28344516e-02,   2.84981262e-02,  -3.25366914e-01,
                          7.27975443e-02,  -4.37785545e-03,  -3.05186689e-01,
                         -3.80790327e-04,   5.39124645e-02,  -3.59284043e-01,
                         -3.06089651e-02,   6.56814501e-02,  -3.77385408e-01,
                         -1.23393266e-02,  -8.28201976e-03,  -3.70337933e-01,
                          6.16069464e-03,  -3.64768468e-02,  -3.54346544e-01,
                          8.18334334e-03,  -3.16088647e-02,  -3.60448807e-01,
                          8.12689960e-02,  -9.89668593e-02,  -2.92075932e-01,
                          2.13110428e-02,  -1.12685405e-01,  -3.54535371e-01,
                          5.45026436e-02,   1.13291806e-02,  -3.11331511e-01,
                          5.05202375e-02,   1.52362240e-02,  -3.15099716e-01,
                          7.17158169e-02,  -9.55302268e-03,  -2.92599410e-01,
                         -3.86146232e-02,   6.58149645e-02,  -3.86243224e-01,
                         -4.11984399e-02,   6.35637790e-02,  -3.89053851e-01,
                          8.18825215e-02,  -8.39850456e-02,  -2.86896318e-01,
                         -8.67130118e-04,  -1.34862997e-02,  -3.62291664e-01,
                         -3.51552851e-03,  -1.81156956e-02,  -3.62252265e-01,
                          1.85819622e-02,  -3.90054062e-02,  -3.52159590e-01,
                          9.12765320e-03,  -4.08623442e-02,  -3.47338200e-01,
                          7.13442042e-02,  -1.13307469e-01,  -2.92246133e-01,
                          1.60727296e-02,  -9.93071273e-02,  -3.52239758e-01,
                          3.20901349e-02,  -1.50715813e-01,  -3.35285395e-01,
                          2.91877389e-02,   3.12659107e-02,  -3.30463469e-01,
                          2.65832152e-02,   3.48384194e-02,  -3.37369323e-01,
                          7.17756385e-03,   4.75669801e-02,  -3.54427010e-01,
                         -2.91888434e-02,   6.88884333e-02,  -3.79569829e-01,
                          8.37089196e-02,  -7.94933736e-02,  -2.86793977e-01,
                          1.50582567e-02,  -4.92345914e-02,  -3.42178136e-01,
                          2.58705541e-02,  -5.41343130e-02,  -3.44121248e-01,
                          7.77917951e-02,  -1.04900539e-01,  -2.92265594e-01,
                          1.96265765e-02,  -7.06865489e-02,  -3.46744686e-01,
                          7.45325312e-02,  -1.16509221e-01,  -2.98018545e-01,
                          2.33201422e-02,  -1.36574239e-01,  -3.55228454e-01,
                          1.83571614e-02,  -1.45412356e-01,  -3.47265452e-01,
                          5.88493235e-03,   4.15513664e-02,  -3.41842890e-01,
                         -2.07800400e-02,   6.24869540e-02,  -3.74917716e-01,
                         -4.25553098e-02,   5.89725375e-02,  -3.88973325e-01,
                         -4.17689756e-02,   5.81041984e-02,  -3.92021209e-01,
                         -2.80174203e-02,   2.78236009e-02,  -3.84136677e-01,
                          8.01116750e-02,  -9.29584578e-02,  -2.87497312e-01,
                          1.17496420e-02,  -7.23753721e-02,  -3.44901383e-01,
                          1.95889659e-02,  -7.85979331e-02,  -3.48637551e-01,
                          1.33655528e-02,  -9.01984572e-02,  -3.48623097e-01,
                         -7.25055858e-03,   5.50087206e-02,  -3.59778136e-01,
                          8.23071077e-02,  -3.11997272e-02,  -2.86057264e-01,
                         -2.66440958e-02,   8.90888926e-03,  -3.80389303e-01,
                         -4.85492731e-03,  -1.15227969e-02,  -3.65470558e-01,
                         -1.51595660e-02,  -4.78842249e-03,  -3.72964054e-01,
                         -3.06715490e-04,  -3.17452438e-02,  -3.60276550e-01,
                          1.39101930e-02,  -5.89627996e-02,  -3.40135008e-01,
                          7.63108283e-02,  -1.09410845e-01,  -2.90338874e-01,
                          6.69542104e-02,  -1.18486196e-01,  -2.92534113e-01,
                          1.88910849e-02,  -7.72285908e-02,  -3.50315928e-01,
                          2.86635738e-02,  -1.50350094e-01,  -3.42955977e-01,
                          2.80025918e-02,  -1.49119332e-01,  -3.47420871e-01,
                          2.06415076e-02,   3.45014296e-02,  -3.28892469e-01,
                          6.73554242e-02,   2.69846455e-03,  -3.02366734e-01,
                          1.04683200e-02,   4.28943671e-02,  -3.46830457e-01,
                         -7.53552048e-03,  -1.73139647e-02,  -3.60836387e-01,
                          6.90205842e-02,  -1.23892032e-01,  -2.99985021e-01,
                          1.85380392e-02,  -1.04164734e-01,  -3.53860676e-01,
                          3.51888351e-02,  -1.49529323e-01,  -3.26625168e-01,
                          2.47154981e-02,  -1.50131747e-01,  -3.32671493e-01,
                          5.93703352e-02,   7.59996707e-03,  -3.07005644e-01,
                         -1.71718933e-03,   4.53536175e-02,  -3.42861414e-01,
                         -3.48972678e-02,   6.69564605e-02,  -3.77798498e-01,
                         -3.89432050e-02,   6.76158294e-02,  -3.81921411e-01,
                         -4.54868227e-02,   6.33865967e-02,  -3.88883144e-01,
                          8.05964842e-02,  -5.82692921e-02,  -2.85434395e-01,
                          8.46418291e-02,  -6.69030547e-02,  -2.87187189e-01,
                          8.26068372e-02,  -7.20357001e-02,  -2.86208183e-01,
                         -2.85541676e-02,  -5.01059368e-03,  -3.71834487e-01,
                         -1.90848205e-02,  -1.07664522e-02,  -3.65051001e-01,
                          7.76416957e-02,  -8.64238292e-02,  -2.84697324e-01,
                          1.80156548e-02,  -4.69955355e-02,  -3.43402773e-01,
                         -5.60702570e-03,  -3.44792306e-02,  -3.54266822e-01,
                          9.05667618e-03,  -1.07171647e-01,  -3.52516383e-01,
                          2.19345484e-02,  -1.50153622e-01,  -3.37623477e-01,
                          1.98000483e-02,  -1.45202741e-01,  -3.44970763e-01,
                          4.35955338e-02,   1.96067058e-02,  -3.17063063e-01,
                          5.73095717e-02,   4.78592142e-03,  -3.02910239e-01,
                          1.86853316e-02,   3.80775556e-02,  -3.39998484e-01,
                          6.45379499e-02,  -4.89288149e-03,  -2.92296916e-01,
                         -4.10029627e-02,   6.82727322e-02,  -3.79476130e-01,
                         -4.33366112e-02,   6.86571300e-02,  -3.84111285e-01,
                         -4.97657880e-02,   5.71467169e-02,  -3.86859983e-01,
                         -4.88574468e-02,   5.40986210e-02,  -3.86304170e-01,
                          8.47930089e-02,  -4.77699675e-02,  -2.90598691e-01,
                         -4.05636467e-02,   4.66856360e-02,  -3.86853218e-01,
                         -3.58805358e-02,   4.14771140e-02,  -3.85641903e-01,
                          8.00177678e-02,  -7.24600032e-02,  -2.83916950e-01,
                          7.92325139e-02,  -7.49924183e-02,  -2.84230798e-01,
                         -7.23946001e-03,  -2.75107529e-02,  -3.57403725e-01,
                          6.20998733e-04,  -3.63231488e-02,  -3.54193449e-01,
                          5.06700762e-02,  -1.42862275e-01,  -3.21023643e-01,
                          1.37292445e-02,  -1.15092970e-01,  -3.53451580e-01,
                          4.27941093e-03,  -1.18218549e-01,  -3.50956082e-01,
                          1.27687501e-02,  -1.27835810e-01,  -3.51626694e-01,
                          3.11781392e-02,  -1.50604293e-01,  -3.27072024e-01,
                          2.13880986e-02,  -1.47943273e-01,  -3.42194319e-01,
                          9.16669052e-03,   3.91998477e-02,  -3.34464103e-01,
                          6.48401752e-02,  -2.35825195e-03,  -2.95763522e-01,
                          7.34592602e-02,  -1.26728108e-02,  -2.91220516e-01,
                         -4.64965105e-02,   6.52766898e-02,  -3.82511079e-01,
                         -4.41840291e-02,   6.40015155e-02,  -3.84546518e-01,
                         -4.95085083e-02,   6.06804341e-02,  -3.87153894e-01,
                          7.76423886e-02,  -5.45615368e-02,  -2.82120794e-01,
                         -1.32006370e-02,  -1.79610290e-02,  -3.59773189e-01,
                          6.54151142e-02,  -1.21635906e-01,  -2.96577096e-01,
                          2.74308231e-02,   2.91388333e-02,  -3.22021723e-01,
                          4.54094484e-02,   1.53306779e-02,  -3.10085744e-01,
                         -8.48829094e-03,   4.83148471e-02,  -3.40060562e-01,
                         -4.11747135e-02,   3.13765034e-02,  -3.82877797e-01,
                          8.04582164e-02,  -7.62386173e-02,  -2.85111547e-01,
                         -1.01123452e-02,  -1.60829537e-02,  -3.63042086e-01,
                         -2.24951953e-02,  -6.81410870e-03,  -3.69902432e-01,
                          1.17788147e-02,  -4.29538377e-02,  -3.46125275e-01,
                         -1.05020078e-02,  -3.31001878e-02,  -3.53878617e-01,
                          5.79840876e-03,  -3.96943539e-02,  -3.48771542e-01,
                          7.13484362e-02,  -1.11116938e-01,  -2.90257007e-01,
                          8.83246679e-03,  -7.60097876e-02,  -3.44339937e-01,
                          5.47175258e-02,  -1.34802356e-01,  -3.00516874e-01,
                          5.72212264e-02,  -1.39211729e-01,  -3.11883301e-01,
                          4.87691015e-02,  -1.41545430e-01,  -3.12466711e-01,
                          3.96761447e-02,  -1.46307334e-01,  -3.20896447e-01,
                         -3.44134420e-02,   6.24414645e-02,  -3.67194951e-01,
                          7.64086396e-02,  -3.23757939e-02,  -2.81319082e-01,
                         -1.94152966e-02,  -1.10044880e-02,  -3.68264079e-01,
                          5.69640892e-03,  -4.21866998e-02,  -3.45534354e-01,
                          3.52627635e-02,  -1.48633122e-01,  -3.19463283e-01,
                          1.62824579e-02,  -1.31041482e-01,  -3.53134423e-01,
                          1.70042459e-02,   3.90943959e-02,  -3.39067191e-01,
                          6.11800589e-02,   1.39129441e-03,  -2.95480609e-01,
                         -3.66298296e-02,   6.36134669e-02,  -3.68144274e-01,
                          7.85883367e-02,  -2.77958475e-02,  -2.86368340e-01,
                         -4.99024615e-02,   6.27958253e-02,  -3.82697284e-01,
                         -5.48545644e-02,   5.76853380e-02,  -3.84648532e-01,
                         -5.13507985e-02,   4.32729423e-02,  -3.81973058e-01,
                         -4.08262163e-02,   2.48874836e-02,  -3.79340261e-01,
                          7.66843930e-02,  -6.88077211e-02,  -2.78879285e-01,
                         -3.36056463e-02,   7.61831878e-03,  -3.74226451e-01,
                          7.55255371e-02,  -8.95793140e-02,  -2.84417808e-01,
                          5.44928527e-03,  -4.37719598e-02,  -3.39360684e-01,
                          1.13152042e-02,  -4.49101664e-02,  -3.42046887e-01,
                          1.57206003e-02,  -5.76153174e-02,  -3.41750085e-01,
                          1.20760761e-02,  -8.15278366e-02,  -3.47417951e-01,
                          4.54958342e-03,  -9.56231728e-02,  -3.45500082e-01,
                          6.35806769e-02,  -1.36728704e-01,  -3.10148627e-01,
                          6.90358458e-03,  -1.33709103e-01,  -3.48185867e-01,
                          1.81777775e-02,  -1.50472224e-01,  -3.38736027e-01,
                          3.71993110e-02,   2.13090330e-02,  -3.16304624e-01,
                         -2.38361824e-02,   6.02737851e-02,  -3.65185767e-01,
                         -4.29866873e-02,   6.49165213e-02,  -3.79811138e-01,
                         -3.72462161e-02,   1.63866021e-02,  -3.77772242e-01,
                          6.51438255e-04,  -3.91540676e-02,  -3.51040155e-01,
                          1.12000471e-02,  -5.00582382e-02,  -3.39190602e-01,
                          7.92401284e-03,  -5.30336685e-02,  -3.35582107e-01,
                          9.82555933e-03,  -6.17694259e-02,  -3.37878257e-01,
                          1.28720487e-02,  -6.23412840e-02,  -3.41816783e-01,
                          6.30581528e-02,  -1.17786884e-01,  -2.88267344e-01,
                          1.18376995e-02,  -1.33088768e-01,  -3.52525830e-01,
                          3.92814875e-02,   2.27034520e-02,  -3.19485664e-01,
                          2.58240215e-02,   2.34392993e-02,  -3.16829205e-01,
                          5.66144809e-02,   4.82603302e-03,  -2.97877520e-01,
                          4.53233048e-02,   6.24135323e-03,  -2.99653500e-01,
                          4.94473800e-02,   8.41424055e-03,  -3.02017570e-01,
                         -1.06459763e-02,   5.64719886e-02,  -3.61111909e-01,
                          7.44570345e-02,  -2.24138591e-02,  -2.84800917e-01,
                          7.52380341e-02,  -3.79584953e-02,  -2.80586541e-01,
                         -4.92127575e-02,   6.30441383e-02,  -3.80222797e-01,
                         -1.80059113e-02,  -1.55584160e-02,  -3.59037936e-01,
                          7.06494004e-02,  -1.01432592e-01,  -2.83209831e-01,
                          7.84662180e-03,  -1.42525986e-01,  -3.40973705e-01,
                         -2.59748865e-02,   5.48152998e-02,  -3.47320706e-01,
                         -4.05699164e-02,   5.97501546e-02,  -3.71175349e-01,
                         -5.64956255e-02,   5.52150495e-02,  -3.84678662e-01,
                          7.66298026e-02,  -5.91909476e-02,  -2.79156595e-01,
                          7.20992312e-02,  -7.63239637e-02,  -2.78277904e-01,
                         -4.27205488e-02,   1.32141560e-02,  -3.74861985e-01,
                         -3.70559506e-02,  -2.82492815e-03,  -3.70079756e-01,
                         -3.10935639e-02,  -9.56188794e-03,  -3.66413653e-01,
                         -2.19827574e-02,  -1.62416026e-02,  -3.59739274e-01,
                         -3.66234966e-02,  -5.55454241e-03,  -3.66453558e-01,
                          1.64093648e-03,  -4.20113467e-02,  -3.41598153e-01,
                          5.67468591e-02,  -1.27176002e-01,  -2.95439392e-01,
                          5.68409748e-02,  -1.28573284e-01,  -2.98081219e-01,
                          1.43420761e-02,  -1.37697309e-01,  -3.47422212e-01,
                          8.85571726e-03,   3.53256799e-02,  -3.26966345e-01,
                          5.17212339e-02,   2.61876005e-04,  -2.90400594e-01,
                         -8.89584422e-03,   4.47607115e-02,  -3.36955756e-01,
                         -2.69643869e-02,   5.89607470e-02,  -3.60292375e-01,
                          7.71678016e-02,  -4.11435999e-02,  -2.78058439e-01,
                          7.44799823e-02,  -4.95261848e-02,  -2.80316293e-01,
                         -5.61203547e-02,   4.92884628e-02,  -3.85004848e-01,
                         -5.93289360e-02,   3.71752270e-02,  -3.81299704e-01,
                         -2.17961017e-02,  -1.67508367e-02,  -3.54507178e-01,
                         -1.76213160e-02,  -2.46012416e-02,  -3.53578925e-01,
                          7.51071470e-03,  -4.75023389e-02,  -3.38708192e-01,
                          1.34958979e-03,  -1.21941246e-01,  -3.49575311e-01,
                          1.11637777e-03,  -1.34877607e-01,  -3.41688067e-01,
                          1.39624383e-02,  -1.45837471e-01,  -3.40427309e-01,
                          2.17823088e-02,   3.18401679e-02,  -3.21904063e-01,
                          5.52367531e-02,  -1.04543362e-02,  -2.80354410e-01,
                         -3.53003144e-02,   5.84308803e-02,  -3.55700552e-01,
                          6.89040795e-02,  -2.42612157e-02,  -2.74707764e-01,
                         -5.74841686e-02,   5.74622713e-02,  -3.78914505e-01,
                          6.25320598e-02,  -9.49220732e-02,  -2.78626442e-01,
                         -1.36217866e-02,  -3.25109996e-02,  -3.49846661e-01,
                          6.44780695e-02,  -1.11530960e-01,  -2.88103759e-01,
                          4.76848371e-02,  -1.43968239e-01,  -3.09642255e-01,
                          4.14829403e-02,  -1.45128682e-01,  -3.11974883e-01,
                          6.75277039e-03,   3.73218395e-02,  -3.26619208e-01,
                          6.82169273e-02,  -1.58574861e-02,  -2.84021616e-01,
                         -1.72858909e-02,   5.66416942e-02,  -3.55300099e-01,
                         -5.63621297e-02,   5.33350110e-02,  -3.85209531e-01,
                         -6.28068745e-02,   5.46170287e-02,  -3.83904159e-01,
                          7.03422427e-02,  -9.21483263e-02,  -2.80771822e-01,
                         -3.11494805e-02,  -1.38882035e-02,  -3.59207511e-01,
                         -5.08680381e-03,  -4.02144343e-02,  -3.41534704e-01,
                          1.01846857e-02,  -5.82864545e-02,  -3.35371405e-01,
                          6.86907992e-02,  -1.03641376e-01,  -2.81913072e-01,
                          5.04852831e-03,  -1.07839614e-01,  -3.49520355e-01,
                          3.14039295e-03,  -1.27143249e-01,  -3.51551801e-01,
                         -4.70497180e-04,  -1.28161371e-01,  -3.46133322e-01,
                          1.83487218e-02,   2.93782614e-02,  -3.19541931e-01,
                          2.58844830e-02,   2.21300796e-02,  -3.13091159e-01,
                         -1.44034382e-02,   5.15554026e-02,  -3.42972755e-01,
                          4.37743291e-02,   9.59631987e-03,  -3.06270272e-01,
                          6.52911738e-02,  -3.17591764e-02,  -2.71521211e-01,
                         -5.16310520e-02,   6.08200245e-02,  -3.73163402e-01,
                         -5.58045655e-02,   6.09972104e-02,  -3.79669487e-01,
                          6.75328299e-02,  -6.73654452e-02,  -2.72114605e-01,
                         -4.63588536e-02,   2.89686583e-02,  -3.81113857e-01,
                         -2.06277408e-02,  -2.01688167e-02,  -3.53895664e-01,
                         -3.54763195e-02,  -9.10203997e-03,  -3.63071859e-01,
                          7.48583407e-04,  -4.57492918e-02,  -3.34917456e-01,
                          6.06190115e-02,  -1.24460846e-01,  -2.95960903e-01,
                          7.33865891e-04,  -9.51843709e-02,  -3.44866812e-01,
                          3.16499770e-02,  -1.49043009e-01,  -3.15126270e-01,
                         -5.76912845e-03,  -1.19446144e-01,  -3.43534201e-01,
                          1.40452478e-02,  -1.49791375e-01,  -3.26367617e-01,
                          1.02703366e-02,  -1.49537787e-01,  -3.34245324e-01,
                          5.95078692e-02,  -1.08611025e-02,  -2.81122118e-01,
                          6.86187595e-02,  -3.82501520e-02,  -2.70543903e-01,
                         -5.15955985e-02,   2.35347860e-02,  -3.76082897e-01,
                         -5.19403405e-02,   1.52562456e-02,  -3.71730030e-01,
                         -4.15515751e-02,   6.10403670e-03,  -3.70341420e-01,
                         -2.42254976e-02,  -1.40112136e-02,  -3.56205463e-01,
                          5.72857037e-02,  -1.12226941e-01,  -2.79927194e-01,
                          4.77253925e-03,  -7.62109458e-02,  -3.45525742e-01,
                          4.89216298e-02,  -1.31149784e-01,  -2.93036997e-01,
                         -4.39813687e-03,   4.26252075e-02,  -3.26784611e-01,
                         -5.85088991e-02,   5.82595319e-02,  -3.84986699e-01,
                          7.09647536e-02,  -8.19673315e-02,  -2.77629286e-01,
                          1.36092608e-03,  -4.93532382e-02,  -3.29588622e-01,
                          6.35967553e-02,  -1.09790526e-01,  -2.83538461e-01,
                         -2.67009553e-03,  -8.66945684e-02,  -3.43043089e-01,
                          4.73508574e-02,  -1.40455127e-01,  -2.98517108e-01,
                          2.39468385e-02,  -1.51180714e-01,  -3.23482066e-01,
                          5.59963435e-02,  -2.15134723e-03,  -2.90114880e-01,
                         -1.62682850e-02,   4.94740866e-02,  -3.33229363e-01,
                          3.80953476e-02,   1.44489314e-02,  -3.08075219e-01,
                          5.78399450e-02,  -1.40240844e-02,  -2.76390821e-01,
                         -3.20377275e-02,   5.76819964e-02,  -3.55492175e-01,
                         -4.06760201e-02,   6.18240610e-02,  -3.67483705e-01,
                          7.23396167e-02,  -5.33390939e-02,  -2.72543013e-01,
                         -6.82900846e-02,   5.61808497e-02,  -3.83028060e-01,
                         -4.38635759e-02,  -1.00285839e-03,  -3.68717432e-01,
                          1.33860647e-03,  -4.48876061e-02,  -3.33039433e-01,
                         -1.77744892e-03,  -7.36972392e-02,  -3.37977856e-01,
                          3.78006184e-03,  -7.18337148e-02,  -3.40501517e-01,
                          2.54878309e-04,  -8.22874978e-02,  -3.45965892e-01,
                         -1.62304379e-03,  -1.11087985e-01,  -3.48259181e-01,
                         -5.74862864e-03,  -1.25157967e-01,  -3.41912001e-01,
                          1.01702120e-02,   3.31837907e-02,  -3.20519656e-01,
                          3.96673121e-02,   9.18309204e-03,  -3.02183002e-01,
                         -4.15417589e-02,   5.51939607e-02,  -3.53911042e-01,
                         -4.75205891e-02,   5.65039068e-02,  -3.60932529e-01,
                         -5.69766723e-02,   5.96161857e-02,  -3.76151383e-01,
                         -6.30330965e-02,   5.83919287e-02,  -3.76559228e-01,
                         -6.74734265e-02,   5.84381819e-02,  -3.81414384e-01,
                          6.95087090e-02,  -6.00677729e-02,  -2.72025943e-01,
                         -6.36590570e-02,   4.98671345e-02,  -3.83232683e-01,
                         -1.34299574e-02,  -3.60541008e-02,  -3.44856918e-01,
                         -1.82525497e-02,  -3.56747173e-02,  -3.45800996e-01,
                          2.27823202e-03,  -5.99373914e-02,  -3.31538469e-01,
                         -4.49295063e-03,  -1.02825940e-01,  -3.43962997e-01,
                         -5.24564972e-03,  -1.10573396e-01,  -3.43845308e-01,
                         -2.76472513e-03,  -1.19634196e-01,  -3.45353037e-01,
                          2.28519877e-03,   4.02902141e-02,  -3.26047093e-01,
                          5.33654280e-02,  -5.89502556e-03,  -2.77578890e-01,
                          3.41703780e-02,   1.04495334e-02,  -3.04316670e-01,
                         -5.14843501e-02,   5.51051423e-02,  -3.66958857e-01,
                         -6.82960823e-02,   3.74570489e-02,  -3.81218046e-01,
                         -3.00567411e-02,  -1.40237259e-02,  -3.56022418e-01,
                         -1.07380711e-02,  -3.69956493e-02,  -3.48804832e-01,
                          5.67639386e-03,  -6.85060099e-02,  -3.38290870e-01,
                         -1.16601563e-03,  -1.02304444e-01,  -3.45836490e-01,
                          3.04018683e-03,  -1.39946610e-01,  -3.39033216e-01,
                          5.58080804e-03,   3.29073183e-02,  -3.19383979e-01,
                          4.30023596e-02,   1.62591657e-03,  -2.85708368e-01,
                          4.51322123e-02,   2.94811931e-03,  -2.91197866e-01,
                         -1.74289923e-02,   5.07439375e-02,  -3.41969073e-01,
                          2.79626027e-02,   1.04435347e-02,  -2.98570842e-01,
                         -4.97519076e-02,   5.84349446e-02,  -3.69578511e-01,
                         -5.95788546e-02,   1.71363093e-02,  -3.74651790e-01,
                          6.59088716e-02,  -8.55912194e-02,  -2.75233209e-01,
                         -6.02778494e-02,   1.13743357e-02,  -3.71567309e-01,
                         -2.40847338e-02,  -1.70440935e-02,  -3.55301112e-01,
                         -1.91782247e-02,  -3.45333330e-02,  -3.41442227e-01,
                          4.23900932e-02,  -1.33612379e-01,  -2.89709300e-01,
                          1.78978797e-02,  -1.53183475e-01,  -3.24405789e-01,
                          7.62419030e-03,  -1.46715254e-01,  -3.25112581e-01,
                          1.59428082e-02,   2.44775582e-02,  -3.13178152e-01,
                          6.62792847e-02,  -1.76136009e-02,  -2.78610885e-01,
                          6.17942289e-02,  -2.33240109e-02,  -2.72811353e-01,
                          6.52663931e-02,  -4.88119759e-02,  -2.68609822e-01,
                         -7.47442469e-02,   5.57294860e-02,  -3.82254928e-01,
                         -2.74716802e-02,  -1.80938505e-02,  -3.47373933e-01,
                         -3.66694480e-02,  -1.23144602e-02,  -3.59188884e-01,
                         -4.69815470e-02,  -6.38112286e-03,  -3.63564789e-01,
                          6.19366392e-02,  -9.68563482e-02,  -2.76039749e-01,
                         -9.64928605e-03,  -3.86058837e-02,  -3.39452088e-01,
                          1.45222712e-03,  -5.35143837e-02,  -3.28427911e-01,
                          6.05023466e-02,  -1.05205119e-01,  -2.78279126e-01,
                          1.21737854e-03,  -6.36702031e-02,  -3.32543433e-01,
                          2.37024054e-02,  -1.52151138e-01,  -3.13952297e-01,
                          9.16997902e-03,  -1.50599614e-01,  -3.23882520e-01,
                          5.03873229e-02,  -1.03892712e-03,  -2.84669816e-01,
                          3.68959345e-02,   3.39754252e-03,  -2.87735075e-01,
                          4.14936170e-02,   6.23952597e-03,  -2.97839046e-01,
                          3.27209868e-02,   8.56935233e-03,  -2.97233194e-01,
                         -5.53790815e-02,   5.78410700e-02,  -3.70927870e-01,
                         -7.33963773e-02,   5.74782379e-02,  -3.81068885e-01,
                          6.40439540e-02,  -6.19905181e-02,  -2.67499924e-01,
                          6.78629801e-02,  -7.59882405e-02,  -2.72513598e-01,
                         -2.02657133e-02,  -2.94357911e-02,  -3.46590161e-01,
                         -4.23965510e-03,  -7.59581551e-02,  -3.38390112e-01,
                         -9.58673283e-03,  -9.04837698e-02,  -3.39568883e-01,
                          3.07288542e-02,  -1.48070842e-01,  -3.09674978e-01,
                          1.43896900e-02,   1.65126063e-02,  -3.01844656e-01,
                         -2.66964659e-02,   5.30710369e-02,  -3.41557443e-01,
                          5.95097244e-02,  -3.75671498e-02,  -2.62813568e-01,
                         -6.43415153e-02,   5.97845614e-02,  -3.74732614e-01,
                         -7.40113407e-02,   5.09990528e-02,  -3.81613225e-01,
                         -7.37977102e-02,   4.70377766e-02,  -3.79496425e-01,
                         -2.19582468e-02,  -2.27738731e-02,  -3.48469883e-01,
                         -5.16662262e-02,  -3.67025821e-03,  -3.63500029e-01,
                         -6.29842374e-03,  -3.94199118e-02,  -3.34913611e-01,
                          2.18145875e-03,  -5.16141430e-02,  -3.28226686e-01,
                          5.55331558e-02,  -1.07544705e-01,  -2.74970829e-01,
                          5.37014380e-02,  -1.16177186e-01,  -2.79646724e-01,
                         -6.42341562e-03,  -6.96027353e-02,  -3.33948463e-01,
                          2.50746757e-02,  -1.50670081e-01,  -3.06252211e-01,
                         -2.54960847e-03,  -1.34408832e-01,  -3.35828543e-01,
                         -7.29163364e-03,   3.84011678e-02,  -3.21175456e-01,
                          5.25621399e-02,  -1.03632454e-02,  -2.70774484e-01,
                         -5.70876226e-02,   5.15796654e-02,  -3.57248545e-01,
                          6.30302727e-02,  -4.40205671e-02,  -2.65550762e-01,
                         -5.99462725e-02,   6.07489645e-02,  -3.78617674e-01,
                         -2.78801434e-02,  -1.36786783e-02,  -3.50010276e-01,
                         -5.11910766e-02,   6.79719914e-03,  -3.68894815e-01,
                         -3.14657241e-02,  -1.56915840e-02,  -3.44977021e-01,
                         -3.02083343e-02,  -1.89545974e-02,  -3.43079001e-01,
                         -1.36608426e-02,  -3.93197387e-02,  -3.26750278e-01,
                         -7.39031518e-03,  -4.55052219e-02,  -3.27720046e-01,
                          4.23244163e-02,  -1.41366720e-01,  -3.02262455e-01,
                          3.89387235e-02,  -1.44019425e-01,  -3.05664480e-01,
                          3.83984856e-03,  -1.43416047e-01,  -3.32963198e-01,
                          1.14340782e-02,   2.71690674e-02,  -3.08499098e-01,
                         -3.07364576e-02,   5.05154505e-02,  -3.39142203e-01,
                          5.63389845e-02,  -2.05096155e-02,  -2.65747577e-01,
                         -6.56778961e-02,   2.77954489e-02,  -3.78807604e-01,
                          5.24145663e-02,  -1.12639785e-01,  -2.74388194e-01,
                         -1.02405418e-02,  -7.28724599e-02,  -3.37037534e-01,
                         -1.26464935e-02,  -1.07450090e-01,  -3.38682443e-01,
                          1.99432243e-02,  -1.47909597e-01,  -3.06416839e-01,
                          1.45861730e-02,  -1.51153043e-01,  -3.19404602e-01,
                         -7.69351702e-03,   4.18553501e-02,  -3.28597516e-01,
                          3.94575149e-02,  -1.09030865e-04,  -2.80009687e-01,
                         -5.35736345e-02,   5.16937003e-02,  -3.61088961e-01,
                         -7.64588341e-02,   5.75972833e-02,  -3.76696378e-01,
                         -7.96354711e-02,   4.94229719e-02,  -3.81912649e-01,
                         -5.56703769e-02,  -1.39230001e-03,  -3.63484740e-01,
                         -5.49013466e-02,  -6.32910291e-03,  -3.59408498e-01,
                          5.05823940e-02,  -1.20658927e-01,  -2.82268494e-01,
                         -1.40155600e-02,  -9.21028331e-02,  -3.36924613e-01,
                         -1.13817798e-02,  -1.16788343e-01,  -3.40383440e-01,
                          2.86502428e-02,  -1.46103159e-01,  -3.05048823e-01,
                         -6.67943805e-03,  -1.26686603e-01,  -3.36934745e-01,
                          5.22069447e-03,   2.61834860e-02,  -3.09662759e-01,
                          2.21701898e-02,   1.26803499e-02,  -2.98640728e-01,
                         -4.49347980e-02,   5.10219485e-02,  -3.50642681e-01,
                         -6.31526560e-02,   5.71252480e-02,  -3.70311141e-01,
                          6.26453385e-02,  -4.89831418e-02,  -2.63813317e-01,
                         -6.88688383e-02,   5.79964481e-02,  -3.70804876e-01,
                         -7.98385516e-02,   4.83554192e-02,  -3.80280226e-01,
                          6.17700443e-02,  -7.67066628e-02,  -2.65439391e-01,
                         -6.66961074e-02,   1.55262137e-02,  -3.72031957e-01,
                         -4.91065867e-02,  -9.60243400e-03,  -3.56081903e-01,
                          4.68015596e-02,  -3.17902141e-03,  -2.79419214e-01,
                         -2.14233398e-02,   4.66864519e-02,  -3.33383858e-01,
                         -2.99403947e-02,   4.85463664e-02,  -3.38012159e-01,
                         -4.29401807e-02,   5.51219881e-02,  -3.50757033e-01,
                          5.71610145e-02,  -3.20152827e-02,  -2.60808259e-01,
                         -5.16637452e-02,   5.13115637e-02,  -3.55132133e-01,
                          5.91882616e-02,  -6.00433759e-02,  -2.59255469e-01,
                          6.08221814e-02,  -6.60754070e-02,  -2.62953609e-01,
                         -6.37872741e-02,   2.21641008e-02,  -3.78181249e-01,
                         -2.67097801e-02,  -2.67068334e-02,  -3.40807080e-01,
                         -1.56768840e-02,  -3.76883298e-02,  -3.28636229e-01,
                          6.09780326e-02,  -9.93931442e-02,  -2.76929557e-01,
                         -3.85574996e-03,  -6.43249825e-02,  -3.28762323e-01,
                         -7.58351153e-03,  -8.04497972e-02,  -3.37617695e-01,
                         -1.09587349e-02,  -8.12723413e-02,  -3.35352719e-01,
                          3.85466665e-02,  -1.38016671e-01,  -2.92028487e-01,
                         -1.40038580e-02,  -1.02872491e-01,  -3.37706327e-01,
                          1.69137977e-02,  -1.50196716e-01,  -3.12063664e-01,
                         -1.38919251e-02,   4.37879562e-02,  -3.29190195e-01,
                          4.57463972e-02,  -5.48993330e-03,  -2.73244232e-01,
                          5.56393825e-02,  -2.41688862e-02,  -2.67925233e-01,
                         -4.07905318e-02,   5.10962270e-02,  -3.42642754e-01,
                         -7.83915892e-02,   3.55416164e-02,  -3.76648068e-01,
                          5.89457303e-02,  -8.33394900e-02,  -2.66078472e-01,
                         -7.41424039e-02,   1.57416258e-02,  -3.72112334e-01,
                         -7.93273970e-02,   2.16606744e-02,  -3.72263849e-01,
                         -3.96035239e-02,  -1.34677812e-02,  -3.51785630e-01,
                         -5.61152734e-02,   3.76246590e-03,  -3.65382642e-01,
                         -5.32524735e-02,  -4.26122546e-03,  -3.61172795e-01,
                         -5.14485547e-03,  -4.25599068e-02,  -3.29183996e-01,
                         -2.05136910e-02,  -3.63447703e-02,  -3.38046163e-01,
                         -2.58065853e-02,  -3.39493863e-02,  -3.37766647e-01,
                         -2.92783719e-03,  -5.70852682e-02,  -3.28425586e-01,
                          5.04486002e-02,  -1.01823524e-01,  -2.68896818e-01,
                         -1.00878682e-02,  -6.98178709e-02,  -3.31768900e-01,
                          3.20394151e-02,  -1.38394326e-01,  -2.96242356e-01,
                          2.41331253e-02,  -1.47039011e-01,  -2.98108786e-01,
                          6.63124071e-03,  -1.49996921e-01,  -3.15356433e-01,
                          2.97535118e-03,   3.13002951e-02,  -3.13215107e-01,
                          2.41455324e-02,   8.59644637e-03,  -2.95407027e-01,
                         -7.71511346e-02,   6.00651987e-02,  -3.74212354e-01,
                         -8.31823945e-02,   4.75558154e-02,  -3.80755991e-01,
                          5.56782037e-02,  -6.85371235e-02,  -2.62351573e-01,
                          5.49953654e-02,  -9.13391560e-02,  -2.67546296e-01,
                         -1.43427150e-02,  -3.75639535e-02,  -3.32876116e-01,
                         -5.73412850e-02,  -7.89524242e-03,  -3.55754554e-01,
                         -8.26719962e-03,  -5.88352121e-02,  -3.22991788e-01,
                         -1.65599696e-02,  -9.01720300e-02,  -3.31924856e-01,
                         -1.35385469e-02,  -1.19189717e-01,  -3.34604710e-01,
                         -2.94459285e-04,  -1.40745729e-01,  -3.23066622e-01,
                          5.91076277e-02,  -4.77718525e-02,  -2.61139244e-01,
                         -7.78801069e-02,   5.88572957e-02,  -3.71743470e-01,
                         -7.26750121e-02,   2.44818386e-02,  -3.75581563e-01,
                         -6.39193133e-02,   4.58726427e-03,  -3.62531930e-01,
                         -6.40766248e-02,   6.51445892e-03,  -3.67393553e-01,
                         -7.92701822e-03,  -6.14624284e-02,  -3.25572699e-01,
                          4.34900671e-02,  -1.31330729e-01,  -2.84619808e-01,
                         -8.93003959e-03,  -1.29143655e-01,  -3.34043235e-01,
                         -1.22143738e-02,   3.82356420e-02,  -3.15152138e-01,
                          1.30507890e-02,   1.52744828e-02,  -2.99985915e-01,
                          5.74443955e-03,   1.94045231e-02,  -3.04166079e-01,
                         -3.32637168e-02,   5.05079478e-02,  -3.41163456e-01,
                         -5.87049015e-02,   4.87016477e-02,  -3.53947341e-01,
                         -8.60775486e-02,   5.53625412e-02,  -3.76243114e-01,
                         -8.40204060e-02,   5.36421239e-02,  -3.76835287e-01,
                         -8.73463899e-02,   5.04846983e-02,  -3.79617989e-01,
                         -9.17981863e-02,   4.21600714e-02,  -3.78705263e-01,
                         -3.79235893e-02,  -1.32740177e-02,  -3.44476283e-01,
                         -3.56516205e-02,  -1.63011253e-02,  -3.38611573e-01,
                          5.15583158e-02,  -8.91468897e-02,  -2.62984723e-01,
                         -4.94577512e-02,  -1.17720896e-02,  -3.48135352e-01,
                         -8.38127919e-03,  -4.85711433e-02,  -3.23780954e-01,
                          4.74464931e-02,  -1.17384829e-01,  -2.75733650e-01,
                         -1.29961222e-02,  -6.98167682e-02,  -3.29296410e-01,
                          3.58919352e-02,  -1.27373531e-01,  -2.76844591e-01,
                          2.81850193e-02,  -1.41732544e-01,  -2.91881442e-01,
                         -1.47493016e-02,  -1.13896489e-01,  -3.34135205e-01,
                          1.40427426e-02,  -1.47638321e-01,  -3.04600626e-01,
                          5.72770834e-03,  -1.49086952e-01,  -3.19976330e-01,
                          3.89257185e-02,  -3.08000157e-03,  -2.74644345e-01,
                         -1.49690034e-03,   2.30944138e-02,  -3.05014253e-01,
                         -2.45122537e-02,   4.67025824e-02,  -3.29322815e-01,
                          3.95169742e-02,  -1.25343585e-02,  -2.64394939e-01,
                          1.61367878e-02,   9.30474326e-03,  -2.90685773e-01,
                          5.72753847e-02,  -3.66559923e-02,  -2.59607702e-01,
                         -6.66016340e-02,   5.11447527e-02,  -3.59128356e-01,
                         -8.13462287e-02,   2.71400604e-02,  -3.74900073e-01,
                         -2.64421050e-02,  -3.10789440e-02,  -3.39465678e-01,
                         -1.40895303e-02,  -4.36675623e-02,  -3.23256850e-01,
                          4.81232516e-02,  -1.09373987e-01,  -2.71235704e-01,
                          4.23035659e-02,  -1.17655784e-01,  -2.74479210e-01,
                          4.63401563e-02,  -1.25496507e-01,  -2.81062275e-01,
                         -1.68652013e-02,  -1.11714803e-01,  -3.34164828e-01,
                          1.75188221e-02,  -1.45225644e-01,  -2.98553109e-01,
                          1.28678279e-02,  -1.47343099e-01,  -3.10315907e-01,
                         -4.61965799e-03,  -1.34482980e-01,  -3.22988957e-01,
                          5.14159352e-03,  -1.43798128e-01,  -3.12512964e-01,
                         -1.65310819e-02,   4.41989079e-02,  -3.26025963e-01,
                          2.10281834e-02,   6.47968659e-03,  -2.87244439e-01,
                         -3.32926847e-02,   4.58783247e-02,  -3.31680834e-01,
                         -4.56537157e-02,   4.50196117e-02,  -3.40920061e-01,
                         -5.16282320e-02,   4.65645455e-02,  -3.50435883e-01,
                         -7.01492205e-02,   5.55137098e-02,  -3.65597755e-01,
                         -7.98735470e-02,   5.71879856e-02,  -3.73389304e-01,
                         -3.43620181e-02,  -2.40938161e-02,  -3.31466347e-01,
                         -1.14680659e-02,  -4.56264541e-02,  -3.23261887e-01,
                         -8.55140947e-03,  -5.37979938e-02,  -3.24404299e-01,
                         -1.42513374e-02,  -7.68215135e-02,  -3.30506653e-01,
                          3.25927548e-02,  -1.32744625e-01,  -2.84383744e-01,
                         -1.51716694e-02,  -1.06586367e-01,  -3.32461774e-01,
                          3.59548489e-03,  -1.43807977e-01,  -3.23403984e-01,
                          2.50617918e-02,   8.98789242e-03,  -2.91624010e-01,
                         -7.91766588e-03,   2.91633680e-02,  -3.10731113e-01,
                          2.54576467e-02,   2.34556478e-03,  -2.82010853e-01,
                          7.21568288e-03,   1.97753459e-02,  -3.02445918e-01,
                          4.98325527e-02,  -2.61006039e-02,  -2.61160493e-01,
                         -4.87592071e-02,   4.34520543e-02,  -3.41835588e-01,
                          5.42369522e-02,  -3.73191461e-02,  -2.57355124e-01,
                          5.71319424e-02,  -5.98435588e-02,  -2.58821398e-01,
                         -9.40500498e-02,   5.06740846e-02,  -3.78977329e-01,
                          5.81557192e-02,  -7.69823045e-02,  -2.62765110e-01,
                         -8.64625424e-02,   3.88841517e-02,  -3.78101408e-01,
                         -3.67530212e-02,  -1.21027604e-02,  -3.45511973e-01,
                         -3.41323167e-02,  -2.01207623e-02,  -3.34091157e-01,
                         -4.86392602e-02,  -1.16675375e-02,  -3.46848965e-01,
                         -7.28992745e-02,   1.05767725e-02,  -3.68124157e-01,
                         -6.28164411e-02,  -5.93603821e-03,  -3.58677566e-01,
                          4.87539954e-02,  -9.96285677e-02,  -2.67135739e-01,
                         -1.21415686e-02,  -6.34301901e-02,  -3.25759977e-01,
                         -1.81948617e-02,  -8.67958143e-02,  -3.30063134e-01,
                          2.84220390e-02,  -1.32469058e-01,  -2.85579771e-01,
                         -1.65731534e-02,  -9.85454619e-02,  -3.30243528e-01,
                         -1.04011949e-02,  -1.24218300e-01,  -3.30431014e-01,
                          3.67467999e-02,   3.01116798e-03,  -2.85073668e-01,
                         -1.23333447e-02,   3.75130549e-02,  -3.17993015e-01,
                          4.60909344e-02,  -1.65169667e-02,  -2.63742536e-01,
                         -2.80825123e-02,   4.19608876e-02,  -3.22352380e-01,
                         -2.77457051e-02,   4.63318452e-02,  -3.26518714e-01,
                         -3.66734304e-02,   4.47135083e-02,  -3.31581324e-01,
                          5.49648963e-02,  -3.09462938e-02,  -2.58845508e-01,
                         -5.41909337e-02,   4.00391817e-02,  -3.44165981e-01,
                         -2.64508482e-02,  -3.09088044e-02,  -3.29006016e-01,
                         -3.06024812e-02,  -2.70156041e-02,  -3.30499649e-01,
                         -1.97075251e-02,  -3.88033241e-02,  -3.21571380e-01,
                         -5.62473126e-02,  -1.23331789e-02,  -3.47584784e-01,
                         -5.94259389e-02,  -5.93085354e-03,  -3.56721908e-01,
                         -1.04399938e-02,  -5.20662777e-02,  -3.18075567e-01,
                         -1.49104092e-02,  -1.20632775e-01,  -3.30496490e-01,
                          3.12010199e-02,   1.73670799e-03,  -2.78380811e-01,
                         -9.38725993e-02,   5.20963818e-02,  -3.74656945e-01,
                         -9.31726322e-02,   4.77662385e-02,  -3.78388494e-01,
                         -9.62266922e-02,   3.62265073e-02,  -3.79259109e-01,
                         -8.58747736e-02,   1.95116606e-02,  -3.69268149e-01,
                         -2.20083632e-02,  -3.67000923e-02,  -3.21371526e-01,
                         -2.95910444e-02,  -3.04392893e-02,  -3.26430142e-01,
                         -8.19423050e-02,   1.32794017e-02,  -3.67599487e-01,
                         -7.30845332e-02,   5.70234703e-03,  -3.63797069e-01,
                         -1.92919429e-02,  -1.03993401e-01,  -3.27778965e-01,
                         -1.66019481e-02,  -1.17282942e-01,  -3.26605082e-01,
                         -3.47254827e-04,  -1.37994289e-01,  -3.16156030e-01,
                         -1.33081982e-02,   3.30703370e-02,  -3.12248260e-01,
                          1.58289969e-02,   4.89066727e-03,  -2.84097254e-01,
                         -2.13222466e-02,   3.91203947e-02,  -3.19673210e-01,
                         -4.48967367e-02,   4.02558073e-02,  -3.36844355e-01,
                         -8.69045928e-02,   5.41867614e-02,  -3.75826448e-01,
                         -8.88192430e-02,   2.37233080e-02,  -3.71205211e-01,
                         -1.62045304e-02,  -6.87818751e-02,  -3.25653553e-01,
                         -1.87857952e-02,  -7.45372996e-02,  -3.22568625e-01,
                         -1.83589086e-02,  -1.14597216e-01,  -3.27646255e-01,
                         -5.84419770e-03,  -1.32535547e-01,  -3.16387802e-01,
                          8.89265910e-04,   1.71388388e-02,  -3.01943421e-01,
                          5.78892082e-02,  -4.67298888e-02,  -2.60585189e-01,
                         -6.00427650e-02,   4.09807265e-02,  -3.48240644e-01,
                         -6.51823804e-02,   4.71708141e-02,  -3.53763163e-01,
                         -8.29652622e-02,   5.47976308e-02,  -3.69965881e-01,
                          4.96870093e-02,  -6.71624243e-02,  -2.57378161e-01,
                          4.96729538e-02,  -7.10942224e-02,  -2.56827772e-01,
                          5.30129448e-02,  -7.54392520e-02,  -2.60912627e-01,
                         -3.96026038e-02,  -1.51448846e-02,  -3.37158054e-01,
                         -5.27549535e-02,  -1.21548828e-02,  -3.41955304e-01,
                         -1.64887626e-02,  -4.56177965e-02,  -3.16397399e-01,
                          4.29738015e-02,  -1.11252055e-01,  -2.67626166e-01,
                          6.57571107e-03,  -1.41732812e-01,  -3.07463467e-01,
                         -1.88577361e-02,   3.44315805e-02,  -3.16232324e-01,
                          1.12498319e-02,   1.26764253e-02,  -2.95116931e-01,
                         -8.04278329e-02,   5.01068532e-02,  -3.61082911e-01,
                         -9.90327373e-02,   5.05847856e-02,  -3.75538588e-01,
                         -1.01843879e-01,   4.21971157e-02,  -3.76858860e-01,
                         -4.97270189e-02,  -1.32562593e-02,  -3.39422941e-01,
                         -8.62409100e-02,   9.56609100e-03,  -3.67937565e-01,
                         -7.96713009e-02,   1.45246508e-03,  -3.62140656e-01,
                         -1.70419961e-02,  -5.76032028e-02,  -3.21106166e-01,
                         -1.59886349e-02,  -6.34950772e-02,  -3.20442677e-01,
                          3.09955385e-02,  -1.29810274e-01,  -2.78140336e-01,
                         -1.73515603e-02,  -9.23970491e-02,  -3.23897988e-01,
                          1.05788745e-02,  -1.39954194e-01,  -2.98764914e-01,
                         -4.75424947e-03,  -1.35048211e-01,  -3.13593656e-01,
                          3.29008931e-03,  -1.44318461e-01,  -3.10071230e-01,
                          2.28194036e-02,  -1.67289225e-03,  -2.73116022e-01,
                         -3.18666957e-02,   3.81472595e-02,  -3.25013012e-01,
                         -3.45646441e-02,   3.89422849e-02,  -3.26783270e-01,
                          4.86590266e-02,  -3.26427482e-02,  -2.55723447e-01,
                         -4.29198891e-02,  -1.31568639e-02,  -3.36975247e-01,
                         -9.37621370e-02,   3.21721211e-02,  -3.76354575e-01,
                         -9.55040455e-02,   2.56069172e-02,  -3.74205530e-01,
                          4.87658344e-02,  -9.55993980e-02,  -2.64004052e-01,
                         -6.05813675e-02,  -1.18314307e-02,  -3.51005256e-01,
                         -6.86991662e-02,  -4.33042133e-03,  -3.60003442e-01,
                         -1.19663067e-02,  -5.33952229e-02,  -3.17234606e-01,
                          3.47562619e-02,  -1.18479066e-01,  -2.70229429e-01,
                         -2.05672607e-02,  -9.94897857e-02,  -3.26988578e-01,
                          2.26879660e-02,  -1.39963493e-01,  -2.89804876e-01,
                          1.38801476e-02,  -1.40259564e-01,  -2.92418748e-01,
                         -9.23503470e-03,  -1.22302726e-01,  -3.18052500e-01,
                          3.41498293e-02,  -3.31277261e-03,  -2.67070979e-01,
                         -8.06747004e-03,   2.30596345e-02,  -3.07997078e-01,
                         -2.34839823e-02,   3.58160995e-02,  -3.16803008e-01,
                         -3.33751715e-03,   1.76718999e-02,  -3.01723808e-01,
                         -4.30505313e-02,   2.72270441e-02,  -3.24449331e-01,
                         -4.38705757e-02,   3.67537811e-02,  -3.32431406e-01,
                          4.06400301e-02,  -4.88254838e-02,  -2.51566529e-01,
                         -6.32265285e-02,   3.46991047e-02,  -3.46184373e-01,
                         -7.20498413e-02,   4.72406968e-02,  -3.55873168e-01,
                         -8.84435177e-02,   4.51709591e-02,  -3.62141490e-01,
                         -3.76169123e-02,  -2.14191154e-02,  -3.30965012e-01,
                         -2.51239836e-02,  -3.64613347e-02,  -3.17118853e-01,
                         -5.87024987e-02,  -1.27388369e-02,  -3.45027417e-01,
                         -1.50336400e-02,  -4.82428037e-02,  -3.16591322e-01,
                         -6.80156872e-02,  -8.17799941e-03,  -3.56543660e-01,
                         -2.07829271e-02,  -9.94218513e-02,  -3.23721617e-01,
                         -1.95105448e-02,  -1.01780519e-01,  -3.17542672e-01,
                         -1.08920573e-03,  -1.38116211e-01,  -3.01470071e-01,
                         -1.51910167e-03,  -1.37116730e-01,  -3.08273077e-01,
                          1.10960482e-02,   5.77211194e-03,  -2.88871258e-01,
                         -2.12599710e-02,   3.51426341e-02,  -3.11973125e-01,
                          4.16341648e-02,  -1.98975950e-02,  -2.57197350e-01,
                         -9.60364342e-02,   5.13106063e-02,  -3.75902236e-01,
                         -9.72685888e-02,   3.79058383e-02,  -3.77666861e-01,
                         -6.69629350e-02,  -1.03567932e-02,  -3.55117381e-01,
                          3.38554010e-02,  -1.06979415e-01,  -2.64487833e-01,
                         -1.77217275e-02,   3.00290957e-02,  -3.10280085e-01,
                          2.71114567e-03,   1.33358119e-02,  -2.93993115e-01,
                         -2.83729676e-02,   3.09747308e-02,  -3.16577345e-01,
                         -4.93897796e-02,   2.88569722e-02,  -3.32655013e-01,
                         -5.59915826e-02,   3.36857550e-02,  -3.39771658e-01,
                         -6.45972863e-02,   3.18771750e-02,  -3.41479391e-01,
                         -7.19759762e-02,   4.25489396e-02,  -3.51088256e-01,
                         -8.70441049e-02,   4.91915792e-02,  -3.61569315e-01,
                         -1.06328309e-01,   4.11872901e-02,  -3.74255031e-01,
                          4.41950746e-02,  -8.11681077e-02,  -2.56962478e-01,
                         -1.77487992e-02,  -4.17742282e-02,  -3.16967160e-01,
                          4.18890156e-02,  -9.30437297e-02,  -2.59968847e-01,
                         -6.80961162e-02,  -1.60961840e-02,  -3.48584950e-01,
                          2.77987812e-02,  -1.26796916e-01,  -2.75716096e-01,
                         -1.98967140e-02,  -9.36902910e-02,  -3.23858589e-01,
                          1.93914212e-02,  -1.32733822e-01,  -2.85334080e-01,
                         -1.79634355e-02,  -1.13395400e-01,  -3.19020778e-01,
                          5.40987961e-03,  -1.39268294e-01,  -2.94555902e-01,
                         -1.35952793e-02,   1.72444303e-02,  -3.06536049e-01,
                         -3.51282321e-02,   3.08605675e-02,  -3.19127232e-01,
                         -8.08383748e-02,   4.17781807e-02,  -3.53166163e-01,
                         -9.93531272e-02,   4.91640270e-02,  -3.73081267e-01,
                         -1.03972808e-01,   3.61210629e-02,  -3.75848174e-01,
                         -3.38180289e-02,  -2.87097655e-02,  -3.25558394e-01,
                         -1.03025854e-01,   3.08247302e-02,  -3.77704680e-01,
                         -9.39453468e-02,   1.31868273e-02,  -3.70473295e-01,
                         -1.03447519e-01,   6.94534602e-03,  -3.66991222e-01,
                         -8.46564621e-02,   4.09222534e-03,  -3.63883793e-01,
                         -2.19327286e-02,  -7.26656169e-02,  -3.15594971e-01,
                         -2.26074588e-02,  -7.44380057e-02,  -3.18823665e-01,
                          1.34282326e-02,  -1.33047119e-01,  -2.85743743e-01,
                         -2.32915971e-02,   2.87219491e-02,  -3.13511193e-01,
                         -4.81920782e-03,   1.13490978e-02,  -2.97950864e-01,
                          4.09420505e-02,  -3.44142243e-02,  -2.51322240e-01,
                         -5.23425043e-02,   2.45249979e-02,  -3.33127111e-01,
                         -5.59459329e-02,   2.79521532e-02,  -3.37503940e-01,
                         -1.00879073e-01,   4.57437709e-02,  -3.67237538e-01,
                         -4.38060276e-02,  -1.41805252e-02,  -3.30289930e-01,
                          4.50774953e-02,  -8.73646215e-02,  -2.57567793e-01,
                         -5.75034581e-02,  -1.04059651e-02,  -3.40483546e-01,
                         -2.41704863e-02,  -4.07691300e-02,  -3.12561810e-01,
                         -6.14051372e-02,  -1.10091567e-02,  -3.42439890e-01,
                         -9.75283980e-02,   6.47265557e-03,  -3.63986343e-01,
                         -7.56493807e-02,  -1.17198704e-02,  -3.54781538e-01,
                         -1.90979019e-02,  -5.51112629e-02,  -3.14059943e-01,
                          2.92436313e-02,  -1.16466731e-01,  -2.68037409e-01,
                         -2.10000630e-02,  -8.22799578e-02,  -3.16221416e-01,
                          1.94575153e-02,  -1.25827193e-01,  -2.77434170e-01,
                          9.36248805e-03,  -1.35294214e-01,  -2.89248705e-01,
                         -1.86087601e-02,  -1.05701089e-01,  -3.10119152e-01,
                         -1.00193508e-02,  -1.25378370e-01,  -3.11762422e-01,
                         -4.86145914e-03,  -1.33646205e-01,  -3.07175934e-01,
                          3.01253460e-02,  -1.43370517e-02,  -2.59994239e-01,
                         -3.99760529e-02,   2.67295670e-02,  -3.21496516e-01,
                          4.11318652e-02,  -6.81994855e-02,  -2.50629216e-01,
                         -1.04178488e-01,   4.71840575e-02,  -3.75653148e-01,
                         -4.69766557e-02,  -1.11093130e-02,  -3.30378801e-01,
                         -5.51950745e-02,  -1.09025845e-02,  -3.36944818e-01,
                         -1.05013527e-01,   1.84324570e-02,  -3.70733917e-01,
                         -1.72267444e-02,  -5.03611676e-02,  -3.14199984e-01,
                         -7.70478547e-02,  -1.47187728e-02,  -3.52009326e-01,
                          7.07693398e-05,  -1.28854886e-01,  -2.95242250e-01,
                          4.32160730e-03,   6.42817002e-03,  -2.87197173e-01,
                         -2.81640645e-02,   2.41027456e-02,  -3.12766701e-01,
                         -2.42342241e-02,   1.80564355e-02,  -3.07615846e-01,
                         -3.60117033e-02,   2.49164030e-02,  -3.18613470e-01,
                         -4.66779582e-02,   2.15334296e-02,  -3.26838702e-01,
                         -5.31854816e-02,   1.74960531e-02,  -3.31603020e-01,
                         -7.08662197e-02,   3.15268300e-02,  -3.46669495e-01,
                          5.23048192e-02,  -5.07132933e-02,  -2.56037444e-01,
                         -7.01284707e-02,   3.76784354e-02,  -3.47032934e-01,
                         -5.83563522e-02,   2.12774873e-02,  -3.35463464e-01,
                         -6.53097257e-02,   2.30580810e-02,  -3.38794470e-01,
                         -9.42083150e-02,   3.99797447e-02,  -3.59121084e-01,
                         -6.29553348e-02,   1.63367912e-02,  -3.37188751e-01,
                         -1.00669913e-01,   4.94932607e-02,  -3.70965391e-01,
                         -4.08378206e-02,  -2.01921742e-02,  -3.26443225e-01,
                         -5.95412627e-02,  -8.45370349e-03,  -3.34596097e-01,
                         -1.09000474e-01,   3.19314227e-02,  -3.76404971e-01,
                         -2.81052422e-02,  -3.53783779e-02,  -3.15419763e-01,
                         -1.81085300e-02,  -4.83894460e-02,  -3.10357571e-01,
                         -6.79888502e-02,  -1.38152270e-02,  -3.43809217e-01,
                         -6.65835366e-02,  -1.39684416e-02,  -3.46248150e-01,
                         -7.32659847e-02,  -1.35901803e-02,  -3.46122950e-01,
                          3.78181338e-02,  -9.56606194e-02,  -2.57963240e-01,
                          2.99071744e-02,  -1.05847403e-01,  -2.64951676e-01,
                         -2.36800630e-02,  -8.90920684e-02,  -3.13164592e-01,
                          2.72994861e-03,  -1.29392490e-01,  -2.89913744e-01,
                         -1.58404540e-02,  -1.17054150e-01,  -3.13176870e-01,
                          3.45337130e-02,  -1.97291877e-02,  -2.56363332e-01,
                         -8.65746988e-04,   5.91112021e-03,  -2.90700883e-01,
                         -1.58903506e-02,   1.25171486e-02,  -3.02438647e-01,
                         -2.34492235e-02,   1.00771543e-02,  -3.04024696e-01,
                          4.58757058e-02,  -6.56615421e-02,  -2.55303800e-01,
                         -7.20393956e-02,   2.82764994e-02,  -3.43669653e-01,
                         -1.02820642e-01,   4.91846018e-02,  -3.70120615e-01,
                         -4.63045537e-02,  -1.33682061e-02,  -3.24121654e-01,
                         -3.49956416e-02,  -2.05322877e-02,  -3.23810428e-01,
                         -3.62114832e-02,  -2.36423314e-02,  -3.18764687e-01,
                         -3.38372104e-02,  -2.71011349e-02,  -3.22420985e-01,
                         -3.55358012e-02,  -2.96368618e-02,  -3.16527009e-01,
                         -1.99184306e-02,  -4.50824350e-02,  -3.12758505e-01,
                         -6.01827465e-02,  -1.15499860e-02,  -3.38623464e-01,
                         -1.10312603e-01,   2.73672268e-02,  -3.75336319e-01,
                         -7.07653686e-02,  -1.42086130e-02,  -3.43728483e-01,
                         -9.54440609e-02,  -8.98288004e-03,  -3.60327929e-01,
                         -8.12293068e-02,  -1.51119288e-02,  -3.49973917e-01,
                         -8.46234486e-02,  -1.58345420e-02,  -3.54410201e-01,
                          2.41027568e-02,  -1.14422739e-01,  -2.67339587e-01,
                          3.47529724e-02,  -1.03577506e-02,  -2.63286948e-01,
                          1.41345058e-02,  -7.30215898e-03,  -2.66166687e-01,
                          9.04094800e-03,   5.46076102e-04,  -2.78806180e-01,
                          3.72075588e-02,  -2.68050805e-02,  -2.54811317e-01,
                         -2.88639087e-02,   1.34112798e-02,  -3.09677869e-01,
                         -4.40779179e-02,   1.60160791e-02,  -3.21933895e-01,
                          5.05861230e-02,  -4.49573621e-02,  -2.55174041e-01,
                         -5.06078042e-02,   1.30565502e-02,  -3.23485970e-01,
                         -5.64062335e-02,   8.56709294e-03,  -3.31818968e-01,
                         -8.32958966e-02,   3.35004292e-02,  -3.50614607e-01,
                         -8.14838409e-02,   2.75860708e-02,  -3.50344241e-01,
                         -6.26120046e-02,   1.61872562e-02,  -3.38889390e-01,
                         -6.74385205e-02,   1.80739798e-02,  -3.39189678e-01,
                         -9.44369137e-02,   4.56018932e-02,  -3.64901066e-01,
                         -7.09750354e-02,   1.20395906e-02,  -3.42025876e-01,
                         -5.08501567e-02,  -1.08379284e-02,  -3.26473147e-01,
                         -1.08973317e-01,   4.08237837e-02,  -3.75766128e-01,
                         -5.05612046e-02,  -1.12854196e-02,  -3.32587361e-01,
                         -2.25973725e-02,  -4.23786677e-02,  -3.07780772e-01,
                         -3.15590911e-02,  -3.40155028e-02,  -3.19216043e-01,
                         -8.24514478e-02,  -6.97825197e-03,  -3.56680125e-01,
                         -2.19934639e-02,  -5.96512407e-02,  -3.08902353e-01,
                         -1.78312194e-02,  -1.11920066e-01,  -3.08509827e-01,
                         -5.09657990e-03,  -1.28731191e-01,  -3.00285131e-01,
                          1.30438954e-02,  -1.08784391e-03,  -2.74172843e-01,
                         -1.30289812e-02,   8.09101295e-03,  -3.00043225e-01,
                         -3.59683558e-02,   1.38733704e-02,  -3.15082848e-01,
                          3.57481539e-02,  -5.17126247e-02,  -2.50426829e-01,
                         -7.51767308e-02,   2.12631337e-02,  -3.43954474e-01,
                         -1.08005300e-01,   4.29582335e-02,  -3.71475816e-01,
                         -4.36741635e-02,  -1.25818811e-02,  -3.23190987e-01,
                          3.53465155e-02,  -8.69404301e-02,  -2.56424636e-01,
                         -6.64323121e-02,  -8.38039164e-03,  -3.38261247e-01,
                         -1.10641882e-01,   1.75879337e-02,  -3.71245056e-01,
                         -7.60540068e-02,  -1.53321829e-02,  -3.45329314e-01,
                         -1.10169470e-01,   1.38001461e-02,  -3.69174719e-01,
                         -2.31153481e-02,  -7.20378757e-02,  -3.13485205e-01,
                          2.12293714e-02,  -1.17392354e-01,  -2.70469964e-01,
                         -5.55639043e-02,   6.30908925e-03,  -3.25618654e-01,
                         -5.72015233e-02,   7.02657783e-03,  -3.28789681e-01,
                         -6.41522035e-02,   6.89734006e-04,  -3.35775048e-01,
                          3.64712700e-02,  -7.53733814e-02,  -2.52503663e-01,
                         -6.36296272e-02,   8.92756321e-03,  -3.35729718e-01,
                         -8.95422325e-02,   3.97893302e-02,  -3.57966453e-01,
                         -5.30420542e-02,  -1.80579303e-03,  -3.28142613e-01,
                         -8.95078778e-02,   3.60807851e-02,  -3.53731781e-01,
                         -6.36000708e-02,   6.54814020e-03,  -3.35307300e-01,
                         -1.06530145e-01,   4.34098542e-02,  -3.68669212e-01,
                         -6.10394329e-02,  -2.93209357e-03,  -3.36216658e-01,
                         -5.92066608e-02,  -6.78616110e-03,  -3.34045172e-01,
                         -5.37318774e-02,  -4.79658134e-03,  -3.26693565e-01,
                         -6.72355145e-02,   5.45817241e-03,  -3.37384820e-01,
                         -1.10151894e-01,   4.00443077e-02,  -3.71776909e-01,
                         -5.45313284e-02,  -7.73315784e-03,  -3.30058903e-01,
                         -3.98477688e-02,  -1.85888335e-02,  -3.21078032e-01,
                         -3.56559195e-02,  -2.21951753e-02,  -3.14024150e-01,
                         -6.95843697e-02,  -5.67171536e-03,  -3.38510573e-01,
                         -2.88420729e-02,  -3.64286378e-02,  -3.09589595e-01,
                         -3.19043174e-02,  -3.29261646e-02,  -3.16281945e-01,
                         -1.11732319e-01,   2.07138676e-02,  -3.72217715e-01,
                          2.92984638e-02,  -9.83973742e-02,  -2.59637296e-01,
                         -2.22653337e-02,  -5.54764047e-02,  -3.06395501e-01,
                         -7.19033107e-02,  -1.27551137e-02,  -3.45810711e-01,
                         -2.34013330e-02,  -7.41517544e-02,  -3.12675208e-01,
                         -2.33385414e-02,  -8.86638388e-02,  -3.08550775e-01,
                          9.53191984e-03,  -1.22929558e-01,  -2.80372530e-01,
                          7.03056389e-03,  -1.30762681e-01,  -2.88144141e-01,
                         -8.63862410e-03,  -1.22944869e-01,  -3.04398745e-01,
                         -2.69078230e-03,  -1.22635610e-01,  -2.91127890e-01,
                          2.54637301e-02,  -6.53543696e-03,  -2.65274078e-01,
                         -1.92920025e-02,   5.78789925e-03,  -3.00735384e-01,
                         -1.13891341e-01,   3.25755589e-02,  -3.71183813e-01,
                         -7.24013001e-02,  -1.10148117e-02,  -3.42596710e-01,
                         -8.96430239e-02,  -1.53475180e-02,  -3.54552358e-01,
                         -3.30381468e-02,   7.88191985e-03,  -3.09675723e-01,
                         -3.95761207e-02,   5.19377925e-03,  -3.11703414e-01,
                         -4.38193195e-02,   6.94866478e-03,  -3.15624028e-01,
                         -4.76944335e-02,   1.67715130e-03,  -3.19390386e-01,
                          3.51603478e-02,  -5.97003363e-02,  -2.48437285e-01,
                         -4.95250300e-02,   4.67458833e-03,  -3.21206093e-01,
                         -5.17437384e-02,  -1.91559157e-04,  -3.24297637e-01,
                         -4.78836372e-02,  -1.94396242e-03,  -3.19886655e-01,
                         -9.36025530e-02,   3.31689045e-02,  -3.55613023e-01,
                         -8.49792212e-02,   1.81725603e-02,  -3.47461611e-01,
                         -1.00581080e-01,   3.44335586e-02,  -3.60354334e-01,
                         -4.10204083e-02,  -1.63102522e-02,  -3.15778702e-01,
                         -4.73129600e-02,  -9.12329555e-03,  -3.23977172e-01,
                         -9.54182670e-02,   2.70605050e-02,  -3.54140192e-01,
                         -7.49083832e-02,   8.32988881e-03,  -3.41751277e-01,
                         -7.72532970e-02,   1.43172797e-02,  -3.44708592e-01,
                         -2.99971923e-02,  -2.61941478e-02,  -3.09018016e-01,
                         -9.81603041e-02,   2.43928377e-02,  -3.52560401e-01,
                         -1.09603576e-01,   3.76266614e-02,  -3.63503575e-01,
                          3.34505327e-02,  -7.77106062e-02,  -2.53111809e-01,
                         -2.80019250e-02,  -3.19631919e-02,  -3.11705709e-01,
                         -8.16881508e-02,   1.23107294e-02,  -3.44758600e-01,
                          2.86442861e-02,  -7.86851645e-02,  -2.51349479e-01,
                         -7.33592510e-02,  -1.57726184e-03,  -3.37306827e-01,
                         -7.67251030e-02,   5.99383726e-04,  -3.43721449e-01,
                         -7.54446909e-02,  -6.94584986e-03,  -3.43088925e-01,
                         -1.15849994e-01,   2.71940939e-02,  -3.74118805e-01,
                         -2.43139639e-02,  -5.18669449e-02,  -3.06115746e-01,
                         -1.14424668e-01,   2.51106527e-02,  -3.70831162e-01,
                         -8.28746483e-02,  -1.47701213e-02,  -3.44204217e-01,
                         -8.14657882e-02,  -1.45493764e-02,  -3.44288468e-01,
                         -1.14810005e-01,   1.68072395e-02,  -3.68654490e-01,
                         -1.06302440e-01,   6.52008038e-03,  -3.66297841e-01,
                         -9.13853496e-02,  -1.20508494e-02,  -3.56185019e-01,
                         -8.94753858e-02,  -1.47585021e-02,  -3.52714032e-01,
                          1.33484844e-02,  -1.13089085e-01,  -2.72265673e-01,
                          2.64467224e-02,  -2.10768487e-02,  -2.53055036e-01,
                         -2.25262530e-02,  -4.29179966e-02,  -3.02146912e-01,
                         -9.04899240e-02,  -1.70712899e-02,  -3.51489484e-01,
                         -1.04928762e-01,  -5.52809983e-03,  -3.59877735e-01,
                          2.88194139e-03,  -1.23857021e-01,  -2.85927415e-01,
                         -1.13388803e-02,  -1.12798870e-01,  -2.98321396e-01,
                          1.87796876e-02,  -1.10330423e-02,  -2.60420710e-01,
                          1.19399978e-03,  -2.92630680e-03,  -2.77011901e-01,
                         -4.19724919e-03,  -6.08714763e-04,  -2.81794369e-01,
                         -1.22604333e-02,  -2.68386770e-03,  -2.84425437e-01,
                         -9.32264328e-03,   2.36915518e-03,  -2.91357547e-01,
                         -2.93907505e-02,   6.56133099e-03,  -3.02747786e-01,
                         -3.09354905e-02,   3.69617017e-03,  -3.05742919e-01,
                         -3.23923863e-02,   2.21035397e-03,  -3.07422787e-01,
                         -3.94222215e-02,   5.63853886e-04,  -3.09608877e-01,
                         -4.61123325e-02,  -2.58105341e-03,  -3.18025023e-01,
                         -4.26166728e-02,  -1.05026336e-02,  -3.14103693e-01,
                         -1.07362166e-01,   3.31461243e-02,  -3.64562541e-01,
                         -2.55996361e-02,  -3.53580751e-02,  -3.06147248e-01,
                         -8.54417533e-02,   9.52170882e-03,  -3.46599042e-01,
                         -2.52742935e-02,  -4.02546264e-02,  -3.02466214e-01,
                         -1.06347650e-01,   2.90874653e-02,  -3.65143836e-01,
                         -9.37998965e-02,   1.60559211e-02,  -3.52774084e-01,
                         -1.03576519e-01,   2.55293772e-02,  -3.59885991e-01,
                         -1.13264523e-01,   3.32424492e-02,  -3.69970471e-01,
                         -1.12639785e-01,   2.66256481e-02,  -3.66371721e-01,
                         -8.19389373e-02,  -3.91717348e-03,  -3.44013989e-01,
                         -1.13845721e-01,   3.01714502e-02,  -3.69477153e-01,
                         -8.50628242e-02,   1.03751617e-03,  -3.46366972e-01,
                         -8.47437829e-02,  -9.73917358e-03,  -3.45076233e-01,
                         -8.08052644e-02,  -7.16171134e-03,  -3.44082236e-01,
                         -1.14818498e-01,   1.46095976e-02,  -3.71047944e-01,
                         -1.13659143e-01,   1.54485572e-02,  -3.69586378e-01,
                         -9.25644636e-02,  -1.26603618e-02,  -3.46861869e-01,
                         -1.03328705e-01,  -2.59784376e-03,  -3.62895787e-01,
                         -2.32369956e-02,  -6.78839386e-02,  -3.07093382e-01,
                         -9.46109369e-02,  -1.56545285e-02,  -3.52620929e-01,
                          4.50361799e-03,  -1.15315393e-01,  -2.76897848e-01,
                         -1.89148188e-02,  -1.03433266e-01,  -3.05588901e-01,
                         -6.69063628e-03,  -1.21145338e-01,  -2.98444092e-01,
                          3.39755900e-02,  -3.62240598e-02,  -2.49364436e-01,
                          3.73944491e-02,  -4.30612639e-02,  -2.48780683e-01,
                         -3.36935818e-02,  -1.84149202e-02,  -3.06778371e-01,
                         -1.14706993e-01,   2.64131725e-02,  -3.66477758e-01,
                         -8.27602819e-02,  -1.57845113e-02,  -3.48257333e-01,
                          2.51844414e-02,  -9.98161882e-02,  -2.61281848e-01,
                         -2.26365030e-02,  -9.35086161e-02,  -3.02864522e-01,
                         -7.39230076e-03,  -1.11294270e-01,  -2.86598146e-01,
                          9.16521903e-03,  -7.75878970e-03,  -2.64271796e-01,
                         -2.48113330e-02,   3.25602386e-03,  -2.97902912e-01,
                         -3.43066454e-02,  -7.30684376e-04,  -3.03581297e-01,
                         -3.90574336e-02,  -5.21854591e-03,  -3.14018101e-01,
                         -2.67243981e-02,  -2.80020293e-02,  -3.04413378e-01,
                         -9.73920301e-02,   6.66844379e-03,  -3.51119876e-01,
                         -9.79945809e-02,   1.36771034e-02,  -3.54451030e-01,
                         -1.10470526e-01,   2.50135772e-02,  -3.62777621e-01,
                          2.77004130e-02,  -8.87060389e-02,  -2.55289882e-01,
                         -1.11944690e-01,   1.83856729e-02,  -3.61671150e-01,
                         -1.14768147e-01,   1.80254560e-02,  -3.67563039e-01,
                          2.35287212e-02,  -9.49117690e-02,  -2.58053273e-01,
                         -1.16759561e-01,   1.93679072e-02,  -3.69856626e-01,
                         -8.98027569e-02,  -7.31049385e-03,  -3.44681472e-01,
                         -1.14112183e-01,   1.00366483e-02,  -3.64379764e-01,
                         -9.20144543e-02,  -1.14996741e-02,  -3.48700374e-01,
                         -1.12154491e-01,   6.67009735e-03,  -3.66891176e-01,
                         -1.09880187e-01,   2.59762025e-03,  -3.65396857e-01,
                         -1.07891813e-01,  -3.19331000e-03,  -3.60028058e-01,
                         -9.89820287e-02,  -1.18862092e-02,  -3.50717247e-01,
                         -2.17455849e-02,  -8.39762688e-02,  -3.01802993e-01,
                         -8.39213375e-04,  -1.20717578e-01,  -2.85992742e-01,
                         -6.09301357e-03,  -5.37794456e-03,  -2.81448722e-01,
                         -1.63851082e-02,  -1.17105967e-03,  -2.91008770e-01,
                          2.85463389e-02,  -6.29384220e-02,  -2.46634662e-01,
                         -3.22409384e-02,  -2.11848784e-02,  -3.01976830e-01,
                         -2.47048717e-02,  -4.60466743e-02,  -3.00766379e-01,
                          1.93487275e-02,  -7.90972337e-02,  -2.52898544e-01,
                         -1.16969250e-01,   1.66079979e-02,  -3.67048562e-01,
                         -1.17836818e-01,   1.38012543e-02,  -3.69492114e-01,
                         -9.19064060e-02,  -1.53487353e-02,  -3.48635405e-01,
                         -9.80669260e-02,  -8.89169984e-03,  -3.57156605e-01,
                         -1.06140107e-01,  -7.14720041e-03,  -3.57196927e-01,
                          1.38826594e-02,  -1.07291810e-01,  -2.64868289e-01,
                          1.72026884e-02,  -2.02161148e-02,  -2.51836240e-01,
                          2.45864317e-02,  -3.30946110e-02,  -2.47215524e-01,
                         -2.22680867e-02,  -7.19689857e-03,  -2.92272449e-01,
                         -2.83955000e-02,  -3.09814466e-03,  -2.99734533e-01,
                          2.94220895e-02,  -4.92878892e-02,  -2.47109130e-01,
                         -3.62243764e-02,  -7.62608927e-03,  -3.05032969e-01,
                         -3.40625457e-02,  -1.70592442e-02,  -2.98343867e-01,
                         -1.13475285e-01,   1.45663423e-02,  -3.60194802e-01,
                         -1.01316780e-01,   1.19606396e-02,  -3.53704363e-01,
                         -1.16923727e-01,   1.16655324e-02,  -3.63572389e-01,
                         -9.33617502e-02,  -8.11313838e-03,  -3.47996324e-01,
                         -9.88465399e-02,  -8.44133180e-03,  -3.51211339e-01,
                         -1.02783114e-01,  -1.06775379e-02,  -3.55299085e-01,
                         -2.43426189e-02,  -6.82664812e-02,  -3.00322413e-01,
                          8.22969712e-03,  -1.03383347e-01,  -2.68203378e-01,
                         -2.27251686e-02,  -7.61132389e-02,  -3.02729815e-01,
                          1.76608004e-03,  -1.07395209e-01,  -2.75038779e-01,
                         -1.01156114e-03,  -1.08634904e-01,  -2.80293822e-01,
                          2.68426873e-02,  -3.76615375e-02,  -2.46575058e-01,
                         -3.38656940e-02,  -1.06610190e-02,  -3.01539034e-01,
                         -3.46575007e-02,  -1.31622078e-02,  -3.04392368e-01,
                         -2.77827475e-02,  -4.52394336e-02,  -2.96115220e-01,
                          1.69633683e-02,  -9.60425064e-02,  -2.59693176e-01,
                         -2.59107314e-02,  -6.26724064e-02,  -3.03091288e-01,
                         -1.02809452e-01,  -6.73109759e-03,  -3.49868298e-01,
                         -2.55828109e-02,  -5.62169179e-02,  -2.97390312e-01,
                         -1.14802115e-01,   6.57903450e-03,  -3.63588631e-01,
                         -1.13620870e-01,   3.32942838e-03,  -3.63622189e-01,
                         -9.91268549e-03,  -1.06751777e-01,  -2.88695276e-01,
                          1.46707119e-02,  -1.75351612e-02,  -2.55389065e-01,
                         -1.38804391e-02,  -3.94290593e-03,  -2.84225285e-01,
                         -3.01333554e-02,  -2.22373083e-02,  -3.02808017e-01,
                         -2.70775296e-02,  -2.74881776e-02,  -2.98490465e-01,
                         -2.55934764e-02,  -4.07753102e-02,  -2.97490418e-01,
                         -2.81028058e-02,  -5.09406067e-02,  -2.97168374e-01,
                         -1.05384767e-01,   3.27719562e-03,  -3.53932142e-01,
                         -1.12719752e-01,   3.32708051e-03,  -3.57908189e-01,
                         -1.11919224e-01,   1.08072031e-02,  -3.60182345e-01,
                         -2.64929757e-02,  -5.72212525e-02,  -2.93844223e-01,
                         -1.16199486e-01,   6.70872256e-03,  -3.62745821e-01,
                         -1.13006316e-01,   6.70413487e-04,  -3.62945497e-01,
                         -1.08186208e-01,  -7.50266667e-03,  -3.54625762e-01,
                         -1.10473111e-01,  -2.01901468e-03,  -3.58861059e-01,
                         -2.18565520e-02,  -7.76642412e-02,  -2.94431388e-01,
                          7.01823086e-03,  -1.06812343e-01,  -2.72697777e-01,
                         -1.87961627e-02,  -9.22433212e-02,  -2.93125212e-01,
                         -1.53850717e-02,  -9.82636437e-02,  -2.89599627e-01,
                         -2.67184600e-02,  -1.14351623e-02,  -2.94259131e-01,
                          2.10859310e-02,  -6.12687245e-02,  -2.46616945e-01,
                         -2.74784788e-02,  -4.61863391e-02,  -2.95848966e-01,
                         -1.09949738e-01,  -4.66266414e-03,  -3.52241158e-01,
                         -1.13920309e-01,   2.81309150e-03,  -3.58372480e-01,
                         -1.01575352e-01,  -8.62171501e-03,  -3.51945043e-01,
                          5.98045578e-03,  -9.37927235e-03,  -2.65196949e-01,
                         -1.11175664e-02,  -9.87771712e-03,  -2.79772282e-01,
                         -1.07564442e-02,  -9.54236183e-03,  -2.82596231e-01,
                         -2.17893682e-02,  -1.11409063e-02,  -2.87103415e-01,
                         -3.05190198e-02,  -1.56023875e-02,  -2.96615601e-01,
                          2.28595957e-02,  -7.93330595e-02,  -2.50349700e-01,
                          1.31723816e-02,  -7.67718628e-02,  -2.51315296e-01,
                         -2.44406164e-02,  -6.57752678e-02,  -2.94492394e-01,
                         -1.13048829e-01,   3.63448821e-03,  -3.60100091e-01,
                         -1.09126292e-01,  -3.88022605e-03,  -3.54639888e-01,
                         -1.86118893e-02,  -8.53050277e-02,  -2.91638732e-01,
                         -1.66931171e-02,  -8.63062441e-02,  -2.86677569e-01,
                         -5.89026744e-03,  -9.67208147e-02,  -2.76360840e-01,
                         -1.42407175e-02,  -9.18070972e-02,  -2.81480312e-01,
                          2.24320497e-02,  -5.06000631e-02,  -2.46271566e-01,
                         -2.86541060e-02,  -1.89931989e-02,  -2.95225918e-01,
                         -2.78564375e-02,  -2.94467211e-02,  -2.95410872e-01,
                         -2.69671194e-02,  -5.09788617e-02,  -2.94887364e-01,
                          5.69324149e-03,  -9.13580060e-02,  -2.65167356e-01,
                          8.99281632e-03,  -1.39552113e-02,  -2.61330009e-01,
                          6.69823028e-04,  -1.64664183e-02,  -2.59448588e-01,
                         -1.48603343e-03,  -9.84987989e-03,  -2.70453662e-01,
                         -8.51606857e-03,  -1.30978404e-02,  -2.73522288e-01,
                          2.06325091e-02,  -4.05844711e-02,  -2.44986609e-01,
                         -2.71074288e-02,  -1.98838115e-02,  -2.87984252e-01,
                         -2.92384233e-02,  -3.00947949e-02,  -2.92051375e-01,
                          2.23499406e-02,  -6.86802715e-02,  -2.48858988e-01,
                         -2.75045410e-02,  -4.08536345e-02,  -2.92356819e-01,
                         -2.68620010e-02,  -5.07889278e-02,  -2.90652186e-01,
                          1.19424667e-02,  -8.89196098e-02,  -2.60211289e-01,
                          2.64334027e-03,  -9.91158560e-02,  -2.66696334e-01,
                         -1.88803710e-02,  -7.18238652e-02,  -2.86128432e-01,
                         -2.93618534e-03,  -9.23374295e-02,  -2.71234512e-01,
                         -1.34090949e-02,  -8.38166922e-02,  -2.77455449e-01,
                         -2.38368940e-03,  -9.87594351e-02,  -2.73943305e-01,
                          1.28435288e-02,  -2.58979760e-02,  -2.49800175e-01,
                         -2.89812554e-02,  -3.96540500e-02,  -2.91007817e-01,
                          1.07809808e-02,  -1.91045534e-02,  -2.52440721e-01,
                          1.47016682e-02,  -2.40267366e-02,  -2.50965774e-01,
                          1.98176019e-02,  -3.05406824e-02,  -2.47590989e-01,
                         -1.38854180e-02,  -1.94223635e-02,  -2.75871962e-01,
                         -2.07922217e-02,  -1.86019652e-02,  -2.81583577e-01,
                          1.62374452e-02,  -6.12006187e-02,  -2.50409633e-01,
                         -2.99777351e-02,  -2.36037634e-02,  -2.91571140e-01,
                         -2.68419757e-02,  -4.46137413e-02,  -2.87405640e-01,
                         -2.76988912e-02,  -4.41536084e-02,  -2.91318476e-01,
                         -2.40495242e-02,  -5.67321144e-02,  -2.89212435e-01,
                          8.14218540e-03,  -7.50230104e-02,  -2.56659538e-01,
                          3.10251326e-03,  -8.49222615e-02,  -2.63498545e-01,
                         -1.99993514e-02,  -6.09811768e-02,  -2.84032494e-01,
                         -4.53633443e-03,  -8.86638612e-02,  -2.70090014e-01,
                         -1.21693322e-02,  -8.16575214e-02,  -2.77022719e-01,
                         -1.53160654e-02,  -7.31068701e-02,  -2.79582202e-01,
                         -8.94992426e-03,  -8.77972171e-02,  -2.73641348e-01,
                         -2.75432449e-02,  -3.46865505e-02,  -2.85428703e-01,
                         -2.63628867e-02,  -3.98368873e-02,  -2.84488589e-01,
                          2.73653679e-03,  -7.50049129e-02,  -2.58677095e-01,
                         -1.28699243e-02,  -8.15923959e-02,  -2.72237629e-01,
                          1.18976235e-02,  -3.55520137e-02,  -2.46229231e-01,
                          1.77075695e-02,  -4.22579385e-02,  -2.44714826e-01,
                         -1.17047951e-02,  -2.42675003e-02,  -2.72892714e-01,
                         -1.56420283e-02,  -3.09673734e-02,  -2.75034577e-01,
                         -2.42663473e-02,  -2.30318867e-02,  -2.82132357e-01,
                         -1.84239428e-02,  -2.60304324e-02,  -2.77941853e-01,
                         -2.64986232e-02,  -2.77621839e-02,  -2.82961875e-01,
                         -2.31963918e-02,  -3.79048660e-02,  -2.80556440e-01,
                         -2.82688122e-02,  -2.83484161e-02,  -2.85569817e-01,
                         -2.04857662e-02,  -4.23045419e-02,  -2.82008231e-01,
                         -2.31577884e-02,  -4.33608256e-02,  -2.85485834e-01,
                         -2.09951736e-02,  -5.31970896e-02,  -2.82498688e-01,
                         -4.31279652e-03,  -7.67601803e-02,  -2.66387224e-01,
                         -9.63411294e-03,  -7.75261223e-02,  -2.68606752e-01,
                         -5.28696692e-03,  -8.48855153e-02,  -2.70509034e-01,
                          3.98043683e-03,  -1.99552849e-02,  -2.55823731e-01,
                         -3.31806950e-03,  -1.79472025e-02,  -2.65349090e-01,
                          1.17035480e-02,  -4.73735780e-02,  -2.48949051e-01,
                         -1.67850242e-03,  -2.06508748e-02,  -2.60355622e-01,
                         -9.15067643e-03,  -2.13081576e-02,  -2.66794026e-01,
                         -9.53310076e-03,  -3.43947597e-02,  -2.67374337e-01,
                         -1.39200669e-02,  -3.52072157e-02,  -2.72809476e-01,
                          1.18282996e-02,  -6.79827780e-02,  -2.50692457e-01,
                         -1.54096643e-02,  -3.92735079e-02,  -2.72961229e-01,
                         -1.66726410e-02,  -4.98342514e-02,  -2.78793037e-01,
                         -2.02128477e-02,  -4.63713109e-02,  -2.80722797e-01,
                         -1.93286669e-02,  -6.08547777e-02,  -2.76954263e-01,
                         -1.09938290e-02,  -6.43337145e-02,  -2.69104004e-01,
                         -4.44146944e-03,  -7.10996762e-02,  -2.62902081e-01,
                         -1.34091610e-02,  -7.19468594e-02,  -2.73622900e-01,
                         -1.27007756e-02,  -2.63179298e-02,  -2.68239677e-01,
                         -1.53711308e-02,  -5.43499850e-02,  -2.73193806e-01,
                         -1.67192961e-03,  -2.93015502e-02,  -2.55538762e-01,
                          3.72045836e-03,  -2.84692161e-02,  -2.50976443e-01,
                          4.72872332e-03,  -3.28648835e-02,  -2.49413311e-01,
                          7.90852681e-03,  -4.57524695e-02,  -2.47886926e-01,
                         -8.52059852e-03,  -3.03183775e-02,  -2.60220587e-01,
                          9.30050761e-03,  -5.37448153e-02,  -2.47491851e-01,
                          6.01643883e-03,  -6.00506701e-02,  -2.51412779e-01,
                         -9.61721782e-03,  -3.82383838e-02,  -2.63796180e-01,
                         -1.37809552e-02,  -4.52018566e-02,  -2.67287523e-01,
                          7.28773931e-03,  -7.02850223e-02,  -2.53518760e-01,
                         -3.41575127e-04,  -6.99465200e-02,  -2.57540733e-01,
                          1.84505619e-03,  -6.08858243e-02,  -2.53312349e-01,
                         -1.60976797e-02,  -4.48330007e-02,  -2.70636797e-01,
                         -1.07533755e-02,  -6.26998693e-02,  -2.64846653e-01,
                         -7.92256184e-03,  -7.03317076e-02,  -2.65427083e-01,
                          7.25858565e-03,  -3.75386477e-02,  -2.45548069e-01,
                          3.02125933e-03,  -3.73040326e-02,  -2.49454305e-01,
                         -7.02404790e-03,  -3.52950990e-02,  -2.58410037e-01,
                          3.33673810e-03,  -4.95805331e-02,  -2.49503240e-01,
                         -1.11329872e-02,  -4.40222360e-02,  -2.66152531e-01,
                          1.14542432e-04,  -5.43865860e-02,  -2.52677947e-01,
                         -5.43944910e-03,  -5.47337793e-02,  -2.58551776e-01,
                         -8.60888883e-03,  -5.15756719e-02,  -2.58364797e-01,
                         -1.13685448e-02,  -5.19394353e-02,  -2.65937537e-01,
                         -4.94685397e-03,  -6.10616058e-02,  -2.55594432e-01,
                         -8.87793116e-03,  -6.20532110e-02,  -2.62042642e-01,
                         -1.18289357e-02,  -5.88264838e-02,  -2.66547322e-01,
                         -4.16910741e-03,  -2.99466960e-02,  -2.57966489e-01,
                         -2.34145252e-03,  -3.67356725e-02,  -2.50966460e-01,
                          1.81835075e-03,  -4.77311239e-02,  -2.49718413e-01,
                         -8.59724544e-03,  -3.97725143e-02,  -2.55605102e-01,
                         -4.68802359e-03,  -5.00612296e-02,  -2.52340168e-01,
                         -6.12230459e-03,  -5.06342910e-02,  -2.55833685e-01
            };

            int i = 0;

            for (Model::Vertex* pVertex : model.Vertices())
            {
                pVertex->_Pos[0] = vertices[i * 3];
                pVertex->_Pos[1] = vertices[i * 3 + 1];
                pVertex->_Pos[2] = vertices[i * 3 + 2];

                ++i;
            }

#endif




            model.BuildVertexData();

#if 0
            // Load the corresponding ply file for triangulation.
            char* pName = new char[modelName.length() + 1];
            pName[modelName.length() - 4] = '.';
            pName[modelName.length() - 3] = 'p';
            pName[modelName.length() - 2] = 'l';
            pName[modelName.length() - 1] = 'y';
            pName[modelName.length()] = '\0';

            for (int i = 0; i < modelName.length() - 4; ++i)
            {
                pName[i] = modelName.at(i).toLatin1();
            }

            tetgenio in, out;
            in.load_ply(pName);
            tetrahedralize("pYV", &in, &out);

            Eigen::MatrixXf& x1 = m_MaterialModelSolver.X1();
            Eigen::MatrixXf& x2 = m_MaterialModelSolver.X2();
            Eigen::MatrixXf& x3 = m_MaterialModelSolver.X3();
            Eigen::MatrixXf& x4 = m_MaterialModelSolver.X4();

            x1.resize(3, out.numberoftetrahedra);
            x2.resize(3, out.numberoftetrahedra);
            x3.resize(3, out.numberoftetrahedra);
            x4.resize(3, out.numberoftetrahedra);

            for (int i = 0; i < out.numberoftetrahedra; ++i)
            {
                x1(0, i) = (float)out.pointlist[out.tetrahedronlist[i * 4]];
                x1(1, i) = (float)out.pointlist[out.tetrahedronlist[i * 4] + 1];
                x1(2, i) = (float)out.pointlist[out.tetrahedronlist[i * 4] + 2];

                x2(0, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 1]];
                x2(1, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 1] + 1];
                x2(2, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 1] + 2];

                x3(0, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 2]];
                x3(1, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 2] + 1];
                x3(2, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 2] + 2];

                x4(0, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 3]];
                x4(1, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 3] + 1];
                x4(2, i) = (float)out.pointlist[out.tetrahedronlist[i * 4 + 3] + 2];
            }

            delete pName;

#endif

            // TODO: Temp.
            model.SetTexture(texture[0]);

            m_Models.push_back(model);

            Model groundTruthModel;
            m_GroundTruthModels.push_back(groundTruthModel);

            ComputeModelCentroid();
            m_PrevModelCentroid = m_ModelCentroid;
            m_ModelCentroidDiff.setZero();

            makeCurrent();

            // Create a vertex array object. In OpenGL ES 2.0 and OpenGL 2.x
            // implementations this is optional and support may not be present
            // at all. Nonetheless the below code works in all cases and makes
            // sure there is a VAO when one is needed.

            QOpenGLVertexArrayObject* pModelVAO = new QOpenGLVertexArrayObject;
            pModelVAO->create();
            QOpenGLVertexArrayObject::Binder modelVAOBinder(pModelVAO);

            // Setup vertex buffer object.
            QOpenGLBuffer* pModelVBO = new QOpenGLBuffer;
            pModelVBO->create();
            pModelVBO->bind();
            pModelVBO->allocate(model.VertexData().data(), model.VertexData().size() * sizeof(GLfloat));

            // Store the vertex attribute bindings for the program.
            QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
            f->glEnableVertexAttribArray(0);
            f->glEnableVertexAttribArray(1);
            f->glEnableVertexAttribArray(2);
            f->glEnableVertexAttribArray(3);
            f->glEnableVertexAttribArray(4);
            f->glEnableVertexAttribArray(5);
            f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), 0);
            f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(3 * sizeof(GLfloat)));
            f->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(6 * sizeof(GLfloat)));
            f->glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(8 * sizeof(GLfloat)));
            f->glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(9 * sizeof(GLfloat)));
            f->glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(12 * sizeof(GLfloat)));

            pModelVBO->release();

            m_ModelVAOs.push_back(pModelVAO);
            m_ModelVBOs.push_back(pModelVBO);

            // For the ground truth model.
            pModelVAO = new QOpenGLVertexArrayObject;
            pModelVAO->create();
            QOpenGLVertexArrayObject::Binder groundTruthModelVAOBinder(pModelVAO);

            // Setup vertex buffer object.
            pModelVBO = new QOpenGLBuffer;
            pModelVBO->create();
            pModelVBO->bind();
            pModelVBO->allocate(model.VertexData().data(), model.VertexData().size() * sizeof(GLfloat));

            // Store the vertex attribute bindings for the program.
            f = QOpenGLContext::currentContext()->functions();
            f->glEnableVertexAttribArray(0);
            f->glEnableVertexAttribArray(1);
            f->glEnableVertexAttribArray(2);
            f->glEnableVertexAttribArray(3);
            f->glEnableVertexAttribArray(4);
            f->glEnableVertexAttribArray(5);
            f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), 0);
            f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(3 * sizeof(GLfloat)));
            f->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(6 * sizeof(GLfloat)));
            f->glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(8 * sizeof(GLfloat)));
            f->glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(9 * sizeof(GLfloat)));
            f->glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 16 * sizeof(GLfloat), reinterpret_cast<void *>(12 * sizeof(GLfloat)));

            pModelVBO->release();

            m_GroundTruthModelVAOs.push_back(pModelVAO);
            m_GroundTruthModelVBOs.push_back(pModelVBO);

            UpdateModel();

            doneCurrent();

            update();
        }

        if(modelsListInit.contains(modelName))
        {
            QMessageBox::warning(this, "Model already loaded", "This model is already loaded. Please select a different model.");
        }
    } while(modelsListInit.contains(modelName));

    emit modelsChanged();
}

void GLWidget::RemoveModel(GLint modelNumber)
{
    makeCurrent();

    QOpenGLBuffer* pModelVBO = m_ModelVBOs[modelNumber];
    pModelVBO->destroy();
    m_ModelVBOs.erase(m_ModelVBOs.begin() + modelNumber);
    delete pModelVBO;

    QOpenGLVertexArrayObject* pModelVAO = m_ModelVAOs[modelNumber];
    pModelVAO->destroy();
    m_ModelVAOs.erase(m_ModelVAOs.begin() + modelNumber);
    delete pModelVAO;

    m_Models.erase(m_Models.begin() + modelNumber);

    // For the ground truth model.
    pModelVBO = m_GroundTruthModelVBOs[modelNumber];
    pModelVBO->destroy();
    m_GroundTruthModelVBOs.erase(m_GroundTruthModelVBOs.begin() + modelNumber);
    delete pModelVBO;

    pModelVAO = m_GroundTruthModelVAOs[modelNumber];
    pModelVAO->destroy();
    m_GroundTruthModelVAOs.erase(m_GroundTruthModelVAOs.begin() + modelNumber);
    delete pModelVAO;

    if (m_GroundTruthModelExisting)
    {
        m_GroundTruthModels.erase(m_GroundTruthModels.begin() + modelNumber);
    }

    doneCurrent();
}

void GLWidget::RemoveModels()
{
//    if (!m_SelectedModel.isEmpty())
//    {
//        unsigned int modelNumber = modelsList.indexOf(m_SelectedModel,0);
//        modelsList.removeAt(modelNumber);
//        RemoveModel(modelNumber);

//        emit modelsChanged();

//        update();
//    }
//    else
    if (m_CheckedModels.size() > 0)
    {
        for(int i = m_CheckedModels.size()-1; i >= 0; i--)
        {
            modelsList.removeAt(m_CheckedModels.at(i));
            RemoveModel(m_CheckedModels.at(i));

            emit modelsChanged();

            update();
        }
    }
}

void GLWidget::AddFrontierContour()
{
    m_SelectingModelContour = MODEL_CONTOUR_TYPE_FRONTIER;

    for (Model& model : m_Models)
    {
        model.UnselectAllVertices();
    }

    std::vector<std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> > frontierContour;
    m_ModelContours.push_back(frontierContour);

    SelectHighCurvatureVertices();
}

void GLWidget::AddOccludingContour()
{
    m_SelectingModelContour = MODEL_CONTOUR_TYPE_OCCLUDING;

    for (Model& model : m_Models)
    {
        model.UnselectAllVertices();
    }

    std::vector<std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> > occludingContour;
    m_ModelContours.push_back(occludingContour);
}

void GLWidget::AddLigamentContour()
{
    m_SelectingModelContour = MODEL_CONTOUR_TYPE_LIGAMENT;

    for (Model& model : m_Models)
    {
        model.UnselectAllVertices();
    }

    std::vector<std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> > ligamentContour;
    m_ModelContours.push_back(ligamentContour);
}

void GLWidget::RemoveModelContours()
{
//    if (!m_SelectedModel.isEmpty())
//    {
//        unsigned int modelNumber = modelsList.indexOf(m_SelectedModel,0);
//        modelsList.removeAt(modelNumber);
//        RemoveModel(modelNumber);

//        emit modelsChanged();

//        update();
//    }
//    else
    if (m_CheckedModelContours.size() > 0)
    {
        for(int i = m_CheckedModelContours.size()-1; i >= 0; i--)
        {
            std::cout << "Remove model contour: " << m_CheckedModelContours.at(i) << std::endl;

            if (m_ModelToImageContourMap.find(m_ModelContoursList.at(m_CheckedModelContours.at(i))) != m_ModelToImageContourMap.end())
            {
                std::cout << "Remove model to image contour map - key: " << m_ModelContoursList.at(m_CheckedModelContours.at(i)).toUtf8().constData() << ", value: " << m_ModelToImageContourMap[m_ModelContoursList.at(m_CheckedModelContours.at(i))].toUtf8().constData() << std::endl;

                m_ModelToImageContourMap.erase(m_ModelContoursList.at(m_CheckedModelContours.at(i)));
            }

            m_ModelContoursList.removeAt(m_CheckedModelContours.at(i));
            m_ModelContours.erase(m_ModelContours.begin() + m_CheckedModelContours.at(i));

            emit ModelContoursChanged();

            for (Model& model : m_Models)
            {
                model.UnselectAllVertices();
            }
        }

        UpdateModel(false, false, false);
        update();
    }
}

void GLWidget::ShowModelContour(unsigned int Index)
{
    for (Model& model : m_Models)
    {
        model.UnselectAllVertices();
    }

    for (std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> item : m_ModelContours[Index])
    {
        m_Models[std::get<1>(item)].Vertices()[std::get<0>(item)]->_Selected = true;
    }

    if (m_ModelToImageContourMap.find(m_ModelContoursList.at(Index)) != m_ModelToImageContourMap.end())
    {
        int index = m_ImageContoursList.indexOf(m_ModelToImageContourMap.at(m_ModelContoursList.at(Index)));
        ShowImageContour(index);
    }

    UpdateModel(false, false, false);
    update();
}

void GLWidget::AddImageContour()
{
    emit OpenContourSelectionTool();
}

void GLWidget::RemoveImageContours()
{
//    if (!m_SelectedModel.isEmpty())
//    {
//        unsigned int modelNumber = modelsList.indexOf(m_SelectedModel,0);
//        modelsList.removeAt(modelNumber);
//        RemoveModel(modelNumber);

//        emit modelsChanged();

//        update();
//    }
//    else
    if (m_CheckedImageContours.size() > 0)
    {
        for(int i = m_CheckedImageContours.size()-1; i >= 0; i--)
        {
            std::cout << "Remove image contour: " << m_CheckedImageContours.at(i) << std::endl;

            for (std::map<QString, QString>::iterator it = m_ModelToImageContourMap.begin(); it != m_ModelToImageContourMap.end(); ++it)
            {
                if (it->second == m_ImageContoursList.at(m_CheckedImageContours.at(i)))
                {
                    QString key = it->first;

                    std::cout << "Remove model to image contour map - key: " << key.toUtf8().constData() << ", value: " << m_ModelToImageContourMap[key].toUtf8().constData() << std::endl;

                    m_ModelToImageContourMap.erase(key);

                    break;
                }
            }

            m_ImageContoursList.removeAt(m_CheckedImageContours.at(i));
            m_ImageContours.erase(m_ImageContours.begin() + m_CheckedImageContours.at(i));

            emit ImageContoursChanged();
        }

        update();
    }
}

void GLWidget::ShowImageContour(unsigned int Index)
{
    m_SelectedImageContourIndex = Index;

    update();
}

void GLWidget::LinkModelToImageContour()
{
    if (m_CheckedModelContours.size() == 1 && m_CheckedImageContours.size() == 1)
    {
        m_ModelToImageContourMap[m_ModelContoursList.at(m_CheckedModelContours.back())] = m_ImageContoursList.at(m_CheckedImageContours.back());

        std::cout << "Link " << m_ModelContoursList.at(m_CheckedModelContours.back()).toUtf8().constData() << " to " << m_ImageContoursList.at(m_CheckedImageContours.back()).toUtf8().constData() << std::endl;

        // Fix the two end points on the model and image contour when moving frontier vertices to the image contour.
        std::vector<std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> > modelContour = m_ModelContours[m_ModelContoursList.indexOf(m_ModelContoursList.at(m_CheckedModelContours.back()))];
        std::vector<Eigen::Vector2f> imageContour = m_ImageContours[m_ImageContoursList.indexOf(m_ImageContoursList.at(m_CheckedImageContours.back()))];
        GLfloat scale = frame_picture_Ratio * cameraParameters[7];
        Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

        if (std::get<2>(modelContour[0]) == MODEL_CONTOUR_TYPE_FRONTIER)
        {
//            Eigen::Vector2f pointStart = imageContour.front();
//            Eigen::Vector2f pointEnd = imageContour.back();

//            pointStart *= cameraParameters[7];
//            pointEnd *= cameraParameters[7];
//            pointStart += scale * dimension;
//            pointEnd += scale * dimension;

//            float minDistStart = std::numeric_limits<float>::max();
//            float minDistEnd = std::numeric_limits<float>::max();
            Model::Vertex* pMinVertexStart = NULL;
            Model::Vertex* pMinVertexEnd = NULL;

//            for (std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> mc : modelContour)
//            {
//                Model::Vertex* pVertex = m_Models[std::get<1>(mc)].Vertices()[std::get<0>(mc)];

//                Eigen::Vector2f p;
//                Point point2D = m_Models[std::get<1>(mc)].ProjectVertexOnto2D(pVertex, m_ModelView, m_proj, m_Viewport);
//                p << point2D.x(), point2D.y();


////                             // Use only a subset of points on the contour.
////                             if (index % 10 > 0)
////                             {
////                                 ++index;

////                                 continue;
////                             }

////                 if (InsideFOV(point))
//                 {
//                     float distStart = (pointStart - p).norm();
//                     float distEnd = (pointEnd - p).norm();

//                     if (distStart < minDistStart)
//                     {
//                         pMinVertexStart = pVertex;
//                         minDistStart = distStart;
//                     }

//                     if (distEnd < minDistEnd)
//                     {
//                         pMinVertexEnd = pVertex;
//                         minDistEnd = distEnd;
//                     }
//                 }
//            }

            pMinVertexStart = m_Models.back().Vertices()[m_HighCurvatureVertexIndices[m_HighCurvatureStartPosition]];
            pMinVertexEnd = m_Models.back().Vertices()[m_HighCurvatureVertexIndices[m_HighCurvatureEndPosition - 1]];

            m_FrontierContourFixedPoints[pMinVertexStart] = imageContour.front();
            m_FrontierContourFixedPoints[pMinVertexEnd] = imageContour.back();
        }
    }
}

void GLWidget::resetTransformations()
{
//    QVector3D origin = model.getCenter(0);
//    Eigen::Vector3f origin = m_Models[0].Centroid();
//    coordModels = origin;

    coordModels.setZero();

    trackball.reset();
    resetTumor();

//    updateGL();
    update();
}

// Put the model on the widget centre.
void GLWidget::CentreModel(void)
{
//    coordModels[2] = -0.3f/scaleFactor;

//    coordModels[1] = ((GLfloat)height()/2-(cameraParameters[4]+m_pBackgroundTexture->height()*frame_picture_Ratio)*cameraParameters[7])*(-coordModels[2])/(cameraParameters[1]*cameraParameters[7]);
//    coordModels[0] = ((GLfloat)width()/2-coordModels[1]*cameraParameters[2]*cameraParameters[7]/(-coordModels[2])-(cameraParameters[3]+m_pBackgroundTexture->width()*frame_picture_Ratio)*cameraParameters[7])*(-coordModels[2])/(cameraParameters[0]*cameraParameters[7]);

    coordModels << 0.0f, 0.0f, -0.3f;

    trackball.reset();

//    updateGL();
    update();
}

void GLWidget::saveObj()        // Saves the model with its transformations
{
    /*QString newModelName;
    int boucle = 0;

    do      // Checking if selected file is the source file
    {
      if(boucle!=0)
           QMessageBox::warning(this,"Source file selected!",
          "Please select a recording file different from the source file or create a new one.");
      boucle = 1;
      newModelName = QFileDialog::getSaveFileName(this, "Save File", "/home/ismael/Qt", "3D Object (*.obj)");
    }while(newModelName==model.getModelName());

    model.saveModel(newModelName,trackball.rotation(),coordModels);*/
}

void GLWidget::rotateX()        // Rotates the model at 360° around X axis
{
    QQuaternion r = trackball.rotation();
    QVector3D orthonormal(1,0,0);
    QMatrix4x4 m;

    m.rotate(r);
    orthonormal =  m.transposed() * orthonormal;

    m_RotationAngle = 0.0f;
    m_RotationQuat = r;
    m_RotationAxis = orthonormal;
    m_RotationTimer.start(10);
    m_RotatingCamera = true;

    m_PrevCameraRotation.setToIdentity();
    m_PrevCameraRotation.rotate(r);
}

void GLWidget::rotateY()        // Rotates the model at 360° around Y axis
{
    QQuaternion r = trackball.rotation();
    QVector3D orthonormal(0,1,0);
    QMatrix4x4 m;

    m.rotate(r);
    orthonormal = m.transposed() * orthonormal;

    m_RotationAngle = 0.0f;
    m_RotationQuat = r;
    m_RotationAxis = orthonormal;
    m_RotationTimer.start(10);
    m_RotatingCamera = true;

    m_PrevCameraRotation.setToIdentity();
    m_PrevCameraRotation.rotate(r);
}

void GLWidget::Rotate()
{
    m_RotationAngle += 5.0f * rotationSpeed;

    if (m_RotationAngle >= 360.0f)
    {
        m_RotationAngle = 0.0f;
        m_RotatingCamera = false;
        m_RotationTimer.stop();
        trackball.setRotation(m_RotationQuat);
    }
    else
    {
        trackball.setRotation(QQuaternion::fromAxisAndAngle(m_RotationAxis, -m_RotationAngle));
    }

    update();
}

/* ============================ OTHER MODELS ============================ */
void GLWidget::createTumor(bool buttonChecked)
{
    tumorMode = buttonChecked;

    if(tumorMode)
    {
        distanceMode = false;
        emit distanceModeIsON(false);
    }

    tumor = glGenLists(1);
    GLUquadric* params = gluNewQuadric();
    gluQuadricDrawStyle(params, GLU_FILL);

    glNewList(tumor, GL_COMPILE);
        gluSphere(params, tumorRadius/scaleFactor, 20, 20);
    glEndList();

//    updateGL();
    update();
}
void GLWidget::createCrosshair(QPointF screenCoordinates)
{
    crosshair = glGenLists(1);
    glLineWidth(2);
    glNewList(crosshair, GL_COMPILE);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width(), 0, height(), -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glBegin(GL_LINES);
            glVertex2f(screenCoordinates.x() - 7, height()-screenCoordinates.y());
            glVertex2f(screenCoordinates.x() + 7, height()-screenCoordinates.y());
            glVertex2f(screenCoordinates.x(), height()-screenCoordinates.y() + 7);
            glVertex2f(screenCoordinates.x(), height()-screenCoordinates.y() - 7);
        glEnd();
    glEndList();

//    updateGL();
    update();
}
void GLWidget::resetTumor()
{
    coordTumor = QVector3D(0,0,0);
    tumor = 0;
}

//void GLWidget::UpdateTumorTransform(void)
//{
//    // Update the tumor model transform w.r.t. the liver model.
//    Eigen::Vector3f currentPos(coordTumor.x(), coordTumor.y(), coordTumor.z());
//    Eigen::Vector3f vec = currentPos - m_ModelCentroid;

//    for (Model& model : m_Models)
//    {
//        model.ComputeCentroid();
//    }

//    ComputeModelCentroid();

//    coordTumor.setX(m_ModelCentroid[0] + vec[0]);
//    coordTumor.setY(m_ModelCentroid[1] + vec[1]);
//    coordTumor.setZ(m_ModelCentroid[2] + vec[2]);
//}

void GLWidget::setDistanceMode(bool buttonChecked)
{
    distanceMode = buttonChecked;

    if(distanceMode)
    {
        tumorMode = false;
        emit tumorModeIsON(false);
    }
}
void GLWidget::createTags(QPointF screenCoordinates)
{
    tags = glGenLists(1);
    GLUquadric* params = gluNewQuadric();
    gluQuadricDrawStyle(params, GLU_FILL);

    QMatrix4x4 m;
    m.rotate(trackball.rotation());


    if(distanceCoordinates1.isNull())   // First tag
    {
        distanceBetweenTags = 0;
        distanceCoordinates1 = screenToModelPixel(screenCoordinates);
        distanceCoordinates1 = m.transposed() * QVector3D(distanceCoordinates1.x(), distanceCoordinates1.y(), distanceCoordinates1.z());

        glNewList(tags, GL_COMPILE);
            glTranslatef(distanceCoordinates1.x(), distanceCoordinates1.y(), distanceCoordinates1.z());
            gluSphere(params, tagsRadius/scaleFactor, 20, 20);
        glEndList();
    }
    else    // First and second tags
    {
        GLfloat radius = tagsRadius/scaleFactor;
        QVector3D distanceCoordinates2 = screenToModelPixel(screenCoordinates);
        distanceCoordinates2 = m.transposed() * QVector3D(distanceCoordinates2.x(), distanceCoordinates2.y(), distanceCoordinates2.z());

        glNewList(tags, GL_COMPILE);
            glPushMatrix();
                glTranslatef(distanceCoordinates1.x(), distanceCoordinates1.y(), distanceCoordinates1.z());
                gluSphere(params, radius, 20, 20);
            glPopMatrix();

            glTranslatef(distanceCoordinates2.x(), distanceCoordinates2.y(), distanceCoordinates2.z());
            gluSphere(params, radius, 20, 20);
        glEndList();

        distanceBetweenTags = sqrt(pow(distanceCoordinates1.x()-distanceCoordinates2.x(),2)
                +pow(distanceCoordinates1.y()-distanceCoordinates2.y(),2)
                +pow(distanceCoordinates1.z()-distanceCoordinates2.z(),2));

        distanceMode = false;
        distanceCoordinates1 = QVector3D(0,0,0);
        emit distanceModeIsON(false);
    }

//    updateGL();
    update();
}

/************** For generating training set for CNNs. **************/

void GLWidget::DistributePointsOnSphere(const unsigned int NumOfPoints, const unsigned int Index, float& Latitude, float& Longitude)
{
    Latitude = acos(-1.0f + (float)(2 * Index + 1) / (float)NumOfPoints);
    Longitude = sqrt((float)NumOfPoints * M_PI) * Latitude;

//    std::cout << "Lat: " << (90.0f - Latitude * 180.0f / PI) << ", lon: " << (180.0f + fmod(Longitude * 180.0f / PI, 360.0f)) << std::endl;
}

void GLWidget::RandomlyDeformModel(void)
{
    // Load the initial undeformed configuration of the mesh.
    QString fileName = QString("./../tensorflow/liver_data/fine_registration/initial_model");
    LoadModelData(fileName, false);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(0, 3);

    // Randomly rotate the initial mesh.
    QVector3D rotationAxis;
    rotationAxis.setX(dist(gen) / 3.0f - 0.5f);
    rotationAxis.setY(dist(gen) / 3.0f - 0.5f);
    rotationAxis.setZ(dist(gen) / 3.0f - 0.5f);
    rotationAxis.normalize();

    QMatrix4x4 rotationMat;
    rotationMat.setToIdentity();
    rotationMat.rotate((dist(gen) / 3.0f) * 180.0f, rotationAxis);
    m_LoadedModelView = m_LoadedModelView * rotationMat;

    QMatrix3x3 rotMat;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            rotMat(i, j) = m_LoadedModelView(i, j);
        }
    }

    trackball.setRotation(QQuaternion::fromRotationMatrix(rotMat));


    // Project vertices of the model on 2D.
    std::vector<Point> points2D;
    m_ModelForSimulation.ProjectVerticesOnto2D(points2D, m_LoadedModelView, m_proj, m_Viewport);

    // There are three cases to deform the mesh:
    enum DEFORM_TYPE
    {
        DEFORM_TYPE_BOTH_ENDS, // Pull/push both ends of the mesh.
        DEFORM_TYPE_ONE_END,   // Pull/push one end of the mesh while the other end is fixed.
        DEFORM_TYPE_REGION     // Pull/push a region while both ends of the mesh are fixed.
    };

    DEFORM_TYPE type = (DEFORM_TYPE)floor(dist(gen));

    std::cout << "Deformation type: " << (int)type << std::endl;

    // Select a number of vertices on the left/right end of the mesh.
    int index = 0;
    std::vector<std::pair<float, int> > pairs;

    for (Point& point : points2D)
    {
        pairs.push_back(std::pair<float, int>(point.x(), index));

        ++index;
    }

    std::sort(pairs.begin(), pairs.end());

    std::vector<std::pair<float, int> >::iterator it = pairs.begin();
    float offset = (dist(gen) / 3.0f) * 0.05f;
    float sign = dist(gen) / 3.0f;

    if (sign < 0.5f)
    {
        // Pull.
        sign = -1.0f;
    }
    else
    {
        // Push.
        sign = 1.0;
    }

    int numOfSelectedVertices = 10;

    switch (type)
    {
        case DEFORM_TYPE_BOTH_ENDS:
        {
            // Choose the left vertices.
            for (int i = 0; i < numOfSelectedVertices; ++i)
            {
                m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                QVector3D vertex;
                QVector4D vertex3D;

                Eigen::Vector3f& pos = m_ModelForSimulation.Vertices()[it->second]->_Pos;
                vertex3D.setX(pos[0]);
                vertex3D.setY(pos[1]);
                vertex3D.setZ(pos[2]);
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView * vertex3D;
                vertex3D /= vertex3D.w();
                vertex.setX(vertex3D.x());
                vertex.setY(vertex3D.y());
                vertex.setZ(vertex3D.z());

                if (sign > 0.0f)
                {
                    // Push the ends.
                    vertex.setX(vertex.x() + offset);
                }
                else
                {
                    // Pull the ends.
                    vertex.setX(vertex.x() - offset);
                }

                vertex3D.setX(vertex.x());
                vertex3D.setY(vertex.y());
                vertex3D.setZ(vertex.z());
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView.inverted() * vertex3D;
                vertex3D /= vertex3D.w();

                pos[0] = vertex3D.x();
                pos[1] = vertex3D.y();
                pos[2] = vertex3D.z();

                ++it;
            }

            // Choose the right vertices.
            it = pairs.end() - 1;

            for (int i = numOfSelectedVertices; i > 0; --i)
            {
                m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                QVector3D vertex;
                QVector4D vertex3D;

                Eigen::Vector3f& pos = m_ModelForSimulation.Vertices()[it->second]->_Pos;
                vertex3D.setX(pos[0]);
                vertex3D.setY(pos[1]);
                vertex3D.setZ(pos[2]);
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView * vertex3D;
                vertex3D /= vertex3D.w();
                vertex.setX(vertex3D.x());
                vertex.setY(vertex3D.y());
                vertex.setZ(vertex3D.z());

                if (sign > 0.0f)
                {
                    // Push the ends.
                    vertex.setX(vertex.x() - offset);
                }
                else
                {
                    // Pull the ends.
                    vertex.setX(vertex.x() + offset);
                }

                vertex3D.setX(vertex.x());
                vertex3D.setY(vertex.y());
                vertex3D.setZ(vertex.z());
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView.inverted() * vertex3D;
                vertex3D /= vertex3D.w();

                pos[0] = vertex3D.x();
                pos[1] = vertex3D.y();
                pos[2] = vertex3D.z();

                --it;
            }
        }
        break;
        case DEFORM_TYPE_ONE_END:
        {
            std::vector<Model::Vertex*> movingVertices;

            float end = dist(gen) / 3.0f;

            if (end < 0.5f)
            {
                // Fix the left end of the mesh.
                // Choose the left vertices.
                for (int i = 0; i < numOfSelectedVertices * 20; ++i)
                {
                    m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                    m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                    ++it;
                }

                // Choose the right vertices to be moved.
                it = pairs.end() - 1;

                for (int i = numOfSelectedVertices; i > 0; --i)
                {
                    m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                    m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                    movingVertices.push_back(m_ModelForSimulation.Vertices()[it->second]);

                    --it;
                }
            }
            else
            {
                // Fix the right end of the mesh.
                // Choose the right vertices.
                it = pairs.end() - 1;

                for (int i = numOfSelectedVertices * 20; i > 0; --i)
                {
                    m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                    m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                    --it;
                }

                // Choose the left vertices to be moved.
                it = pairs.begin();

                for (int i = 0; i < numOfSelectedVertices; ++i)
                {
                    m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                    m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                    movingVertices.push_back(m_ModelForSimulation.Vertices()[it->second]);

                    ++it;
                }
            }

            // Translate the end.

            // Get a translation vector.
            QVector3D axis;
            axis.setX(dist(gen) / 3.0f - 0.5f);
            axis.setY(dist(gen) / 3.0f - 0.5f);
            axis.setZ(dist(gen) / 3.0f - 0.5f);
            axis.normalize();
            axis *= offset;

            for (Model::Vertex* pVertex : movingVertices)
            {
                QVector3D vertex;
                QVector4D vertex3D;

                Eigen::Vector3f& pos = pVertex->_Pos;
                vertex3D.setX(pos[0]);
                vertex3D.setY(pos[1]);
                vertex3D.setZ(pos[2]);
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView * vertex3D;
                vertex3D /= vertex3D.w();
                vertex.setX(vertex3D.x());
                vertex.setY(vertex3D.y());
                vertex.setZ(vertex3D.z());

                vertex += axis;

                vertex3D.setX(vertex.x());
                vertex3D.setY(vertex.y());
                vertex3D.setZ(vertex.z());
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView.inverted() * vertex3D;
                vertex3D /= vertex3D.w();

                pos[0] = vertex3D.x();
                pos[1] = vertex3D.y();
                pos[2] = vertex3D.z();
            }
        }
        break;
        case DEFORM_TYPE_REGION:
        {
            // Choose the left vertices.
            for (int i = 0; i < numOfSelectedVertices; ++i)
            {
                m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                ++it;
            }

            // Choose the right vertices.
            it = pairs.end() - 1;

            for (int i = numOfSelectedVertices; i > 0; --i)
            {
                m_ModelForSimulation.Vertices()[it->second]->_Selected = true;
                m_ModelForSimulation.Vertices()[it->second]->_Moved = true;

                --it;
            }

            // Choose a region to rotate & translate.
            bool selected = false;
            std::vector<Model::Vertex*> neighbours;

            while (!selected)
            {
                neighbours.clear();
                int index = floor((dist(gen) / 3.0f) * m_ModelForSimulation.Vertices().size());

                Model::Vertex* pVertex = m_ModelForSimulation.Vertices()[index];
                m_ModelForSimulation.OneRingNeighbours(pVertex, neighbours);
                selected = true;

                for (Model::Vertex* pVertex : neighbours)
                {
                    if (pVertex->_Selected)
                    {
                        selected = false;

                        break;
                    }
                }
            }

            // Rotate & translate the region.
            std::vector<QVector3D> vertices;

            for (Model::Vertex* pVertex : neighbours)
            {
                pVertex->_Selected = true;
                pVertex->_Moved = true;

                QVector3D vertex;
                QVector4D vertex3D;

                Eigen::Vector3f& pos = pVertex->_Pos;
                vertex3D.setX(pos[0]);
                vertex3D.setY(pos[1]);
                vertex3D.setZ(pos[2]);
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView * vertex3D;
                vertex3D /= vertex3D.w();
                vertex.setX(vertex3D.x());
                vertex.setY(vertex3D.y());
                vertex.setZ(vertex3D.z());

                vertices.push_back(vertex);
            }

            // Rotate the region around the axis (from centre vertex to one of the neighbouring vertex) and translate.
            QVector3D centreVertex = vertices[0];

            int index = 0;

            do
            {
                index = floor((dist(gen) / 3.0f) * (float)neighbours.size());
            }
            while (index == 0);

            QVector3D axis = vertices[index] - centreVertex;
            axis.normalize();

            float angle = (dist(gen) / 3.0f - 0.5f) * 180.0f;

            QMatrix4x4 mat;
            mat.setToIdentity();
            mat.translate(centreVertex);
            mat.rotate(angle, axis);
            mat.translate(-centreVertex);

            // Get a translation vector.
            axis.setX(dist(gen) / 3.0f - 0.5f);
            axis.setY(dist(gen) / 3.0f - 0.5f);
            axis.setZ(dist(gen) / 3.0f - 0.5f);
            axis.normalize();
            axis *= offset;

            index = 0;

            for (QVector3D& vertex : vertices)
            {
                vertex = mat * vertex;
                vertex += axis;

                QVector4D vertex3D;
                vertex3D.setX(vertex.x());
                vertex3D.setY(vertex.y());
                vertex3D.setZ(vertex.z());
                vertex3D.setW(1.0);

                vertex3D = m_LoadedModelView.inverted() * vertex3D;
                vertex3D /= vertex3D.w();

                Eigen::Vector3f& pos = neighbours[index]->_Pos;
                pos[0] = vertex3D.x();
                pos[1] = vertex3D.y();
                pos[2] = vertex3D.z();

                ++index;
            }
        }
        break;
        default:
        break;
    }

    // Deform the model.
    m_MaterialModelSolver.SetNumOfInnerIterations(4);
    m_MaterialModelSolver.SetNumOfOuterIterations(400);

//        m_MaterialModelSolver.Solve();
    m_MaterialModelSolver.SolveCUDA();
    m_MaterialModelSolver.SetAllParticlesUnMoved();
    m_ModelForSimulation.UnselectAllVertices();

//        UpdateTumorTransform();
    UpdateModel();
    update();

    // Save an image.
    QImage image = this->grabFramebuffer();
    QString format = "png";
    image.save(QString("./../tensorflow/liver_data/images/liver_%1.").arg(m_TrainingImageIndex) + format, qPrintable(format));

    (*m_pDataGenerationFileStream) << m_TrainingImageIndex;

    QVector4D vertex3D;

    for (Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
    {
        vertex3D.setX(pVertex->_Pos[0]);
        vertex3D.setY(pVertex->_Pos[1]);
        vertex3D.setZ(pVertex->_Pos[2]);
        vertex3D.setW(1.0);

        vertex3D = m_LoadedModelView * vertex3D;
        vertex3D /= vertex3D.w();

        (*m_pDataGenerationFileStream) << "," << vertex3D.x() << "," << vertex3D.y() << "," << vertex3D.z();
    }

    (*m_pDataGenerationFileStream) << endl;

    ++m_TrainingImageIndex;

    if (m_TrainingImageIndex == m_NumOfTrainingImages)
    {
        m_DataGenerationTimer.stop();

        std::cout << "Generating deformed model training images finished." << std::endl;
    }
}

void GLWidget::GenerateTrainingSet(void)
{
    std::cout << "Generating training images starts." << std::endl;

    m_TrainingImageIndex = 0;
    m_FrameCount = 0;
    m_GeneratingTrainingSet = true;
    GLdouble modelViewMatrix[16];

    QString fileName = "./../tensorflow/liver_data/camera_transforms_dual_quaternion.txt";
    QFile file(fileName);
    file.open(QIODevice::ReadWrite);
    QTextStream stream(&file);

    // Random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 0.05);

    unsigned int numOfTrainingImagesPerRadius = m_NumOfTrainingImages / 1000;

    for (unsigned int j = 0; j < numOfTrainingImagesPerRadius; ++j)
    {
        m_CameraRadius = 0.35f - j * (0.15f / (float)numOfTrainingImagesPerRadius) * 1.5f; // Closest: 0.15f;
        m_PointsOnSphereIndex = 1000 * 2 - 1;

        for (int i = 0; i < 1000; ++i)
        {
            m_CameraPositionNoise << dist(gen) - 0.025, dist(gen) - 0.025, dist(gen) - 0.025;
            m_CameraLookAtNoise << dist(gen) - 0.025, dist(gen) - 0.025, dist(gen) - 0.025;
            m_CameraRollNoise = (dist(gen) - 0.025) * 20.0;

//            std::cout << "m_CameraPositionNoise: " << m_CameraPositionNoise << "m_CameraLookAtNoise: " << m_CameraLookAtNoise << std::endl;

//            updateGL();
            update();

            // Save an image.
            QImage image = this->grabFramebuffer();
            QString format = "png";
            image.save(QString("./../tensorflow/liver_data/images/liver_%1.").arg(m_TrainingImageIndex) + format, qPrintable(format));

            // Save camera extrinsic matrix as dual quaternion.
            // First, convert the angle axis representation of the rotation into quaternion and save it.
            Eigen::Quaternionf quat(m_CameraRotation);

            stream << i + j * 1000;
            stream << "," << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z();

            // Save the camera position.
            stream << "," << m_CameraPosition[0] << "," << m_CameraPosition[1] << "," << m_CameraPosition[2];
            stream << endl;

            ++m_TrainingImageIndex;
            --m_PointsOnSphereIndex;
        }
    }

    m_GeneratingTrainingSet = false;

    std::cout << "Generating training images finished." << std::endl;
}

void GLWidget::GenerateTestSet(void)
{
    std::cout << "Generating test images starts." << std::endl;

    m_TrainingImageIndex = 0;
    m_FrameCount = 0;
    m_GeneratingTrainingSet = true;
    GLdouble modelViewMatrix[16];

    QString fileName = "./../tensorflow/liver_data/camera_transforms_dual_quaternion_test.txt";
    QFile file(fileName);
    file.open(QIODevice::ReadWrite);
    QTextStream stream(&file);

    // Random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 0.05);

    unsigned int numOfTrainingImagesPerRadius = m_NumOfTrainingImages / 100;

    for (unsigned int j = 0; j < numOfTrainingImagesPerRadius; ++j)
    {
        m_CameraRadius = 0.2f - j * (0.1f / (float)numOfTrainingImagesPerRadius) * 1.0f; // Closest: 0.15f;
        m_PointsOnSphereIndex = 100 * 2 - 1;

        for (int i = 0; i < 100; ++i)
        {
            m_CameraPositionNoise << dist(gen) - 0.025, dist(gen) - 0.025, dist(gen) - 0.025;
            m_CameraLookAtNoise << dist(gen) - 0.025, dist(gen) - 0.025, dist(gen) - 0.025;
            m_CameraRollNoise = (dist(gen) - 0.025) * 20.0;

//            std::cout << "m_CameraPositionNoise: " << m_CameraPositionNoise << "m_CameraLookAtNoise: " << m_CameraLookAtNoise << std::endl;

//            updateGL();
            update();

            // Save an image.
            QImage image = this->grabFramebuffer();
            QString format = "png";
            image.save(QString("./../tensorflow/liver_data/test_images/liver_%1.").arg(m_TrainingImageIndex) + format, qPrintable(format));

            // Save camera extrinsic matrix as dual quaternion.
            // First, convert the angle axis representation of the rotation into quaternion and save it.
            Eigen::Quaternionf quat(m_CameraRotation);

            stream << i + j * 100;
            stream << "," << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z();

            // Save the camera position.
            stream << "," << m_CameraPosition[0] << "," << m_CameraPosition[1] << "," << m_CameraPosition[2];
            stream << endl;

            ++m_TrainingImageIndex;
            --m_PointsOnSphereIndex;
        }
    }

    m_GeneratingTrainingSet = false;

    std::cout << "Generating test images finished." << std::endl;
}

void GLWidget::GenerateDeformedModelTrainingSet(void)
{
    std::cout << "Generating deformed model training images starts." << std::endl;

    m_TrainingImageIndex = 0;

    if (m_pDataGenerationFile)
    {
        delete m_pDataGenerationFile;
    }

    m_pDataGenerationFile = new QFile(QString("./../tensorflow/liver_data/model_vertices.txt"));
    m_pDataGenerationFile->open(QIODevice::ReadWrite);
    m_pDataGenerationFile->resize(0);

    if (m_pDataGenerationFileStream)
    {
        delete m_pDataGenerationFileStream;
    }

    m_pDataGenerationFileStream = new QTextStream(m_pDataGenerationFile);

    m_DataGenerationTimer.start(1);
}

/****************************/

void GLWidget::GetContour(std::vector<Eigen::Vector2f>& Contour)
{
    int numOfPixels = this->width() * this->height();
    GLfloat winZ[numOfPixels];

    // First, segment the model using depth values.
    glReadPixels(0, 0, this->width(), this->height(), GL_DEPTH_COMPONENT, GL_FLOAT, winZ);

    for (int i = 0; i < numOfPixels; ++i)
    {
        if (winZ[i] < 1.0f)
        {
            // Model.
            winZ[i] = 1.0f;
        }
        else
        {
            // Background.
            winZ[i] = 0.0f;
        }
    }

    // Extract the contour using Canny edge detector.
    cv::Mat src(this->height(), this->width(), CV_8U);

    for (int row = 0; row < this->height(); ++row)
    {
        for (int col = 0; col < this->width(); ++col)
        {
            src.at<unsigned char>(row, col) = (unsigned char)(255.0f * winZ[col + this->width() * row]);
        }
    }

    int lowThreshold = 50;
    int ratio = 3;
    int kernel_size = 3;

    Contour.clear();

    cv::Canny(src, src, lowThreshold, lowThreshold * ratio, kernel_size);

    for (int row = 0; row < this->height(); ++row)
    {
        for (int col = 0; col < this->width(); ++col)
        {
            if ((unsigned int)src.at<unsigned char>(row, col) == 255)
            {
                Contour.push_back(Eigen::Vector2f(col, row));
            }
        }
    }
}

void GLWidget::GetContourFromImage(const QImage& Image, std::vector<Eigen::Vector2f>& Contour)
{
    // Extract the contour using Canny edge detector.
    cv::Mat src(Image.height(), Image.width(), CV_8U);

    for (int row = 0; row < Image.height(); ++row)
    {
        for (int col = 0; col < Image.width(); ++col)
        {
            src.at<unsigned char>(row, col) = (unsigned char)(Image.pixelColor(col, Image.height() - 1 - row)).blue();
        }
    }

    int lowThreshold = 50;
    int ratio = 3;
    int kernel_size = 3;

    Contour.clear();

    cv::Canny(src, src, lowThreshold, lowThreshold * ratio, kernel_size);

    for (int row = 0; row < Image.height(); ++row)
    {
        for (int col = 0; col < Image.width(); ++col)
        {
            if ((unsigned int)src.at<unsigned char>(row, col) == 255)
            {
                Contour.push_back(Eigen::Vector2f(col, row));
            }
        }
    }
}

bool GLWidget::InsideFOV(Eigen::Vector2f& Point2D)
{
    float scale = frame_picture_Ratio * cameraParameters[7];
    GLint viewport[4] = { m_InputImage.width() * scale, m_InputImage.height() * scale,
                          (GLint)width() - (2 * m_InputImage.width() * scale), (GLint)height() - (2 * m_InputImage.height() * scale) };

    if (Point2D[0] >= viewport[0] + viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0] && Point2D[0] < viewport[0] + viewport[2] - viewport[2] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[0]
     && Point2D[1] >= viewport[1] + viewport[3] * (1.0f - m_FOVScale) * 0.5f  + m_FOVPosOffset[1] && Point2D[1] < viewport[1] + viewport[3] - viewport[3] * (1.0f - m_FOVScale) * 0.5f + m_FOVPosOffset[1])
    {
        return true;
    }
    else
    {
        return false;
    }
}

void GLWidget::FilterInvisibleFacesInModel()
{
    std::vector<QVector3D> points2D, points3D, visiblePixelCoords;
    QVector3D point2D, point3D;
    GLfloat pDepthData[m_Viewport[2] * m_Viewport[3]];
    std::vector<Model::Vertex*> vertices;

    makeCurrent();

    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, pDepthData);

    doneCurrent();

    for (Model& model : m_Models)
    {
        vertices = model.Vertices();

        for (Model::Face* pFace : model.Faces())
        {
            pFace->_Sampled = false;

            if (!Model::FaceFrontFacing(pFace, m_ModelView, m_proj, m_Viewport))
            {
                continue;
            }

            for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
            {
                point3D = QVector3D(vertices[*it]->_Pos[0], vertices[*it]->_Pos[1], vertices[*it]->_Pos[2]);
                point2D = Model::ProjectPointOnto2D(point3D, m_ModelView, m_proj, m_Viewport);

                points3D.push_back(point3D);
                points2D.push_back(point2D);
            }

            // Check if the face is completely out of the viewport.
            bool outOfViewport = true;

            for (QVector3D& point : points2D)
            {
                if (point.x() >= 0.0 && point.x() < m_Viewport[2] && point.y() >= 0.0 && point.y() < m_Viewport[3])
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

            Model::SamplePointsOnFace(points2D, points3D, visiblePixelCoords, pDepthData, m_ModelView, m_proj, m_Viewport);

            if (visiblePixelCoords.size() == 0)
            {
                points2D.clear();
                points3D.clear();

                continue;
            }

            pFace->_Sampled = true;

            points2D.clear();
            points3D.clear();
            visiblePixelCoords.clear();
        }
    }
}

void GLWidget::MoveClosestVerticesOnMeshToContour(bool Using2DSearch)
{
    // Find the closest vertices on the contour of the mesh to the ground truth model contour.
//    std::map<int, int> closestVertexIndicesToContour;
//    int i = 0;
    QVector3D near, far;
    GLfloat scale = frame_picture_Ratio * cameraParameters[7];
    Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

    m_ContourOffsetVectors.clear();

    std::vector<int> contourProcessedIndices;
//    std::vector<Model::Face*> faces;

    // Choose vertices, on the contour of the model, which are closest to the points on the ground truth model contour.
    m_MaterialModelSolver.SetAllParticlesUnMoved();

//    std::vector<Eigen::Vector2f> contourPoints = m_GroundTruthModelContour;

//    Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());
//    float scale = frame_picture_Ratio * cameraParameters[7];

//    for (Eigen::Vector2f& point : contourPoints)
//    {
//            point *= cameraParameters[7];
//            point += scale * dimension;
//    }

    for (std::map<QString, QString>::const_iterator it = m_ModelToImageContourMap.begin(); it != m_ModelToImageContourMap.end(); ++it)
    {
        std::vector<std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> > modelContour = m_ModelContours[m_ModelContoursList.indexOf(it->first)];
        std::vector<Eigen::Vector2f> imageContour = m_ImageContours[m_ImageContoursList.indexOf(it->second)];

        switch (std::get<2>(modelContour[0]))
        {
            case MODEL_CONTOUR_TYPE_NULL:
            {
            }
                break;
            case MODEL_CONTOUR_TYPE_FRONTIER:
            case MODEL_CONTOUR_TYPE_LIGAMENT:
            {
                int minIndex = -1;

                for (std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> mc : modelContour)
                {
                    Model::Vertex* pVertex = m_Models[std::get<1>(mc)].Vertices()[std::get<0>(mc)];

//                    faces.clear();

//                    if (!pVertex->_Moved)
                    {
//                        m_Models[std::get<1>(mc)].OneRingNeighbourFaces(pVertex, faces);

//                        bool visible = false;

//                        for (Model::Face* pFace : faces)
//                        {
//                            if (pFace->_Sampled)
//                            {
//                                visible = true;

//                                break;
//                            }
//                        }

//                        if (!visible)
//                        {
//                            continue;
//                        }

                        Eigen::Vector2f newPoint, p;
                        Point point2D = m_Models[std::get<1>(mc)].ProjectVertexOnto2D(pVertex, m_ModelView, m_proj, m_Viewport);
                        p << point2D.x(), point2D.y();

#if 1
                        // Fix the start/end contour points.
                        if (std::get<2>(modelContour[0]) == MODEL_CONTOUR_TYPE_FRONTIER && m_FrontierContourFixedPoints.find(pVertex) != m_FrontierContourFixedPoints.end())
                        {
                            newPoint = m_FrontierContourFixedPoints[pVertex];

                            newPoint *= cameraParameters[7];
                            newPoint += scale * dimension;

                            if (m_FrontierContourFixedPoints.begin()->first == pVertex)
                            {
                                minIndex = 0;
                            }
                            else
                            {
                                minIndex = imageContour.size() - 1;
                            }

                            // For displaying contour offsets.
                            m_ContourOffsetVectors.push_back(p);
                            m_ContourOffsetVectors.push_back(newPoint);

                            // Move the vertex on the mesh to the point on the contour.
                            Utils::Unproject2DPointOnto3D(newPoint[0], newPoint[1], 0.0f, m_ModelView, m_proj, m_Viewport, near);
                            Utils::Unproject2DPointOnto3D(newPoint[0], newPoint[1], 1.0f, m_ModelView, m_proj, m_Viewport, far);
                        }
                        else
#endif
                        {
                            int index = 0;
                            float minDist = std::numeric_limits<float>::max();

            //                if (!InsideFOV(p))
            //                {
            //                    continue;
            //                }

                            std::vector<Eigen::Vector2f> scaledImageContour;

                             for (Eigen::Vector2f point : imageContour)
                             {
                                 point *= cameraParameters[7];
                                 point += scale * dimension;

                                 scaledImageContour.push_back(point);

#if 0
                                 if (index == 0 || index == imageContour.size() - 1)
                                 {
                                     ++index;

                                     continue;
                                 }
#endif

//                                 // Use only a subset of points on the contour.
//                                 if (index % 5 > 0)
//                                 {
//                                     ++index;

//                                     continue;
//                                 }

                                 if (InsideFOV(point))
                                 {
                                     // TODO: Temp - currently no 1D search.
                                     if (0)//!Using2DSearch)
                                     {
                                         // Search along the vertex normal direction for the closest contour point.
                                         QVector3D normal(pVertex->_Normal[0], pVertex->_Normal[1], pVertex->_Normal[2]);
                                         QVector3D pos3D(pVertex->_Pos[0], pVertex->_Pos[1], pVertex->_Pos[2]);
                                         normal += pos3D;
                                         normal = Model::ProjectPointOnto2D(normal, m_ModelView, m_proj, m_Viewport);

                                         normal.setX(normal.x() - p[0]);
                                         normal.setY(normal.y() - p[1]);

                                         float norm = sqrt(normal.x() * normal.x() + normal.y() * normal.y());
                                         normal.setX(normal.x() / norm);
                                         normal.setY(normal.y() / norm);
                                         normal *= 350.f * cameraParameters[7];

                                         Eigen::Vector2f lineSegmentStart(p[0] - normal.x(), p[1] - normal.y());
                                         Eigen::Vector2f lineSegmentEnd(p[0] + normal.x(), p[1] + normal.y());

                                         if (Utils::PointOnLineSegment(point, lineSegmentStart, lineSegmentEnd))
                                         {
                                             float dist = (point - p).norm();

                                             if (dist < minDist)
                                             {
                                                 // The point on the image contour is on the vertex normal line segment.
                                                 if (std::find(contourProcessedIndices.begin(), contourProcessedIndices.end(), index) == contourProcessedIndices.end())
                                                 {
                                                     minDist = dist;
                                                     minIndex = index;
                                                 }
                                             }
                                         }
                                     }
                                     else
                                     {
                                         // 2D search for the closest point.
                                         float dist = (point - p).norm();

                                         if (dist < minDist)
                                         {
                                             if (std::find(contourProcessedIndices.begin(), contourProcessedIndices.end(), index) == contourProcessedIndices.end())
                                             {
                                                 minDist = dist;
                                                 minIndex = index;
                                             }
                                         }
                                     }
                                 }

                                 ++index;
                             }

                             if (minIndex == -1)
                             {
                                 continue;
                             }

                             // TODO: Temp - Currently, no 1D search
//                             if (!Using2DSearch && (minDist >= 350.0f * cameraParameters[7]))
//                             {
//                                 continue;
//                             }

                             // This is to address the zigzag vertices on the frontier contour.
                             // Do not project vertices close to the image contour.
//                             if (std::get<2>(modelContour[0]) == MODEL_CONTOUR_TYPE_FRONTIER)
                             {
                                 if (minDist <= 50.0f * cameraParameters[7])
                                 {
                                     continue;
                                 }
                             }

                            // For displaying contour offsets.
                            m_ContourOffsetVectors.push_back(p);
                            m_ContourOffsetVectors.push_back(scaledImageContour[minIndex]);

                            // Move the vertex on the mesh to the point on the contour.
                            Utils::Unproject2DPointOnto3D((scaledImageContour[minIndex])[0], (scaledImageContour[minIndex])[1], 0.0f, m_ModelView, m_proj, m_Viewport, near);
                            Utils::Unproject2DPointOnto3D((scaledImageContour[minIndex])[0], (scaledImageContour[minIndex])[1], 1.0f, m_ModelView, m_proj, m_Viewport, far);
                        }

                        Eigen::Vector3f point3D(far.x() - near.x(), far.y() - near.y(), far.z() - near.z());
                        point3D.normalize();

                        // Project the corresponding vertice onto the vector from the 3D contour point.
                        Eigen::Vector3f vertex = pVertex->_Pos;
                        vertex[0] -= near.x();
                        vertex[1] -= near.y();
                        vertex[2] -= near.z();

                        float norm = vertex.dot(point3D);
                        Eigen::Vector3f newPos = norm * point3D;
                        newPos[0] += near.x();
                        newPos[1] += near.y();
                        newPos[2] += near.z();

                        pVertex->_Pos = newPos;
                        pVertex->_Moved = true;

                        contourProcessedIndices.push_back(minIndex);
                    }
                }
            }
                break;
            case MODEL_CONTOUR_TYPE_OCCLUDING:
            {
                // Project the occluding contour vertices of the model on 2D. TODO: Refactor this later because this projection is computed again when the contour is computed in the next lines.
                std::vector<Point> points2D;

                for (Model& model : m_Models)
                {
                    model.ProjectVerticesOnto2D(points2D, m_ModelView, m_proj, m_Viewport);
                }

                // Find the contour vertices of the model.
            //    m_Models[0].ExtractVerticesOnContour(m_ModelView, m_proj, m_Viewport);
                makeCurrent();
                GetContour(m_ModelContour);
                doneCurrent();

                m_ModelForSimulation.SelectVerticesOnContour(m_ModelContour, m_ModelView, m_proj, m_Viewport);

//                for (std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> mc : modelContour)
                for (Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
                {
//                    Model::Vertex* pVertex = m_Models[std::get<1>(mc)].Vertices()[std::get<0>(mc)];

    //                    faces.clear();

    //                    if (!pVertex->_Moved)
                    if (pVertex->_Selected)
                    {
    //                        m_Models[std::get<1>(mc)].OneRingNeighbourFaces(pVertex, faces);

    //                        bool visible = false;

    //                        for (Model::Face* pFace : faces)
    //                        {
    //                            if (pFace->_Sampled)
    //                            {
    //                                visible = true;

    //                                break;
    //                            }
    //                        }

    //                        if (!visible)
    //                        {
    //                            continue;
    //                        }

                        int index = 0;
                        int minIndex = -1;
                        Eigen::Vector2f p;
                        float minDist = std::numeric_limits<float>::max();

//                        Point point2D = m_Models[std::get<1>(mc)].ProjectVertexOnto2D(pVertex, m_ModelView, m_proj, m_Viewport);
                        Point point2D = m_ModelForSimulation.ProjectVertexOnto2D(pVertex, m_ModelView, m_proj, m_Viewport);
                        p << point2D.x(), point2D.y();

        //                if (!InsideFOV(p))
        //                {
        //                    continue;
        //                }

                        std::vector<Eigen::Vector2f> scaledImageContour;

                        for (Eigen::Vector2f point : imageContour)
                        {
                            point *= cameraParameters[7];
                            point += scale * dimension;

                            scaledImageContour.push_back(point);

//                            // Use only a subset of points on the contour.
//                            if (index % 5 > 0)
//                            {
//                                ++index;

//                                continue;
//                            }

                            if (InsideFOV(point))
                            {
                                // Search along the vertex normal direction for the closest contour point.
                                QVector3D normal(pVertex->_Normal[0], pVertex->_Normal[1], pVertex->_Normal[2]);
                                QVector3D pos3D(pVertex->_Pos[0], pVertex->_Pos[1], pVertex->_Pos[2]);
                                normal += pos3D;
                                normal = Model::ProjectPointOnto2D(normal, m_ModelView, m_proj, m_Viewport);

                                normal.setX(normal.x() - p[0]);
                                normal.setY(normal.y() - p[1]);

                                float norm = sqrt(normal.x() * normal.x() + normal.y() * normal.y());
                                normal.setX(normal.x() / norm);
                                normal.setY(normal.y() / norm);
                                normal *= 150.f * cameraParameters[7];

                                Eigen::Vector2f lineSegmentStart(p[0] - normal.x(), p[1] - normal.y());
                                Eigen::Vector2f lineSegmentEnd(p[0] + normal.x(), p[1] + normal.y());

                                if (Utils::PointOnLineSegment(point, lineSegmentStart, lineSegmentEnd))
                                {
                                    float dist = (point - p).norm();

                                    if (dist < minDist)
                                    {
                                        // The point on the image contour is on the vertex normal line segment.
                                        if (std::find(contourProcessedIndices.begin(), contourProcessedIndices.end(), index) == contourProcessedIndices.end())
                                        {
                                            minDist = dist;
                                            minIndex = index;
                                        }
                                    }
                                }
                            }

                            ++index;
                        }

                        if (minIndex == -1)
                        {
                            continue;
                        }

                        if (minDist >= 150.0f * cameraParameters[7])
                        {
                            continue;
                        }

                        // For displaying contour offsets.
                        m_ContourOffsetVectors.push_back(p);
                        m_ContourOffsetVectors.push_back(scaledImageContour[minIndex]);

                        // Move the vertex on the mesh to the point on the contour.
                        Utils::Unproject2DPointOnto3D((scaledImageContour[minIndex])[0], (scaledImageContour[minIndex])[1], 0.0f, m_ModelView, m_proj, m_Viewport, near);
                        Utils::Unproject2DPointOnto3D((scaledImageContour[minIndex])[0], (scaledImageContour[minIndex])[1], 1.0f, m_ModelView, m_proj, m_Viewport, far);

                        Eigen::Vector3f point3D(far.x() - near.x(), far.y() - near.y(), far.z() - near.z());
                        point3D.normalize();

                        // Project the corresponding vertice onto the vector from the 3D contour point.
                        Eigen::Vector3f vertex = pVertex->_Pos;
                        vertex[0] -= near.x();
                        vertex[1] -= near.y();
                        vertex[2] -= near.z();

                        float norm = vertex.dot(point3D);
                        Eigen::Vector3f newPos = norm * point3D;
                        newPos[0] += near.x();
                        newPos[1] += near.y();
                        newPos[2] += near.z();

                        pVertex->_Pos = newPos;
                        pVertex->_Moved = true;

                        contourProcessedIndices.push_back(minIndex);
                    }
                }
            }
                break;
            default:
                break;
        }

//        if (it->second == m_ImageContoursList.at(m_CheckedImageContours.at(i)))
//        {
//            QString key = it->first;

//            std::cout << "Remove model to image contour map - key: " << key.toUtf8().constData() << ", value: " << m_ModelToImageContourMap[key].toUtf8().constData() << std::endl;

//            m_ModelToImageContourMap.erase(key);

//            break;
//        }



//        if (m_ModelToImageContourMap.find(m_ModelContoursList.at(Index)) != m_ModelToImageContourMap.end())
//        {
//            int index = m_ImageContoursList.indexOf(m_ModelToImageContourMap.at(m_ModelContoursList.at(Index)));
//            ShowImageContour(index);
//        }



//        for (std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE> item : m_ModelContours[Index])
//        {
//            m_Models[std::get<1>(item)].Vertices()[std::get<0>(item)]->_Selected = true;
//        }
    }





//    for (Model& model : m_Models)
//    {
//        for (Model::Vertex*& pVertex : model.Vertices())
//        {
//            faces.clear();

//            if (pVertex->_Selected && !pVertex->_Moved)
//            {
//                model.OneRingNeighbourFaces(pVertex, faces);

//                bool visible = false;

//                for (Model::Face* pFace : faces)
//                {
//                    if (pFace->_Sampled)
//                    {
//                        visible = true;

//                        break;
//                    }
//                }

//                if (!visible)
//                {
//                    continue;
//                }

//                int index = 0;
//                int minIndex = 0;
//                Eigen::Vector2f p;
//                float minDist = std::numeric_limits<float>::max();

//                Point point2D = model.ProjectVertexOnto2D(pVertex, m_ModelView, m_proj, m_Viewport);
//                p << point2D.x(), point2D.y();

////                if (!InsideFOV(p))
////                {
////                    continue;
////                }

//                 bool skipping = false;

//                 for (Eigen::Vector2f& point : contourPoints)
//                 {
//                     if (InsideFOV(point))
//                     {
//                         float dist = (point - p).norm();

//                         if (dist < minDist)
//                         {
//                             if (std::find(contourProcessedIndices.begin(), contourProcessedIndices.end(), index) == contourProcessedIndices.end())
//                             {
//                                 minDist = dist;
//                                 minIndex = index;
//                             }
//                             else
//                             {
//                                 skipping = true;

//                                 break;
//                             }
//                         }
//                     }

//                     ++index;
//                 }

//                 if (skipping)
//                 {
//                     continue;
//                 }

//    //            if (minDist >= meanDistances)
////                 if (minDist >= 300.0f * cameraParameters[7])
////                {
////                    continue;
////                }

//                // For displaying contour offsets.
//                m_ContourOffsetVectors.push_back(p);
//                m_ContourOffsetVectors.push_back(contourPoints[minIndex]);

//                // Move the vertex on the mesh to the point on the contour.
//                Utils::Unproject2DPointOnto3D((contourPoints[minIndex])[0], (contourPoints[minIndex])[1], 0.0f, m_ModelView, m_proj, m_Viewport, near);
//                Utils::Unproject2DPointOnto3D((contourPoints[minIndex])[0], (contourPoints[minIndex])[1], 1.0f, m_ModelView, m_proj, m_Viewport, far);

//                Eigen::Vector3f point3D(far.x() - near.x(), far.y() - near.y(), far.z() - near.z());
//                point3D.normalize();

//                // Project the corresponding vertice onto the vector from the 3D contour point.
//                Eigen::Vector3f vertex = pVertex->_Pos;
//                vertex[0] -= near.x();
//                vertex[1] -= near.y();
//                vertex[2] -= near.z();

//                float norm = vertex.dot(point3D);
//                Eigen::Vector3f newPos = norm * point3D;
//                newPos[0] += near.x();
//                newPos[1] += near.y();
//                newPos[2] += near.z();

//                pVertex->_Pos = newPos;
//                pVertex->_Moved = true;

//                contourProcessedIndices.push_back(minIndex);
//            }
//        }
//    }

//    m_FreeVertexIndicesInModel.clear();

//    for (Model& model : m_Models)
//    {
//        int index = 0;

//        for (Model::Vertex*& vertex : model.Vertices())
//        {
//            if (!vertex->_Moved)
//            {
//                m_FreeVertexIndicesInModel.push_back(index);
//            }

//            ++index;
//        }
//    }

    UpdateModel();
    update();
}

void GLWidget::PreCameraCalibration()
{
    // Compute c = l (light intensity) * k (camera response) * a (albedo) for the sysnthetic input images.
    // N.B. For now, we do not use the rendered image by OpenGL as the reflectance values by it differs from the Lambertian equation.
    // I suspect that we compute the reflectance using the face normal but OpenGL interpolate the normal from the vertex normals consisting the face.
    // And, therefore, we set c to some value ourselves.
    std::vector<QVector3D> points2D, points3D, sampledPixelCoords, sampledPoints3D;
    QVector3D point2D, point3D;
    GLfloat depthData[m_Viewport[2] * m_Viewport[3]];
    std::vector<GLfloat> reflectances, inputImageReflectances;
    float RMSE = 0.0f;
    int count = 0;
    std::vector<float> cs;

    m_allSampledPixelCoords.clear();
    m_allSampledPoints3D.clear();

    makeCurrent();

    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, depthData);

    doneCurrent();

    for (Model& model : m_Models)
    {
        for (Model::Face*& pFace : model.Faces())
        {
            pFace->_Sampled = false;

            if (!Model::FaceFrontFacing(pFace, m_ModelView, m_proj, m_Viewport))
            {
                continue;
            }

            QVector3D Q(pFace->_Centroid[0], pFace->_Centroid[1], pFace->_Centroid[2]);
            QVector3D centre2D = Model::ProjectPointOnto2D(Q, m_ModelView, m_proj, m_Viewport);
            Eigen::Vector2f point(centre2D.x(), centre2D.y());

            float scale = frame_picture_Ratio * cameraParameters[7];
            Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());
            Eigen::Vector2f regionCentre = m_ShadingOptimisationRegionCentre;
            regionCentre *= cameraParameters[7];
            regionCentre += scale * dimension;

            // Skip if the face is out of the specified optimisation region.
            if ((point - regionCentre).squaredNorm() > m_ShadingOptimisationRegionRadius * m_ShadingOptimisationRegionRadius)
            {
                continue;
            }

            bool skipping = false;

            for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
            {
                if (model.Vertices()[*it]->_Moved)
                {
                    // Skip the fixed vertices on the contour.
                    skipping = true;

                    break;
                }

                point3D = QVector3D(model.Vertices()[*it]->_Pos[0], model.Vertices()[*it]->_Pos[1], model.Vertices()[*it]->_Pos[2]);
                point2D = Model::ProjectPointOnto2D(point3D, m_ModelView, m_proj, m_Viewport);

                points3D.push_back(point3D);
                points2D.push_back(point2D);
            }

            if (skipping)
            {
                points2D.clear();
                points3D.clear();

                continue;
            }

            // Check if the face is completely out of the viewport.
            bool outOfViewport = true;

            for (QVector3D& point : points2D)
            {
                if (point.x() >= 0.0 && point.x() < m_Viewport[2] && point.y() >= 0.0 && point.y() < m_Viewport[3])
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

            sampledPoints3D = Model::SamplePointsOnFace(points2D, points3D, sampledPixelCoords, depthData, m_ModelView, m_proj, m_Viewport);

            if (sampledPixelCoords.size() > 0)
            {
                pFace->_Sampled = true;

                m_allSampledPoints3D.push_back(sampledPoints3D);
                m_allSampledPixelCoords.push_back(sampledPixelCoords);

                // Read the reflectance of the input image at the location corresponding to the face centre.
//                QVector3D Q(pFace->_Centroid[0], pFace->_Centroid[1], pFace->_Centroid[2]);
//                QVector3D centre2D = Model::ProjectPointOnto2D(Q, m_ModelView, m_proj, m_Viewport);
    //            QVector3D normal3D(pFace->_Normal[0], pFace->_Normal[1], pFace->_Normal[2]);

                Eigen::Vector2f ct(centre2D.x(), centre2D.y());

                if (!InsideFOV(ct))
                {
                    points2D.clear();
                    points3D.clear();
                    sampledPixelCoords.clear();

                    continue;
                }

                QVector4D centre3D(Q.x(), Q.y(), Q.z(), 1.0);
                QVector4D normal((pFace->_Normal)[0], (pFace->_Normal)[1], (pFace->_Normal)[2], 0.0);

                centre3D = m_ModelView * centre3D;
                centre3D /= centre3D.w();
                Q.setX(centre3D.x());
                Q.setY(centre3D.y());
                Q.setZ(centre3D.z());

                QMatrix3x3 normalMat = m_ModelView.normalMatrix();
                QMatrix4x4 mat(normalMat(0, 0), normalMat(0, 1), normalMat(0, 2), 0.0,
                               normalMat(1, 0), normalMat(1, 1), normalMat(1, 2), 0.0,
                               normalMat(2, 0), normalMat(2, 1), normalMat(2, 2), 0.0,
                               0.0            , 0.0            , 0.0            , 1.0);
                normal = mat * normal;
                QVector3D n(normal.x(), normal.y(), normal.z());
                n.normalize();

                QVector3D q = Q.normalized();

                // Reflectance of the face in the model.
    //            float I = -m_c * QVector3D::dotProduct(n, q) / Q.lengthSquared();

    #if USING_SYNTHETIC_INPUT_IMAGE

                // Make a triangle in the image, corresponding the face in the model,
                // and compute the reflectance at the face centre in the input image.
                Eigen::Vector3f a = model.Vertices()[pFace->_VertexIndices[0]]->_Pos;
                Eigen::Vector3f b = model.Vertices()[pFace->_VertexIndices[1]]->_Pos;
                Eigen::Vector3f c = model.Vertices()[pFace->_VertexIndices[2]]->_Pos;

                QVector3D A(a[0], a[1], a[2]);
                QVector3D B(b[0], b[1], b[2]);
                QVector3D C(c[0], c[1], c[2]);

                QVector3D a2D = Model::ProjectPointOnto2D(A, m_ModelView, m_proj, m_Viewport);
                QVector3D b2D = Model::ProjectPointOnto2D(B, m_ModelView, m_proj, m_Viewport);
                QVector3D c2D = Model::ProjectPointOnto2D(C, m_ModelView, m_proj, m_Viewport);

                GLfloat aZ = m_pGroundTruthDepthData[(int)a2D.x() + (int)a2D.y() * m_Viewport[2]];
                GLfloat bZ = m_pGroundTruthDepthData[(int)b2D.x() + (int)b2D.y() * m_Viewport[2]];
                GLfloat cZ = m_pGroundTruthDepthData[(int)c2D.x() + (int)c2D.y() * m_Viewport[2]];

                if (aZ >= 1.0f || bZ >= 1.0f || cZ >= 1.0f)
                {
                    points2D.clear();
                    points3D.clear();
                    sampledPixelCoords.clear();

                    continue;
                }

                QVector3D a3D, b3D, c3D;
                Utils::Unproject2DPointOnto3D(a2D.x(), a2D.y(), aZ, m_ModelView, m_proj, m_Viewport, a3D);
                Utils::Unproject2DPointOnto3D(b2D.x(), b2D.y(), bZ, m_ModelView, m_proj, m_Viewport, b3D);
                Utils::Unproject2DPointOnto3D(c2D.x(), c2D.y(), cZ, m_ModelView, m_proj, m_Viewport, c3D);

                QVector3D QStar = (a3D + b3D + c3D) / 3.0;
                QVector4D centre(QStar.x(), QStar.y(), QStar.z(), 1.0);

                //std::cout << "pFace->Centroid: " << pFace->_Centroid[0] << ", " << pFace->_Centroid[1] << ", " << pFace->_Centroid[2] << std::endl << "QStar: " << QStar.x() << ", " << QStar.y() << ", " << QStar.z() << std::endl;
    //            std::cout << "GT - model: " << m_pGroundTruthDepthData[(int)a2D.x() + (int)a2D.y() * m_Viewport[2]] - a2D.z() << std::endl;

                QVector3D nStar = QVector3D::crossProduct(b3D - a3D, c3D - a3D);
                nStar.normalize();

                QVector4D normal3D(nStar.x(), nStar.y(), nStar.z(), 0.0);

                centre = m_ModelView * centre;
                centre /= centre.w();
                QStar.setX(centre.x());
                QStar.setY(centre.y());
                QStar.setZ(centre.z());

                normal3D = mat * normal3D;
                nStar.setX(normal3D.x());
                nStar.setY(normal3D.y());
                nStar.setZ(normal3D.z());
                nStar.normalize();

                QVector3D qStar = QStar.normalized();

//                float IStar = -m_c * QVector3D::dotProduct(nStar, qStar) / QStar.lengthSquared();
                float IStar = -0.015f * QVector3D::dotProduct(nStar, qStar) / QStar.lengthSquared();
    #else

    //            float IStar = (m_InputImage.pixelColor((int)centre2D.x(), m_Viewport[3] - (int)centre2D.y())).blueF();

                // Compute the median reflectance on the face of the model and in the input image.
                for (QVector3D& coord : sampledPixelCoords)
                {
                    Eigen::Vector2f p(coord.x(), coord.y());

                    if (!InsideFOV(p))
                    {
                        continue;
                    }

    //                reflectances.push_back(currentFrameBuffer[(int)coord.x() + (int)coord.y() * m_Viewport[2]]);


                    float scale = frame_picture_Ratio * cameraParameters[7];
                    Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

                    p -= scale * dimension;
                    p /= cameraParameters[7];

//                    GLint viewport[4] = { m_InputImage.width() * scale, m_InputImage.height() * scale,
//                                          (GLint)width() - (2 * m_InputImage.width() * scale), (GLint)height() - (2 * m_InputImage.height() * scale) };

                    inputImageReflectances.push_back((m_InputImageMedianFilteredY.pixelColor((int)p[0], dimension[1] - 1 - (int)p[1])).blueF());

    //                std::cout << "Model: " << currentFrameBuffer[(int)coord.x() + (int)coord.y() * m_Viewport[2]] << ", input image: " << (m_InputImageMedianFilteredY.pixelColor((int)coord.x(), m_Viewport[3] - (int)coord.y())).blueF() << std::endl;
                }

    //            std::nth_element(reflectances.begin(), reflectances.begin() + reflectances.size() / 2, reflectances.end());
    //            float I = reflectances[reflectances.size() / 2];

    //            // Median reflectance from the input image.
                std::nth_element(inputImageReflectances.begin(), inputImageReflectances.begin() + inputImageReflectances.size() / 2, inputImageReflectances.end());
                float IStar = inputImageReflectances[inputImageReflectances.size() / 2];
    #endif

                float cc = -IStar * Q.lengthSquared() / QVector3D::dotProduct(n, q);
                cs.push_back(cc);

                //            std::cout << "I: " << I << ", IStar: " << IStar << std::endl;

    //            // Compute the error.
    //            float error = IStar - I;
    //            RMSE += error * error;
            }

            points2D.clear();
            points3D.clear();
            sampledPixelCoords.clear();
            inputImageReflectances.clear();
    //        reflectances.clear();
        }
    }

    m_c = 0.0f;

    for (float c : cs)
    {
        m_c += c;
    }

    m_c /= (float)cs.size();

    std::cout << "Computed c: " << m_c << std::endl;


// Not using this.
#if 0

    // Sample points on the faces which are projected onto 2D from the model.
    std::vector<QVector3D> points2D, points3D, sampledPixelCoords, sampledPoints3D;
    QVector3D point2D, point3D;
    GLfloat depthData[m_Viewport[2] * m_Viewport[3]];
    GLfloat currentFrameBuffer[m_Viewport[2] * m_Viewport[3]]; // Reflectance (in greyscale) read from the current framebuffer.

    m_allSampledPixelCoords.clear();
    m_allSampledPoints3D.clear();

    makeCurrent();

    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, depthData);
    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_BLUE, GL_FLOAT, currentFrameBuffer);

    int num = 0;

    for (Model::Face*& pFace : m_Models[0].Faces())
    {
        pFace->_Sampled = false;

        if (!Model::FaceFrontFacing(pFace, m_ModelView, m_proj, m_Viewport))
        {
            continue;
        }

        for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
        {
            point3D = QVector3D(m_Models[0].Vertices()[*it]->_Pos[0], m_Models[0].Vertices()[*it]->_Pos[1], m_Models[0].Vertices()[*it]->_Pos[2]);
            point2D = Model::ProjectPointOnto2D(point3D, m_ModelView, m_proj, m_Viewport);

            points3D.push_back(point3D);
            points2D.push_back(point2D);
        }

        // Check if the face is completely out of the viewport.
        bool outOfViewport = true;

        for (QVector3D& point : points2D)
        {
            if (point.x() >= 0.0 && point.x() < m_Viewport[2] && point.y() >= 0.0 && point.y() < m_Viewport[3])
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

        sampledPoints3D = Model::SamplePointsOnFace(points2D, points3D, sampledPixelCoords, depthData, m_ModelView, m_proj, m_Viewport);

        if (sampledPixelCoords.size() > 0)
        {
            pFace->_Sampled = true;

            m_allSampledPoints3D.push_back(sampledPoints3D);
            m_allSampledPixelCoords.push_back(sampledPixelCoords);

            ++num;
        }

        points2D.clear();
        points3D.clear();
        sampledPixelCoords.clear();
    }

    // Compute one c from each face and get the median.
    int index = 0;
    std::vector<float> cs;

    for (Model::Face*& pFace : m_Models[0].Faces())
    {
        if (pFace->_Sampled)
        {
            // Just choose the first sampled point.
            QVector3D coord = (m_allSampledPixelCoords[index]).front();
            QVector3D point3D = (m_allSampledPoints3D[index]).front();
            QVector4D point(point3D.x(), point3D.y(), point3D.z(), 1.0);
            QVector4D normal((pFace->_Normal)[0], (pFace->_Normal)[1], (pFace->_Normal)[2], 0.0);

            point = m_ModelView * point;
            point /= point.w();
            point3D.setX(point.x());
            point3D.setY(point.y());
            point3D.setZ(point.z());

            QMatrix3x3 normalMat = m_ModelView.normalMatrix();
            QMatrix4x4 mat(normalMat(0, 0), normalMat(0, 1), normalMat(0, 2), 0.0,
                           normalMat(1, 0), normalMat(1, 1), normalMat(1, 2), 0.0,
                           normalMat(2, 0), normalMat(2, 1), normalMat(2, 2), 0.0,
                           0.0            , 0.0            , 0.0            , 1.0);
            normal = mat * normal;
            QVector3D normal3D(normal.x(), normal.y(), normal.z());
            normal3D.normalize();

            QVector3D eyeDirection = point3D.normalized();

            float I = currentFrameBuffer[(int)coord.x() + (int)coord.y() * m_Viewport[2]];

            float c = -(I * point3D.lengthSquared()) / QVector3D::dotProduct(normal3D, eyeDirection);

            std::cout << "I: " << I << ", computed I: " << -QVector3D::dotProduct(normal3D, eyeDirection) * 0.025 / point3D.lengthSquared() << ", c: " << c << ", coord: " << (int)coord.x() << ", " << (int)coord.y() << std::endl;

            cs.push_back(c);

//            reflectances.push_back(currentFrameBuffer[(int)coord.x() + (int)coord.y() * m_Viewport[2]]);


    //        std::nth_element(reflectances.begin(), reflectances.begin() + reflectances.size() / 2, reflectances.end());
    //        float median = reflectances[reflectances.size() / 2];

            // Get a relfectance of the input image at the centre

            // TODO: Not complete.

            ++index;
        }

    }

    // Get the median c.
    std::nth_element(cs.begin(), cs.begin() + cs.size() / 2, cs.end());
    m_c = cs[cs.size() / 2];
    std::cout << "Median c: " << m_c << std::endl;

    doneCurrent();

#endif

}

void GLWidget::OptimiseMeshWithShading(bool UpdatingDepth)
{
    // Rotate each face to match the reflectance, or move it to match the depth, of the pixels projected from the model to those of the input image.
    // First, sample points on the faces which are projected onto 2D from the model
    // and select a pixel with the median reflectance for each face.
    std::vector<QVector3D> points2D, points3D, sampledPixelCoords, sampledPoints3D;
    QVector3D point2D, point3D;
    GLfloat depthData[m_Viewport[2] * m_Viewport[3]];
//    GLfloat currentFrameBuffer[m_Viewport[2] * m_Viewport[3]]; // Reflectance (in greyscale) read from the current framebuffer.
    std::vector<GLfloat> reflectances, inputImageReflectances;
    float RMSE = 0.0f;
    int count = 0;

    // Compute c.
//    PreCameraCalibration();

    m_allSampledPixelCoords.clear();
    m_allSampledPoints3D.clear();

    makeCurrent();

    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, depthData);
//    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_BLUE, GL_FLOAT, currentFrameBuffer);

    doneCurrent();

    int index = 0;

    for (Model& model : m_Models)
    {
        for (Model::Face*& pFace : model.Faces())
        {
            ++index;
            pFace->_Sampled = false;

            if (!Model::FaceFrontFacing(pFace, m_ModelView, m_proj, m_Viewport))
            {
                continue;
            }

            QVector3D Q(pFace->_Centroid[0], pFace->_Centroid[1], pFace->_Centroid[2]);
            QVector3D centre2D = Model::ProjectPointOnto2D(Q, m_ModelView, m_proj, m_Viewport);
            Eigen::Vector2f point(centre2D.x(), centre2D.y());

            float scale = frame_picture_Ratio * cameraParameters[7];
            Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());
            Eigen::Vector2f regionCentre = m_ShadingOptimisationRegionCentre;
            regionCentre *= cameraParameters[7];
            regionCentre += scale * dimension;

            // Skip if the face is out of the specified optimisation region.
            if ((point - regionCentre).squaredNorm() > m_ShadingOptimisationRegionRadius * m_ShadingOptimisationRegionRadius)
            {
                continue;
            }

            bool skipping = false;

            for (std::vector<int>::iterator it = pFace->_VertexIndices.begin(); it != pFace->_VertexIndices.end(); ++it)
            {
                if (model.Vertices()[*it]->_Moved)
                {
                    // Skip the fixed vertices on the contour.
                    skipping = true;

                    break;
                }

                point3D = QVector3D(model.Vertices()[*it]->_Pos[0], model.Vertices()[*it]->_Pos[1], model.Vertices()[*it]->_Pos[2]);
                point2D = Model::ProjectPointOnto2D(point3D, m_ModelView, m_proj, m_Viewport);

                points3D.push_back(point3D);
                points2D.push_back(point2D);
            }

            if (skipping)
            {
                points2D.clear();
                points3D.clear();

                continue;
            }

            // Check if the face is completely out of the viewport.
            bool outOfViewport = true;

            for (QVector3D& point : points2D)
            {
                if (point.x() >= 0.0 && point.x() < m_Viewport[2] && point.y() >= 0.0 && point.y() < m_Viewport[3])
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

            sampledPoints3D = Model::SamplePointsOnFace(points2D, points3D, sampledPixelCoords, depthData, m_ModelView, m_proj, m_Viewport);

            if (sampledPixelCoords.size() > 0)
            {
                pFace->_Sampled = true;

                m_allSampledPoints3D.push_back(sampledPoints3D);
                m_allSampledPixelCoords.push_back(sampledPixelCoords);

                // Read the reflectance of the input image at the location corresponding to the face centre.
                QVector3D Q(pFace->_Centroid[0], pFace->_Centroid[1], pFace->_Centroid[2]);
                QVector3D centre2D = Model::ProjectPointOnto2D(Q, m_ModelView, m_proj, m_Viewport);
    //            QVector3D normal3D(pFace->_Normal[0], pFace->_Normal[1], pFace->_Normal[2]);

                Eigen::Vector2f ct(centre2D.x(), centre2D.y());

                if (!InsideFOV(ct))
                {
                    points2D.clear();
                    points3D.clear();
                    sampledPixelCoords.clear();

                    continue;
                }

                QVector4D centre3D(Q.x(), Q.y(), Q.z(), 1.0);
                QVector4D normal((pFace->_Normal)[0], (pFace->_Normal)[1], (pFace->_Normal)[2], 0.0);

                centre3D = m_ModelView * centre3D;
                centre3D /= centre3D.w();
                Q.setX(centre3D.x());
                Q.setY(centre3D.y());
                Q.setZ(centre3D.z());

                QMatrix3x3 normalMat = m_ModelView.normalMatrix();
                QMatrix4x4 mat(normalMat(0, 0), normalMat(0, 1), normalMat(0, 2), 0.0,
                               normalMat(1, 0), normalMat(1, 1), normalMat(1, 2), 0.0,
                               normalMat(2, 0), normalMat(2, 1), normalMat(2, 2), 0.0,
                               0.0            , 0.0            , 0.0            , 1.0);
                normal = mat * normal;
                QVector3D n(normal.x(), normal.y(), normal.z());
                n.normalize();

                QVector3D q = Q.normalized();


    #if USING_SYNTHETIC_INPUT_IMAGE

                // Make a triangle in the image, corresponding the face in the model,
                // and compute the reflectance at the face centre in the input image.
                Eigen::Vector3f a = model.Vertices()[pFace->_VertexIndices[0]]->_Pos;
                Eigen::Vector3f b = model.Vertices()[pFace->_VertexIndices[1]]->_Pos;
                Eigen::Vector3f c = model.Vertices()[pFace->_VertexIndices[2]]->_Pos;

                QVector3D A(a[0], a[1], a[2]);
                QVector3D B(b[0], b[1], b[2]);
                QVector3D C(c[0], c[1], c[2]);

                QVector3D a2D = Model::ProjectPointOnto2D(A, m_ModelView, m_proj, m_Viewport);
                QVector3D b2D = Model::ProjectPointOnto2D(B, m_ModelView, m_proj, m_Viewport);
                QVector3D c2D = Model::ProjectPointOnto2D(C, m_ModelView, m_proj, m_Viewport);

                GLfloat aZ = m_pGroundTruthDepthData[(int)a2D.x() + (int)a2D.y() * m_Viewport[2]];
                GLfloat bZ = m_pGroundTruthDepthData[(int)b2D.x() + (int)b2D.y() * m_Viewport[2]];
                GLfloat cZ = m_pGroundTruthDepthData[(int)c2D.x() + (int)c2D.y() * m_Viewport[2]];

                if (aZ >= 1.0f || bZ >= 1.0f || cZ >= 1.0f)
                {
                    points2D.clear();
                    points3D.clear();
                    sampledPixelCoords.clear();

                    continue;
                }

                QVector3D a3D, b3D, c3D;
                Utils::Unproject2DPointOnto3D(a2D.x(), a2D.y(), aZ, m_ModelView, m_proj, m_Viewport, a3D);
                Utils::Unproject2DPointOnto3D(b2D.x(), b2D.y(), bZ, m_ModelView, m_proj, m_Viewport, b3D);
                Utils::Unproject2DPointOnto3D(c2D.x(), c2D.y(), cZ, m_ModelView, m_proj, m_Viewport, c3D);

                QVector3D QStar = (a3D + b3D + c3D) / 3.0;
                QVector4D centre(QStar.x(), QStar.y(), QStar.z(), 1.0);

                //std::cout << "pFace->Centroid: " << pFace->_Centroid[0] << ", " << pFace->_Centroid[1] << ", " << pFace->_Centroid[2] << std::endl << "QStar: " << QStar.x() << ", " << QStar.y() << ", " << QStar.z() << std::endl;
    //            std::cout << "GT - model: " << m_pGroundTruthDepthData[(int)a2D.x() + (int)a2D.y() * m_Viewport[2]] - a2D.z() << std::endl;

                QVector3D nStar = QVector3D::crossProduct(b3D - a3D, c3D - a3D);
                nStar.normalize();

                QVector4D normal3D(nStar.x(), nStar.y(), nStar.z(), 0.0);

                centre = m_ModelView * centre;
                centre /= centre.w();
                QStar.setX(centre.x());
                QStar.setY(centre.y());
                QStar.setZ(centre.z());

                normal3D = mat * normal3D;
                nStar.setX(normal3D.x());
                nStar.setY(normal3D.y());
                nStar.setZ(normal3D.z());
                nStar.normalize();

                QVector3D qStar = QStar.normalized();

                float IStar = -0.015f * QVector3D::dotProduct(nStar, qStar) / QStar.lengthSquared();


    //            QVector3D near, far;
    //            Utils::Unproject2DPointOnto3D(centre2D.x(), centre2D.y(), 0.0f, m_ModelView, m_proj, m_Viewport, near);
    //            Utils::Unproject2DPointOnto3D(centre2D.x(), centre2D.y(), 1.0f, m_ModelView, m_proj, m_Viewport, far);

    //            QVector3D direction = far - near;
    //            QVector3D intersection;
    //            direction.normalize();
    //            float t = 0.0f;

    //            QVector4D faceCentre, faceNormal;
    //            std::vector<Model::Face*> intersectingFaces;

    //            for (Model::Face*& pGTFace : m_GroundTruthModel.Faces())
    //            {
    //                Eigen::Vector3f a = m_GroundTruthModel.Vertices()[pGTFace->_VertexIndices[0]]->_Pos;
    //                Eigen::Vector3f b = m_GroundTruthModel.Vertices()[pGTFace->_VertexIndices[1]]->_Pos;
    //                Eigen::Vector3f c = m_GroundTruthModel.Vertices()[pGTFace->_VertexIndices[2]]->_Pos;

    //                QVector3D A(a[0], a[1], a[2]);
    //                QVector3D B(b[0], b[1], b[2]);
    //                QVector3D C(c[0], c[1], c[2]);

    //                if (Utils::TriangleRayIntersection(A, B, C, near, direction, &t) == 1)
    //                {
    //                    // There is an intersection.
    //                    intersection = near + t * direction;

    //                    QVector3D point = Model::ProjectPointOnto2D(intersection, m_ModelView, m_proj, m_Viewport);

    //        //            std::cout << "pixel depth: " << pDepthData[(int)pixelCoord.x() + (int)pixelCoord.y() * pViewport[2]] << ", face depth: " << point.z() << std::endl;

    //                    if (fabs(depthData[(int)centre2D.x() + (int)centre2D.y() * m_Viewport[2]] - point.z()) >= 1e-5)
    //                    {
    //                        // The pixel is covered.
    //                        continue;
    //                    }

    //                    intersectingFaces(pGTFace);


    //                    break;
    //                }
    //            }


    //            faceCentre.setX(pGTFace->_Centroid[0]);
    //            faceCentre.setY(pGTFace->_Centroid[1]);
    //            faceCentre.setZ(pGTFace->_Centroid[2]);
    //            faceCentre.setW(1.0);

    //            faceNormal.setX((pGTFace->_Normal)[0]);
    //            faceNormal.setY((pGTFace->_Normal)[1]);
    //            faceNormal.setZ((pGTFace->_Normal)[2]);
    //            faceNormal.setW(0.0);


    //            QVector3D QStar;

    //            faceCentre = m_ModelView * faceCentre;
    //            faceCentre /= faceCentre.w();
    //            QStar.setX(faceCentre.x());
    //            QStar.setY(faceCentre.y());
    //            QStar.setZ(faceCentre.z());

    ////            QMatrix3x3 normalMat = m_ModelView.normalMatrix();
    ////            QMatrix4x4 mat(normalMat(0, 0), normalMat(0, 1), normalMat(0, 2), 0.0,
    ////                           normalMat(1, 0), normalMat(1, 1), normalMat(1, 2), 0.0,
    ////                           normalMat(2, 0), normalMat(2, 1), normalMat(2, 2), 0.0,
    ////                           0.0            , 0.0            , 0.0            , 1.0);
    //            faceNormal = mat * faceNormal;
    //            QVector3D nStar(faceNormal.x(), faceNormal.y(), faceNormal.z());
    //            nStar.normalize();

    //            QVector3D qStar = QStar.normalized();



    #else

                // Compute the median reflectance on the face of the model and in the input image.
                for (QVector3D& coord : sampledPixelCoords)
                {
                    Eigen::Vector2f p(coord.x(), coord.y());

                    if (!InsideFOV(p))
                    {
                        continue;
                    }

                    float scale = frame_picture_Ratio * cameraParameters[7];
                    Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

                    p -= scale * dimension;
                    p /= cameraParameters[7];

//                    GLint viewport[4] = { m_InputImage.width() * scale, m_InputImage.height() * scale,
//                                          (GLint)width() - (2 * m_InputImage.width() * scale), (GLint)height() - (2 * m_InputImage.height() * scale) };

                    inputImageReflectances.push_back((m_InputImageMedianFilteredY.pixelColor((int)p[0], dimension[1] - 1 - (int)p[1])).blueF());

    //                std::cout << "Model: " << currentFrameBuffer[(int)coord.x() + (int)coord.y() * m_Viewport[2]] << ", input image: " << (m_InputImageMedianFilteredY.pixelColor((int)coord.x(), m_Viewport[3] - (int)coord.y())).blueF() << std::endl;
                }

    //            std::nth_element(reflectances.begin(), reflectances.begin() + reflectances.size() / 2, reflectances.end());
    //            float I = reflectances[reflectances.size() / 2];

    //            // Median reflectance from the input image.
                std::nth_element(inputImageReflectances.begin(), inputImageReflectances.begin() + inputImageReflectances.size() / 2, inputImageReflectances.end());
                float IStar = inputImageReflectances[inputImageReflectances.size() / 2];

    #endif

                float I = -m_c * QVector3D::dotProduct(n, q) / Q.lengthSquared();

//                std::cout << "I: " << I << ", IStar: " << IStar << ", c: " << cc << std::endl;


                // Compute the error.
                float error = IStar - I;
                RMSE += error * error;

    #if !USING_SYNTHETIC_INPUT_IMAGE

                // Save the reflectance error for each face.
    //            if (m_pExperimentResultsFileStream)
    //            {
    //                (*m_pExperimentResultsFileStream) << index - 1 << "," << error << endl;
    //            }

    #endif

    //            if (UpdatingDepth)
                if (m_UsingShadingOptimisation)
                {
                    // Move the centroid of the face so that the reflectance matched IStar.
                    float t = -m_c * QVector3D::dotProduct(n, q) / IStar;

                    if (t < 0.0f || IStar < 1e-1)
                    {
                        points2D.clear();
                        points3D.clear();
                        sampledPixelCoords.clear();
                        reflectances.clear();

                        ++count;

                        continue;
                    }

                    float diff = sqrt(t) - Q.length();
    //                float diff = QStar.length() - Q.length();

#if 0

                    // TODO: Temp for using random value for the depth correction.
                    static std::random_device rd;
                    static std::mt19937 gen(rd());
                    static std::uniform_real_distribution<> dist(0, 1);
                    diff = dist(gen) * 0.01f;
                    float sign = dist(gen);
                    if (sign < 0.5f)
                    {
                        sign = -1.0f;
                    }
                    else
                    {
                        sign = 1.0f;
                    }
                    diff *= sign;

#endif



                    QVector3D vec = q * diff * 0.1;

                    QVector3D vertex;
                    QVector4D vertex3D;

                    for (int i : {0, 1, 2})
                    {
                        Eigen::Vector3f& pos = model.Vertices()[pFace->_VertexIndices[i]]->_Pos;
                        vertex3D.setX(pos[0]);
                        vertex3D.setY(pos[1]);
                        vertex3D.setZ(pos[2]);
                        vertex3D.setW(1.0);

                        vertex3D = m_ModelView * vertex3D;
                        vertex3D /= vertex3D.w();
                        vertex.setX(vertex3D.x());
                        vertex.setY(vertex3D.y());
                        vertex.setZ(vertex3D.z());

                        vertex += vec;

                        vertex3D.setX(vertex.x());
                        vertex3D.setY(vertex.y());
                        vertex3D.setZ(vertex.z());
                        vertex3D.setW(1.0);

                        vertex3D = m_ModelView.inverted() * vertex3D;
                        vertex3D /= vertex3D.w();

                        pos[0] = vertex3D.x();
                        pos[1] = vertex3D.y();
                        pos[2] = vertex3D.z();
                    }

                    // Update Q.
                    Q += vec;
                }
    //            else
                // TODO: Temp - no orientation correction for now.
                if (0) //m_UsingShadingOptimisation)
                {
                    // Update the orientation of the face.
                    // For estimating the normal in the input image.
                    float cosAlphaStar = (IStar * Q.lengthSquared()) / m_c;

                    if (cosAlphaStar > 1.0f)
                    {
                        cosAlphaStar = 1.0f;
                    }
                    else if (cosAlphaStar < -1.0f)
                    {
                        cosAlphaStar = -1.0f;
                    }

                    // From the model.
                    float cosAlpha = QVector3D::dotProduct(n, -q);

                    float alphaStar = acos(cosAlphaStar);
                    float alpha = acos(cosAlpha);

                    // The angle between the normals of the face in the input image and model.
                    float theta =  0.005f * (alphaStar - alpha);

                    // Convert to degrees from radians.
                    theta *= 180.f / M_PI;

        //            std::cout << "cosAlphaStar: " << cosAlphaStar << ", cosAlpha: " << cosAlpha << ", theta (degrees): " << theta << std::endl;

                    // Rotate the face in the model by theta around the vector u.
                    QVector3D u = (QVector3D::crossProduct(n, -q)).normalized();
                    QVector3D vertex;
                    QVector4D vertex3D;

                    QMatrix4x4 rotation;
                    rotation.setToIdentity();
                    rotation.rotate(theta, u);

                    for (int i : {0, 1, 2})
                    {
                        Eigen::Vector3f& pos = model.Vertices()[pFace->_VertexIndices[i]]->_Pos;
                        vertex3D.setX(pos[0]);
                        vertex3D.setY(pos[1]);
                        vertex3D.setZ(pos[2]);
                        vertex3D.setW(1.0);

                        vertex3D = m_ModelView * vertex3D;
                        vertex3D /= vertex3D.w();
                        vertex.setX(vertex3D.x());
                        vertex.setY(vertex3D.y());
                        vertex.setZ(vertex3D.z());

                        vertex -= Q;

                        vertex = rotation * vertex;

                        vertex += Q;

                        vertex3D.setX(vertex.x());
                        vertex3D.setY(vertex.y());
                        vertex3D.setZ(vertex.z());
                        vertex3D.setW(1.0);

                        vertex3D = m_ModelView.inverted() * vertex3D;
                        vertex3D /= vertex3D.w();

                        pos[0] = vertex3D.x();
                        pos[1] = vertex3D.y();
                        pos[2] = vertex3D.z();

        //                m_Models[0].Vertices()[pFace->_VertexIndices[i]]->_Moved = true;
                    }
                }

                // TODO: compute rmse here!!

                ++count;


    //            makeCurrent();

    //            m_Models[0].ComputeFaceNormals();
    //            m_Models[0].ComputeVertexNormals();
    //            m_Models[0].BuildVertexData();
    //            UpdateModelVBO();

    //            doneCurrent();

    //            update();

    //            makeCurrent();

    //            glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, depthData);
    //            glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_BLUE, GL_FLOAT, currentFrameBuffer);

    //            doneCurrent();
            }

            points2D.clear();
            points3D.clear();
            sampledPixelCoords.clear();
            inputImageReflectances.clear();
            reflectances.clear();
        }
    }

    RMSE /= (float)count;
    RMSE = sqrt(RMSE);

#if !USING_SYNTHETIC_INPUT_IMAGE

    std::cout << "RMSE (shading optimisation): " << RMSE << std::endl;

    if (m_pExperimentResultsFileStream)
    {
        (*m_pExperimentResultsFileStream) << RMSE << ",-1" << endl;
    }

#endif

}

void GLWidget::SaveContourToFile(std::vector<Eigen::Vector2f>& Contour)
{
    // Save the contour to a file for computing Hausdorff distance.
    if (m_pExperimentResultsFileStream)
    {
        (*m_pExperimentResultsFileStream) << "-1," << Contour.size() << endl;

        for (const Eigen::Vector2f& point : Contour)
        {
            (*m_pExperimentResultsFileStream) << point[0] << "," << point[1] << endl;
        }
    }
}

void GLWidget::UpdateModel(bool FaceNormals, bool VertexNormals, bool FaceCentroids)
{
    int index = 0;

    for (Model& model : m_Models)
    {
        if (FaceNormals)
        {
            model.ComputeFaceNormals();
        }

        if (VertexNormals)
        {
            model.ComputeVertexNormals();
        }

        if (FaceCentroids)
        {
            model.ComputeFaceCentroids();
        }

        model.BuildVertexData();
        UpdateModelVBO(index);

        ++index;
    }

    ComputeModelCentroid();
}

void GLWidget::FineRegistration()
{
    int numOfIterationsStage1 = 20; //100; // This should be a multiple of 2.
    int numOfIterationsStage2 = 30; //150; //300 // Real image. 150 // Synthetic image ; // This should be a multiple of 3.
    static bool s_UpdatingDepth = false;

    switch (m_FineRegistrationStage)
    {
        case 0:
        {
            // First stage: Deformation (no fixed vertices) + contour projection.
            //              This is to bring the model to a good configuration.

            std::cout << "#Iterations (fine registration stage 1): " << m_FineRegistrationCountStage1 / 2 << std::endl;

            if (m_FineRegistrationCountStage1 % 2 == 1)
            {
                // Run a 'free optimisation' where all vertices are free, in order to adjust the orientation.
                m_MaterialModelSolver.SetAllParticlesUnMoved();
                m_MaterialModelSolver.SolveCUDA();
                m_MaterialModelSolver.SetAllParticlesUnMoved();

                int i = 0;

                for (const Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
                {
                    m_DiffModelVertices[i] = pVertex->_Pos - m_PrevModelVertices[i];
                    m_PrevModelVertices[i] = pVertex->_Pos;

                    ++i;
                }

                std::cout << "Deformation without fixed vertices (stage 1)." << std::endl;

                // TODO: Do not save the contour for Hausdorff distance as we do not have the ground truth contour.
#if !USING_SYNTHETIC_INPUT_IMAGE

                UpdateModel();
                update();

                makeCurrent();
                GetContour(m_ModelContour);
                doneCurrent();

                SaveContourToFile(m_ModelContour);
#endif

            }
            else if(m_FineRegistrationCountStage1 % 2 == 0)
            {
                int i = 0;
                float k = 0.0f;

                // For accelerating the convergence.
                for (Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
                {
                    k = 0.75f * ((float)m_FineRegistrationCountStage1 * 0.5f) / ((float)m_FineRegistrationCountStage1 * 0.5f + 3.0f);
                    pVertex->_Pos += k * m_DiffModelVertices[i];

                    ++i;
                }

//                std::cout << "k: " << k << std::endl;

                // Move vertices on the mesh onto the contour in the input image, and run the simulation for deformation with the moved vertices fixed.
                FilterInvisibleFacesInModel();
                MoveClosestVerticesOnMeshToContour(true);

                std::cout << "Move vertices to the contour (stage 1)." << std::endl;

                m_MaterialModelSolver.SolveCUDA();

                //// Do not call m_MaterialModelSolver.SetAllParticlesUnMoved() to retain the fixed vertices on the contour.

                m_MaterialModelSolver.SetAllParticlesUnMoved();
                m_ModelForSimulation.UnselectAllVertices();

                std::vector<Eigen::Vector3f> vertices;

#if 1 //USING_SYNTHETIC_INPUT_IMAGE

                for (Model& model : m_GroundTruthModels)
                {
                    for (const Model::Vertex* pVertex : model.Vertices())
                    {
                        vertices.push_back(pVertex->_Pos);
                    }
                }

#else

                for (const Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
                {
                    vertices.push_back(pVertex->_Pos);
                }

#endif

                float RMSE = ComputeRMSEForFineRegistration(vertices);

//                if (m_FineRegistrationCountStage1 > 0 && RMSE < 0.005)
//                {
//                    // Move to the next stage.
//                    ++m_FineRegistrationStage;

//                    if (m_pExperimentResultsFileStream)
//                    {
//                        (*m_pExperimentResultsFileStream) << "# Stage 2 #" << endl;
//                    }

//                    m_MaterialModelSolver.SetNumOfOuterIterations(40);

//                    UpdateTumorTransform();

//                    m_Models[0].ComputeFaceNormals();
//                    m_Models[0].ComputeVertexNormals();
//                    m_Models[0].ComputeFaceCentroids();
//                    m_Models[0].BuildVertexData();
//                    UpdateModelVBO();

//                    update();

//                    return;
//                }

                if (m_FineRegistrationCountStage1 >= numOfIterationsStage1 - 2/* || RMSE < 0.001*/)
                {
                    if (!m_UsingShadingOptimisation)
                    {
//                        // Stop if using only contour.
//                        m_FineRegistrationTimer.stop();

//                        m_MaterialModelSolver.SetAllParticlesUnMoved();
//                        m_ModelForSimulation.UnselectAllVertices();

//                        UpdateModel();
//                        update();

//                        if (m_pExperimentResultsFile)
//                        {
//                            m_pExperimentResultsFile->close();
//                        }

//                        // Save the optimised model to a file.
//                        QString fileName = QString("./../tensorflow/liver_data/fine_registration/optimised_model");
//                        SaveModelData(fileName, false);

//                        return;
                    }

                    // Move to the next stage.
                    ++m_FineRegistrationStage;

//                    if (m_pExperimentResultsFileStream)
//                    {
//                        (*m_pExperimentResultsFileStream) << "# Stage 2 #" << endl;
//                    }

//                    m_MaterialModelSolver.SetNumOfOuterIterations(100);
                }
            }

            ++m_FineRegistrationCountStage1;
        }
        break;
        case 1:
        {
            // Second stage: Contour projection + deformation (with fixed vertices on the contour) + shading optimisation + deformation (no fixed vertices).

            std::cout << "#Iterations (fine registration stage 2): " << m_FineRegistrationCountStage2 / 3 << std::endl;

            if (m_FineRegistrationCountStage2 % 3 == 1)
            {
                // Run a 'free optimisation' where all vertices are free, in order to adjust the orientation.
                m_MaterialModelSolver.SetAllParticlesUnMoved();
                m_MaterialModelSolver.SolveCUDA();
                m_MaterialModelSolver.SetAllParticlesUnMoved();

                std::cout << "Deformation without fixed vertices (stage 2)." << std::endl;
            }
            else if(m_FineRegistrationCountStage2 % 3 == 2)
            {
                // Move vertices on the mesh onto the contour in the input image, and run the simulation for deformation with the moved vertices fixed.
                FilterInvisibleFacesInModel();
                MoveClosestVerticesOnMeshToContour(false);

                std::cout << "Move vertices to the contour (stage 2)." << std::endl;

                m_MaterialModelSolver.SolveCUDA();

                std::cout << "Deformation with fixed vertices on the contour (stage 2)." << std::endl;

                // Do not call m_MaterialModelSolver.SetAllParticlesUnMoved() to retain the fixed vertices on the contour.

                m_ModelForSimulation.UnselectAllVertices();

                std::vector<Eigen::Vector3f> vertices;

#if 1 //USING_SYNTHETIC_INPUT_IMAGE

                for (Model& model : m_GroundTruthModels)
                {
                    for (const Model::Vertex* pVertex : model.Vertices())
                    {
                        vertices.push_back(pVertex->_Pos);
                    }
                }

#else

                for (const Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
                {
                    vertices.push_back(pVertex->_Pos);
                }

#endif

                float RMSE = 0.0f;

                if (m_UsingShadingOptimisation)
                {
                    if (m_FineRegistrationCountStage2 >= numOfIterationsStage2 - 2/* || RMSE < 0.001*/)
                    {
                        s_UpdatingDepth = false;
                        m_FineRegistrationTimer.stop();
                        m_ContourOffsetVectors.clear();

                        m_MaterialModelSolver.SetNumOfOuterIterations(300);
                        m_MaterialModelSolver.SolveCUDA();

                        m_MaterialModelSolver.SetAllParticlesUnMoved();
                        m_ModelForSimulation.UnselectAllVertices();

//                        // Compute the final model vertices by the linear combination of vertices from deformation and shading optimisation.
//                        float alphaD = 0.9f; // Coefficient for deformation.
//                        float alphaS = 1.0f - alphaD; // Coefficient for shading.
//                        int i = 0;

//                        for (Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
//                        {
//                            pVertex->_Pos = alphaD * pVertex->_Pos + alphaS * m_ShadingOptimisationVertices[i];

//                            ++i;
//                        }

                        ComputeRMSEForFineRegistration(vertices);

//                        UpdateTumorTransform();
                        UpdateModel();
                        update();

                        if (m_pExperimentResultsFile)
                        {
                            m_pExperimentResultsFile->close();
                        }

                        // Save the optimised model to a file.
                        QString fileName = QString("./../tensorflow/liver_data/fine_registration/optimised_model");
                        SaveModelData(fileName, false);

                        return;
                    }
                    else
                    {
                        RMSE = ComputeRMSEForFineRegistration(vertices);
                    }
                }
                else
                {
                    if (m_FineRegistrationCountStage2 >= numOfIterationsStage2 - 2 /* || RMSE < 0.001*/)
                    {
                        s_UpdatingDepth = false;
                        m_FineRegistrationTimer.stop();
                        m_ContourOffsetVectors.clear();

                        m_MaterialModelSolver.SetNumOfOuterIterations(300);
                        m_MaterialModelSolver.SolveCUDA();

                        m_MaterialModelSolver.SetAllParticlesUnMoved();
                        m_ModelForSimulation.UnselectAllVertices();

                        ComputeRMSEForFineRegistration(vertices);

//                        UpdateTumorTransform();
                        UpdateModel();
                        update();

                        if (m_pExperimentResultsFile)
                        {
                            m_pExperimentResultsFile->close();
                        }

                        // Save the optimised model to a file.
                        QString fileName = QString("./../tensorflow/liver_data/fine_registration/optimised_model");
                        SaveModelData(fileName, false);

                        return;
                    }
                    else
                    {
                        RMSE = ComputeRMSEForFineRegistration(vertices);
                    }
                }
            }
            else if (m_FineRegistrationCountStage2 % 3 == 0)
            {

#if !USING_SYNTHETIC_INPUT_IMAGE

                makeCurrent();
                GetContour(m_ModelContour);
                doneCurrent();

                SaveContourToFile(m_ModelContour);
#endif

                if (m_UsingShadingOptimisation)
                {
//                    m_MaterialModelSolver.SetAllParticlesUnMoved();

                    // Finish with depth update.
                    if (m_FineRegistrationCountStage2 == numOfIterationsStage2 - 6)
                    {
                        s_UpdatingDepth = true;
                    }

                    if (s_UpdatingDepth)
                    {
                        std::cout << "Shading optimisation - depth (stage 2)." << std::endl;
                    }
                    else
                    {
                        std::cout << "Shading optimisation - orientation (stage 2)." << std::endl;
                    }

#if !USING_SYNTHETIC_INPUT_IMAGE


//                    if (m_pExperimentResultsFileStream)
//                    {

//                        (*m_pExperimentResultsFileStream) << "iteration," << (int)(m_FineRegistrationCountStage2 / 3) << endl;
//                    }

#endif

                    // Compute c.
                    PreCameraCalibration();

                    for (int i = 0; i < 1; ++i)
                    {
                        OptimiseMeshWithShading(s_UpdatingDepth);

                        UpdateModel();
                        update();
                    }

                    s_UpdatingDepth = !s_UpdatingDepth;

                    m_MaterialModelSolver.SetAllParticlesUnMoved();
                    m_ModelForSimulation.UnselectAllVertices();

                    // Save the last model vertices after shading optimisation.
                    m_ShadingOptimisationVertices.clear();

                    for (const Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
                    {
                        m_ShadingOptimisationVertices.push_back(pVertex->_Pos);
                    }
                }
            }

            ++m_FineRegistrationCountStage2;
        }
        break;
    }

//    UpdateTumorTransform();
    UpdateModel();
    update();
}

float GLWidget::ComputeRMSEForFineRegistration(const std::vector<Eigen::Vector3f>& Vertices)
{

#if !USING_SYNTHETIC_INPUT_IMAGE

    return 0.0f;

#endif

    float RMSE = 0.0f;
    Eigen::Vector3f p0, p1;

#if 1

    for (unsigned int i = 0; i < m_ModelForSimulation.Vertices().size(); ++i)
    {
        p0 = m_ModelForSimulation.Vertices()[i]->_Pos;
//        p1 = m_GroundTruthModel.Vertices()[i]->_Pos;
        p1 = Vertices[i];

        QVector3D a(p0[0], p0[1], p0[2]);
        QVector3D b(p1[0], p1[1], p1[2]);

        a = m_ModelView * a;

        // TODO: Change when using real input images.
        b = m_GroundTruthModelModelView * b;

        RMSE += QVector3D::dotProduct(a - b, a - b);
    }

    RMSE /= (float)m_ModelForSimulation.Vertices().size();
    RMSE = sqrt(RMSE);

    std::cout << "RMSE (fine registration): " << RMSE << std::endl;

#else

    std::vector<Eigen::Vector3f> vertices, groundTruthModelVertices;

    for (unsigned int i = 0; i < m_ModelForSimulation.Vertices().size(); ++i)
    {
        p0 = m_ModelForSimulation.Vertices()[i]->_Pos;
        p1 = Vertices[i];

        QVector3D a(p0[0], p0[1], p0[2]);
        QVector3D b(p1[0], p1[1], p1[2]);

        a = m_ModelView * a;

        // TODO: Change when using real input images.
        b = m_GroundTruthModelModelView * b;

        vertices.push_back(Eigen::Vector3f(a.x(), a.y(), a.z()));
        groundTruthModelVertices.push_back(Eigen::Vector3f(b.x(), b.y(), b.z()));
    }

    // Use modified Hausdorff distance instead of RMSE.
    RMSE = Utils::ModifiedHausdorffDistance(vertices, groundTruthModelVertices);

    std::cout << "mHausdorff (fine registration): " << RMSE << std::endl;

#endif

    if (m_pExperimentResultsFileStream)
    {
        (*m_pExperimentResultsFileStream) << RMSE << endl;
    }

    return RMSE;
}

void GLWidget::ComputeVertexErrors(const std::vector<Model::Vertex*>& GroundTruthVertices, std::vector<Model::Vertex*>& ModelVertices)
{
    m_VertexErrors.clear();
    m_VertexErrorColours.clear();

    float error = 0.0f;
    Eigen::Vector3f p0, p1;
    QVector4D vertex3D0, vertex3D1;

    for (unsigned int i = 0; i < GroundTruthVertices.size(); ++i)
    {
        p0 = GroundTruthVertices[i]->_Pos;
        p1 = ModelVertices[i]->_Pos;

        vertex3D0.setX(p0[0]);
        vertex3D0.setY(p0[1]);
        vertex3D0.setZ(p0[2]);
        vertex3D0.setW(1.0);
        vertex3D1.setX(p1[0]);
        vertex3D1.setY(p1[1]);
        vertex3D1.setZ(p1[2]);
        vertex3D1.setW(1.0);

        vertex3D0 = m_GroundTruthModelModelView * vertex3D0;
        vertex3D0 /= vertex3D0.w();
        p0 << vertex3D0.x(), vertex3D0.y() , vertex3D0.z();

        vertex3D1 = m_ModelView * vertex3D1;
        vertex3D1 /= vertex3D1.w();
        p1 << vertex3D1.x(), vertex3D1.y(), vertex3D1.z();

        error = (p0 - p1).norm();
        m_VertexErrors.push_back(error);
    }

    // Normalise the values by the maxium value.
    std::vector<float>::iterator itMax = std::max_element(m_VertexErrors.begin(), m_VertexErrors.end());
    float max = *itMax;
    std::cout << "Max vertex error: " << max << std::endl;

    // TODO: Temp.
//    max = 0.0755021;

    for (float& error : m_VertexErrors)
    {
        error /= max;

//        std::cout << error << std::endl;
    }

    // Set the colour map.
//    cv::Mat errors, colours;
    unsigned int size = m_VertexErrors.size();

//    errors.create(size, 1, CV_32FC1);
//    colours.create(size, 1, CV_32FC3);

//    for (unsigned int i = 0; i < size; ++i)
//    {
//        errors.at<float>(i, 0) = m_VertexErrors[i];
//    }

//    cv::applyColorMap(errors, colours, cv::COLORMAP_AUTUMN);

    Eigen::Vector4f colour;

    for (unsigned int i = 0; i < size; ++i)
    {
//        m_VertexErrorColours.push_back(Eigen::Vector3f(colours.at<cv::Vec3f>(i, 0)[0], colours.at<cv::Vec3f>(i, 0)[1], colours.at<cv::Vec3f>(i, 0)[2]));
        colour << m_VertexErrors[i], 0.0f, 1 - m_VertexErrors[i], 1.0f;

        if (m_VertexErrors[i] <= 0.5f)
        {
            colour[1] = 2.0f * colour[0];
        }
        else if (m_VertexErrors[i] > 0.5f)
        {
            colour[1] = 2.0f * colour[2];
        }

        m_VertexErrorColours.push_back(colour);
    }

    int index = 0;

    for (Model::Vertex* pVertex : ModelVertices)
    {
        pVertex->_Colour = m_VertexErrorColours[index];

        ++index;
    }

    m_cForRendering = 0.03f;

    UpdateModel(false, false, false);
    update();

    std::cout << "Vertex errors computed." << std::endl;
}

void GLWidget::LoadModelData(QString& FileName, bool IsGroundTruth)
{
    // Load the model vertices, depth values, contour and camera transform from a file.
//    QString fileName = "./../tensorflow/liver_data/fine_registration/ground_truth.txt";
    QFile file(FileName + QString(".txt"));
    file.open(QIODevice::ReadOnly);
    QTextStream stream(&file);

//    while (!stream.atEnd())

    if (IsGroundTruth)
    {
        std::cout << "Loading a ground truth model data..." << std::endl;
    }
    else
    {
        std::cout << "Loading a model data..." << std::endl;
    }

    // Model data.
    QString line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    line = stream.readLine(); // Empty line.

    int num = 0;
    QStringList list;

    if (IsGroundTruth)
    {
        // Depth values;
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        // Number of values;
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        line = stream.readLine();
        num = line.toInt();
        std::cout << num << std::endl;

        // Values.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        if (m_pGroundTruthDepthData)
        {
            delete m_pGroundTruthDepthData;
        }

        int size = m_Viewport[2] * m_Viewport[3];
        m_pGroundTruthDepthData = new GLfloat[size];

        for (int i = 0; i < m_Viewport[2]; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            for (int j = 0; j < list.size(); ++j)
            {
                m_pGroundTruthDepthData[j + i * m_Viewport[3]] = list.at(j).toFloat();
            }
        }

        line = stream.readLine(); // Empty line.
    }

    // Contour.
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    // Number of points.
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    line = stream.readLine();
    num = line.toInt();
    std::cout << num << std::endl;

    // Points.
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    if (IsGroundTruth)
    {
        m_GroundTruthModelContour.clear();

        for (int i = 0; i < num; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            m_GroundTruthModelContour.push_back(Eigen::Vector2f(list.at(0).toFloat(), list.at(1).toFloat()));
        }
    }
    else
    {
        m_ModelContour.clear();

        for (int i = 0; i < num; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            m_ModelContour.push_back(Eigen::Vector2f(list.at(0).toFloat(), list.at(1).toFloat()));
        }
    }

    line = stream.readLine(); // Empty line.

    // Model.
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    // Number of models.
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    line = stream.readLine();
    int numOfModels = line.toInt();
    std::cout << numOfModels << std::endl;

    for (int i = 0; i < numOfModels; ++i)
    {
        // Number of vertices.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        line = stream.readLine();
        num = line.toInt();
        std::cout << num << std::endl;

        // Vertices.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        Model* pModel = &m_Models[i];

        if (IsGroundTruth)
        {
            pModel = &m_GroundTruthModels[i];
            pModel->CleanUp();
        }

        Model::Vertex* pNewVertex = NULL;

        for (int i = 0; i < num; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            if (IsGroundTruth)
            {
                pNewVertex = new Model::Vertex(list.at(0).toFloat(), list.at(1).toFloat(), list.at(2).toFloat());
                pNewVertex->_Normal[0] = list.at(3).toFloat();
                pNewVertex->_Normal[1] = list.at(4).toFloat();
                pNewVertex->_Normal[2] = list.at(5).toFloat();

                pModel->Vertices().push_back(pNewVertex);
            }
            else
            {
                pModel->Vertices()[i]->_Pos = Eigen::Vector3f(list.at(0).toFloat(), list.at(1).toFloat(), list.at(2).toFloat());
                pModel->Vertices()[i]->_Normal = Eigen::Vector3f(list.at(3).toFloat(), list.at(4).toFloat(), list.at(5).toFloat());
            }
        }

        // Vertex normals.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        for (int i = 0; i < num; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            if (IsGroundTruth)
            {
                pModel->VertexNormals().push_back(Eigen::Vector3f(list.at(0).toFloat(), list.at(1).toFloat(), list.at(2).toFloat()));
            }
            else
            {
                pModel->VertexNormals()[i] = Eigen::Vector3f(list.at(0).toFloat(), list.at(1).toFloat(), list.at(2).toFloat());
            }
        }

        // Texture coordinates.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        for (int i = 0; i < num; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            if (IsGroundTruth)
            {
                pModel->TexCoords().push_back(Model::TexCoord(list.at(0).toFloat(), list.at(1).toFloat()));
            }
            else
            {
                pModel->TexCoords()[i] = Model::TexCoord(list.at(0).toFloat(), list.at(1).toFloat());
            }
        }

        // Number of faces.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        line = stream.readLine();
        num = line.toInt();
        std::cout << num << std::endl;

        // Faces.
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        Model::Face* pNewFace = NULL;

        for (int i = 0; i < num; ++i)
        {
            line = stream.readLine();
            list = line.split(',');

            if (IsGroundTruth)
            {
                pNewFace = new Model::Face;
                pNewFace->_VertexIndices.push_back(list.at(0).toInt());
                pNewFace->_VertexIndices.push_back(list.at(1).toInt());
                pNewFace->_VertexIndices.push_back(list.at(2).toInt());

                pNewFace->_TexCoordIndices.push_back(list.at(3).toInt());
                pNewFace->_TexCoordIndices.push_back(list.at(4).toInt());
                pNewFace->_TexCoordIndices.push_back(list.at(5).toInt());

                pNewFace->_VertexNormalIndices.push_back(list.at(6).toInt());
                pNewFace->_VertexNormalIndices.push_back(list.at(7).toInt());
                pNewFace->_VertexNormalIndices.push_back(list.at(8).toInt());

                pNewFace->_Centroid[0] = list.at(9).toFloat();
                pNewFace->_Centroid[1] = list.at(10).toFloat();
                pNewFace->_Centroid[2] = list.at(11).toFloat();

                pModel->Faces().push_back(pNewFace);
            }
            else
            {
                pModel->Faces()[i]->_VertexIndices[0] = list.at(0).toInt();
                pModel->Faces()[i]->_VertexIndices[1] = list.at(1).toInt();
                pModel->Faces()[i]->_VertexIndices[2] = list.at(2).toInt();

                pModel->Faces()[i]->_TexCoordIndices[0] = list.at(3).toInt();
                pModel->Faces()[i]->_TexCoordIndices[1] = list.at(4).toInt();
                pModel->Faces()[i]->_TexCoordIndices[2] = list.at(5).toInt();

                pModel->Faces()[i]->_VertexNormalIndices[0] = list.at(6).toInt();
                pModel->Faces()[i]->_VertexNormalIndices[1] = list.at(7).toInt();
                pModel->Faces()[i]->_VertexNormalIndices[2] = list.at(8).toInt();

                pModel->Faces()[i]->_Centroid[0] = list.at(9).toFloat();
                pModel->Faces()[i]->_Centroid[1] = list.at(10).toFloat();
                pModel->Faces()[i]->_Centroid[2] = list.at(11).toFloat();
            }
        }

        pModel->ComputeFaceNormals();
        pModel->BuildVertexData();

        if (IsGroundTruth)
        {
            pModel->BuildHalfedges();

            UpdateGroundTruthModelVBO(i);

            m_GroundTruthModelExisting = true;
            m_RenderingGroundTruthModelContour = 2;
        }
        else
        {
            UpdateModelVBO(i);
        }
    }

    line = stream.readLine(); // Empty line.

    // Camera transform.
    QMatrix4x4* pMat = &m_LoadedModelView;

    if (IsGroundTruth)
    {
        pMat = &m_GroundTruthModelModelView;
    }

    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    for (int i = 0; i < 4; ++i)
    {
        line = stream.readLine();
        list = line.split(',');

        for (int j = 0; j < 4; ++j)
        {
            (*pMat)(i, j) = list.at(j).toFloat();

            std::cout << (*pMat)(i, j) << ",";
        }

        std::cout << std::endl;
    }

    line = stream.readLine(); // Empty line.

    QMatrix3x3 rotationMat;

    // Load the model translation.
    if (!IsGroundTruth)
    {
        line = stream.readLine();
        std::cout << line.toStdString() << std::endl;

        line = stream.readLine();
        list = line.split(',');

        coordModels << list.at(0).toFloat(), list.at(1).toFloat(), list.at(2).toFloat();
        std::cout << coordModels << std::endl;

//        (*pMat)(0, 3) = 0.0f;
//        (*pMat)(1, 3) = 0.0f;
//        (*pMat)(2, 3) = 0.0f;

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (i < 3 && j < 3)
                {
                    rotationMat(i, j) = (*pMat)(i, j);
                }

                std::cout << (*pMat)(i, j) << ",";
            }

            std::cout << std::endl;
        }
    }

    trackball.setRotation(QQuaternion::fromRotationMatrix(rotationMat));

    line = stream.readLine(); // Empty line.

    // FOV scale.
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    line = stream.readLine();
    m_FOVScale = line.toFloat();
    std::cout << m_FOVScale << std::endl;

    emit FOVScaleChanged(m_FOVScale);

    file.close();

    m_ModelDataLoaded = true;

    if (IsGroundTruth)
    {
        std::cout << "Ground truth model data loaded." << std::endl;
    }
    else
    {
        std::cout << "Model data loaded." << std::endl;
    }

    update();
}

void GLWidget::SaveModelData(QString& FileName, bool IsGroundTruth)
{
    // Save the model vertices, depth values, contour and camera transform to a file.
//    QString fileName = "./../tensorflow/liver_data/fine_registration/ground_truth.txt";
    QFile file(FileName + QString(".txt"));
    file.open(QIODevice::ReadWrite);
    file.resize(0);

    QTextStream stream(&file);

    if (IsGroundTruth)
    {
        stream << "########## Ground truth model data ##########" << endl << endl;
    }
    else
    {
        stream << "########## Model data ##########" << endl << endl;
    }

    makeCurrent();

    // Save the ground truth depth values.
    if (IsGroundTruth)
    {
        if (m_pGroundTruthDepthData)
        {
            delete m_pGroundTruthDepthData;
        }

        int size = m_Viewport[2] * m_Viewport[3];
        m_pGroundTruthDepthData = new GLfloat[size];
        glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, m_pGroundTruthDepthData);

        stream << "##### Depth values #####" << endl;

        stream << "# Number of values #" << endl;
        stream << size << endl;

        stream << "# Values #" << endl;

        for (int i = 0; i < m_Viewport[2]; ++i)
        {
            for (int j = 0; j < m_Viewport[3]; ++j)
            {
                stream << m_pGroundTruthDepthData[j + i * m_Viewport[3]];

                if (j < m_Viewport[3] - 1)
                {
                    stream << ",";
                }
            }

            stream << endl;
        }

        stream << endl;

        GetContour(m_GroundTruthModelContour);
    }
    else
    {
        GetContour(m_ModelContour);
    }

    doneCurrent();

    stream << "##### Contour #####" << endl;

    stream << "# Number of points #" << endl;

    if (IsGroundTruth)
    {
        stream << m_GroundTruthModelContour.size() << endl;
    }
    else
    {
        stream << m_ModelContour.size() << endl;
    }

    stream << "# Points #" << endl;

    if (IsGroundTruth)
    {
        for (const Eigen::Vector2f& point : m_GroundTruthModelContour)
        {
            stream << point[0] << "," << point[1] << endl;
        }
    }
    else
    {
        for (const Eigen::Vector2f& point : m_ModelContour)
        {
            stream << point[0] << "," << point[1] << endl;
        }
    }

    stream << endl;

    // Save the model.

    if (IsGroundTruth)
    {
        for (Model& model : m_GroundTruthModels)
        {
            model.CleanUp();
        }
    }

    Eigen::Vector3f pos;
    Model::Vertex* pNewVertex = NULL;

    stream << "##### Model #####" << endl;

    stream << "##### Number of models #####" << endl;
    int numOfModels = m_Models.size();
    stream << numOfModels << endl;

    for (int i = 0; i < numOfModels; ++i)
    {
        stream << "# Number of vertices #" << endl;
        stream << m_Models[i].Vertices().size() << endl;

        stream << "# Vertices #" << endl;

        for (Model::Vertex* pVertex : m_Models[i].Vertices())
        {
            if (IsGroundTruth)
            {
                pNewVertex = new Model::Vertex(pos[0], pos[1], pos[2]);
                pNewVertex->_Pos = pVertex->_Pos;
                pNewVertex->_Normal = pVertex->_Normal;
                m_GroundTruthModels[i].Vertices().push_back(pNewVertex);
            }
            else
            {
                pNewVertex = pVertex;
            }

            stream << pNewVertex->_Pos[0] << "," << pNewVertex->_Pos[1] << "," << pNewVertex->_Pos[2] << ","
                   << pNewVertex->_Normal[0] << "," << pNewVertex->_Normal[1] << "," << pNewVertex->_Normal[2] << endl;
        }

        if (IsGroundTruth)
        {
            m_GroundTruthModels[i].SetVertexNormals(m_Models[i].VertexNormals());
            m_GroundTruthModels[i].SetTexCoords(m_Models[i].TexCoords());
        }

        Model* pModel = &m_Models[i];

        if (IsGroundTruth)
        {
            pModel = &m_GroundTruthModels[i];
        }

        stream << "# Vertex normals #" << endl;

        for (const Eigen::Vector3f& normal : pModel->VertexNormals())
        {
            stream << normal[0] << "," << normal[1] << "," << normal[2] << endl;
        }

        stream << "# Texture coordinates #" << endl;

        for (const Model::TexCoord& coord : pModel->TexCoords())
        {
            stream << coord._u << "," << coord._v << endl;
        }

        stream << "# Number of faces #" << endl;
        stream << m_Models[i].Faces().size() << endl;

        stream << "# Faces #" << endl;

        Model::Face* pNewFace;

        for (Model::Face* pFace : m_Models[i].Faces())
        {
            if (IsGroundTruth)
            {
                pNewFace = new Model::Face;
                pNewFace->_VertexIndices = pFace->_VertexIndices;
                pNewFace->_TexCoordIndices = pFace->_TexCoordIndices;
                pNewFace->_VertexNormalIndices = pFace->_VertexNormalIndices;
                pNewFace->_Centroid = pFace->_Centroid;

                m_GroundTruthModels[i].Faces().push_back(pNewFace);
            }
            else
            {
                pNewFace = pFace;
            }

            stream << pNewFace->_VertexIndices[0] << "," << pNewFace->_VertexIndices[1] << "," << pNewFace->_VertexIndices[2] << ","
                   << pNewFace->_TexCoordIndices[0] << "," << pNewFace->_TexCoordIndices[1] << "," << pNewFace->_TexCoordIndices[2] << ","
                   << pNewFace->_VertexNormalIndices[0] << "," << pNewFace->_VertexNormalIndices[1] << "," << pNewFace->_VertexNormalIndices[2] << ","
                   << pNewFace->_Centroid[0] << "," << pNewFace->_Centroid[1] << "," << pNewFace->_Centroid[2] << endl;
        }

        if (IsGroundTruth)
        {
            m_GroundTruthModels[i].ComputeFaceNormals();
            m_GroundTruthModels[i].BuildHalfedges();
            m_GroundTruthModels[i].BuildVertexData();

            UpdateGroundTruthModelVBO(i);
            m_GroundTruthModelExisting = true;
            m_GroundTruthModelModelView = m_ModelView;
            m_RenderingGroundTruthModelContour = 2;
        }
    }

    update();

    stream << endl;

    // Save the camera transform.
    stream << "# Camera transform (ModelView matrix) #" << endl;

    QMatrix4x4* pMat = &m_ModelView;

    if (IsGroundTruth)
    {
        pMat = &m_GroundTruthModelModelView;
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            stream << (*pMat)(i, j);

            if (j < 3)
            {
                stream << ",";
            }
        }

        stream << endl;
    }

    stream << endl;

    // Save the model translation.
    if (!IsGroundTruth)
    {
        stream << "# Model translation #" << endl;
        stream << coordModels[0] << "," << coordModels[1] << "," << coordModels[2] << endl;
        stream << endl;
    }

    // Save FOV scale.
    stream << "# FOV scale #" << endl;
    stream << m_FOVScale << endl;

    file.close();

    // Save the model image to a file.
    m_SavingImage = this->grabFramebuffer(); // Saved Image format is QImage::Format_RGB32.
    QString format = "png";
//    m_SavingImage.save(QString("./../tensorflow/liver_data/fine_registration/ground_truth.") + format, qPrintable(format));
    m_SavingImage.save(FileName + QString(".") + format, qPrintable(format));

//            std::cout << "Image format: " << m_SavingImage.format() << std::endl;

    if (IsGroundTruth)
    {
        std::cout << "Ground truth model data saved." << std::endl;
    }
    else
    {
        std::cout << "Model data saved." << std::endl;
    }
}

void GLWidget::SaveDisconnectedCellGroupsForSimulation(void)
{
    QString fileName = "./../tensorflow/liver_data/fine_registration/disconnected_cell_groups.txt";
    QFile file(fileName);
    file.open(QIODevice::ReadWrite);
    file.resize(0);

    QTextStream stream(&file);

    stream << "########## Disconnected cell groups for simulation ##########" << endl << endl;

    stream << "# Number of disconnected cell groups #" << endl;
    stream << m_MaterialModelSolver.DisconnectedCellGroups().size() << endl;

    stream << "# Disconnected cell groups #" << endl;

    for (std::vector<int> group : m_MaterialModelSolver.DisconnectedCellGroups())
    {
        stream << group.size();

        for (int cellIndex : group)
        {
            stream << "," << cellIndex;
        }

        stream << endl;
    }

    file.close();

    std::cout << "Disconnected cell groups for simulation saved." << std::endl;
}

void GLWidget::LoadDisconnectedCellGroupsForSimulation(void)
{
    QString fileName = "./../tensorflow/liver_data/fine_registration/disconnected_cell_groups.txt";
    QFile file(fileName);
    file.open(QIODevice::ReadOnly);

    QTextStream stream(&file);

//    stream << "########## Disconnected cell groups for simulation ##########" << endl << endl;
    QString line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    line = stream.readLine(); // Empty line.

//    stream << "# Number of disconnected cell groups #" << endl;
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    line = stream.readLine();
    int num = line.toInt();
    std::cout << num << std::endl;

//    stream << "# Disconnected cell groups #" << endl;
    line = stream.readLine();
    std::cout << line.toStdString() << std::endl;

    std::vector<std::vector<int> >& disconnectedCellGroups = m_MaterialModelSolver.DisconnectedCellGroups();
    disconnectedCellGroups.clear();

    std::vector<int> group;

    QStringList list;

    for (int i = 0; i < num; ++i)
    {
        line = stream.readLine();
        list = line.split(',');

        std::cout << list.at(0).toInt();

        group.clear();

        for (int j = 1; j < list.size(); ++j)
        {
            group.push_back(list.at(j).toInt());

            std::cout << "," << list.at(j).toInt();
        }

        disconnectedCellGroups.push_back(group);

        std::cout << endl;
    }

    file.close();

    std::cout << "Disconnected cell groups for simulation loaded." << std::endl;
}

void GLWidget::RunOptimisation(bool UsingContourAndShading)
{
    // Save the current model, contour and camera transfrom to a file.

    if (!m_RenderingGroundTruthModel)
    {
        QString fileName = QString("./../tensorflow/liver_data/fine_registration/initial_model");
        SaveModelData(fileName, false);

        camera();

        m_MaterialModelSolver.SetNumOfInnerIterations(3);
        m_MaterialModelSolver.SetNumOfOuterIterations(100);
        m_FineRegistrationStage = 0;
        m_ShadingOptimisationVertices.clear();

        m_DiffModelVertices.clear();
        m_PrevModelVertices.clear();

        // Store the current model vertices.
        for (const Model::Vertex* pVertex : m_ModelForSimulation.Vertices())
        {
            m_PrevModelVertices.push_back(pVertex->_Pos);
            m_DiffModelVertices.push_back(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
        }

        if (!UsingContourAndShading)
        {
            m_UsingShadingOptimisation = false;

            std::cout << "Starting fine registration (without shading optimisation)..." << std::endl;
        }
        else
        {
            m_UsingShadingOptimisation = true;

            std::cout << "Starting fine registration (with shading optimisation)..." << std::endl;
        }

        m_FineRegistrationCountStage1 = 0;
        m_FineRegistrationCountStage2 = 0;
        m_FineRegistrationTimer.start(1);
        m_OptimisingShading = false;

        // Save the experiment results (RMSE) to a file.
        if (m_pExperimentResultsFile)
        {
            delete m_pExperimentResultsFile;
        }

        m_pExperimentResultsFile = new QFile(QString("./../tensorflow/liver_data/fine_registration/results.txt"));
        m_pExperimentResultsFile->open(QIODevice::ReadWrite);
        m_pExperimentResultsFile->resize(0);

        if (m_pExperimentResultsFileStream)
        {
            delete m_pExperimentResultsFileStream;
        }

        m_pExperimentResultsFileStream = new QTextStream(m_pExperimentResultsFile);

        // TODO: Do not use the contour for Hausdorff distance as we do not have the ground truth contour.
#if !USING_SYNTHETIC_INPUT_IMAGE

        if (m_pExperimentResultsFileStream)
        {
            (*m_pExperimentResultsFileStream) << "c:" << m_c << ",FOVScale:" << m_FOVScale << ",FOVPosOffset:" << m_FOVPosOffset[0] << "," << m_FOVPosOffset[1] << endl;
        }

        // Save the ground truth contour.
        SaveContourToFile(m_GroundTruthModelContour);

#endif

//            if (m_pExperimentResultsFileStream)
//            {
//                (*m_pExperimentResultsFileStream) << "# Initial #" << endl;
//            }

        std::vector<Eigen::Vector3f> vertices;

        for (Model& model : m_GroundTruthModels)
        {
            for (const Model::Vertex* pVertex : model.Vertices())
            {
                vertices.push_back(pVertex->_Pos);
            }
        }

        ComputeRMSEForFineRegistration(vertices);

//            if (m_pExperimentResultsFileStream)
//            {
//                (*m_pExperimentResultsFileStream) << "# Stage 1 #" << endl;
//            }
    }
    else
    {
        std::cout << "Turn off rendering the ground truth model first." << std::endl;
    }
}

/* ============================ MOUSE AND KEYBOARD TRANSFORMATIONS ============================ */
void GLWidget::mouseMoveEvent(QMouseEvent *e)
{
    float move = 0.0f;

    if(e->modifiers() & Qt::ControlModifier)
    {
        move = sensibilityPlus;
    }
    else
    {
        move = sensibility;
    }

    if(e->buttons() & Qt::LeftButton)
    {
        if (m_IsContourSelectionOn)
        {
        }
        else if (m_EditingModel)
        {
            QVector3D vec(e->localPos().x() - m_PrevMousePos[0], -(e->localPos().y() - m_PrevMousePos[1]), 0.0);
            QMatrix4x4 r;
            r.rotate(trackball.rotation());
            r = r.transposed(); // Inverse.
            vec = r * vec;
            vec *= move/scaleFactor/1000.0f;

            if (m_TranslatingVertices)
            {
                // Translate the selected vertices in the model.
                for (Model& model : m_Models)
                {
                    model.TranslateSelectedVertices(Eigen::Vector3f(vec.x(), vec.y(), vec.z()));
                }
            }
            else if (m_RotatingVertices)
            {
                // Rotate the selected vertices in the model.
                if (vec.length() >= 1e-6)
                {
                    QVector3D vecPerpendicular(0.0, 0.0, -1.0);
                    vecPerpendicular = r * vecPerpendicular;

                    for (Model& model : m_Models)
                    {
                        model.RotateSelectedVertices(Eigen::Vector3f(vec.x() * 100.0f, vec.y() * 100.0f, vec.z() * 100.0f), Eigen::Vector3f(vecPerpendicular.x(), vecPerpendicular.y(), vecPerpendicular.z()));
                    }
                }
            }
            else if (m_SelectingMultipleVertices)
            {
                // Update vertex selection rect.
                m_VertexSelectionRect.setRight(e->localPos().x());
                m_VertexSelectionRect.setBottom(m_Viewport[3] - e->localPos().y());
            }

            UpdateModel();
        }
        else
        {
            trackball.move(pixelPosToViewPos(e->localPos()), QQuaternion());
        }

//        updateGL();
        update();
    }

    m_PrevMousePos[0] = e->localPos().x();
    m_PrevMousePos[1] = e->localPos().y();
}

void GLWidget::mousePressEvent(QMouseEvent *e)
{
    if(e->buttons() & Qt::LeftButton)
    {
        const QPointF screenCoordinates = e->localPos();
        bool modelPicked = false;
        QVector3D pos = screenToModelPixel(screenCoordinates, &modelPicked);
        surfaceCoordinates = pos;
        surfaceCoordinates.setZ(-surfaceCoordinates.z());

        createCrosshair(screenCoordinates);

        if (m_SelectingShadingOptimisationRegion)
        {
            Eigen::Vector2f point(e->localPos().x(), m_Viewport[3] - e->localPos().y());
            float scale = frame_picture_Ratio * cameraParameters[7];
            Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

            point -= scale * dimension;
            point /= cameraParameters[7];

            m_ShadingOptimisationRegionCentre = point;
        }
        else if (m_IsContourSelectionOn)
        {
            Eigen::Vector2f point(e->localPos().x(), m_Viewport[3] - e->localPos().y());
            float scale = frame_picture_Ratio * cameraParameters[7];
            Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

            point -= scale * dimension;
            point /= cameraParameters[7];

            m_ContourSelectionPoints.push_back(point);
        }
        else if (m_EditingModel && modelPicked) // Select a vertex in the model.
        {
            float distance = 0.0f;
            float minDist = std::numeric_limits<float>::max();
            int minIndex = -1;
            int modelIndex = -1;

            if (m_CheckedModels.size() > 0)
            {
                for(GLuint i = 0; i < (GLuint)m_CheckedModels.size(); i++)
                {
                    Model& model = m_Models[m_CheckedModels.at(i)];

                    int index = model.ClosestVertexIndexToPoint(Eigen::Vector2f(e->localPos().x(), m_Viewport[3] - e->localPos().y()), m_ModelView, m_proj, m_Viewport, distance);

                    if (distance < minDist)
                    {
                        minDist = distance;
                        minIndex = index;
                        modelIndex = m_CheckedModels.at(i);
                    }
                }
            }


//            for (Model& model : m_Models)
//            {
//                int index = model.ClosestVertexIndexToPoint(Eigen::Vector2f(e->localPos().x(), m_Viewport[3] - e->localPos().y()), m_ModelView, m_proj, m_Viewport, distance);

//                if (distance < minDist)
//                {
//                    minDist = distance;
//                    minIndex = index;
//                    modelIndex = i;
//                }

//                ++i;

//    //            std::cout << "Selected vertex index: " << index << std::endl;
//            }

            if (minIndex >= 0 && modelIndex >= 0)
            {
                if ((m_Models[modelIndex].Vertices()[minIndex])->_Selected)
                {
                    if(e->modifiers() & Qt::ShiftModifier)
                    {
                        // Rotate the selected vertices in the model.
                        m_RotatingVertices = true;
                    }
                    else
                    {
                        // Translate the selected vertices in the model.
                        m_TranslatingVertices = true;
                    }

                    UpdateModel();
                }
                else
                {
                    GLfloat pDepthData[m_Viewport[2] * m_Viewport[3]];

                    makeCurrent();

                    glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, pDepthData);

                    doneCurrent();

                    if (Model::VertexVisible(m_Models[modelIndex].Vertices()[minIndex], pDepthData, m_ModelView, m_proj, m_Viewport))
                    {
                        if (m_SelectingModelContour > MODEL_CONTOUR_TYPE_NULL && !(m_Models[modelIndex].Vertices()[minIndex])->_Selected)
                        {
                            SelectModelContour(minIndex, modelIndex, m_SelectingModelContour);
                        }
                        else
                        {
                            (m_Models[modelIndex].Vertices()[minIndex])->_Selected = true;
                            m_Models[modelIndex].SetHasSelectedVertices(true);

        //                   m_Models[modelIndex].SelectVertexWithOneRingNeighbours(m_Models[modelIndex].Vertices()[minIndex]);
                        }
                    }

                    UpdateModel(false, false, false);
                }
            }

            update();
        }
        else if (m_EditingModel && !modelPicked)
        {
            // Start selecting multiple vertices.
            m_SelectingMultipleVertices = true;
            m_VertexSelectionRect.setLeft(e->localPos().x());
            m_VertexSelectionRect.setRight(e->localPos().x());
            m_VertexSelectionRect.setTop(m_Viewport[3] - e->localPos().y());
            m_VertexSelectionRect.setBottom(m_Viewport[3] - e->localPos().y());
        }
        else if (!m_EditingModel)
        {
            setCursor(Qt::ClosedHandCursor);      
            trackball.push(pixelPosToViewPos(screenCoordinates));
        }
    }
    else if(e->buttons() & Qt::RightButton && distanceMode)
    {
        update();
        createTags(e->localPos());
    }

    m_PrevMousePos[0] = e->localPos().x();
    m_PrevMousePos[1] = e->localPos().y();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *e)
{
    if(e->buttons() && Qt::LeftButton)
    {
        if (!m_EditingModel)
        {
            trackball.release(pixelPosToViewPos(e->localPos()),QQuaternion());
        }
    }

    if (m_TranslatingVertices || m_RotatingVertices)
    {
        UpdateModel();

        m_TranslatingVertices = false;
        m_RotatingVertices = false;

        for (Model& model : m_Models)
        {
            for (Model::Vertex*& pVertex : model.Vertices())
            {
                if (pVertex->_Selected)
                {
                    pVertex->_Moved = true;
                }
            }
        }
    }
    else if (!m_EditingModel && m_SelectingShadingOptimisationRegion)
    {
        Eigen::Vector2f point(e->localPos().x(), m_Viewport[3] - e->localPos().y());
        float scale = frame_picture_Ratio * cameraParameters[7];
        Eigen::Vector2f dimension(m_InputImage.width(), m_InputImage.height());

        point -= scale * dimension;
        point /= cameraParameters[7];

        m_ShadingOptimisationRegionRadius = (point - m_ShadingOptimisationRegionCentre).norm();
    }
    else if (!m_EditingModel && m_SelectingContour)
    {
//        m_world.setToIdentity();

////        m_camera.scale(scaleFactor);
//        m_camera.setToIdentity();
//        m_camera.translate(coordModels[0], coordModels[1], coordModels[2]); // Keyboard and wheel translation

//        QMatrix4x4 m;
//        m.rotate(trackball.rotation());
//        m_camera *= m;

//        m_ModelView = m_camera * m_world;

//        m_Models[0].ExtractVerticesOnContour(m_ModelView, m_proj, m_Viewport);

//        std::vector<GLfloat> contour;

//        for (Segment& segment : m_Models[0].Contour())
//        {
//            contour.push_back(segment.source().x());
//            contour.push_back(segment.source().y());
//            contour.push_back(segment.target().x());
//            contour.push_back(segment.target().y());
//        }

//        QOpenGLVertexArrayObject::Binder contourVAOBinder(&m_ContourVAO);
//        m_ContourVBO.bind();
//        m_ContourVBO.allocate(contour.data(), contour.size() * sizeof(GLfloat));
    }
    else if (m_SelectingContour && m_SelectingMultipleVertices
         && (m_VertexSelectionRect.left() - m_VertexSelectionRect.right()) != 0.0f && (m_VertexSelectionRect.top() - m_VertexSelectionRect.bottom()) != 0.0f)
    {
//        for (Model& model : m_Models)
        {
            m_ModelForSimulation.SelectVerticesOnContour(m_ModelContour, m_VertexSelectionRect, m_ModelView, m_proj, m_Viewport);
        }

//        m_Models[0].SelectVerticesOnContour(m_VertexSelectionRect, m_ModelViewMatrix, m_ProjectionMatrix, m_Viewport);

        UpdateModel(false, false, false);
    }
    else if (m_SelectingMultipleVertices
         && (m_VertexSelectionRect.left() - m_VertexSelectionRect.right()) != 0.0f && (m_VertexSelectionRect.top() - m_VertexSelectionRect.bottom()) != 0.0f)
    {
        // Select the vertices included in the rect.
        Point point2D;

        if (m_CheckedModels.size() > 0)
        {
            for(GLuint i = 0; i < (GLuint)m_CheckedModels.size(); i++)
            {
                Model& model = m_Models[m_CheckedModels.at(i)];
                int index = 0;

                for (Model::Vertex*& pVertex : model.Vertices())
                {
                    if (!pVertex->_Selected)
                    {
        //                Eigen::Vector3f windowPos;
        //                GLdouble x, y, z;

                        point2D = model.ProjectVertexOnto2D(pVertex, m_ModelView, m_proj, m_Viewport);

        //                gluProject((GLdouble)pVertex->_Pos[0], (GLdouble)pVertex->_Pos[1], (GLdouble)pVertex->_Pos[2], m_ModelViewMatrix, m_ProjectionMatrix, m_Viewport, &x, &y, &z);
        //                windowPos << (float)x, (float)y, (float)z;

                        if (point2D.x() < m_VertexSelectionRect.center().x() - fabs(m_VertexSelectionRect.width()) * 0.5f || point2D.x() > m_VertexSelectionRect.center().x() + fabs(m_VertexSelectionRect.width()) * 0.5f
                         || point2D.y() < m_VertexSelectionRect.center().y() - fabs(m_VertexSelectionRect.height()) * 0.5f || point2D.y() > m_VertexSelectionRect.center().y() + fabs(m_VertexSelectionRect.height()) * 0.5f)
                        {
                            // The vertex is outside the rect.
        //                    std::cout << "Vertex outside." << std::endl;
                        }
                        else
                        {
                            GLfloat pDepthData[m_Viewport[2] * m_Viewport[3]];

                            makeCurrent();

                            glReadPixels(0, 0, m_Viewport[2], m_Viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, pDepthData);

                            doneCurrent();

                            if (Model::VertexVisible(pVertex, pDepthData, m_ModelView, m_proj, m_Viewport))
                            {
                                if (m_SelectingModelContour > MODEL_CONTOUR_TYPE_NULL)
                                {
                                    SelectModelContour(index, m_CheckedModels.at(i), m_SelectingModelContour);
                                }
                                else
                                {
                                    pVertex->_Selected = true;
                                    model.SetHasSelectedVertices(true);
                                }

            //                    std::cout << "Vertex inside." << std::endl;
                            }
                        }
                    }

                    ++index;
                }
            }
        }

        UpdateModel(false, false, false);
    }

    m_SelectingMultipleVertices = false;
    m_VertexSelectionRect.setLeft(0.0f);
    m_VertexSelectionRect.setRight(0.0f);
    m_VertexSelectionRect.setTop(0.0f);
    m_VertexSelectionRect.setBottom(0.0f);

    setCursor(Qt::PointingHandCursor);

//    updateGL();
    update();

    m_PrevMousePos[0] = e->localPos().x();
    m_PrevMousePos[1] = e->localPos().y();
}

void GLWidget::wheelEvent(QWheelEvent *e)
{
    GLfloat move;

    if(e->modifiers().testFlag(Qt::ControlModifier))
        move = sensibilityPlus;
    else
        move = sensibility;

    if(tumorMode)
    {
        if(e->modifiers().testFlag(Qt::AltModifier))
        {
            tumorRadius -= (((GLfloat)e->delta()/120))*move/scaleFactor/10000;
            createTumor(true);
        }

        else
        {
            QVector3D coordTumorVect;

            coordTumorVect.setZ(-(((GLfloat)e->delta()/120))*move/scaleFactor);

            QMatrix4x4 r;
            r.rotate(trackball.rotation());
            r = r.transposed();
            coordTumorVect = r * coordTumorVect;

            coordTumor.setX(coordTumor.x()+coordTumorVect.x());
            coordTumor.setY(coordTumor.y()+coordTumorVect.y());
            coordTumor.setZ(coordTumor.z()+coordTumorVect.z());

//            updateGL();
            update();
        }
    }
    else
    {
        if (!m_EditingModel)
        {
            coordModels[2] = coordModels.z()-(((GLfloat)e->delta()/120))*move/scaleFactor;
        }
        else
        {
            QVector3D vec(0.0, 0.0, -(((GLfloat)e->delta()/120))*move/scaleFactor);
            QMatrix4x4 r;
            r.rotate(trackball.rotation());
            r = r.transposed(); // Inverse.
            vec = r * vec;

            // Translate the selected vertices in the model.
            for (Model& model : m_Models)
            {
                model.TranslateSelectedVertices(Eigen::Vector3f(vec.x(), vec.y(), vec.z()));

                for (Model::Vertex*& pVertex : model.Vertices())
                {
                    if (pVertex->_Selected)
                    {
                        pVertex->_Moved = true;
                    }
                }
            }

            UpdateModel();
        }
    }

    if (!m_EditingModel && m_SelectingContour)
    {
//        m_world.setToIdentity();

////        m_camera.scale(scaleFactor);
//        m_camera.setToIdentity();
//        m_camera.translate(coordModels[0], coordModels[1], coordModels[2]); // Keyboard and wheel translation

//        QMatrix4x4 m;
//        m.rotate(trackball.rotation());
//        m_camera *= m;

//        m_ModelView = m_camera * m_world;

//        m_Models[0].ExtractVerticesOnContour(m_ModelView, m_proj, m_Viewport);
    }

    //        updateGL();
    update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
    GLfloat move = 0.0f;

    if(e->modifiers() & Qt::ControlModifier)
        move = sensibilityPlus;
    else
        move = sensibility;

    // Toggle model editing.
    if ((e->key() == Qt::Key_E) && !e->isAutoRepeat())
    {
        m_EditingModel = !m_EditingModel;

        if (m_EditingModel)
        {
            std::cout << "Model editing on." << std::endl;
        }
        else
        {
            std::cout << "Model editing off." << std::endl;

            m_TranslatingVertices = false;
            m_SelectingMultipleVertices = false;
            m_VertexSelectionRect.setRect(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Toggle rendering background image.
    if ((e->key() == Qt::Key_B) && !e->isAutoRepeat())
    {
        m_RenderingBackgroundImage = !m_RenderingBackgroundImage;

        if (m_RenderingBackgroundImage)
        {
            std::cout << "Rendering background on." << std::endl;
        }
        else
        {
            std::cout << "Rendering background image off." << std::endl;
        }
    }

    if ((e->key() == Qt::Key_R) && !e->isAutoRepeat())
    {
        // Unselect all the vertices in the model.
        for (Model& model : m_Models)
        {
            model.UnselectAllVertices();
        }

        UpdateModel(false, false, false);
    }

    if ((e->key() == Qt::Key_I) && !e->isAutoRepeat())
    {
        if (e->modifiers() & Qt::ShiftModifier)
        {
            RunOptimisation(false);
        }
        else
        {
            RunOptimisation(true);
        }
    }

    // Toggle contour selection.
    if ((e->key() == Qt::Key_C) && !e->isAutoRepeat())
    {
        m_SelectingContour = !m_SelectingContour;

        if (m_SelectingContour)
        {
//            m_Models[0].ExtractVerticesOnContour(m_ModelView, m_proj, m_Viewport);

//            std::vector<GLfloat> contour;

//            for (Segment& segment : m_Models[0].Contour())
//            {
//                contour.push_back(segment.source().x());
//                contour.push_back(segment.source().y());
//                contour.push_back(segment.target().x());
//                contour.push_back(segment.target().y());
//            }

//            QOpenGLVertexArrayObject::Binder contourVAOBinder(&m_ContourVAO);
//            m_ContourVBO.bind();
//            m_ContourVBO.allocate(contour.data(), contour.size() * sizeof(GLfloat));

            std::cout << "Contour selection on." << std::endl;
        }
        else
        {
//            m_Models[0].RemoveContour();

            std::cout << "Contour selection off." << std::endl;
        }
    }

    // Pre-calibrate for the light intensity and camera response.
    if ((e->key() == Qt::Key_X) && !e->isAutoRepeat())
    {
        PreCameraCalibration();
    }

    if ((e->key() == Qt::Key_J) && !e->isAutoRepeat())
    {
        m_SelectingShadingOptimisationRegion = !m_SelectingShadingOptimisationRegion;

        if (m_SelectingShadingOptimisationRegion)
        {
            std::cout << "Selecting shading optimisation region on." << std::endl;
        }
        else
        {
            std::cout << "Selecting shading optimisation region off." << std::endl;
        }
    }

    // Load a ground truth/initial model data from a file.
    if ((e->key() == Qt::Key_L) && !e->isAutoRepeat())
    {
        if (e->modifiers() & Qt::ShiftModifier)
        {
            QString fileName = QString("./../tensorflow/liver_data/fine_registration/ground_truth_model");
            LoadModelData(fileName, true);
        }
        else
        {
            QString fileName = QString("./../tensorflow/liver_data/fine_registration/initial_model");
            LoadModelData(fileName, false);
        }
    }

    // Save the ground truth/initial model data to a file.
    if ((e->key() == Qt::Key_O) && !e->isAutoRepeat())
    {
        if (e->modifiers() & Qt::ShiftModifier)
        {
            if (!m_RenderingGroundTruthModel)
            {
                QString fileName = QString("./../tensorflow/liver_data/fine_registration/ground_truth_model");
                SaveModelData(fileName, true);
            }
            else
            {
                std::cout << "Turn off rendering the ground truth model first." << std::endl;
            }
        }
        else
        {
            QString fileName = QString("./../tensorflow/liver_data/fine_registration/initial_model");
            SaveModelData(fileName, false);
        }
    }

    // Toggle rendering the ground truth model.
    if ((e->key() == Qt::Key_G) && !e->isAutoRepeat())
    {
        if (m_GroundTruthModelExisting)
        {
            m_RenderingGroundTruthModel = !m_RenderingGroundTruthModel;

            if (m_RenderingGroundTruthModel)
            {
                std::cout << "Rendering ground truth model on." << std::endl;
            }
            else
            {
                std::cout << "Rendering ground truth model off." << std::endl;
            }
        }
    }

    // Toggle rendering the ground truth model contour.
    if ((e->key() == Qt::Key_H) && !e->isAutoRepeat())
    {
        if (m_GroundTruthModelExisting)
        {
            if (m_RenderingGroundTruthModelContour < 2)
            {
                ++m_RenderingGroundTruthModelContour;
            }
            else
            {
                m_RenderingGroundTruthModelContour = 0;
            }

            switch (m_RenderingGroundTruthModelContour)
            {
                case 0:
                {
                    std::cout << "Rendering ground truth model contour off." << std::endl;
                }
                break;
                case 1:
                {
                    std::cout << "Rendering ground truth model contour on." << std::endl;
                }
                break;
                case 2:
                {
                    std::cout << "Rendering ground truth model contour inside FOV on." << std::endl;
                }
                break;
            }
        }
    }

    // Compute the vertex errors between the ground truth and current model.
    if ((e->key() == Qt::Key_U) && !e->isAutoRepeat())
    {
        std::vector<Model::Vertex*> vertices;

        for (Model& model : m_GroundTruthModels)
        {
            for (Model::Vertex* pVertex : model.Vertices())
            {
                vertices.push_back(pVertex);
            }
        }

        ComputeVertexErrors(vertices, m_ModelForSimulation.Vertices());
    }

    // Run material model solver.
    if ((e->key() == Qt::Key_P) && !e->isAutoRepeat())
    {
        m_MaterialModelSolver.SetNumOfInnerIterations(4);
        m_MaterialModelSolver.SetNumOfOuterIterations(400);

//        m_MaterialModelSolver.Solve();
        m_MaterialModelSolver.SolveCUDA();
        m_MaterialModelSolver.SetAllParticlesUnMoved();

//        UpdateTumorTransform();
        UpdateModel();
    }

    if ((e->key() == Qt::Key_Q) && !e->isAutoRepeat())
    {
        if (e->modifiers() & Qt::ShiftModifier)
        {
            // Load the disconnected cell groups for simulation from a file.
            LoadDisconnectedCellGroupsForSimulation();
        }
        else
        {
            // Save the disconnected cell groups for simulation to a file.
            SaveDisconnectedCellGroupsForSimulation();
        }
    }

    // Generate training set for CNNs.
    if ((e->key() == Qt::Key_T) && !e->isAutoRepeat())
    {
//        GenerateTrainingSet();
//        GenerateTestSet();
        GenerateDeformedModelTrainingSet();
    }

    bool modelEdited = false;

    if(tumorMode)
    {
        QVector3D coordTumorVect;

        switch (e->key())
        {
            case Qt::Key_Left:
            case Qt::Key_A:
                coordTumorVect.setX(-move/scaleFactor);
                break;
            case Qt::Key_Right:
            case Qt::Key_D:
                coordTumorVect.setX(move/scaleFactor);
                break;
            case Qt::Key_Down:
            case Qt::Key_S:
                coordTumorVect.setY(-move/scaleFactor);
                break;
            case Qt::Key_Up:
            case Qt::Key_W:
                coordTumorVect.setY(move/scaleFactor);
                break;
            case Qt::Key_Enter:
            case Qt::Key_Return:
                emit tumorModeIsON(false);
                break;
            case Qt::Key_Escape:
                tumorMode=false;
                emit tumorModeIsON(false);
                resetTumor();
                break;
            default:
                break;
        }

        if(e->key()==Qt::Key_Left || e->key()==Qt::Key_Right || e->key()==Qt::Key_Down || e->key()==Qt::Key_Up
        || e->key()==Qt::Key_A || e->key()==Qt::Key_D || e->key()==Qt::Key_S || e->key()==Qt::Key_W)
        {
            QMatrix4x4 r;
            r.rotate(trackball.rotation());
            r = r.transposed();
            coordTumorVect = r * coordTumorVect;

            coordTumor.setX(coordTumor.x()+coordTumorVect.x());
            coordTumor.setY(coordTumor.y()+coordTumorVect.y());
            coordTumor.setZ(coordTumor.z()+coordTumorVect.z());
        }
    }
    else
    {      
        if (!m_EditingModel)
        {
            switch (e->key())
            {
                case Qt::Key_Left:
                case Qt::Key_A:
                    if(e->modifiers() & Qt::ShiftModifier)
                    {
                        // Move the FOV viewport.
                        m_FOVPosOffset[0] -= 500 * move / scaleFactor;
                    }
                    else
                    {
                        coordModels[0] = coordModels.x()-move/scaleFactor;
                        modelEdited = true;
                    }

                    break;
                case Qt::Key_Right:
                case Qt::Key_D:
                    if(e->modifiers() & Qt::ShiftModifier)
                    {
                        // Move the FOV viewport.
                        m_FOVPosOffset[0] += 500 * move / scaleFactor;
                    }
                    else
                    {
                        coordModels[0] = coordModels.x()+move/scaleFactor;
                        modelEdited = true;
                    }
                    break;
                case Qt::Key_Down:
                case Qt::Key_S:
                    if(e->modifiers() & Qt::ShiftModifier)
                    {
                        // Move the FOV viewport.
                        m_FOVPosOffset[1] -= 500 * move / scaleFactor;
                    }
                    else
                    {
                        coordModels[1] = coordModels.y()-move/scaleFactor;
                        modelEdited = true;
                    }
                    break;
                case Qt::Key_Up:
                case Qt::Key_W:
                    if(e->modifiers() & Qt::ShiftModifier)
                    {
                        // Move the FOV viewport.
                        m_FOVPosOffset[1] += 500 * move / scaleFactor;
                    }
                    else
                    {
                        coordModels[1] = coordModels.y()+move/scaleFactor;
                        modelEdited = true;
                    }
                    break;
                case Qt::Key_Escape:
                    if(distanceMode)
                    {
                        distanceMode=false;
                        tags = 0;
                        distanceCoordinates1 = QVector3D(0,0,0);
                        distanceBetweenTags = 0;
                        emit distanceModeIsON(false);
                    }
                    else
                        qApp->quit();
                    break;
                default:
                    break;
            }
        }
        else
        {
            float dx = 0.0f, dy = 0.0f;

            switch (e->key())
            {
                case Qt::Key_Left:
                case Qt::Key_A:
                    dx = -move / scaleFactor;
                    modelEdited = true;
                    break;
                case Qt::Key_Right:
                case Qt::Key_D:
                    dx = move / scaleFactor;
                    modelEdited = true;
                    break;
                case Qt::Key_Down:
                case Qt::Key_S:
                    dy = -move / scaleFactor;
                    modelEdited = true;
                    break;
                case Qt::Key_Up:
                case Qt::Key_W:
                    dy = move / scaleFactor;
                    modelEdited = true;
                    break;
                case Qt::Key_Escape:
                    if(distanceMode)
                    {
                        distanceMode=false;
                        tags = 0;
                        distanceCoordinates1 = QVector3D(0,0,0);
                        distanceBetweenTags = 0;
                        emit distanceModeIsON(false);
                    }
                    else
                        qApp->quit();
                    break;
                case Qt::Key_Enter:
                case Qt::Key_Return:
                    {
                        switch (m_SelectingModelContour)
                        {
                            case MODEL_CONTOUR_TYPE_NULL:
                                {

                                }
                                break;
                            case MODEL_CONTOUR_TYPE_FRONTIER:
                                {
                                    // Finish adding a frontier contour.                            
                                    for (int index : m_HighCurvatureVertexIndices)
                                    {
                                        if (m_Models.back().Vertices()[index]->_Selected)
                                        {
                                            m_ModelContours.back().push_back(std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE>(index, m_Models.size() - 1, MODEL_CONTOUR_TYPE_FRONTIER));
                                        }
                                    }

                                    m_ModelContoursList << QString("frontier_") + QString::number(m_FrontierContourCount);
                                    m_SelectingModelContour = MODEL_CONTOUR_TYPE_NULL;
                                    ++m_FrontierContourCount;

                                    for (Model& model : m_Models)
                                    {
                                        model.UnselectAllVertices();
                                    }

                                    m_SelectingHighCurvatureVertices = false;

                                    emit ModelContoursChanged();
                                }
                                break;
                            case MODEL_CONTOUR_TYPE_OCCLUDING:
                                {
                                    // Finish adding an occluding contour.
                                    m_ModelContoursList << QString("occluding_") + QString::number(m_OccludingContourCount);
                                    m_SelectingModelContour = MODEL_CONTOUR_TYPE_NULL;
                                    ++m_OccludingContourCount;

                                    for (Model& model : m_Models)
                                    {
                                        model.UnselectAllVertices();
                                    }

                                    emit ModelContoursChanged();
                                }
                                break;
                            case MODEL_CONTOUR_TYPE_LIGAMENT:
                                {
                                    // Finish adding a ligament contour.
                                    m_ModelContoursList << QString("ligament_") + QString::number(m_LigamentContourCount);
                                    m_SelectingModelContour = MODEL_CONTOUR_TYPE_NULL;
                                    ++m_LigamentContourCount;

                                    for (Model& model : m_Models)
                                    {
                                        model.UnselectAllVertices();
                                    }

                                    emit ModelContoursChanged();
                                }
                                break;
                            default:
                                break;
                        }
                    }
                default:
                    break;
            }

            QVector3D vec(dx, dy, 0.0);
            QMatrix4x4 r;
            r.rotate(trackball.rotation());
            r = r.transposed(); // Inverse.
            vec = r * vec;

            // Translate the selected vertices in the model.
            for (Model& model : m_Models)
            {
                model.TranslateSelectedVertices(Eigen::Vector3f(vec.x(), vec.y(), vec.z()));

                for (Model::Vertex*& pVertex : model.Vertices())
                {
                    if (pVertex->_Selected)
                    {
                        pVertex->_Moved = true;
                    }
                }
            }

            UpdateModel();
        }
    }

    if (modelEdited && m_SelectingContour)
    {
//        m_world.setToIdentity();

////        m_camera.scale(scaleFactor);
//        m_camera.setToIdentity();
//        m_camera.translate(coordModels[0], coordModels[1], coordModels[2]); // Keyboard and wheel translation

//        QMatrix4x4 m;
//        m.rotate(trackball.rotation());
//        m_camera *= m;

//        m_ModelView = m_camera * m_world;

//                m_Models[0].ExtractVerticesOnContour(m_ModelView, m_proj, m_Viewport);
    }

    //    updateGL();
        update();
}

void GLWidget::keyReleaseEvent(QKeyEvent *e)
{
}

/* ============================ OPERATORS ============================ */
void GLWidget::multMatrix(const QMatrix4x4& m)    // Multiplies QMatrix4x4 as a GLfloat matrix
{
     static GLfloat mat[16];
     const GLfloat *data = m.constData();
     for (GLint index = 0; index < 16; ++index)
        mat[index] = data[index];
     glMultMatrixf(mat);
}

QPointF GLWidget::pixelPosToViewPos(const QPointF& p)    // Sets pixel coordinates to view coordinates
{
    return QPointF(2.0*GLfloat(p.x())/width()-1.0, 1.0-2.0*GLfloat(p.y())/height());
}

QVector3D GLWidget::screenToModelPixel(const QPointF& screenCoordinates, bool* pModelPicked) // Sets 2D screen coordinates to 3D model coordinates
{
//    GLint viewport[4];
//    GLdouble modelview[16];
//    GLdouble projection[16];
    GLfloat winZ;
    GLdouble posX, posY, posZ;
//    unsigned char rgba[4] = { 0, };

//    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
//    glGetDoublev(GL_PROJECTION_MATRIX, projection);
//    glGetIntegerv(GL_VIEWPORT, viewport);

    makeCurrent();

//    glPixelStorei(GL_PACK_ALIGNMENT, 1);
//    glReadPixels((GLint)screenCoordinates.x(), (GLint)(m_Viewport[3] - screenCoordinates.y()), 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &rgba);
    glReadPixels((GLint)screenCoordinates.x(), (GLint)(m_Viewport[3] - screenCoordinates.y()), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);

    // TODO: Replace/implement gluUnProject().
//    gluUnProject((GLdouble)screenCoordinates.x(), (GLdouble)(viewport[3] - screenCoordinates.y()), winZ, modelview, projection, viewport, &posX, &posY, &posZ);

//    std::cout << winZ << std::endl;
//    std::cout << (int)rgba[0] << ", " << (int)rgba[1] << ", " << (int)rgba[2] << ", " << (int)rgba[3] << std::endl;

    doneCurrent();

    if (pModelPicked)
    {
        if (winZ < 1.0f)
        {
            // Model picked.
            *pModelPicked = true;
        }
        else
        {
            // Nothing picked.
            *pModelPicked = false;
        }
    }

    QVector3D pos(posX, posY, posZ);
    QMatrix4x4 r;
    r.rotate(trackball.rotation());
    pos = r * pos;

    return QVector3D(pos.x(), pos.y(), pos.z());
}

/* ============================ GETTERS ============================ */
GLfloat GLWidget::getFramePictureRatio()
{
    return frame_picture_Ratio;
}
qreal GLWidget::getCameraSettings(GLint settingNumber)
{
    return cameraParameters[settingNumber];
}
GLfloat GLWidget::getSensibility()
{
    return sensibility;
}
GLfloat GLWidget::getSensibilityPlus()
{
    return sensibilityPlus;
}
GLfloat GLWidget::getRotationSpeed()
{
    return rotationSpeed;
}
GLfloat GLWidget::getTagsRadius()
{
    return tagsRadius;
}
QStringList GLWidget::getModelsList()
{
    return modelsList;
}

QStringList GLWidget::GetModelContoursList()
{
    return m_ModelContoursList;
}

QStringList GLWidget::GetImageContoursList()
{
    return m_ImageContoursList;
}

/* ============================ SETTERS ============================ */
void GLWidget::setOpacity(GLint sliderValue)
{
    opacity = ((float)sliderValue/100);

    Eigen::Vector4f colour;

    if (m_CheckedModels.size() > 0)
    {
        for(unsigned int i = 0; i < (unsigned int)m_CheckedModels.size(); i++)
        {
            Model& model = m_Models[m_CheckedModels.at(i)];

            for (Model::Vertex* pVertex : model.Vertices())
            {
                pVertex->_Colour[3] = opacity;
            }

            m_Models[m_CheckedModels.at(i)].BuildVertexData();
            UpdateModelVBO(m_CheckedModels.at(i));
        }
    }

    update();
}

void GLWidget::setFramePictureRatio(GLfloat new_frame_picture_Ratio)
{
    float scale = 1.0f;

    if (frame_picture_Ratio > 0.0f)
    {
        scale = new_frame_picture_Ratio / frame_picture_Ratio;
    }
    else
    {
        scale = new_frame_picture_Ratio;
    }

    frame_picture_Ratio = new_frame_picture_Ratio;

    resizeWidget();

//    Eigen::Vector2f translation(this->width() * scale, this->height() * scale);

//    if (scale < 1.0f)
//    {
//        translation = -translation;
//    }

//    for (Eigen::Vector2f& point : m_GroundTruthModelContour)
//    {
//        point += translation;
//    }

//    for (Eigen::Vector2f& point : m_ContourSelectionPoints)
//    {
//        point += translation;
//    }

    update();
}

void GLWidget::setCameraSettings(GLint settingNumber, qreal newValue)
{
    if (settingNumber == 7)
    {
        m_WindowScale = newValue;
    }

    cameraParameters[settingNumber] = newValue;

    if (settingNumber == 7)
    {
        resizeWidget();
    }

    update();
}
void GLWidget::setSensibility(GLfloat newValue)
{
    sensibility = newValue;
}
void GLWidget::setSensibilityPlus(GLfloat newValue)
{
    sensibilityPlus = newValue;
}
void GLWidget::setRotationSpeed(GLfloat newValue)
{
    rotationSpeed = newValue;
}
void GLWidget::scaleSliderState(bool newState)
{
    scaleSliderPressed = newState;
}

void GLWidget::FOVScaleSliderState(bool newState)
{
    m_FOVScaleSliderPressed = newState;
}

void GLWidget::SetHighCurvatureStartPosition(float Position)
{
    if (m_SelectingHighCurvatureVertices)
    {
        std::cout << "Start pos: " << Position << std::endl;

        m_HighCurvatureStartPosition = floor(Position * m_HighCurvatureVertexIndices.size());

        UpdateSelectedHighCurvatureVertices();
    }
}

void GLWidget::SetHighCurvatureEndPosition(float Position)
{
    if (m_SelectingHighCurvatureVertices)
    {
        std::cout << "End pos: " << Position << std::endl;

        m_HighCurvatureEndPosition = floor(Position * m_HighCurvatureVertexIndices.size());

        UpdateSelectedHighCurvatureVertices();
    }
}

void GLWidget::SetHighCurvatureRangeReversed(bool Reversed)
{
    m_HighCurvatureRangeReversed = Reversed;

    if (Reversed)
    {
        std::cout << "m_HighCurvatureRangeReversed: true" << std::endl;
    }
    else
    {
        std::cout << "m_HighCurvatureRangeReversed: false" << std::endl;
    }

    UpdateSelectedHighCurvatureVertices();
}

void GLWidget::SetHighCurvatureVertexSearchAreaRadius(float Radius)
{
    m_HighCurvatureVertexSearchAreaRadius = Radius;

    std::cout << "m_HighCurvatureVertexSearchAreaRadius: " << m_HighCurvatureVertexSearchAreaRadius << std::endl;


    Model& model = m_Models.back();
    model.UnselectAllVertices();
    SelectHighCurvatureVertices();
}

void GLWidget::SetHighCurvatureVerticesPolynomialOrder(int Order)
{
    m_HighCurvatureVerticesPolynomialOrder = Order;

    std::cout << "m_HighCurvatureVerticesPolynomialOrder: " << m_HighCurvatureVerticesPolynomialOrder << std::endl;


    Model& model = m_Models.back();
    model.UnselectAllVertices();
    SelectHighCurvatureVertices();
}

void GLWidget::UpdateSelectedHighCurvatureVertices(void)
{
    Model& model = m_Models.back();
    model.UnselectAllVertices();

    std::cout << "m_HighCurvatureStartPosition: " << m_HighCurvatureStartPosition << ", m_HighCurvatureEndPosition: " << m_HighCurvatureEndPosition << std::endl;
    std::cout << "m_HighCurvatureVertexIndices.size(): " << m_HighCurvatureVertexIndices.size() << std::endl;

    if (m_HighCurvatureRangeReversed)
    {
        for (int i = m_HighCurvatureEndPosition + 1; i < m_HighCurvatureVertexIndices.size(); ++i)
        {
            model.Vertices()[m_HighCurvatureVertexIndices[i]]->_Selected = true;

            std::cout << "not: " << i << std::endl;
        }

        for (int i = 0; i < m_HighCurvatureStartPosition; ++i)
        {
            model.Vertices()[m_HighCurvatureVertexIndices[i]]->_Selected = true;

            std::cout << "here" << std::endl;
        }
    }
    else
    {
        for (int i = m_HighCurvatureStartPosition; i < m_HighCurvatureEndPosition; ++i)
        {
            model.Vertices()[m_HighCurvatureVertexIndices[i]]->_Selected = true;
        }
    }

    UpdateModel(false, false, false);
    update();
}

void GLWidget::setTagsRadius(GLfloat newValue)
{
    tagsRadius = newValue;
}

void GLWidget::SetRenderingModelFaces(bool Rendering)
{
    m_RenderingModelFaces = Rendering;
}

void GLWidget::SetFOVScale(float Scale)
{
    m_FOVScale = Scale;

    update();

    std::cout << "FOV Scale: " << Scale << std::endl;
}

void GLWidget::SetCheckedModels(QVector<unsigned int>& CheckedModels)
{
    m_CheckedModels = CheckedModels;
}

void GLWidget::SetSelectedModel(QString& SelectedModel)
{
    m_SelectedModel = SelectedModel;
}

void GLWidget::SetCheckedModelContours(QVector<unsigned int>& CheckedModelContours)
{
    m_CheckedModelContours = CheckedModelContours;
}

void GLWidget::SetSelectedModelContour(QString& SelectedModelContour)
{
    m_SelectedModelContour = SelectedModelContour;
}

void GLWidget::SetCheckedImageContours(QVector<unsigned int>& CheckedImageContours)
{
    m_CheckedImageContours = CheckedImageContours;
}

void GLWidget::SetSelectedImageContour(QString& SelectedImageContour)
{
    m_SelectedImageContour = SelectedImageContour;
}

void GLWidget::ChangeColour(QColor& Colour)
{
    m_cForRendering = 0.015f;

    Eigen::Vector4f colour;

/*    if (!m_SelectedModel.isEmpty())
    {
        colour << Colour.redF(), Colour.greenF(), Colour.blueF(), Colour.alphaF();

        unsigned int modelNumber = modelsList.indexOf(m_SelectedModel,0);
        m_Models[modelNumber].SetColour(colour);

        m_Models[modelNumber].BuildVertexData();
        UpdateModelVBO(modelNumber);
    }
    else */if (m_CheckedModels.size() > 0)
    {
        for(GLuint i = 0; i < (GLuint)m_CheckedModels.size(); i++)
        {
            colour = m_Models[m_CheckedModels.at(i)].Colour();

            colour << Colour.redF(), Colour.greenF(), Colour.blueF(), colour[3];
            m_Models[m_CheckedModels.at(i)].SetColour(colour);

            m_Models[m_CheckedModels.at(i)].BuildVertexData();
            UpdateModelVBO(m_CheckedModels.at(i));
        }
    }

    update();
}

void GLWidget::ContourSelection(bool Enabled, bool Finalised)
{
    m_IsContourSelectionOn = Enabled;

    if (!Enabled && !Finalised)
    {
        ResetContourSelection();

        m_RenderingGroundTruthModelContour = 0;
    }
    else if (!Enabled && Finalised)
    {
        FinaliseContourSelection();

        // Finish adding an image contour.
        m_ImageContoursList << QString("contour_") + QString::number(m_ImageContourCount);
        ++m_ImageContourCount;

        emit ImageContoursChanged();

        m_RenderingGroundTruthModelContour = 2;
    }
}

void GLWidget::ResetContourSelection()
{
    m_ContourSelectionPoints.clear();
    m_GroundTruthModelContour.clear();

    update();
}

void GLWidget::FinaliseContourSelection()
{
    // Rasterise the paths from contour selection points.
    m_GroundTruthModelContour.clear();

    unsigned int numOfPoints = m_ContourSelectionPoints.size();
    Eigen::Vector2f point0, point1;

//    float scale = frame_picture_Ratio * cameraParameters[7];

//    if (scale < 1e-12)
//    {
//        scale = 1.0f;
//    }

    for (unsigned int i = 0; i < numOfPoints - 1; ++i)
    {
        point0 = m_ContourSelectionPoints[i];
        point1 = m_ContourSelectionPoints[i + 1];

        RasteriseLine((int)point0[0], (int)point0[1], (int)point1[0], (int)point1[1], m_GroundTruthModelContour);
    }

    m_ImageContours.push_back(m_GroundTruthModelContour);

    m_ContourSelectionPoints.clear();
}

// Rasterise a line from two 2D points.
// Reference: http://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C
void GLWidget::RasteriseLine(int x0, int y0, int x1, int y1, std::vector<Eigen::Vector2f>& Pixels)
{
    int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
    int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1;
    int err = (dx>dy ? dx : -dy)/2, e2;

    for(;;)
    {
        Pixels.push_back(Eigen::Vector2f(x0, y0));

        if (x0==x1 && y0==y1) break;

        e2 = err;

        if (e2 >-dx)
        {
            err -= dy;
            x0 += sx;
        }

        if (e2 < dy)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void GLWidget::ComputeModelSegmentation()//Model& OriginalModel, Model& SegmentationModel)
{
    // TODO: Temp - currently hard coded.
    Model& OriginalModel = m_Models[2];
    Model& SegmentationModel = m_ModelForSegmentation;

    float threshold = 0.02f;

    for (Model::Vertex* pVertexA : OriginalModel.Vertices())
    {
        for (Model::Vertex* pVertexB : SegmentationModel.Vertices())
        {
            if ((pVertexA->_Pos - pVertexB->_Pos).norm() < threshold)
            {
                pVertexA->_Selected = true;
            }
        }
    }

    OriginalModel.SetHasSelectedVertices(true);

    UpdateModel(false, false, false);
    update();
}

void GLWidget::ComputeGaussianCurvature()
{
    for (Model& model : m_Models)
    {
        model.ComputeGaussianCurvature();
    }

    UpdateModel(false, false, false);
    update();
}

void GLWidget::SelectHighCurvatureVertices(void)
{
    // Project high curvature vertices onto 2D and choose ones that are on the contour.
    // When projecting onto 2D, have to find an orientation of the mesh where the high curvature vertices are mostly on the contour.
    ComputeGaussianCurvature();
    m_SelectingHighCurvatureVertices = true;

    // Compute the principal axes of the mesh.
    int numOfVertices = 0;

    // For now, only consider the liver, not tumour and vein.
    Model& model = m_Models.back();

//    for (Model& model : m_Models)
//    {
//        numOfVertices += model.Vertices().size();
//    }

    numOfVertices = model.Vertices().size();

    Eigen::MatrixXf matVertices;
    matVertices.resize(numOfVertices, 3);

    QVector3D vertex;
    int index = 0;

    //for (Model& model : m_Models)
    {
        for (Model::Vertex* pVertex : model.Vertices())
        {
            vertex.setX(pVertex->_Pos[0]);
            vertex.setY(pVertex->_Pos[1]);
            vertex.setZ(pVertex->_Pos[2]);

            matVertices.row(index) << vertex.x(), vertex.y(), vertex.z();

            ++index;
        }
    }

    Eigen::MatrixXf centred = matVertices.rowwise() - matVertices.colwise().mean();
    Eigen::MatrixXf cov = centred.adjoint() * centred;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);

    // Lowest variance axis.
    Eigen::Vector3f axis = eig.eigenvectors().col(0); // This can be 0 or 1.

    // Align the axis to z-axis.
    QVector3D axis3D;
    axis3D.setX(axis[0]);
    axis3D.setY(axis[1]);
    axis3D.setZ(axis[2]);
    axis3D.normalize();

    QVector3D axisZ(0.0, 0.0, 1.0);
    float angle = acos(QVector3D::dotProduct(axis3D, axisZ));
    axis3D = QVector3D::crossProduct(axis3D, axisZ);

    // Rotate the model around the axis by the angle.
    QMatrix4x4 mat;
    mat.setToIdentity();
    mat.rotate(angle * 180.0f / M_PI, axis3D);

    QMatrix3x3 rotMat;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            rotMat(i, j) = mat(i, j);
        }
    }

    trackball.setRotation(QQuaternion::fromRotationMatrix(rotMat));

//    ComputeModelCentroid();
    coordModels = -model.Centroid(); //-m_ModelCentroid;
    coordModels[2] -= 0.35f;

    QMatrix4x4 translationMat;
    translationMat.setToIdentity();
    translationMat.translate(coordModels[0], coordModels[1], coordModels[2]);
    mat = translationMat * mat;

    // Project high curvature vertices onto 2D.
    std::vector<Point>& points = m_HighCurvaturePoints;
    points.clear();

    for (Model::Vertex* pVertex : model.Vertices())
    {
        if (pVertex->_HighCurvature)
        {
            points.push_back(Model::ProjectVertexOnto2D(pVertex, mat, m_proj, m_Viewport));
        }
    }

    this->repaint();

    // Select the vertices whose 2D projection points are close to the contour of the model.
    makeCurrent();
    GetContour(m_ModelContour);
    doneCurrent();

    Eigen::Vector2f point2D;
    index = 0;
    int numOfSelectedPoints = 0;

    for (Model::Vertex* pVertex : model.Vertices())
    {
        if (pVertex->_HighCurvature)
        {
            point2D << points[index].x(), points[index].y();

            for (Eigen::Vector2f& contourPoint : m_ModelContour)
            {
                if ((point2D - contourPoint).dot(point2D - contourPoint) < 40.0f * cameraParameters[7] * cameraParameters[7])
                {
                    pVertex->_Selected = true;
                    ++numOfSelectedPoints;

                    break;
                }
            }

            ++index;
        }
    }

    // Filter out far points.
    // Compute the distance between two points and remove ones that are big.
    int i = 0, j = 0;
    Eigen::Vector3f pointA, pointB;
    std::vector<float> distances;

    for (Model::Vertex* pVertex : model.Vertices())
    {
        if (pVertex->_Selected)
        {
//            pointA << points[i].x(), points[i].y();
            pointA = pVertex->_Pos;

            j = 0;
            float distanceSum = 0.0f;

            for (Model::Vertex* pVertexB : model.Vertices())
            {
                if (pVertexB->_Selected)
                {
//                    pointB << points[j].x(), points[j].y();
                    pointB = pVertexB->_Pos;

                    distanceSum += (pointA - pointB).dot(pointA - pointB);

                    ++j;
                }
            }

            distances.push_back(distanceSum / (float)numOfSelectedPoints);

            ++i;
        }
    }

    // Compute the Median Absolute Deviation.
    std::vector<float> meanDistances = distances;


    float MAD = Utils::MedianAbsoluteDeviation(meanDistances);
//    std::cout << "MAD: " << MAD << std::endl;

    MAD *= 1.4826f;

    meanDistances = distances;
    std::nth_element(meanDistances.begin(), meanDistances.begin() + meanDistances.size() / 2, meanDistances.end());
    float median = meanDistances[meanDistances.size() / 2];

    index = 0;
    int vertexIndex = 0;
//    std::map<int, Eigen::Vector2f> mapVertexIndexToPoint;

    for (Model::Vertex* pVertex : model.Vertices())
    {
        if (pVertex->_Selected)
        {
            if (distances[index] > 2.0f * MAD + median)
            {
                pVertex->_Selected = false;
            }
            else
            {
//                mapVertexIndexToPoint[vertexIndex] = Eigen::Vector2f(points[index].x(), points[index].y());
            }

            ++index;
        }

        ++vertexIndex;
    }

    // Compute the distances between vertices in 3D.
    std::map<int, std::map<int, float> > mapDistances;
    i = 0;

    for (Model::Vertex* pVertexA : model.Vertices())
    {
        if (pVertexA->_Selected)
        {
            j = 0;
            std::map<int, float> mapSubDistances;

            for (Model::Vertex* pVertexB : model.Vertices())
            {
                if (pVertexA != pVertexB && pVertexB->_Selected)
                {
                    mapSubDistances[j] = (pVertexA->_Pos - pVertexB->_Pos).dot(pVertexA->_Pos - pVertexB->_Pos);
                }

                ++j;
            }

            mapDistances[i] = mapSubDistances;
        }

        ++i;
    }

    // Find a vertex whose mean distance to others are largest.
    float maxDistance = std::numeric_limits<float>::min();
    int maxVertexIndex = 0;

    for (std::map<int, std::map<int, float> >::iterator itMapSubDistances = mapDistances.begin(); itMapSubDistances != mapDistances.end(); ++itMapSubDistances)
    {
        float meanDistance = 0.0f;

        for (std::map<int, float>::iterator itSubDistance = itMapSubDistances->second.begin(); itSubDistance != itMapSubDistances->second.end(); ++itSubDistance)
        {
            meanDistance += itSubDistance->second;
        }

        meanDistance /= (float)itMapSubDistances->second.size();

        if (meanDistance > maxDistance)
        {
            maxDistance = meanDistance;
            maxVertexIndex = itMapSubDistances->first;
        }
    }

    // Add vertices one by one w.r.t. the proximity to the chosen vertex above.
    // Only consider vertex in the same direction as the previous one when adding.
    m_HighCurvatureVertexIndices.clear();
    m_HighCurvatureVertexIndices.push_back(maxVertexIndex);

    std::vector<double> xs, ys, zs, ts, xCoeffs, yCoeffs, zCoeffs;
    xs.push_back(model.Vertices()[maxVertexIndex]->_Pos[0]);
    ys.push_back(model.Vertices()[maxVertexIndex]->_Pos[1]);
    zs.push_back(model.Vertices()[maxVertexIndex]->_Pos[2]);

    for (Model::Vertex* pVertex : model.Vertices())
    {
        pVertex->_HighCurvature = false;
    }

    index = maxVertexIndex;
    model.Vertices()[index]->_HighCurvature = true;

    Eigen::Vector3f prevDirection, direction;

    for (int i = 0; i < mapDistances.size() - 1; ++i)
    {
        float minDistance = std::numeric_limits<float>::max();
        float minDistanceDirection = std::numeric_limits<float>::max();
        int minIndex = 0, tempMinIndex = -1;
        bool found = false;

        for (std::map<int, float>::iterator itMapSubDistance = mapDistances[index].begin(); itMapSubDistance != mapDistances[index].end(); ++itMapSubDistance)
        {
            // For the first vertex, just find the closest one.
            // Otherwise, check the direction to the next vertex
            // if it has the same direction as the previous one.
            if (i > 0)
            {
                if (!model.Vertices()[itMapSubDistance->first]->_HighCurvature && itMapSubDistance->second < m_HighCurvatureVertexSearchAreaRadius/*massicot: 0.001f*/ && itMapSubDistance->second < minDistance)
                {
                    direction = model.Vertices()[itMapSubDistance->first]->_Pos - model.Vertices()[index]->_Pos;
                    direction.normalize();

//                    if (itMapSubDistance->second < minDistance)
//                    {
//                        tempMinIndex = itMapSubDistance->first;
                        minDistance = itMapSubDistance->second;
                        minIndex = itMapSubDistance->first;
//                    }

                    if (direction.dot(prevDirection) >= 0.0f)
                    {
//                        if (itMapSubDistance->second < minDistanceDirection)
//                        {
//                            model.Vertices()[itMapSubDistance->first]->_HighCurvature = true;

//                            minDistanceDirection = itMapSubDistance->second;
//                            minIndex = itMapSubDistance->first;
                            found = true;
//                        }
                    }
                }
            }
            else
            {
                if (itMapSubDistance->second < minDistance/* && itMapSubDistance->second < 0.1f*/)
                {
                    minDistance = itMapSubDistance->second;
                    minIndex = itMapSubDistance->first;
                    found = true;
                }
            }
        }

        if (!found)
        {
            // When there is no vertex whose direction is the same as the previous one,
            // add the closest one.

//            if (tempMinIndex == -1)
//            {
//                tempMinIndex = 0;
//            }

//            minIndex = tempMinIndex;

//            model.Vertices()[minIndex]->_HighCurvature = true;
            found = true;
        }

        if (found)
        {
            m_HighCurvatureVertexIndices.push_back(minIndex);
//            model.Vertices()[minIndex]->_Selected = true;
            model.Vertices()[minIndex]->_HighCurvature = true;
            prevDirection = direction;

            index = minIndex;

            xs.push_back(model.Vertices()[index]->_Pos[0]);
            ys.push_back(model.Vertices()[index]->_Pos[1]);
            zs.push_back(model.Vertices()[index]->_Pos[2]);
        }
    }

    // Fit a curve, on x-, y- and z-axis separately.
    for (int i = 0; i < xs.size(); ++i)
    {
        ts.push_back(i);
    }

    Utils::polyfit(ts, xs, xCoeffs, m_HighCurvatureVerticesPolynomialOrder);
    Utils::polyfit(ts, ys, yCoeffs, m_HighCurvatureVerticesPolynomialOrder);
    Utils::polyfit(ts, zs, zCoeffs, m_HighCurvatureVerticesPolynomialOrder);

//    for (double coeff : xCoeffs)
//    {
//        std::cout << coeff << ", ";
//    }

//    std::cout << std::endl;

    // Evaluation the curve.
    std::vector<double> evalX = Utils::polyval(xCoeffs, ts);
    std::vector<double> evalY = Utils::polyval(yCoeffs, ts);
    std::vector<double> evalZ = Utils::polyval(zCoeffs, ts);
    m_HighCurvatureFittedCurve.clear();

    for (int i = 0; i < ts.size(); ++i)
    {
        m_HighCurvatureFittedCurve.push_back(QVector3D(evalX[i], evalY[i], evalZ[i]));

//        std::cout << evalX[i] << ", " << evalY[i] << ", " << evalZ[i] << std::endl;
    }

    // Choose vertices close to the curve.
    Eigen::Vector3f a, b;
    m_HighCurvatureVertexIndices.clear();
    model.UnselectAllVertices();

    for (int i = 0; i < m_HighCurvatureFittedCurve.size() - 1; ++i)
    {
        a << m_HighCurvatureFittedCurve[i].x(), m_HighCurvatureFittedCurve[i].y(), m_HighCurvatureFittedCurve[i].z();
        b << m_HighCurvatureFittedCurve[i + 1].x(), m_HighCurvatureFittedCurve[i + 1].y(), m_HighCurvatureFittedCurve[i + 1].z();

        index = 0;

        for (Model::Vertex* pVertex : model.Vertices())
        {
            if (!pVertex->_Selected)
            {
                float distance = Utils::PointLineDistance(a, b, pVertex->_Pos);

                if (distance < 0.005f)
                {
                    m_HighCurvatureVertexIndices.push_back(index);
                    pVertex->_Selected = true;

    //                    std::cout << "Distance: " << distance << std::endl;
                }
            }

            ++index;
        }
    }

    m_HighCurvatureStartPosition = 0;
    m_HighCurvatureEndPosition = m_HighCurvatureVertexIndices.size();

    UpdateModel(false, false, false);
    update();
}

void GLWidget::SelectModelContour(unsigned int VertexIndex, unsigned int ModelIndex, MODEL_CONTOUR_TYPE ModelContourType)
{
    Model& model = m_Models[ModelIndex];

    switch (ModelContourType)
    {
        case MODEL_CONTOUR_TYPE_NULL:
            {

            }
            break;
        case MODEL_CONTOUR_TYPE_FRONTIER:
            {
                std::vector<float>::iterator maxIt = std::max_element(model.GaussianCurvatures().begin(), model.GaussianCurvatures().end());
                float max = *maxIt;

                if (model.GaussianCurvatures()[VertexIndex] / max > 0.8f)
                {
                    model.Vertices()[VertexIndex]->_Selected = true;
                    m_ModelContours.back().push_back(std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE>(VertexIndex, ModelIndex, MODEL_CONTOUR_TYPE_FRONTIER));
                }
            }
            break;
        case MODEL_CONTOUR_TYPE_OCCLUDING:
            {
                model.Vertices()[VertexIndex]->_Selected = true;
                m_ModelContours.back().push_back(std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE>(VertexIndex, ModelIndex, MODEL_CONTOUR_TYPE_OCCLUDING));
            }
            break;
        case MODEL_CONTOUR_TYPE_LIGAMENT:
            {
                model.Vertices()[VertexIndex]->_Selected = true;
                m_ModelContours.back().push_back(std::tuple<unsigned int, unsigned int, MODEL_CONTOUR_TYPE>(VertexIndex, ModelIndex, MODEL_CONTOUR_TYPE_LIGAMENT));
            }
            break;
        default:
            break;
    }
}
