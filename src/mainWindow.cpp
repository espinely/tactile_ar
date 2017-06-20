#include "mainWindow.h"

#include <QRadioButton>


MainWindow::MainWindow()
{
    screenshotNumber = 0;       // Number of screenshots taken
    m_pContourSelectionToolWindow = NULL;             // Models List Window OFF
    m_ContourSelectionFinalised = false;

    QWidget *widgetPrincipal = new QWidget;
    setCentralWidget(widgetPrincipal);

    layoutPrincipal = new QGridLayout(widgetPrincipal);
    widgetPrincipal->setLayout(layoutPrincipal);

    object = new GLWidget;
    layoutPrincipal->addWidget(object);
    layoutPrincipal->setMargin(0);

// BOTTOM TOOLBAR

    leftToolBar = addToolBar("Menu");
    leftToolBar->setIconSize(QSize(40,40));

    // LOAD IMAGE
    QAction *actionLoadImage = new QAction(QIcon("../data/img/picture.jpg"), "Load Background &Image", this);
    leftToolBar->addAction(actionLoadImage);
    leftToolBar->addSeparator();

    // LOAD MODEL
    QAction *actionLoadModel = new QAction(QIcon("../data/img/load.png"), "&Load 3D Model", this);
    leftToolBar->addAction(actionLoadModel);
    leftToolBar->addSeparator();

//    // MODELS LIST
////    QAction *actionModelsList = new QAction(QIcon("../data/img/modelslist.png"), "M&odels List", this);
//    QAction *actionModelsList = new QAction(QIcon("../data/img/modelslist.png"), "&Load 3D Models", this);
//    leftToolBar->addAction(actionModelsList);
//    leftToolBar->addSeparator();

//    // SAVE MODEL
//    QAction *actionSaveModel = new QAction(QIcon("../data/img/save.png"), "Save &Model", this);
//    leftToolBar->addAction(actionSaveModel);
//    leftToolBar->addSeparator();

    // Contour selection tool.
    pActionContour = new QAction(QIcon("../data/img/contour.png"), "Contour Selection &Tool", this);
//    pActionContour->setCheckable(true);
    leftToolBar->addAction(pActionContour);
    leftToolBar->addSeparator();

    // Optimisation.
    m_pActionOptimisation = new QAction(QIcon("../data/img/optimisation.png"), "&Optimisation", this);
    leftToolBar->addAction(m_pActionOptimisation);
    leftToolBar->addSeparator();

    // CENTER MODEL
    QAction *actionCenterModel = new QAction(QIcon("../data/img/center.png"), "&Center Model", this);
    leftToolBar->addAction(actionCenterModel);
    leftToolBar->addSeparator();

//    // DISTANCE
//    distance = new QAction(QIcon("../data/img/distance.png"), "&Distance", this);
//    distance->setCheckable(true);
//    leftToolBar->addAction(distance);
//    leftToolBar->addSeparator();

    // ROTATE X
    QAction *rotateX = new QAction(QIcon("../data/img/rotateX.png"), "Rotate Model (&X Axis)", this);
    leftToolBar->addAction(rotateX);
    leftToolBar->addSeparator();

    // ROTATE Y
    QAction *rotateY = new QAction(QIcon("../data/img/rotateY.png"), "Rotate Model (&Y Axis)", this);
    leftToolBar->addAction(rotateY);
    leftToolBar->addSeparator();

    // SCREENSHOT
    QAction *screenshot = new QAction(QIcon("../data/img/screenshot.png"), "Sc&reenshot", this);
    leftToolBar->addAction(screenshot);
    leftToolBar->addSeparator();

    // SETTINGS
    QAction *settings = new QAction(QIcon("../data/img/settings.png"), "&Settings", this);
    leftToolBar->addAction(settings);
    leftToolBar->addSeparator();

    addToolBar(Qt::LeftToolBarArea, leftToolBar);

// RIGHT TOOLBAR

    rightToolBar = addToolBar("Menu");
    const QSize sliderSize(120, 60);

    QWidget *slidersWidget = new QWidget(rightToolBar);     // Widget containing the right toolbar layout
    QVBoxLayout *slidersLayout = new QVBoxLayout(slidersWidget);    // Layout containing the right toolbar elements
    slidersLayout->setAlignment(Qt::AlignCenter);

// OPACITY SLIDER

  // SLIDER TITLE
    QLabel *opacityTitle = new QLabel("Opacity");
    opacityTitle->setAlignment(Qt::AlignCenter);
    slidersLayout->addWidget(opacityTitle);

  // SLIDER
    opacitySlider = new QSlider(Qt::Horizontal);
    opacitySlider->setRange(0,100);
    opacitySlider->setSliderPosition(100);
    opacitySlider->setTickPosition(QSlider::TicksBelow);
    opacitySlider->setTickInterval(10);
    opacitySlider->setMaximumSize(sliderSize);
    slidersLayout->addWidget(opacitySlider);

  // SLIDER TEXTMIN
    QLabel *opacityTextMin = new QLabel("0 %               100 %\n");
    opacityTextMin->setAlignment(Qt::AlignCenter);
    slidersLayout->addWidget(opacityTextMin);

// SCALE SLIDER

  // SLIDER TITLE
    QLabel *scaleTitle = new QLabel("Window Scale");
    scaleTitle->setAlignment(Qt::AlignCenter);
    slidersLayout->addWidget(scaleTitle);

  // SLIDER
    scaleSlider = new QSlider(Qt::Horizontal);
    scaleSlider->setRange(10,200);
    scaleSlider->setSliderPosition(100);
    scaleSlider->setSingleStep(10);
    scaleSlider->setTickPosition(QSlider::TicksBelow);
    scaleSlider->setTickInterval(50);
    scaleSlider->setMaximumSize(sliderSize);
    slidersLayout->addWidget(scaleSlider);

  // SLIDER TEXTMIN
    QLabel *scaleTextMin = new QLabel("0.1 x                  2 x\n");
    scaleTextMin->setAlignment(Qt::AlignCenter);
    slidersLayout->addWidget(scaleTextMin);

// For FOV scale.
// SLIDER TITLE

    QLabel* pFOVScaleTitle = new QLabel("FOV Scale");
    pFOVScaleTitle->setAlignment(Qt::AlignCenter);
    slidersLayout->addWidget(pFOVScaleTitle);

// SLIDER
    m_pFOVScaleSlider = new QSlider(Qt::Horizontal);
    m_pFOVScaleSlider->setRange(1,100);
    m_pFOVScaleSlider->setSliderPosition(100);
    m_pFOVScaleSlider->setSingleStep(5);
    m_pFOVScaleSlider->setTickPosition(QSlider::TicksBelow);
    m_pFOVScaleSlider->setTickInterval(10);
    m_pFOVScaleSlider->setMaximumSize(sliderSize);
    slidersLayout->addWidget(m_pFOVScaleSlider);

// SLIDER TEXTMIN
    QLabel* pFOVScaleTextMin = new QLabel("1 %               100 %\n");
    pFOVScaleTextMin->setAlignment(Qt::AlignCenter);
    slidersLayout->addWidget(pFOVScaleTextMin);

    rightToolBar->addWidget(slidersWidget);
    rightToolBar->addSeparator();

// MODELS LIST
// MODELS LIST TITLE
    QLabel *modelsListTitle = new QLabel("3D Models");
    modelsListTitle->setAlignment(Qt::AlignCenter);
    rightToolBar->addWidget(modelsListTitle);

// MODELS LIST
    list = new ModelsListWidget;    // Creation of a ModelsListWidget object
    list->setFixedWidth(170);
    rightToolBar->addWidget(list);

    // ModelContours list.
    QLabel* pModelContoursListTitle = new QLabel("3D Model Contours");
    pModelContoursListTitle->setAlignment(Qt::AlignCenter);
    rightToolBar->addWidget(pModelContoursListTitle);

    m_pModelContoursList = new ModelContoursListWidget(this);
    m_pModelContoursList->setFixedWidth(170);
    rightToolBar->addWidget(m_pModelContoursList);

    // Image contours list.
    QLabel* pImageContoursListTitle = new QLabel("Image Contours");
    pImageContoursListTitle->setAlignment(Qt::AlignCenter);
    rightToolBar->addWidget(pImageContoursListTitle);

    m_pImageContoursList = new ImageContoursListWidget;
    m_pImageContoursList->setFixedWidth(170);
    rightToolBar->addWidget(m_pImageContoursList);

    addToolBar(Qt::RightToolBarArea, rightToolBar);

    /* ============================ CONNECTIONS ============================ */

// LEFT TOOLBAR BUTTONS
    connect(actionLoadImage, SIGNAL(triggered()), object, SLOT(setTexturePath()));

    connect(actionLoadModel, SIGNAL(triggered()), object, SLOT(addModel()));

//    connect(actionModelsList, SIGNAL(triggered()), this, SLOT(modelsListWindow()));
//    connect(actionSaveModel, SIGNAL(triggered()), object, SLOT(saveObj()));

    connect(pActionContour, SIGNAL(triggered()), this, SLOT(OpenContourSelectionTool()));
    connect(object, SIGNAL(OpenContourSelectionTool()), this, SLOT(OpenContourSelectionTool()));
    connect(m_pActionOptimisation, SIGNAL(triggered()), this, SLOT(OptimisationDialog()));

    connect(actionCenterModel, SIGNAL(triggered()), object, SLOT(CentreModel()));

//    connect(distance, SIGNAL(changed()), this, SLOT(distanceMode()));
//    connect(object, SIGNAL(distanceModeIsON(bool)), distance, SLOT(setChecked(bool)));

    connect(rotateX, SIGNAL(triggered()), object, SLOT(rotateX()));
    connect(rotateY, SIGNAL(triggered()), object, SLOT(rotateY()));

    connect(screenshot, SIGNAL(triggered()), this, SLOT(screenshot()));

    connect(settings, SIGNAL(triggered()), this, SLOT(settingsWindow()));

// AUTO-RESIZE
    connect(object, SIGNAL(pictureChanged(int,int)), this, SLOT(resizeMainWindow(int,int)));

// OPACITY SLIDER
    connect(opacitySlider, SIGNAL(valueChanged(int)), this, SLOT(updateToolTip(int)));
    connect(opacitySlider, SIGNAL(valueChanged(int)), object, SLOT(setOpacity(int)));
    connect(opacitySlider, SIGNAL(sliderReleased()), this, SLOT(focusOFF()));

// SCALE SLIDER
    connect(scaleSlider, SIGNAL(valueChanged(int)), this, SLOT(updateToolTip(int)));
    connect(scaleSlider, SIGNAL(valueChanged(int)), this, SLOT(scaleSliderState()));
    connect(scaleSlider, SIGNAL(sliderReleased()), this, SLOT(scaleSliderState()));
    connect(object, SIGNAL(BackgroundImageChanged()), this, SLOT(ResetWindowScaleSlider()));

// FOV SCALE SLIDER
    connect(m_pFOVScaleSlider, SIGNAL(valueChanged(int)), this, SLOT(updateToolTip(int)));
    connect(m_pFOVScaleSlider, SIGNAL(valueChanged(int)), this, SLOT(FOVScaleSliderState()));
    connect(m_pFOVScaleSlider, SIGNAL(sliderReleased()), this, SLOT(FOVScaleSliderState()));
    connect(object, SIGNAL(FOVScaleChanged(float)), this, SLOT(UpdateFOVSlider(float)));

// MODELS LIST WIDGET
    connect(object, SIGNAL(modelsChanged()), this, SLOT(updateModelsList()));
    connect(list, SIGNAL(updateList()), this, SLOT(updateModelsList()));
    connect(list, SIGNAL(updateCheckedModels(QVector<unsigned int>&)), object, SLOT(SetCheckedModels(QVector<unsigned int>&)));
    connect(list, SIGNAL(selectedModelChanged(QString&)), object, SLOT(SetSelectedModel(QString&)));
    connect(list, SIGNAL(addModel()), object, SLOT(addModel()));
    connect(list, SIGNAL(removeModels()), object, SLOT(RemoveModels()));
    connect(list, SIGNAL(modelColor(QColor&)), object, SLOT(ChangeColour(QColor&)));
    connect(list, SIGNAL(InitialiseSimulation()), object, SLOT(InitialiseSimulation()));
    connect(list, SIGNAL(Segmentation()), object, SLOT(ComputeModelSegmentation()));
    connect(list, SIGNAL(ComputeGaussianCurvature()), object, SLOT(ComputeGaussianCurvature()));

    connect(object, SIGNAL(ModelContoursChanged()), this, SLOT(UpdateModelContoursList()));
    connect(m_pModelContoursList, SIGNAL(UpdateList()), this, SLOT(UpdateModelContoursList()));
    connect(m_pModelContoursList, SIGNAL(UpdateCheckedModelContours(QVector<unsigned int>&)), object, SLOT(SetCheckedModelContours(QVector<unsigned int>&)));
    connect(m_pModelContoursList, SIGNAL(SelectedModelContourChanged(QString&)), object, SLOT(SetSelectedModelContour(QString&)));
    connect(m_pModelContoursList, SIGNAL(AddFrontierContour()), object, SLOT(AddFrontierContour()));
    connect(m_pModelContoursList, SIGNAL(AddFrontierContour()), m_pModelContoursList, SLOT(OpenHighCurvatureSelectionTool()));
    connect(m_pModelContoursList, SIGNAL(AddOccludingContour()), object, SLOT(AddOccludingContour()));
    connect(m_pModelContoursList, SIGNAL(AddLigamentContour()), object, SLOT(AddLigamentContour()));
    connect(m_pModelContoursList, SIGNAL(RemoveModelContours()), object, SLOT(RemoveModelContours()));
    connect(m_pModelContoursList, SIGNAL(ShowModelContour(unsigned int)), object, SLOT(ShowModelContour(unsigned int)));
    connect(m_pModelContoursList, SIGNAL(LinkModelToImageContour()), object, SLOT(LinkModelToImageContour()));
    connect(m_pModelContoursList, SIGNAL(SetHighCurvatureStartPosition(float)), object, SLOT(SetHighCurvatureStartPosition(float)));
    connect(m_pModelContoursList, SIGNAL(SetHighCurvatureEndPosition(float)), object, SLOT(SetHighCurvatureEndPosition(float)));
    connect(m_pModelContoursList, SIGNAL(SetHighCurvatureRangeReversed(bool)), object, SLOT(SetHighCurvatureRangeReversed(bool)));
    connect(m_pModelContoursList, SIGNAL(SetHighCurvatureVerticesPolynomialOrder(int)), object, SLOT(SetHighCurvatureVerticesPolynomialOrder(int)));
    connect(m_pModelContoursList, SIGNAL(SetVertexSearchAreaRadius(float)), object, SLOT(SetHighCurvatureVertexSearchAreaRadius(float)));

    connect(object, SIGNAL(ImageContoursChanged()), this, SLOT(UpdateImageContoursList()));
    connect(m_pImageContoursList, SIGNAL(UpdateList()), this, SLOT(UpdateImageContoursList()));
    connect(m_pImageContoursList, SIGNAL(UpdateCheckedImageContours(QVector<unsigned int>&)), object, SLOT(SetCheckedImageContours(QVector<unsigned int>&)));
    connect(m_pImageContoursList, SIGNAL(SelectedImageContourChanged(QString&)), object, SLOT(SetSelectedImageContour(QString&)));
    connect(m_pImageContoursList, SIGNAL(AddImageContour()), object, SLOT(AddImageContour()));
    connect(m_pImageContoursList, SIGNAL(RemoveImageContours()), object, SLOT(RemoveImageContours()));
    connect(m_pImageContoursList, SIGNAL(ShowImageContour(unsigned int)), object, SLOT(ShowImageContour(unsigned int)));

    connect(this, SIGNAL(ContourSelection(bool, bool)), object, SLOT(ContourSelection(bool, bool)));
    connect(this, SIGNAL(ResetContourSelection()), object, SLOT(ResetContourSelection()));
}

void MainWindow::resizeMainWindow(int newWidth, int newHeight)  // Resizes main window when GLWidget size changes
{
    QDesktopWidget rec;
    QRect mainScreenSize = rec.availableGeometry(rec.primaryScreen());

    if((newHeight+20)<mainScreenSize.height())
        this->setFixedHeight(newHeight+20);
    else
        this->setFixedHeight(mainScreenSize.height()-28);

    if(newWidth+rightToolBar->width()+leftToolBar->width()+20<mainScreenSize.width())
        this->setFixedWidth(newWidth+rightToolBar->width()+leftToolBar->width()+20);
    else
        this->setFixedWidth(mainScreenSize.width());
}

void MainWindow::ResetWindowScaleSlider()
{
    scaleSlider->setSliderPosition(100);
}

/* ============================ LEFT TOOLBAR ============================ */

void MainWindow::OpenContourSelectionTool()
{
    m_pContourSelectionToolWindow = new ContourSelectionToolDialog(this);
    m_pContourSelectionToolWindow->setWindowTitle("Contour Selection Tool");
    QGridLayout* pGridLayout = new QGridLayout(m_pContourSelectionToolWindow);

//    list = new QListWidget(m_pContourSelectionToolWindow);
    /*item->setFlags(item->flags() | Qt::ItemIsUserCheckable); // set checkable flag
    item->setCheckState(Qt::Unchecked); // AND initialize check state*/

//    this->setContextMenuPolicy(Qt::ActionsContextMenu);


    QVBoxLayout *buttonsLayout = new QVBoxLayout;
    QPushButton *pResetButton = new QPushButton("&Reset Contour", this);
//    QPushButton *saveModelButton = new QPushButton("&Save Model", this);
    QPushButton *pFinaliseButton = new QPushButton("&Finalise Contour", this);

    buttonsLayout->addWidget(pResetButton);
//    buttonsLayout->addWidget(saveModelButton);
    buttonsLayout->addWidget(pFinaliseButton);

//    QDialogButtonBox *saveQuitButtons = new QDialogButtonBox(QDialogButtonBox::Ok);
//    connect(saveQuitButtons, SIGNAL(accepted()), m_pContourSelectionToolWindow, SLOT(accept()));


//    updateModelsList();     // LIST ITEMS

//    pGridLayout->addWidget(list, 0, 0, 0, 1);
    pGridLayout->addLayout(buttonsLayout, 0, 1);
//    pGridLayout->addWidget(saveQuitButtons, 1, 1);

    /* ============================ CONNECTIONS ============================ */

    connect(pResetButton, SIGNAL(clicked(bool)), this, SLOT(EmitResetContourSelection()));
    connect(pFinaliseButton, SIGNAL(clicked(bool)), this, SLOT(EmitFinaliseContourSelection()));

    emit ContourSelection(true, m_ContourSelectionFinalised);

//      connect(pResetButton, SIGNAL(triggered()), this, SLOT(EmitResetContourSelection()));

//    buttonsLayout->setSizePolicy(QSizePolicy::Expanding);
    m_pContourSelectionToolWindow->resize(250, 60);

    m_pContourSelectionToolWindow->show();
}

void MainWindow::EmitResetContourSelection()
{
    m_ContourSelectionFinalised = false;

    emit ResetContourSelection();
}

void MainWindow::EmitFinaliseContourSelection()
{
    m_ContourSelectionFinalised = true;

    m_pContourSelectionToolWindow->reject();

//    emit ContourSelection(false, true);
}

void MainWindow::ContourSelectionToolClosed()
{
//    if (!m_ContourSelectionFinalised)
//    {
//        emit ResetContourSelection();
//    }

    emit ContourSelection(false, m_ContourSelectionFinalised);
}

void MainWindow::OptimisationDialog()
{
    QDialog* pDialog = new QDialog(this);
    pDialog->setWindowTitle("Optimisation");
//    pDialog->setWindowFlags(Qt::WindowStaysOnTopHint);
    QGridLayout* pGridLayout = new QGridLayout(pDialog);

//    list = new QListWidget(m_pContourSelectionToolWindow);
    /*item->setFlags(item->flags() | Qt::ItemIsUserCheckable); // set checkable flag
    item->setCheckState(Qt::Unchecked); // AND initialize check state*/

//    this->setContextMenuPolicy(Qt::ActionsContextMenu);

    QVBoxLayout* buttonsLayout = new QVBoxLayout;
    QRadioButton* pContourButton = new QRadioButton("&Contour", this);
    QRadioButton* pContourAndShadingButton = new QRadioButton("Contour + &Shading", this);
    QPushButton* pRunButton = new QPushButton("&Run", this);
    pContourAndShadingButton->setChecked(true);

    buttonsLayout->addWidget(pContourButton);
    buttonsLayout->addWidget(pContourAndShadingButton);
    buttonsLayout->addWidget(pRunButton);

    m_OptimisationUsingContourAndShading = true;


//    QDialogButtonBox *saveQuitButtons = new QDialogButtonBox(QDialogButtonBox::Ok);
//    connect(saveQuitButtons, SIGNAL(accepted()), m_pContourSelectionToolWindow, SLOT(accept()));


//    updateModelsList();     // LIST ITEMS

//    pGridLayout->addWidget(list, 0, 0, 0, 1);
    pGridLayout->addLayout(buttonsLayout, 0, 1);
//    pGridLayout->addWidget(saveQuitButtons, 1, 1);

    /* ============================ CONNECTIONS ============================ */

    connect(pContourButton, SIGNAL(clicked(bool)), this, SLOT(SetOptimisationUsingContour()));
    connect(pContourAndShadingButton, SIGNAL(clicked(bool)), this, SLOT(SetOptimisationUsingContourAndShading()));
    connect(pRunButton, SIGNAL(clicked(bool)), this, SLOT(EmitRunOptimisation()));
    connect(this, SIGNAL(RunOptimisation(bool)), object, SLOT(RunOptimisation(bool)));

//    emit ContourSelection(true, m_ContourSelectionFinalised);

//      connect(pResetButton, SIGNAL(triggered()), this, SLOT(EmitResetContourSelection()));

//    buttonsLayout->setSizePolicy(QSizePolicy::Expanding);
    pDialog->resize(200, 60);

    pDialog->show();
}

void MainWindow::SetOptimisationUsingContour()
{
    // Optimisation using only contour.
    m_OptimisationUsingContourAndShading = false;
}

void MainWindow::SetOptimisationUsingContourAndShading()
{
    // Optimisation using contour and shading.
    m_OptimisationUsingContourAndShading = true;
}

void MainWindow::EmitRunOptimisation()
{
    emit RunOptimisation(m_OptimisationUsingContourAndShading);
}

void MainWindow::updateModelsList()
{
    list->updateModelsList(object->getModelsList());
}

void MainWindow::UpdateModelContoursList()
{
    m_pModelContoursList->UpdateModelContoursList(object->GetModelContoursList());
}

void MainWindow::UpdateImageContoursList()
{
    m_pImageContoursList->UpdateImageContoursList(object->GetImageContoursList());
}

void MainWindow::distanceMode()
{
   if(distance->isChecked())
       object->setDistanceMode(true);
   else
       object->setDistanceMode(false);
}

void MainWindow::screenshot()
{
    QImage image = object->grabFramebuffer();
    QString format = "png";
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"),
                               QString("screenshot%1.").arg(screenshotNumber) + format,
                               QString("%1 Files (*.%2);;All Files (*)")
                               .arg(format.toUpper())
                               .arg(format));

    if (!fileName.isEmpty())
    {
        image.save(fileName, qPrintable(format));
        screenshotNumber++;
    }
}

void MainWindow::settingsWindow()
{
    QDialog *settings = new QDialog(this);
    settings->setWindowTitle("Settings");
    QVBoxLayout *settingsLayout = new QVBoxLayout(settings);
    QTabWidget *tabs = new QTabWidget(settings);
    QDialogButtonBox *saveQuitButtons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    settingsLayout->addWidget(tabs);
    settingsLayout->addWidget(saveQuitButtons);

    settings->setLayout(settingsLayout);


// Tabs
    QWidget *display = new QWidget;
    QWidget *camera = new QWidget;
    QWidget *sensibility = new QWidget;


    // DISPLAY
        framePictureRatioLineEdit = new QLineEdit(QString("%1").arg(object->getFramePictureRatio()));
        rotationSpeedLineEdit = new QLineEdit(QString("%1").arg(object->getRotationSpeed()));
        tagsRadiusLineEdit = new QLineEdit(QString("%1").arg(object->getTagsRadius()));

        m_pRenderingWireframeCheckBox = new QCheckBox;
        m_pRenderingWireframeCheckBox->setChecked(object->RenderingModelFaces());

        QFormLayout *formLayout1 = new QFormLayout;
        formLayout1->addRow("&Ratio Frame/Picture :", framePictureRatioLineEdit);
        formLayout1->addRow("Rotation &Speed :", rotationSpeedLineEdit);
        formLayout1->addRow("&Tags Radius (m) :", tagsRadiusLineEdit);
        formLayout1->addRow("Rendering &Wireframe:", m_pRenderingWireframeCheckBox);
        display->setLayout(formLayout1);


    // CAMERA
      // GRID LAYOUT
        QString zeroString = "0";

        QLabel *k = new QLabel("K");
        QLabel *equal = new QLabel("=");
        QLabel *zero1 = new QLabel(zeroString);
        QLabel *zero2 = new QLabel(zeroString);
        QLabel *zero3 = new QLabel(zeroString);
        QLabel *zero4 = new QLabel(zeroString);
        QLabel *zero5 = new QLabel(zeroString);
        QLabel *zero6 = new QLabel(zeroString);
        QLabel *zero7 = new QLabel(zeroString);
        QLabel *zero8 = new QLabel(zeroString);
        QLabel *a = new QLabel("A");
        QLabel *b = new QLabel("B");
        QLabel *minusOne = new QLabel("-1");

        alphaX = new QLineEdit(QString("%1").arg(object->getCameraSettings(0)));
        alphaY = new QLineEdit(QString("%1").arg(object->getCameraSettings(1)));
        skewness = new QLineEdit(QString("%1").arg(object->getCameraSettings(2)));
        centerX = new QLineEdit(QString("%1").arg(object->getCameraSettings(3)));
        centerY = new QLineEdit(QString("%1").arg(object->getCameraSettings(4)));

        alphaX->setFixedWidth(70);
        alphaY->setFixedWidth(70);
        skewness->setFixedWidth(70);
        centerX->setFixedWidth(70);
        centerY->setFixedWidth(70);

        QGridLayout *gridLayout = new QGridLayout;
        gridLayout->addWidget(alphaX,0,2);
        gridLayout->addWidget(skewness,0,3);
        gridLayout->addWidget(centerX,0,4);
        gridLayout->addWidget(zero1,0,5);
        gridLayout->addWidget(zero2,1,2);
        gridLayout->addWidget(alphaY,1,3);
        gridLayout->addWidget(centerY,1,4);
        gridLayout->addWidget(zero3,1,5);
        gridLayout->addWidget(k,2,0);
        gridLayout->addWidget(equal,2,1);
        gridLayout->addWidget(zero4,2,2);
        gridLayout->addWidget(zero5,2,3);
        gridLayout->addWidget(a,2,4);
        gridLayout->addWidget(b,2,5);
        gridLayout->addWidget(zero6,3,2);
        gridLayout->addWidget(zero7,3,3);
        gridLayout->addWidget(minusOne,3,4);
        gridLayout->addWidget(zero8,3,5);


      // TEXT LAYOUT
        QLabel *A = new QLabel("\nA = near + far");
        QLabel *B = new QLabel("B = near * far\n");
        QVBoxLayout *vTextLayout = new QVBoxLayout;
        vTextLayout->addWidget(A);
        vTextLayout->addWidget(B);


      // FORM LAYOUT
        near = new QLineEdit(QString("%1").arg(object->getCameraSettings(5)));
        far = new QLineEdit(QString("%1").arg(object->getCameraSettings(6)));

        QFormLayout *formLayout2 = new QFormLayout;
        formLayout2->addRow("&Near :", near);
        formLayout2->addRow("&Far :", far);


      // MAIN LAYOUT
        QVBoxLayout *vBoxLayout = new QVBoxLayout;
        vBoxLayout->addLayout(gridLayout);
        vBoxLayout->addLayout(vTextLayout);
        vBoxLayout->addLayout(formLayout2);


        camera->setLayout(vBoxLayout);

    // SENSIBILITY
        sensibilityLineEdit = new QLineEdit(QString("%1").arg(object->getSensibility()));
        sensibilityPlusLineEdit = new QLineEdit(QString("%1").arg(object->getSensibilityPlus()));
        QFormLayout *formLayout3 = new QFormLayout;
        formLayout3->addRow("&Sensibility (m):", sensibilityLineEdit);
        formLayout3->addRow("&Precision Sensibility (m) :", sensibilityPlusLineEdit);
        sensibility->setLayout(formLayout3);


    tabs->addTab(display, "Display");
    tabs->addTab(camera, "Camera");
    tabs->addTab(sensibility, "Sensibility");
    tabs->adjustSize();


    connect(saveQuitButtons, SIGNAL(accepted()), settings, SLOT(accept()));
    connect(saveQuitButtons, SIGNAL(rejected()), settings, SLOT(reject()));

    if(settings->exec() == QDialog::Accepted)
        sendSettings();
}
void MainWindow::sendSettings()
{
  // DISPLAY
    object->setFramePictureRatio(framePictureRatioLineEdit->text().toFloat());
    object->setRotationSpeed(rotationSpeedLineEdit->text().toFloat());
    object->setTagsRadius(tagsRadiusLineEdit->text().toFloat());
    object->SetRenderingModelFaces(m_pRenderingWireframeCheckBox->isChecked());

  // CAMERA
    object->setCameraSettings(0, alphaX->text().toFloat());
    object->setCameraSettings(1, alphaY->text().toFloat());
    object->setCameraSettings(2, skewness->text().toFloat());
    object->setCameraSettings(3, centerX->text().toFloat());
    object->setCameraSettings(4, centerY->text().toFloat());
    object->setCameraSettings(5, near->text().toFloat());
    object->setCameraSettings(6, far->text().toFloat());

  // SENSIBILITY
    object->setSensibility(sensibilityLineEdit->text().toFloat());
    object->setSensibilityPlus(sensibilityPlusLineEdit->text().toFloat());

    // Save to a file.
    QFile file("./../tensorflow/liver_data/fine_registration/camera_settings.txt");
    file.open(QIODevice::ReadWrite);
    file.resize(0);

    QTextStream stream(&file);

    stream << "########## Camera settings ##########" << endl << endl;
    stream << "# Focal length x, focal length y, skewness, optical centre x, optical centre y, near, far #" << endl;

    stream << alphaX->text() << endl << alphaY->text() << endl << skewness->text() << endl << centerX->text() << endl << centerY->text() << endl << near->text() << endl << far->text();

    file.close();
}


/* ============================ RIGHT TOOLBAR ============================ */
void MainWindow::updateToolTip(int sliderValue)     // Updates opacity tooltip
{
    QToolTip::showText(QCursor::pos(),QString("%1%").arg(sliderValue));
}

void MainWindow::scaleSliderState()
{
    object->scaleSliderState(scaleSlider->isSliderDown());
    object->setCameraSettings(7, (GLfloat)scaleSlider->value()/100);
    object->setFocus();
}

void MainWindow::FOVScaleSliderState()
{
    object->FOVScaleSliderState(m_pFOVScaleSlider->isSliderDown());
    object->SetFOVScale((float)m_pFOVScaleSlider->value() / 100.0f);
    object->setFocus();
}

void MainWindow::UpdateFOVSlider(float Scale)
{
    m_pFOVScaleSlider->setSliderPosition(100 * Scale);
}

void MainWindow::focusOFF()     // Disables sliders focus
{
    object->setFocus();
}
