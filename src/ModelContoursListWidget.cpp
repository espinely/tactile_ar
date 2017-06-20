#include "ModelContoursListWidget.h"
#include "mainWindow.h"

#include <QGridLayout>
#include <QPushButton>
#include <QLabel>
#include <QFormLayout>


ModelContoursListWidget::ModelContoursListWidget(QWidget *parent) : QListWidget(parent)
{
    this->setContextMenuPolicy(Qt::ActionsContextMenu);

  // RIGHT CLICK MENU ACTIONS
    QAction* pAddFrontierContourButton = new QAction("Add &Frontier Contour", this);
    QAction* pAddOccludingContourButton = new QAction("Add &Occluding Contour", this);
    QAction* pAddLigamentContourButton = new QAction("Add &Ligament Contour", this);
    QAction* pRemoveModelContourButton = new QAction("&Remove Contour", this);
    QAction* pLinkToImageContourButton = new QAction("&Link to Image Contour", this);
//    QAction *changeColorButton = new QAction("&Change Colour", this);
//    QAction* pInitialiseSimulationButton = new QAction("&Initialise Simulation", this);
//    QAction* pSegmentationButton = new QAction("&Segmentation", this);
//    QAction* pComputeGaussianCurvatureButton = new QAction("Compute &Gaussian Curvature", this);

    this->addAction(pAddFrontierContourButton);
    this->addAction(pAddOccludingContourButton);
    this->addAction(pAddLigamentContourButton);
    this->addAction(pRemoveModelContourButton);
    this->addAction(pLinkToImageContourButton);
//    this->addAction(changeColorButton);
//    this->addAction(pInitialiseSimulationButton);
//    this->addAction(pSegmentationButton);
//    this->addAction(pComputeGaussianCurvatureButton);

    emit UpdateList();

    /* ============================ CONNECTIONS ============================ */

  // ITEM CLICK EVENTS
    connect(this, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(SetSelected(QListWidgetItem*)));
    connect(this, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(UpdateCheckedModelContoursSlot()));

  // RIGHT CLICK MENU
    connect(pAddFrontierContourButton, SIGNAL(triggered()), this, SLOT(EmitAddFrontierContour()));
    connect(pAddOccludingContourButton, SIGNAL(triggered()), this, SLOT(EmitAddOccludingContour()));
    connect(pAddLigamentContourButton, SIGNAL(triggered()), this, SLOT(EmitAddLigamentContour()));
    connect(pRemoveModelContourButton, SIGNAL(triggered()), this, SLOT(EmitRemoveModelContours()));
    connect(pLinkToImageContourButton, SIGNAL(triggered()), this, SLOT(EmitLinkToImageContour()));

//    connect(changeColorButton, SIGNAL(triggered()), this, SLOT(emitChangeColor()));
//    connect(pInitialiseSimulationButton, SIGNAL(triggered()), this, SLOT(EmitInitialiseSimulation()));
//    connect(pSegmentationButton, SIGNAL(triggered()), this, SLOT(EmitSegmentation()));
//    connect(pComputeGaussianCurvatureButton, SIGNAL(triggered()), this, SLOT(EmitComputeGaussianCurvature()));
}

void ModelContoursListWidget::mousePressEvent(QMouseEvent *event)  // MOUSE EVENTS
{
/*    if(event->button() == Qt::RightButton)
    {
        QListWidgetItem *clickedItem = this->itemAt(event->pos());

        if(clickedItem) // Detection of the clicked item
        {
            for(unsigned int i = 0; i < (unsigned int)pathsList.size(); i++)
                if(pathsList.at(i).contains(clickedItem->text()))
                    selectedItem = pathsList.at(i);

//            referenceModelButton->setEnabled(true);
        }

//        else
//            referenceModelButton->setEnabled(false);
    }
    else */if(event->button() == Qt::LeftButton)
    {
        QListWidgetItem *clickedItem = this->itemAt(event->pos());

        if (clickedItem) // Detection of the clicked item
        {
            for (unsigned int i = 0; i < (unsigned int)pathsList.size(); i++)
            {
                if (pathsList.at(i).contains(clickedItem->text()))
                {
                    selectedItem = pathsList.at(i);

                    emit ShowModelContour(i);
                }
            }
        }
    }
    else
    {
        selectedItem = QString("");
    }

    QListWidget::mousePressEvent(event);
}

/* ============================ UPDATE ============================ */
void ModelContoursListWidget::UpdateModelContoursList(QStringList items)  // Updates the model contours list from OpenGLWidget model contours list, via MainWindow connection
{
    unsigned int initialSize = this->count();
    this->clear();

    pathsList = items;

    if(initialSize < (unsigned int)items.size())
        checked.push_back(items.size()-1);

    for(unsigned int i = 0; i < (unsigned int)items.size(); i++)
    {
//        QStringList temp = items.at(i).split("/");
        QListWidgetItem *currentItem = new QListWidgetItem(items.at(i));

        currentItem->setFlags(currentItem->flags() | Qt::ItemIsUserCheckable);
        if(checked.contains(i))
            currentItem->setCheckState(Qt::Checked);
        else
            currentItem->setCheckState(Qt::Unchecked);

        this->addItem(currentItem);
    }

    UpdateCheckedModelContoursSlot();
}

void ModelContoursListWidget::UpdateCheckedModelContoursSlot()    // Updates the checked model contours list and send it to OpenGLWidget via MainWindow connection
{
    checked.clear();
    for(unsigned int i = 0; i < (unsigned int)this->count(); i++)
        if(this->item(i)->checkState())
            checked.push_back(i);

    emit UpdateCheckedModelContours(checked);
}

void ModelContoursListWidget::SetSelected(QListWidgetItem* item)
{
    for(unsigned int i = 0; i < (unsigned int)pathsList.size(); i++)
        if(pathsList.at(i).contains(item->text()))
            selectedItem = pathsList.at(i);
}


/* ============================ MENU SIGNALS ============================ */
void ModelContoursListWidget::EmitAddFrontierContour()
{
    emit AddFrontierContour();
}

void ModelContoursListWidget::EmitAddOccludingContour()
{
    emit AddOccludingContour();
}

void ModelContoursListWidget::EmitAddLigamentContour()
{
    emit AddLigamentContour();
}

//void ModelContoursListWidget::emitChangeColor()
//{
//    if(!checked.isEmpty() || !selectedItem.isEmpty())
//    {
//        QColorDialog *colorWindow = new QColorDialog(this);
//        colorWindow->setWindowTitle("3D Model Colour");

//        connect(colorWindow, SIGNAL(colorSelected(QColor)), this, SLOT(changeColor(QColor)));
//        colorWindow->exec();
//    }
//}

//void ModelContoursListWidget::changeColor(QColor newColor)
//{
//    emit selectedModelChanged(selectedItem);
//    emit modelColor(newColor);
//}

void ModelContoursListWidget::EmitRemoveModelContours()
{
    emit UpdateCheckedModelContours(checked);
    emit SelectedModelContourChanged(selectedItem);
    emit RemoveModelContours();

    checked.clear();
    emit UpdateList();
}

void ModelContoursListWidget::EmitLinkToImageContour()
{
    emit LinkModelToImageContour();
}

//void ModelContoursListWidget::EmitInitialiseSimulation()
//{
//    emit InitialiseSimulation();
//}

//void ModelContoursListWidget::EmitSegmentation()
//{
//    // TODO: Temp - fix to check if the liver model is checked.
//    if(!checked.isEmpty() || !selectedItem.isEmpty())
//    {
//        emit Segmentation();
//    }
//}

//void ModelContoursListWidget::EmitComputeGaussianCurvature()
//{
//    emit ComputeGaussianCurvature();
//}

void ModelContoursListWidget::OpenHighCurvatureSelectionTool(void)
{
    m_pHighCurvatureSelectionToolWindow = new QDialog;
    m_pHighCurvatureSelectionToolWindow->setWindowTitle("High Curvature Selection Tool");
    QGridLayout* pGridLayout = new QGridLayout(m_pHighCurvatureSelectionToolWindow);
    QVBoxLayout* pSlidersLayout = new QVBoxLayout;
    QVBoxLayout* pSlidersLayout2 = new QVBoxLayout;
    pSlidersLayout->setAlignment(Qt::AlignCenter);
    const QSize sliderSize(250, 60);

//    list = new QListWidget(m_pContourSelectionToolWindow);
    /*item->setFlags(item->flags() | Qt::ItemIsUserCheckable); // set checkable flag
    item->setCheckState(Qt::Unchecked); // AND initialize check state*/

//    this->setContextMenuPolicy(Qt::ActionsContextMenu);


    // SLIDER TITLE
      QLabel* pHighCurvatureTitle = new QLabel("High Curvature Vertices");
      pHighCurvatureTitle->setAlignment(Qt::AlignCenter);
      pSlidersLayout->addWidget(pHighCurvatureTitle);

    // SLIDER
      m_pHighCurvatureStartSlider = new QSlider(Qt::Horizontal);
      m_pHighCurvatureStartSlider->setRange(0,1000);
      m_pHighCurvatureStartSlider->setSliderPosition(0);
      m_pHighCurvatureStartSlider->setSingleStep(1);
      m_pHighCurvatureStartSlider->setTickPosition(QSlider::TicksBelow);
      m_pHighCurvatureStartSlider->setTickInterval(100);
      m_pHighCurvatureStartSlider->setMaximumSize(sliderSize);
      pSlidersLayout->addWidget(m_pHighCurvatureStartSlider);

      pSlidersLayout->addSpacing(30);

      m_pHighCurvatureEndSlider = new QSlider(Qt::Horizontal);
      m_pHighCurvatureEndSlider->setRange(0,1000);
      m_pHighCurvatureEndSlider->setSliderPosition(1000);
      m_pHighCurvatureEndSlider->setSingleStep(1);
      m_pHighCurvatureEndSlider->setTickPosition(QSlider::TicksBelow);
      m_pHighCurvatureEndSlider->setTickInterval(100);
      m_pHighCurvatureEndSlider->setMaximumSize(sliderSize);
      pSlidersLayout->addWidget(m_pHighCurvatureEndSlider);

      pSlidersLayout->addSpacing(30);

    // SLIDER TEXTMIN
  //    QLabel *scaleTextMin = new QLabel("0.1 x                  2 x\n");
  //    scaleTextMin->setAlignment(Qt::AlignCenter);
  //    pSlidersLayout->addWidget(scaleTextMin);

//      rightToolBar->addWidget(slidersWidget);
//      rightToolBar->addSeparator();

      QLabel* pPolynomialOrderTitle = new QLabel("Polynomial Order");
      pPolynomialOrderTitle->setAlignment(Qt::AlignCenter);
      pSlidersLayout2->addWidget(pPolynomialOrderTitle);

      // For reversing the range of vertex selection.
      m_pReverseRangeCheckBox = new QCheckBox;
      QFormLayout* pFormLayout = new QFormLayout;
      pFormLayout->addRow("&Reverse Range:", m_pReverseRangeCheckBox);


      // For changing the polynomial order of the curve.
      m_pPolynomialOrderSlider = new QSlider(Qt::Horizontal);
      m_pPolynomialOrderSlider->setRange(1,50);
      m_pPolynomialOrderSlider->setSliderPosition(20);
      m_pPolynomialOrderSlider->setSingleStep(1);
      m_pPolynomialOrderSlider->setTickPosition(QSlider::TicksBelow);
      m_pPolynomialOrderSlider->setTickInterval(5);
      m_pPolynomialOrderSlider->setMaximumSize(sliderSize);
      pSlidersLayout2->addWidget(m_pPolynomialOrderSlider);

      // SLIDER TEXTMIN
      QLabel *polynomialOrderTextMin = new QLabel("1                                                           50\n");
      polynomialOrderTextMin->setAlignment(Qt::AlignCenter);
      pSlidersLayout2->addWidget(polynomialOrderTextMin);


      // For changing the radius of searching area for the next vertex when selecting vertices.
      QLabel* pSearchRadiusTitle = new QLabel("Vertex Search Area Radius (cm)");
      pSearchRadiusTitle->setAlignment(Qt::AlignCenter);
      pSlidersLayout2->addWidget(pSearchRadiusTitle);

      m_pVertexSearchAreaRadiusSlider = new QSlider(Qt::Horizontal);
      m_pVertexSearchAreaRadiusSlider->setRange(10,100);
      m_pVertexSearchAreaRadiusSlider->setSliderPosition(70);
      m_pVertexSearchAreaRadiusSlider->setSingleStep(1);
      m_pVertexSearchAreaRadiusSlider->setTickPosition(QSlider::TicksBelow);
      m_pVertexSearchAreaRadiusSlider->setTickInterval(10);
      m_pVertexSearchAreaRadiusSlider->setMaximumSize(sliderSize);
      pSlidersLayout2->addWidget(m_pVertexSearchAreaRadiusSlider);

      // SLIDER TEXTMIN
      QLabel *pSearchRadiusTextMin = new QLabel("1                                                           10\n");
      pSearchRadiusTextMin->setAlignment(Qt::AlignCenter);
      pSlidersLayout2->addWidget(pSearchRadiusTextMin);

      pGridLayout->addLayout(pSlidersLayout, 0, 0);
      pGridLayout->addLayout(pFormLayout, 1, 0);
      pGridLayout->addLayout(pSlidersLayout2, 0, 1);

//    QVBoxLayout *buttonsLayout = new QVBoxLayout;
//    QPushButton *pResetButton = new QPushButton("&Reset Contour", this);
////    QPushButton *saveModelButton = new QPushButton("&Save Model", this);
//    QPushButton *pFinaliseButton = new QPushButton("&Finalise Contour", this);

//    buttonsLayout->addWidget(pResetButton);
////    buttonsLayout->addWidget(saveModelButton);
//    buttonsLayout->addWidget(pFinaliseButton);

//    QDialogButtonBox *saveQuitButtons = new QDialogButtonBox(QDialogButtonBox::Ok);
//    connect(saveQuitButtons, SIGNAL(accepted()), m_pContourSelectionToolWindow, SLOT(accept()));


//    updateModelsList();     // LIST ITEMS

//    pGridLayout->addWidget(list, 0, 0, 0, 1);
//    pGridLayout->addLayout(buttonsLayout, 0, 1);
//    pGridLayout->addWidget(saveQuitButtons, 1, 1);

    /* ============================ CONNECTIONS ============================ */

//    connect(pResetButton, SIGNAL(clicked(bool)), this, SLOT(EmitResetContourSelection()));
//    connect(pFinaliseButton, SIGNAL(clicked(bool)), this, SLOT(EmitFinaliseContourSelection()));

//    emit ContourSelection(true, m_ContourSelectionFinalised);

//      connect(pResetButton, SIGNAL(triggered()), this, SLOT(EmitResetContourSelection()));


      // High curvature vertices selection tool.
      connect(m_pHighCurvatureStartSlider, SIGNAL(valueChanged(int)), this, SLOT(HighCurvatureStartSliderState()));
      connect(m_pHighCurvatureStartSlider, SIGNAL(sliderReleased()), this, SLOT(HighCurvatureStartSliderState()));

      connect(m_pHighCurvatureEndSlider, SIGNAL(valueChanged(int)), this, SLOT(HighCurvatureEndSliderState()));
      connect(m_pHighCurvatureEndSlider, SIGNAL(sliderReleased()), this, SLOT(HighCurvatureEndSliderState()));

      connect(m_pPolynomialOrderSlider, SIGNAL(valueChanged(int)), this, SLOT(UpdateToolTip(int)));
//      connect(m_pPolynomialOrderSlider, SIGNAL(valueChanged(int)), this, SLOT(PolynomialOrderSliderState()));
      connect(m_pPolynomialOrderSlider, SIGNAL(sliderReleased()), this, SLOT(PolynomialOrderSliderState()));

      connect(m_pVertexSearchAreaRadiusSlider, SIGNAL(valueChanged(int)), this, SLOT(UpdateVertexSearchAreaToolTip(int)));
//      connect(m_pVertexSearchAreaRadiusSlider, SIGNAL(valueChanged(int)), this, SLOT(VertexSearchAreaRadiusSliderState()));
      connect(m_pVertexSearchAreaRadiusSlider, SIGNAL(sliderReleased()), this, SLOT(VertexSearchAreaRadiusSliderState()));

      connect(m_pReverseRangeCheckBox, SIGNAL(stateChanged(int)), this, SLOT(ReverseRangeState(int)));


//    buttonsLayout->setSizePolicy(QSizePolicy::Expanding);
    m_pHighCurvatureSelectionToolWindow->resize(600, 60);

    m_pHighCurvatureSelectionToolWindow->show();
}

void ModelContoursListWidget::HighCurvatureStartSliderState(void)
{
    emit SetHighCurvatureStartPosition((float)m_pHighCurvatureStartSlider->value() / 1000.0f);

    if (m_pHighCurvatureStartSlider->value() > m_pHighCurvatureEndSlider->value())
    {
        m_pHighCurvatureEndSlider->setValue(m_pHighCurvatureStartSlider->value());
        emit SetHighCurvatureEndPosition((float)m_pHighCurvatureEndSlider->value() / 1000.0f);
    }
}

void ModelContoursListWidget::HighCurvatureEndSliderState(void)
{
    emit SetHighCurvatureEndPosition((float)m_pHighCurvatureEndSlider->value() / 1000.0f);

    if (m_pHighCurvatureStartSlider->value() > m_pHighCurvatureEndSlider->value())
    {
        m_pHighCurvatureStartSlider->setValue(m_pHighCurvatureEndSlider->value());
        emit SetHighCurvatureStartPosition((float)m_pHighCurvatureStartSlider->value() / 1000.0f);
    }
}

void ModelContoursListWidget::ReverseRangeState(int State)
{
    bool reversed = false;

    if (State == Qt::Checked)
    {
        reversed = true;
    }

    emit SetHighCurvatureRangeReversed(reversed);
}

void ModelContoursListWidget::VertexSearchAreaRadiusSliderState(void)
{
    emit SetVertexSearchAreaRadius((float)m_pVertexSearchAreaRadiusSlider->value() * (float)m_pVertexSearchAreaRadiusSlider->value() * 0.000001f);
}

void ModelContoursListWidget::PolynomialOrderSliderState(void)
{
    emit SetHighCurvatureVerticesPolynomialOrder((int)m_pPolynomialOrderSlider->value());
}

void ModelContoursListWidget::UpdateToolTip(int SliderValue)
{
    QToolTip::showText(QCursor::pos(),QString("%1").arg(SliderValue));
}

void ModelContoursListWidget::UpdateVertexSearchAreaToolTip(int SliderValue)
{
    QToolTip::showText(QCursor::pos(),QString("%1").arg((float)SliderValue * 0.1f));
}

