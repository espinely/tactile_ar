#include "ImageContoursListWidget.h"


ImageContoursListWidget::ImageContoursListWidget(QWidget *parent) : QListWidget(parent)
{
    this->setContextMenuPolicy(Qt::ActionsContextMenu);

  // RIGHT CLICK MENU ACTIONS
    QAction* pAddImageContourButton = new QAction("&Add Image Contour", this);
    QAction* pRemoveImageContourButton = new QAction("&Remove Image Contour", this);
//    QAction *changeColorButton = new QAction("&Change Colour", this);
//    QAction* pInitialiseSimulationButton = new QAction("&Initialise Simulation", this);
//    QAction* pSegmentationButton = new QAction("&Segmentation", this);
//    QAction* pComputeGaussianCurvatureButton = new QAction("Compute &Gaussian Curvature", this);

    this->addAction(pAddImageContourButton);
    this->addAction(pRemoveImageContourButton);
//    this->addAction(changeColorButton);
//    this->addAction(pInitialiseSimulationButton);
//    this->addAction(pSegmentationButton);
//    this->addAction(pComputeGaussianCurvatureButton);

    emit UpdateList();

    /* ============================ CONNECTIONS ============================ */

  // ITEM CLICK EVENTS
    connect(this, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(SetSelected(QListWidgetItem*)));
    connect(this, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(UpdateCheckedImageContoursSlot()));

  // RIGHT CLICK MENU
    connect(pAddImageContourButton, SIGNAL(triggered()), this, SLOT(EmitAddImageContour()));
    connect(pRemoveImageContourButton, SIGNAL(triggered()), this, SLOT(EmitRemoveImageContours()));
//    connect(changeColorButton, SIGNAL(triggered()), this, SLOT(emitChangeColor()));
//    connect(pInitialiseSimulationButton, SIGNAL(triggered()), this, SLOT(EmitInitialiseSimulation()));
//    connect(pSegmentationButton, SIGNAL(triggered()), this, SLOT(EmitSegmentation()));
//    connect(pComputeGaussianCurvatureButton, SIGNAL(triggered()), this, SLOT(EmitComputeGaussianCurvature()));
}

void ImageContoursListWidget::mousePressEvent(QMouseEvent *event)  // MOUSE EVENTS
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

                    emit ShowImageContour(i);
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
void ImageContoursListWidget::UpdateImageContoursList(QStringList items)  // Updates the image contours list from OpenGLWidget image contours list, via MainWindow connection
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

    UpdateCheckedImageContoursSlot();
}

void ImageContoursListWidget::UpdateCheckedImageContoursSlot()    // Updates the checked contours list and send it to OpenGLWidget via MainWindow connection
{
    checked.clear();
    for(unsigned int i = 0; i < (unsigned int)this->count(); i++)
        if(this->item(i)->checkState())
            checked.push_back(i);

    emit UpdateCheckedImageContours(checked);
}

void ImageContoursListWidget::SetSelected(QListWidgetItem* item)
{
    for(unsigned int i = 0; i < (unsigned int)pathsList.size(); i++)
        if(pathsList.at(i).contains(item->text()))
            selectedItem = pathsList.at(i);
}


/* ============================ MENU SIGNALS ============================ */
void ImageContoursListWidget::EmitAddImageContour()
{
    emit AddImageContour();
}

//void ImageContoursListWidget::emitChangeColor()
//{
//    if(!checked.isEmpty() || !selectedItem.isEmpty())
//    {
//        QColorDialog *colorWindow = new QColorDialog(this);
//        colorWindow->setWindowTitle("3D Model Colour");

//        connect(colorWindow, SIGNAL(colorSelected(QColor)), this, SLOT(changeColor(QColor)));
//        colorWindow->exec();
//    }
//}

//void ImageContoursListWidget::changeColor(QColor newColor)
//{
//    emit selectedModelChanged(selectedItem);
//    emit modelColor(newColor);
//}

void ImageContoursListWidget::EmitRemoveImageContours()
{
    emit UpdateCheckedImageContours(checked);
    emit SelectedImageContourChanged(selectedItem);
    emit RemoveImageContours();

    checked.clear();
    emit UpdateList();
}

//void ImageContoursListWidget::EmitInitialiseSimulation()
//{
//    emit InitialiseSimulation();
//}

//void ImageContoursListWidget::EmitSegmentation()
//{
//    // TODO: Temp - fix to check if the liver model is checked.
//    if(!checked.isEmpty() || !selectedItem.isEmpty())
//    {
//        emit Segmentation();
//    }
//}

//void ImageContoursListWidget::EmitComputeGaussianCurvature()
//{
//    emit ComputeGaussianCurvature();
//}
