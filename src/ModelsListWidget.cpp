#include "ModelsListWidget.h"


ModelsListWidget::ModelsListWidget(QWidget *parent) : QListWidget(parent)
{
    this->setContextMenuPolicy(Qt::ActionsContextMenu);

  // RIGHT CLICK MENU ACTIONS
    QAction *addModelButton = new QAction("&Add 3D Model", this);
    QAction *removeModelButton = new QAction("&Remove 3D Model", this);
    QAction *changeColorButton = new QAction("&Change Colour", this);
    QAction* pInitialiseSimulationButton = new QAction("&Initialise Simulation", this);
    QAction* pSegmentationButton = new QAction("&Segmentation", this);
    QAction* pComputeGaussianCurvatureButton = new QAction("Compute &Gaussian Curvature", this);

    this->addAction(addModelButton);
    this->addAction(removeModelButton);
    this->addAction(changeColorButton);
    this->addAction(pInitialiseSimulationButton);
    this->addAction(pSegmentationButton);
    this->addAction(pComputeGaussianCurvatureButton);

    emit updateList();

    /* ============================ CONNECTIONS ============================ */

  // ITEM CLICK EVENTS
    connect(this, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(setSelected(QListWidgetItem*)));
    connect(this, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(updateCheckedModelsSlot()));

  // RIGHT CLICK MENU
    connect(addModelButton, SIGNAL(triggered()), this, SLOT(emitAddModel()));
    connect(removeModelButton, SIGNAL(triggered()), this, SLOT(emitRemoveModels()));
    connect(changeColorButton, SIGNAL(triggered()), this, SLOT(emitChangeColor()));
    connect(pInitialiseSimulationButton, SIGNAL(triggered()), this, SLOT(EmitInitialiseSimulation()));
    connect(pSegmentationButton, SIGNAL(triggered()), this, SLOT(EmitSegmentation()));
    connect(pComputeGaussianCurvatureButton, SIGNAL(triggered()), this, SLOT(EmitComputeGaussianCurvature()));
}

void ModelsListWidget::mousePressEvent(QMouseEvent *event)  // MOUSE EVENTS
{
    if(event->button() == Qt::RightButton)
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
    else
    {
        QListWidget::mousePressEvent(event);    // No item selected
        selectedItem = QString("");
    }
}

/* ============================ UPDATE ============================ */
void ModelsListWidget::updateModelsList(QStringList items)  // Updates the models list from OpenGLWidget models list, via MainWindow connection
{
    unsigned int initialSize = this->count();
    this->clear();

    pathsList = items;

    if(initialSize < (unsigned int)items.size())
        checked.push_back(items.size()-1);

    for(unsigned int i = 0; i < (unsigned int)items.size(); i++)
    {
        QStringList temp = items.at(i).split("/");
        QListWidgetItem *currentItem = new QListWidgetItem(temp.at(temp.size()-1));

        currentItem->setFlags(currentItem->flags() | Qt::ItemIsUserCheckable);
        if(checked.contains(i))
            currentItem->setCheckState(Qt::Checked);
        else
            currentItem->setCheckState(Qt::Unchecked);

        this->addItem(currentItem);
    }

    updateCheckedModelsSlot();
}
void ModelsListWidget::updateCheckedModelsSlot()    // Updates the checked models list and send it to OpenGLWidget via MainWindow connection
{
    checked.clear();
    for(unsigned int i = 0; i < (unsigned int)this->count(); i++)
        if(this->item(i)->checkState())
            checked.push_back(i);

    emit updateCheckedModels(checked);
}
void ModelsListWidget::setSelected(QListWidgetItem* item)
{
    for(unsigned int i = 0; i < (unsigned int)pathsList.size(); i++)
        if(pathsList.at(i).contains(item->text()))
            selectedItem = pathsList.at(i);
}


/* ============================ MENU SIGNALS ============================ */
void ModelsListWidget::emitAddModel()
{
    emit addModel();
}

void ModelsListWidget::emitChangeColor()
{
    if(!checked.isEmpty() || !selectedItem.isEmpty())
    {
        QColorDialog *colorWindow = new QColorDialog(this);
        colorWindow->setWindowTitle("3D Model Colour");

        connect(colorWindow, SIGNAL(colorSelected(QColor)), this, SLOT(changeColor(QColor)));
        colorWindow->exec();
    }
}

void ModelsListWidget::changeColor(QColor newColor)
{
    emit selectedModelChanged(selectedItem);
    emit modelColor(newColor);
}

void ModelsListWidget::emitRemoveModels()
{
    referenceModel = -1;
    emit updateCheckedModels(checked);

    emit selectedModelChanged(selectedItem);
    emit removeModels();

    checked.clear();
    emit updateList();
}

void ModelsListWidget::EmitInitialiseSimulation()
{
    emit InitialiseSimulation();
}

void ModelsListWidget::EmitSegmentation()
{
    // TODO: Temp - fix to check if the liver model is checked.
    if(!checked.isEmpty() || !selectedItem.isEmpty())
    {
        emit Segmentation();
    }
}

void ModelsListWidget::EmitComputeGaussianCurvature()
{
    emit ComputeGaussianCurvature();
}
