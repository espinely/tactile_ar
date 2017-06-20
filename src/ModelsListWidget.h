#ifndef MODELSLISTWIDGET_H
#define MODELSLISTWIDGET_H

#include <QAction>
#include <QColorDialog>
#include <QListWidget>
#include <QMouseEvent>


class ModelsListWidget : public QListWidget
{
    Q_OBJECT

private:
    QVector<unsigned int> checked;    // List of checked items number
    QString selectedItem, referenceModel;   // Name of the selected model and the reference model
    QStringList pathsList;  // List of the models name

    void mousePressEvent(QMouseEvent *event);

public:
    ModelsListWidget(QWidget *parent = 0);

signals:
    void addModel();
    void removeModels();
    void modelColor(QColor& newColor);
    void InitialiseSimulation();
    void Segmentation();
    void ComputeGaussianCurvature();

    void selectedModelChanged(QString& selectedItem);

    void updateList();
    void updateCheckedModels(QVector<unsigned int>& checked);

public slots:
    void emitAddModel();
    void emitRemoveModels();
    void emitChangeColor();
    void changeColor(QColor newColor);
    void EmitInitialiseSimulation();
    void EmitSegmentation();
    void EmitComputeGaussianCurvature();

    void updateModelsList(QStringList items);
    void updateCheckedModelsSlot();

    void setSelected(QListWidgetItem* item);
};

#endif // MODELSLISTWIDGET_H
