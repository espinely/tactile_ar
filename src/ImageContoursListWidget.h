#ifndef IMAGECONTOURSLISTWIDGET_H
#define IMAGECONTOURSLISTWIDGET_H

#include <QAction>
#include <QColorDialog>
#include <QListWidget>
#include <QMouseEvent>


class ImageContoursListWidget : public QListWidget
{
    Q_OBJECT

private:
    QVector<unsigned int> checked;    // List of checked items number.
    QString selectedItem;   // Name of the selected item.
    QStringList pathsList;  // List of the item names.

    void mousePressEvent(QMouseEvent *event);

public:
    ImageContoursListWidget(QWidget *parent = 0);

signals:
    void AddImageContour();
    void RemoveImageContours();
//    void modelColor(QColor& newColor);
//    void InitialiseSimulation();
//    void Segmentation();
//    void ComputeGaussianCurvature();

    void SelectedImageContourChanged(QString& selectedItem);

    void UpdateList();
    void UpdateCheckedImageContours(QVector<unsigned int>& checked);
    void ShowImageContour(unsigned int Index);

public slots:
    void EmitAddImageContour();
    void EmitRemoveImageContours();
//    void emitChangeColor();
//    void changeColor(QColor newColor);
//    void EmitInitialiseSimulation();
//    void EmitSegmentation();
//    void EmitComputeGaussianCurvature();

    void UpdateImageContoursList(QStringList items);
    void UpdateCheckedImageContoursSlot();

    void SetSelected(QListWidgetItem* item);
};

#endif // IMAGECONTOURSLISTWIDGET_H
