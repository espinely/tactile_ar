#ifndef MODELCONOUTRSLISTWIDGET_H
#define MODELCONOUTRSLISTWIDGET_H

#include <QAction>
#include <QColorDialog>
#include <QListWidget>
#include <QMouseEvent>
#include <QCheckBox>


class ModelContoursListWidget : public QListWidget
{
    Q_OBJECT

private:
    QVector<unsigned int> checked;    // List of checked items number.
    QString selectedItem;   // Name of the selected item.
    QStringList pathsList;  // List of the item names.
    QDialog* m_pHighCurvatureSelectionToolWindow;
    QSlider *m_pHighCurvatureStartSlider, *m_pHighCurvatureEndSlider, *m_pPolynomialOrderSlider, *m_pVertexSearchAreaRadiusSlider;
    QCheckBox* m_pReverseRangeCheckBox;

    void mousePressEvent(QMouseEvent *event);

public:
    ModelContoursListWidget(QWidget *parent = 0);

signals:
    void AddFrontierContour();
    void AddOccludingContour();
    void AddLigamentContour();
    void RemoveModelContours();
    void LinkModelToImageContour();
//    void modelColor(QColor& newColor);
//    void InitialiseSimulation();
//    void Segmentation();
//    void ComputeGaussianCurvature();

    void SelectedModelContourChanged(QString& selectedItem);

    void UpdateList();
    void UpdateCheckedModelContours(QVector<unsigned int>& checked);
    void ShowModelContour(unsigned int Index);

    void SetHighCurvatureStartPosition(float Position);
    void SetHighCurvatureEndPosition(float Position);
    void SetHighCurvatureRangeReversed(bool Reversed);
    void SetVertexSearchAreaRadius(float Radius);
    void SetHighCurvatureVerticesPolynomialOrder(int Order);

public slots:
    void EmitAddFrontierContour();
    void EmitAddOccludingContour();
    void EmitAddLigamentContour();
    void EmitRemoveModelContours();
    void EmitLinkToImageContour();
//    void emitChangeColor();
//    void changeColor(QColor newColor);
//    void EmitInitialiseSimulation();
//    void EmitSegmentation();
//    void EmitComputeGaussianCurvature();

    void UpdateModelContoursList(QStringList items);
    void UpdateCheckedModelContoursSlot();

    void SetSelected(QListWidgetItem* item);

    void OpenHighCurvatureSelectionTool(void);

private slots:
    void HighCurvatureStartSliderState(void);
    void HighCurvatureEndSliderState(void);
    void ReverseRangeState(int State);
    void VertexSearchAreaRadiusSliderState(void);
    void PolynomialOrderSliderState(void);
    void UpdateToolTip(int SliderValue);
    void UpdateVertexSearchAreaToolTip(int SliderValue);
};

#endif // MODELCONOUTRSLISTWIDGET_H
