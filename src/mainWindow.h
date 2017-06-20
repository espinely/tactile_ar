#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QDesktopWidget>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QPushButton>
#include <QSlider>
#include <QToolBar>
#include <QToolTip>
#include <QCheckBox>

#include "GLWidget.h"
#include "ModelsListWidget.h"
#include "ModelContoursListWidget.h"
#include "ImageContoursListWidget.h"


class ContourSelectionToolDialog : public QDialog
{
    Q_OBJECT

public:
    ContourSelectionToolDialog(QWidget* pParent)
    {
        setWindowFlags(Qt::WindowStaysOnTopHint);

        connect(this, SIGNAL(ContourSelectionToolClosing()), pParent, SLOT(ContourSelectionToolClosed()));
    }

    // Called when the dialog is closed.
    void reject()
    {
        emit ContourSelectionToolClosing();

        QDialog::reject();
    }

signals:
    void ContourSelectionToolClosing();
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();
    GLWidget* Object(void) { return object; }


private:
    QGridLayout *layoutPrincipal;
    GLWidget *object;

    QToolBar *leftToolBar;
    QAction *pActionContour, *distance;
    QAction* m_pActionOptimisation;

    ContourSelectionToolDialog* m_pContourSelectionToolWindow;
    bool m_ContourSelectionFinalised;
    bool m_OptimisationUsingContourAndShading;

    ModelsListWidget *list;
    ModelContoursListWidget* m_pModelContoursList;
    ImageContoursListWidget* m_pImageContoursList;

    QToolBar *rightToolBar;
    QSlider *opacitySlider, *scaleSlider, *m_pFOVScaleSlider;

    GLuint screenshotNumber;    // Number of screenshots taken

  // SETTINGS
    QLineEdit *framePictureRatioLineEdit, *rotationSpeedLineEdit, *tagsRadiusLineEdit;
    QLineEdit *alphaX, *alphaY, *skewness, *centerX, *centerY, *near, *far;
    QLineEdit *sensibilityLineEdit, *sensibilityPlusLineEdit;
    QCheckBox* m_pRenderingWireframeCheckBox;


signals :
    void openModelsListWindow();
    void ContourSelection(bool Enabled, bool Finalised);
    void ResetContourSelection();
    void RunOptimisation(bool UsingContourAndShading);

private slots :
    void resizeMainWindow(int newWidth, int newHeight);
    void ResetWindowScaleSlider();

    void OpenContourSelectionTool();
    void EmitResetContourSelection();
    void EmitFinaliseContourSelection();
    void ContourSelectionToolClosed();
    void OptimisationDialog();
    void SetOptimisationUsingContour();
    void SetOptimisationUsingContourAndShading();
    void EmitRunOptimisation();

    void updateModelsList();
    void UpdateModelContoursList();
    void UpdateImageContoursList();
    void distanceMode();
    void screenshot();
    void settingsWindow();
    void sendSettings();

    void updateToolTip(int sliderValue);
    void scaleSliderState();
    void FOVScaleSliderState();
    void UpdateFOVSlider(float Scale);
    void focusOFF();
};

#endif // MAINWINDOW_H
