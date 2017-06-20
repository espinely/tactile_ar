#ifndef GLMODEL_H
#define GLMODEL_H

#include <QFileDialog>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QTextStream>
#include <QVector2D>
#include <QOpenGLTexture>

using namespace std;


struct Face{
    QVector<QVector4D> v, vn;
    QVector<QVector3D> t;
    Face(){v.resize(4);
           vn.resize(4);
           t.resize(4);}
};


class GLmodel{

public:
    GLmodel();

    void render(QStringList modelsList);
    void saveModel(QString newModelName, QQuaternion rotation, QVector<QVector3D> coordModels);

    GLuint getModel();
    QVector3D getCenter(GLuint modelNumber);
    bool getTextureState(GLuint modelNumber);
    QOpenGLTexture* getTexture(GLuint modelNumber);


private:
    GLuint model;

    QVector<QVector3D> origin;   // Models center coordinates

    QVector<Face> Faces;    //f
//    QVector<GLuint> textures;
    QVector<QOpenGLTexture*> textures;

    void loadModel(QString fileName, GLuint modelNumber);
    void loadMTL(QString MTLPath, QString MTLName);
    void loadTexture(QString textureName);

  // REPETITIVE TASKS
    void calculateNormal(GLuint i, GLuint modelNumber);
    void drawFace(GLuint i, GLuint j, GLuint modelNumber);

    QVector<bool> vnFile, vtFile, squareFile;    // bool=True if the .obj file contains vn/vt/squares
};

#endif // GLMODEL_H
