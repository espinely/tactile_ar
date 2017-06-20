#ifndef GLTEXTURE_H
#define GLTEXTURE_H

#include <QOpenGLWidget>


class GLtexture
{
public:
    GLtexture();

    GLint getWidth();
    GLint getHeight();

    void setTexture(QString);
    GLuint getTexture();


private:
    GLuint texture[1];
    GLint width, height;
};

#endif // GLTEXTURE_H
