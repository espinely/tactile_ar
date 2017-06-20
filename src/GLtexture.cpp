#include "GLtexture.h"

GLtexture::GLtexture(){}


void GLtexture::setTexture(QString textureName)
{
    QImage qim_Texture;
    QImage qim_TempTexture;
    qim_TempTexture.load(textureName);
    qim_Texture = QGLWidget::convertToGLFormat(qim_TempTexture);

    glGenTextures(1, &texture[0]);
    glBindTexture(GL_TEXTURE_2D, texture[0]);

    glTexImage2D(GL_TEXTURE_2D, 0, 3, qim_Texture.width(), qim_Texture.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, qim_Texture.bits());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    width = qim_TempTexture.width();
    height = qim_TempTexture.height();
}


GLint GLtexture::getWidth()
{
    return width;
}
GLint GLtexture::getHeight()
{
    return height;
}


GLuint GLtexture::getTexture()
{
    return texture[0];
}
