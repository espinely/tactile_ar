#include "GLmodel.h"

GLmodel::GLmodel():model(0){}

void GLmodel::render(QStringList modelsList)
{
    vnFile.clear(), vtFile.clear(), squareFile.clear(), textures.clear(), origin.clear();
    model = glGenLists(modelsList.size());
    for(GLuint i = 0; i < (GLuint)modelsList.size(); i++)
        loadModel(modelsList.at(i), i);
}


void GLmodel::loadModel(QString fileName, GLuint modelNumber)
{
    QVector<QVector3D> Vertices, VNormals;  //v, vn
    QVector<QVector2D> VTexture;            //vt

    GLfloat xMin, yMin, zMin, xMax, yMax, zMax;   // Extreme coordinates to calculate origin
    xMin = yMin = zMin = xMax = yMax = zMax = 0;

    QVector3D temp3D;   // Buffer vectors and variables
    QVector2D temp2D;
    bool tempVnFile = false, tempVtFile = false, tempSquareFile = false, tempMtllib = false;
    QString MTLName;


/* ============================ READING FILE ============================ */
    if(!fileName.isEmpty())
    {
        QFile file(fileName);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QTextStream fileText(&file);
            while (!fileText.atEnd())
            {
                QString fileLine = fileText.readLine();
                fileLine = fileLine.simplified();

              // VN
                if(fileLine.startsWith("vn "))
                {
                    tempVnFile = true;

                    while(fileLine.contains(","))       // If necessary, replaces ',' with '.'
                        fileLine = fileLine.replace(fileLine.indexOf(",", 1),1,".");

                    QStringList lineList = fileLine.split(" ");

                    temp3D = QVector3D(lineList[1].toFloat(), lineList[2].toFloat(), lineList[3].toFloat());
                    VNormals.push_back(temp3D);
                }

              // VT
                else if(fileLine.startsWith("vt "))
                {
                    tempVtFile = true;

                    while(fileLine.contains(","))       // If necessary, replaces ',' with '.'
                        fileLine = fileLine.replace(fileLine.indexOf(",", 1),1,".");

                    QStringList lineList = fileLine.split(" ");

                    temp2D = QVector2D(lineList[1].toFloat(), lineList[2].toFloat());
                    VTexture.push_back(temp2D);
                }

              // V
                else if(fileLine.startsWith("v "))
                {
                    while(fileLine.contains(","))       // If necessary, replaces ',' with '.'
                        fileLine = fileLine.replace(fileLine.indexOf(",", 1),1,".");

                    QStringList lineList = fileLine.split(" ");

                    temp3D = QVector3D(lineList[1].toFloat(), lineList[2].toFloat(), lineList[3].toFloat());
                    Vertices.push_back(temp3D);

                    if (xMin==0 || temp3D.x() < xMin)   // Records the min and max coordinates values...
                        xMin = temp3D.x();              //...to calculate the model center "origin"
                    if (yMin==0 || temp3D.y() < yMin)
                        yMin = temp3D.y();
                    if (zMin==0 || temp3D.z() < zMin)
                        zMin = temp3D.z();
                    if (xMax==0 || temp3D.x() > xMax)
                        xMax = temp3D.x();
                    if (yMax==0 || temp3D.y() > yMax)
                        yMax = temp3D.y();
                    if (zMax==0 || temp3D.z() > zMax)
                        zMax = temp3D.z();
                }

              // F
                else if(fileLine.startsWith("f "))
                {
                    Face F;
                    QStringList lineList = fileLine.split(" ");

                    for(int i = 1; i <= 3; i++)
                    {
                        QStringList arg = lineList[i].split("/");

                        F.v[i-1] = Vertices[arg[0].toInt()-1];
                        if(tempVtFile)
                            F.t[i-1] = VTexture[arg[1].toInt()-1];
                        if(tempVnFile)
                            F.vn[i-1] = VNormals[arg[2].toInt()-1];
                    }
                    if(lineList.size()==5)
                    {
                        tempSquareFile = true;
                        QStringList arg = lineList[4].split("/");

                        F.v[3] = Vertices[arg[0].toInt()-1];
                        if(tempVtFile)
                            F.t[3] = VTexture[arg[1].toInt()-1];
                        if(tempVnFile)
                            F.vn[3] = VNormals[arg[2].toInt()-1];
                    }
                    Faces.push_back(F);
                }

              // MTLLIB
                else if(fileLine.startsWith("mtllib "))
                {
                    tempMtllib = true;
                    QStringList lineList = fileLine.split(" ");
                    MTLName = lineList[1];
                }
            }
        }
        file.close();

        QFileInfo fi(fileName);
        QString BaseName = fi.fileName();
        QString MTLPath(fileName);
        MTLPath.remove(MTLPath.size() - BaseName.size(), BaseName.size());
        if(tempVtFile)
            loadMTL(MTLPath, MTLName);

      // Save model parameters
        vnFile.push_back(tempVnFile);
        vtFile.push_back(tempVtFile * tempMtllib);
        squareFile.push_back(tempSquareFile);
    }

  // Calculates the model center coordinates
    temp3D = QVector3D(xMin+((xMax-xMin)/2), yMin+((yMax-yMin)/2), zMin+((zMax-zMin)/2));
    origin.push_back(temp3D);


/* ============================ DRAWING MODEL IN DISPLAYLIST ============================ */
    glNewList(model+modelNumber, GL_COMPILE);
        glBegin(GL_TRIANGLES);
            for(GLuint i = 0; i < (GLuint)Faces.size(); i++)
                if(Faces[i].v[3].isNull())
                {
                    calculateNormal(i, modelNumber);
                    for(GLuint j = 0; j < 3; j++)
                        drawFace(i, j, modelNumber);
                }
        glEnd();

        if(tempSquareFile)
        {
            glBegin(GL_QUADS);
                for(GLuint i = 0; i < (GLuint)Faces.size(); i++)
                    if(!Faces[i].v[3].isNull())
                    {
                        calculateNormal(i, modelNumber);
                        for(GLuint j = 0; j < 4; j++)
                            drawFace(i, j, modelNumber);
                    }
            glEnd();
        }
    glEndList();

    Faces.clear();
}


void GLmodel::saveModel(QString newModelName, QQuaternion rotation, QVector<QVector3D> coordModels)
{
    /*QChar newModelName_ch[newModelName.size()];     // Checking if .obj extension was seized during file creation
    QString extension = "";
    for(int i = 0; i < newModelName.size(); i++)
    {
        newModelName_ch[i] = newModelName.at(i);
        if(newModelName_ch[i]=='.')
            for(;i < newModelName.size(); i++)
                extension += newModelName.at(i);
    }
    if(extension=="")
        QTextStream(&newModelName)<<".obj";


    /* ============================ FILE WRITING ============================ */
  /*  QFile newFile(newModelName);
    if (!newFile.open(QIODevice::WriteOnly | QIODevice::Text))
        return;

    QFileInfo fileInfo(newModelName);


    /* ============================ HEADER ============================ */
    /*QTextStream fluxOut(&newFile);
    fluxOut.setCodec("UTF-8");
    fluxOut << "####" <<endl
    << "#" <<endl
    << "# OBJ File Generated by HEPATAUG" <<endl
    << "#" <<endl
    << "####" <<endl
    << "# Object " << fileInfo.fileName() <<endl
    << "#" <<endl
    << "# Vertices: " << Vertices.size() <<endl
    << "# Faces: " << Faces.size() <<endl
    << "#" <<endl
    << "####" <<endl<<endl;


    /* ============================ BODY - VERTICES ============================ */
    /*QMatrix4x4 r;
    r.rotate(rotation);

    for(int i=0; i < Vertices.size(); i++)
    {
        QVector3D coords = r * (Vertices[i] - origin) + coordModel;

        fluxOut << "v " << QString::number(coords.x(), 'f', 6)
                << " " << QString::number(coords.y(), 'f', 6)
                << " " << QString::number(coords.z(), 'f', 6) <<endl;
    }
    fluxOut<<endl;


    /* ============================ BODY - NORMAL ============================ */
    /*if(vnFile)
    {
        for(int i=0; i < VNormals.size(); i++)
        {
            fluxOut << "vn " << QString::number(VNormals[i].x(), 'f', 6)
            << " " << QString::number(VNormals[i].y(), 'f', 6)
            << " " << QString::number(VNormals[i].z(), 'f', 6) <<endl;
        }
        fluxOut<<endl;
    }


    /* ============================ BODY - TEXTURE ============================ */
    /*if(vtFile)
    {
        fluxOut << "mtllib " << MTLName <<endl;

        for(int i=0; i < VTexture.size(); i++)
        {
            fluxOut << "vt " << QString::number(VTexture[i].x(), 'f', 6)
            << " " << QString::number(VTexture[i].y(), 'f', 6) <<endl;
        }
        fluxOut<<endl;
    }


    /* ============================ BODY - FACES ============================ */
    /*QString modelName = modelsList.at(0);
    QFile file(modelName);
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    QTextStream fileText(&file);


    while(!fileText.atEnd())
    {
        QString fileLine = fileText.readLine();
        fileLine = fileLine.simplified();
        if(fileLine.startsWith("f "))
        {
            QStringList lineList = fileLine.split(" ");

            fluxOut << "f " << lineList[1] <<" "<< lineList[2] <<" "<< lineList[3];

            if(lineList.size()==5)
                fluxOut<<" "<< lineList[4]<<endl;
            else
                fluxOut<<endl;
        }
    }
    file.close();

    /* ============================ FOOTER ============================ */
    /*fluxOut << "\n# End of File ";
    newFile.close();*/
}


/* ============================ TEXTURE ============================ */
void GLmodel::loadMTL(QString MTLPath, QString MTLName)
{
    QFile file(MTLPath + MTLName);
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream fileText(&file);
        bool textureFound = false;

        while (!fileText.atEnd() && !textureFound)
        {
            QString fileLine = fileText.readLine();

            if(fileLine.startsWith("map_Kd "))
            {
                textureFound = true;
                QStringList lineList = fileLine.split(" ");
                QString textureName = MTLPath + lineList[1];

                while(textureName.contains("\\"))    // If necessary, replaces '\' with '/'
                    textureName = textureName.replace(textureName.indexOf("\\", 1),1,"/");

                while(textureName.contains(".."))    // If necessary, replaces '..' with '.'
                    textureName = textureName.remove(textureName.indexOf("..", 1),2);

                while(textureName.contains("//"))    // If necessary, replaces '//' with '/'
                    textureName = textureName.remove(textureName.indexOf("//", 1),1);

                loadTexture(textureName);
            }
        }
    }
    file.close();
}
void GLmodel::loadTexture(QString textureName)
{
    QImage textureImg = QImage(textureName);
//    GLuint texture;

//    textureImg = QGLWidget::convertToGLFormat(textureImg);
    QOpenGLTexture* pTexture = new QOpenGLTexture(textureImg.mirrored());
    pTexture->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    pTexture->setMagnificationFilter(QOpenGLTexture::Linear);


//    glGenTextures(1, &texture);
//    glBindTexture(GL_TEXTURE_2D, texture);

//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, glTexture.width(), glTexture.height(), 0, GL_RGBA,
//                 GL_UNSIGNED_BYTE, glTexture.bits());
//    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
//    glBindTexture(GL_TEXTURE_2D,0);

//    textures.push_back(texture);
    textures.push_back(pTexture);
}


/* ============================ REPETITIVE TASKS ============================ */
void GLmodel::calculateNormal(GLuint i, GLuint modelNumber)
{
    if(!vnFile[modelNumber])
    {
        GLfloat xNorm = (Faces[i].v[1].y()-Faces[i].v[0].y())*(Faces[i].v[2].z()-Faces[i].v[0].z())-(Faces[i].v[1].z()-Faces[i].v[0].z())*(Faces[i].v[2].y()-Faces[i].v[0].y());
        GLfloat yNorm = (Faces[i].v[1].z()-Faces[i].v[0].z())*(Faces[i].v[2].x()-Faces[i].v[0].x())-(Faces[i].v[1].x()-Faces[i].v[0].x())*(Faces[i].v[2].z()-Faces[i].v[0].z());
        GLfloat zNorm = (Faces[i].v[1].x()-Faces[i].v[0].x())*(Faces[i].v[2].y()-Faces[i].v[0].y())-(Faces[i].v[1].y()-Faces[i].v[0].y())*(Faces[i].v[2].x()-Faces[i].v[0].x());

        glNormal3f(xNorm,yNorm,zNorm);
    }
}
void GLmodel::drawFace(GLuint i, GLuint j, GLuint modelNumber)
{
    if(vnFile[modelNumber])
        glNormal3f(Faces[i].vn[j].x(), Faces[i].vn[j].y(), Faces[i].vn[j].z());
    if(vtFile[modelNumber])
        glTexCoord2f(Faces[i].t[j].x(), Faces[i].t[j].y());
    glVertex3f(Faces[i].v[j].x()-origin[modelNumber].x(), Faces[i].v[j].y()-origin[modelNumber].y(), Faces[i].v[j].z()-origin[modelNumber].z());
}


/* ============================ GETTERS ============================ */
GLuint GLmodel::getModel()
{
    return model;
}
QVector3D GLmodel::getCenter(GLuint modelNumber)
{
    return origin.at(modelNumber);
}
bool GLmodel::getTextureState(GLuint modelNumber)
{
    return(vtFile.at(modelNumber));
}

//GLuint GLmodel::getTexture(GLuint modelNumber)
QOpenGLTexture* GLmodel::getTexture(GLuint modelNumber)

{
    return(textures.at(modelNumber));
}
