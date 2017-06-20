#ifndef TRACKBALL_H
#define TRACKBALL_H

//#define PI 3.14159265

#include <QOpenGLWidget>
#include <QQuaternion>
#include <QTime>


class TrackBall
{
public:
    TrackBall();
    TrackBall(GLfloat angularVelocity, const QVector3D& axis);  // Coordinates in [-1,1]x[-1,1]

    void push(const QPointF& p);
    void move(const QPointF& p, const QQuaternion &transformation);
    void release(const QPointF& p, const QQuaternion &transformation);

    void reset();       // Resets rotation to 0
    QQuaternion rotation() const;   // Returns rotation
    void setRotation(QQuaternion r);  // Manually sets rotation to new value


private:
    QQuaternion m_rotation;
    QVector3D m_axis;
    GLfloat m_angularVelocity;

    QPointF m_lastPos;
    QTime m_lastTime;
    bool m_paused, m_pressed;

    void start();       // Starts clock
    void stop();        // Stops clock
};

#endif // TRACKBALL_H
