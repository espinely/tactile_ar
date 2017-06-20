#include "trackBall.h"


TrackBall::TrackBall()
    : m_angularVelocity(0), m_paused(false), m_pressed(false)
{
    m_axis = QVector3D(0, 1, 0);
    m_rotation = QQuaternion();
    m_lastTime = QTime::currentTime();
}
TrackBall::TrackBall(GLfloat angularVelocity, const QVector3D& axis)
    : m_axis(axis), m_angularVelocity(angularVelocity), m_paused(false), m_pressed(false)
{
    m_rotation = QQuaternion();
    m_lastTime = QTime::currentTime();
}


void TrackBall::push(const QPointF& p)
{
    m_rotation = rotation();
    m_pressed = true;
    m_lastTime = QTime::currentTime();
    m_lastPos = p;
    m_angularVelocity = 0.0f;
}
void TrackBall::release(const QPointF& p, const QQuaternion &transformation)
{
    move(p, transformation);
    m_pressed = false;
}


void TrackBall::move(const QPointF& p, const QQuaternion &transformation)
{
    if (!m_pressed)
        return;

    QTime currentTime = QTime::currentTime();
    GLint msecs = m_lastTime.msecsTo(currentTime);
    if (msecs <= 50)
        return;

    QVector3D lastPos3D = QVector3D(m_lastPos.x(), m_lastPos.y(), 0.0f);
    float sqrZ = 1 - QVector3D::dotProduct(lastPos3D, lastPos3D);
    if (sqrZ > 0)
        lastPos3D.setZ(sqrt(sqrZ));
    else
        lastPos3D.normalize();

    QVector3D currentPos3D = QVector3D(p.x(), p.y(), 0.0f);
    sqrZ = 1 - QVector3D::dotProduct(currentPos3D, currentPos3D);
    if (sqrZ > 0)
        currentPos3D.setZ(sqrt(sqrZ));
    else
        currentPos3D.normalize();

    m_axis = QVector3D::crossProduct(lastPos3D, currentPos3D);
    GLfloat angle = 180 / M_PI * asin(sqrt(QVector3D::dotProduct(m_axis, m_axis)));

    m_angularVelocity = angle / msecs;
    m_axis.normalize();
    m_axis = transformation.rotatedVector(m_axis);
    m_rotation = QQuaternion::fromAxisAndAngle(m_axis, angle) * m_rotation;
    m_lastPos = p;
    m_lastTime = currentTime;
}


void TrackBall::start()
{
    m_lastTime = QTime::currentTime();
    m_paused = false;
}
void TrackBall::stop()
{
    m_rotation = rotation();
    m_paused = true;
}


QQuaternion TrackBall::rotation() const     // Returns rotation
{
    if(m_paused || m_pressed)
        return m_rotation;

    QTime currentTime = QTime::currentTime();
    GLfloat angle = m_angularVelocity * m_lastTime.msecsTo(currentTime);
    return QQuaternion::fromAxisAndAngle(m_axis, angle) * m_rotation;
}


void TrackBall::reset() // Resets rotation to 0
{
    m_rotation = QQuaternion();
}


void TrackBall::setRotation(QQuaternion r)  // Manually sets rotation to new value
{
    m_rotation = r;
}
