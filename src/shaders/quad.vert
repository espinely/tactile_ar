attribute vec2 vertex;
const vec2 madd = vec2(0.5, 0.5);
varying mediump vec2 texCoord;

void main()
{
    texCoord = vertex.xy * madd + madd; // Scale vertex attribute to [0-1] range.
    gl_Position = vec4(vertex.xy, 0.0, 1.0);
}
