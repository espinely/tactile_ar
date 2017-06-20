// Reference: http://www.mathematik.uni-marburg.de/~thormae/lectures/graphics1/code/WebGLShaderLightMat/ShaderLightMat.html

attribute vec3 vertex;
//attribute vec2 inputTexCoord;
attribute vec3 normal;
attribute mediump vec4 texCoord;
attribute bool selected;
attribute vec3 faceCentre;
attribute vec4 vertexColour;

//uniform mat4 projMatrix, mvMatrix, normalMatrix;
uniform mat4 projMatrix, mvMatrix;
uniform mat3 normalMatrix;

varying vec4 vertexColourInterp;

varying vec3 normalInterp;
varying vec3 vertPos;
varying mediump vec4 texc;
varying bool vertexSelected = false;
varying vec3 faceCentrePos;

void main(){
    gl_Position = projMatrix * mvMatrix * vec4(vertex, 1.0);
    vec4 vertPos4 = mvMatrix * vec4(vertex, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
//   normalInterp = vec3(normalMatrix * vec4(normal, 0.0));
    normalInterp = normalMatrix * normal;
    texc = texCoord;

    vertexSelected = selected;

    vertexColourInterp = vertexColour;

    vec4 faceCentrePos4 = mvMatrix * vec4(faceCentre, 1.0);
    faceCentrePos = vec3(faceCentrePos4) / faceCentrePos4.w;
}
