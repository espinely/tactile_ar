precision mediump float;

varying vec3 normalInterp;
varying vec3 vertPos;
varying vec3 faceCentrePos;

//uniform int mode;

uniform sampler2D texture;
varying mediump vec4 texc;
uniform highp vec3 lightPos;

//const vec3 lightPos = vec3(0.0,0.0,0.0);
vec3 ambientColor = vec3(0.01, 0.01, 0.01); //vec3(0.02, 0.02, 0.02);

uniform float c = 0.025; // Coefficient for Lambertian model. 0.05; // For rendering vertex error.
vec3 diffuseColor = vec3(c, c, c);
//const vec3 diffuseColor = vec3(1.0, 1.0, 1.0);

const vec3 specColor = vec3(1.0, 1.0, 1.0);

// For scaling the texture randomly.
uniform float texScale = 1.0;

varying bool vertexSelected;

varying vec4 vertexColourInterp;

void main() {

  vec3 normal = normalize(normalInterp);

//  vec3 lightDir = normalize(lightPos - vertPos); // For phong rendering/CNN.
  vec3 lightDir = normalize(lightPos - faceCentrePos); // For computing reflectance.

  float lambertian = max(dot(lightDir,normal), 0.0);
  float specular = 0.0;

  if(lambertian > 0.0) {

    vec3 reflectDir = reflect(-lightDir, normal);

//    vec3 viewDir = normalize(-vertPos); // For phong rendering/CNN.
    vec3 viewDir = normalize(-faceCentrePos); // For computing reflectance.

    float specAngle = max(dot(reflectDir, viewDir), 0.0);
    specular = pow(specAngle, 30.0);

    // the exponent controls the shininess (try mode 2)
//        "    if(mode == 2)  specular = pow(specAngle, 16.0);

    // according to the rendering equation we would need to multiply
    // with the the 'lambertian', but this has little visual effect
//        "    if(mode == 3) specular *= lambertian;

     specular *= lambertian;
    // switch to mode 4 to turn off the specular component
//        "    if(mode == 4) specular *= 0.0;

  }

//  gl_FragColor = vec4(ambientColor + texture2D(texture, texc.st * texScale) * lambertian * diffuseColor + specular * specColor, 1.0) / (length(vertPos) * length(vertPos)); // For phong rendering/CNN.
  // For computing reflectance.
  diffuseColor.rgb = vertexColourInterp.rgb * c;
  gl_FragColor = vec4(lambertian * diffuseColor, 1.0) / (length(faceCentrePos) * length(faceCentrePos));
  gl_FragColor.a = vertexColourInterp.a;

  if (vertexSelected)
  {
      gl_FragColor.rgb = vec3(1.0, 0.0, 0.0);
  }

  // TODO: Temp for rendering vertex error.
//  gl_FragColor = vec4(gl_FragColor.xyz * vertexColourInterp ,1.0);

}
