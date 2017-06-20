precision mediump float;

uniform sampler2D texture;

// For scaling the texture randomly.
uniform float scale;

varying vec2 texCoord;

void main()
{
   vec4 colour = texture2D(texture, texCoord * scale);
   gl_FragColor = colour;
}
