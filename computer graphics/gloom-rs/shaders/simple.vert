#version 430 core

in layout (location = 0) vec3 position;
in layout (location = 1) vec4 color;
in layout (location = 2) vec3 normal;

out layout (location = 1) vec4 out_color;
out layout (location = 2) vec4 out_normal;


uniform mat4 transformation;
uniform mat4 model;
uniform float scale;


void main()
{
    gl_Position = transformation * vec4(position, 1.1f);
    out_color = color;
    mat3 rotation;
    rotation[0] = vec3(model[0][0], model[0][1], model[0][2]);
    rotation[1] = vec3(model[1][0], model[1][1], model[1][2]);
    rotation[2] = vec3(model[2][0], model[2][1], model[2][2]);
    out_normal = vec4(normalize(rotation * normal), 1.0f);

}