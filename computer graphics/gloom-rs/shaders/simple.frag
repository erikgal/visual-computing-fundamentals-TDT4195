#version 430 core

in layout (location = 1) vec4 out_color;
in layout (location = 2) vec4 out_normal;


out vec4 color;

void main()
{
    vec3 normal = vec3(out_normal[0], out_normal[1], out_normal[2]);
    vec3 color3 = vec3(out_color[0], out_color[1], out_color[2]);
    vec3 lightDirection = -normalize(vec3(0.8, -0.5, 0.6));
    color = vec4( color3 * max(0, dot(normal, lightDirection)), out_color[3]);
}