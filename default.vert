#version 330 core

// Positions/Coordinates
layout (location = 0) in vec3 aPos;


// Outputs the color for the Fragment Shader
out vec3 color;

// Imports the camera matrix from the main function
uniform mat4 camMatrix;


void main()
{
	gl_Position = camMatrix * vec4(aPos, 1.0);
}