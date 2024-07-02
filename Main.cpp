#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb/stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

#include "Texture.h"
#include "shaderClass.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "Camera.h"

const unsigned int width = 1280;
const unsigned int height = 720;
const unsigned int stacks = 30;
const unsigned int sectors = 30;
float* radius = new float(0.1);
const float PI = 3.14159;

const unsigned short spheresCount = 3;
int vertexIndex = 0;
int index = 0;

void generateVerticesAndIndices(GLfloat* vertices, GLuint* indices) {
    const float sectorStep = 2 * PI / sectors;
    const float stackStep = PI / stacks;

    for (int i = 0; i <= stacks; ++i) {
        float stackAngle = PI / 2 - i * stackStep;
        float xy = *radius * cosf(stackAngle);
        float z = *radius * sinf(stackAngle);

        for (int j = 0; j <= sectors; ++j) {
            float sectorAngle = j * sectorStep;
            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);
            vertices[vertexIndex++] = x;
            vertices[vertexIndex++] = y;
            vertices[vertexIndex++] = z;
        }
    }

    for (int i = 0; i < stacks; ++i) {
        int k1 = i * (sectors + 1);
        int k2 = k1 + sectors + 1;

        for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices[index++] = k1;
                indices[index++] = k2;
                indices[index++] = k1 + 1;
            }
            if (i != (stacks - 1)) {
                indices[index++] = k1 + 1;
                indices[index++] = k2;
                indices[index++] = k2 + 1;
            }
        }
    }
}

double lastFrame = 0.0;

double calculateFPS(GLFWwindow* window) {
    double currentFrame = glfwGetTime();
    double deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    return 1.0 / deltaTime;
}

struct State {
    float x, y, z;
};

float bezier(float t, float p0, float p1, float p2, float p3) {
    float u = 1 - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;

    float p = uuu * p0;
    p += 3 * uu * t * p1;
    p += 3 * u * tt * p2;
    p += ttt * p3;

    return p;
}

State bezierPosition(float t, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
    State state;
    state.x = bezier(t, p0.x, p1.x, p2.x, p3.x);
    state.y = bezier(t, p0.y, p1.y, p2.y, p3.y);
    state.z = bezier(t, p0.z, p1.z, p2.z, p3.z);
    return state;
}

State bezierVelocity(float t, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
    State state;
    state.x = 3 * pow(1 - t, 2) * (p1.x - p0.x) + 6 * (1 - t) * t * (p2.x - p1.x) + 3 * pow(t, 2) * (p3.x - p2.x);
    state.y = 3 * pow(1 - t, 2) * (p1.y - p0.y) + 6 * (1 - t) * t * (p2.y - p1.y) + 3 * pow(t, 2) * (p3.y - p2.y);
    state.z = 3 * pow(1 - t, 2) * (p1.z - p0.z) + 6 * (1 - t) * t * (p2.z - p1.z) + 3 * pow(t, 2) * (p3.z - p2.z);
    return state;
}

State bezierAcceleration(float t, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
    State state;
    state.x = 6 * (1 - t) * (p2.x - 2 * p1.x + p0.x) + 6 * t * (p3.x - 2 * p2.x + p1.x);
    state.y = 6 * (1 - t) * (p2.y - 2 * p1.y + p0.y) + 6 * t * (p3.y - 2 * p2.y + p1.y);
    state.z = 6 * (1 - t) * (p2.z - 2 * p1.z + p0.z) + 6 * t * (p3.z - 2 * p2.z + p1.z);
    return state;
}

int main() {
    const int numVertices = (sectors + 1) * (stacks + 1) * 3;
    const int numIndices = stacks * sectors * 6;
    GLfloat vertices[numVertices];
    GLuint indices[numIndices];
    generateVerticesAndIndices(vertices, indices);

    // Initialize GLFW
    glfwInit();

    // Set GLFW window properties
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    double time = 0.0;
    double dt = 0.001;

    GLFWwindow* window = glfwCreateWindow(width, height, "OpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    gladLoadGL();
    glViewport(0, 0, width, height);

    Shader shaderProgram("default.vert", "default.frag");
    Shader lineShader("default.vert", "default.frag"); // Shader for drawing the line

    VAO VAO1;
    VAO1.Bind();
    VBO VBO1(vertices, sizeof(vertices));
    EBO EBO1(indices, sizeof(indices));
    VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 3 * sizeof(GLfloat), (void*)0);
    VAO1.Unbind();
    VBO1.Unbind();
    EBO1.Unbind();

    glEnable(GL_DEPTH_TEST);

    Camera camera(width, height, glm::vec3(0.0f, 0.0f, 2.0f));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    float color[4] = { 0.8f, 0.3f, 0.02f, 1.0f };
    float* fov = new float(90);

    glm::vec3 controlPoints[4] = {
        glm::vec3(-0.5f, 0.0f, 0.0f),
        glm::vec3(-0.25f, 1.5f, 0.0f),
        glm::vec3(1.25f, -3.5f, 0.0f),
        glm::vec3(0.5f, 1.0f, 0.0f)
    };

    // Generate points for the Bézier curve
    std::vector<State> curvePoints;
    for (float t = 0; t <= 1.0f; t += dt) {
        curvePoints.push_back(bezierPosition(t, controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3]));
    }

    GLuint curveVBO, curveVAO;
    glGenVertexArrays(1, &curveVAO);
    glGenBuffers(1, &curveVBO);
    glBindVertexArray(curveVAO);

    glBindBuffer(GL_ARRAY_BUFFER, curveVBO);
    glBufferData(GL_ARRAY_BUFFER, curvePoints.size() * sizeof(State), &curvePoints[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(State), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        shaderProgram.Activate();

        camera.Inputs(window);
        camera.Matrix(*fov, 0.1f, 100.0f, shaderProgram, "camMatrix", glm::vec3(0.0f, 0.0f, 0.0f));

        State position = bezierPosition(time, controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3]);
        State velocity = bezierVelocity(time, controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3]);
        State acceleration = bezierAcceleration(time, controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3]);

        glm::vec3 translation(position.x, position.y, position.z);
        camera.Matrix(*fov, 0.1f, 100.0f, shaderProgram, "camMatrix", translation);
        VAO1.Bind();
        EBO1.Bind();
        glDrawElements(GL_TRIANGLE_STRIP, numIndices, GL_UNSIGNED_INT, 0);

        // Draw the Bézier curve
        lineShader.Activate();
        camera.Matrix(*fov, 0.1f, 100.0f, lineShader, "camMatrix", glm::vec3(0.0f, 0.0f, 0.0f));
        glBindVertexArray(curveVAO);
        glDrawArrays(GL_LINE_STRIP, 0, curvePoints.size());

        lineShader.Activate();
        camera.Matrix(*fov, 0.1f, 100.0f, lineShader, "camMatrix", glm::vec3(0.0f, 0.0f, 0.0f));
        glBindVertexArray(curveVAO);
        glDrawArrays(GL_LINE_STRIP, 0, curvePoints.size());

        float fps = calculateFPS(window);

        ImGui::Begin("Parameters");
        ImGui::Text("FPS: %.0f", fps);

        if (ImGui::CollapsingHeader("Info")) {
            ImGui::Text("Window Dimensions: %d x %d", width, height);
        }

        if (ImGui::CollapsingHeader("Timers")) {
            ImGui::Text("Frame Time: %.3f ms", (glfwGetTime() - lastFrame) * 1000.0);
        }

        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Text("Camera Position: x: %.2f, y: %.2f, z: %.2f", camera.Position.x, camera.Position.y, camera.Position.z);
            ImGui::SliderFloat("Field of View", fov, 40.0f, 120.0f);
        }

        if (ImGui::CollapsingHeader("Spheres")) {
            ImGui::Text("Spheres: %d", spheresCount);
            ImGui::ColorEdit4("Color", color);
        }

        if (ImGui::CollapsingHeader("Bezier Curve")) {
            ImGui::Text("Position: x: %.2f, y: %.2f, z: %.2f", position.x, position.y, position.z);
            ImGui::Text("Velocity: x: %.2f, y: %.2f, z: %.2f", velocity.x, velocity.y, velocity.z);
            ImGui::Text("Acceleration: x: %.2f, y: %.2f, z: %.2f", acceleration.x, acceleration.y, acceleration.z);
        }

        ImGui::End();

        shaderProgram.Activate();
        glUniform1f(glGetUniformLocation(shaderProgram.ID, "size"), *radius);
        glUniform4f(glGetUniformLocation(shaderProgram.ID, "color"), color[0], color[1], color[2], color[3]);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();

        time += dt;
        if (time > 1.0) {
            time = 0.0;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    VAO1.Delete();
    VBO1.Delete();
    EBO1.Delete();
    shaderProgram.Delete();
    glDeleteBuffers(1, &curveVBO);
    glDeleteVertexArrays(1, &curveVAO);
    delete[] vertices;
    delete[] indices;
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}