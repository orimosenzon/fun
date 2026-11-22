// GLES2 helpers: shader compile + program link. Works on desktop Mesa/NVIDIA and on PS4 Piglet.
#pragma once
#include <GLES2/gl2.h>
#include <cstdio>
#include <cstdlib>

namespace fohn {

inline GLuint glCompile(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048]; GLsizei n = 0;
    glGetShaderInfoLog(s, sizeof(log), &n, log);
    std::fprintf(stderr, "shader compile error (%s):\n%s\n", type == GL_VERTEX_SHADER ? "vs" : "fs", log);
  }
  return s;
}

inline GLuint glProgram(const char* vs, const char* fs) {
  GLuint v = glCompile(GL_VERTEX_SHADER, vs), f = glCompile(GL_FRAGMENT_SHADER, fs);
  GLuint p = glCreateProgram();
  glAttachShader(p, v); glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[2048]; GLsizei n = 0; glGetProgramInfoLog(p, sizeof(log), &n, log); std::fprintf(stderr, "link error:\n%s\n", log); }
  glDeleteShader(v); glDeleteShader(f);
  return p;
}

} // namespace fohn
