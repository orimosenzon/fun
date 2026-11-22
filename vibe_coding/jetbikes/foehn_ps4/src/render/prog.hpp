// A linked GLES2 program with cached uniform / attribute locations, and the camera of a pass.
#pragma once
#include <GLES2/gl2.h>
#include <string>
#include <unordered_map>
#include "gl.hpp"
#include "mat4.hpp"

namespace fohn {

struct Prog {
  GLuint id = 0;
  std::unordered_map<std::string, GLint> u_, a_;
  void build(const std::string& vs, const std::string& fs) { id = glProgram(vs.c_str(), fs.c_str()); u_.clear(); a_.clear(); }
  void use() const { glUseProgram(id); }
  GLint U(const char* n) { auto it = u_.find(n); if (it != u_.end()) return it->second; GLint l = glGetUniformLocation(id, n); u_[n] = l; return l; }
  GLint A(const char* n) { auto it = a_.find(n); if (it != a_.end()) return it->second; GLint l = glGetAttribLocation(id, n); a_[n] = l; return l; }
  void f1(const char* n, float x) { glUniform1f(U(n), x); }
  void f2(const char* n, float x, float y) { glUniform2f(U(n), x, y); }
  void f3(const char* n, float x, float y, float z) { glUniform3f(U(n), x, y, z); }
  void f4(const char* n, float x, float y, float z, float w) { glUniform4f(U(n), x, y, z, w); }
  void v3(const char* n, const V3& v) { glUniform3f(U(n), (float)v[0], (float)v[1], (float)v[2]); }
  void m4(const char* n, const Mat4& m) { glUniformMatrix4fv(U(n), 1, GL_FALSE, m.data()); }
  void i1(const char* n, int x) { glUniform1i(U(n), x); }
  // bind a float attribute; returns its location (or -1 when the shader does not use it)
  GLint attr(const char* n, int size, GLenum type, bool normalized, int stride, size_t offset) {
    GLint l = A(n);
    if (l < 0) return l;
    glEnableVertexAttribArray(l);
    glVertexAttribPointer(l, size, type, normalized ? GL_TRUE : GL_FALSE, stride, (const void*)offset);
    return l;
  }
  void attrOff(const char* n) { GLint l = A(n); if (l >= 0) glDisableVertexAttribArray(l); }
};

struct Camera {
  V3 eye{0, 0, 0}, fwd{1, 0, 0}, right{0, 0, 1}, up{0, 1, 0};
  float fovY = 1.05f, aspect = 16.0f / 9, znear = 1.0f, zfar = 60000.0f;
  Mat4 view, proj, vp;
  void look(const V3& e, const V3& target, const V3& upHint = {0, 1, 0}) {
    eye = e; fwd = norm(sub(target, e)); right = norm(cross(fwd, upHint)); up = cross(right, fwd);
    view = mat4LookAt(e, target, upHint);
    proj = mat4Perspective(fovY, aspect, znear, zfar);
    vp = mat4Mul(proj, view);
  }
};

} // namespace fohn
