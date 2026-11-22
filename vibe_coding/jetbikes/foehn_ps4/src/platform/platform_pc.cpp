// PC platform: X11 window (or EGL pbuffer) + EGL GLES2 context + keyboard.
#include "platform.hpp"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace fohn {

static EGLConfig chooseConfig(EGLDisplay d, bool window) {
  EGLint attrs[] = {
    EGL_SURFACE_TYPE, window ? EGL_WINDOW_BIT : EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_NONE
  };
  EGLConfig cfg; EGLint n = 0;
  if (!eglChooseConfig(d, attrs, &cfg, 1, &n) || n < 1) { std::fprintf(stderr, "eglChooseConfig failed\n"); return nullptr; }
  return cfg;
}

bool Platform::initWindow(int w, int h, const char* title) {
  w_ = w; h_ = h; offscreen_ = false;
  Display* xd = XOpenDisplay(nullptr);
  if (!xd) { std::fprintf(stderr, "XOpenDisplay failed (DISPLAY unset?)\n"); return false; }
  xdisp_ = xd;
  int screen = DefaultScreen(xd);
  Window root = RootWindow(xd, screen);
  XSetWindowAttributes swa; std::memset(&swa, 0, sizeof(swa));
  swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | StructureNotifyMask;
  Window win = XCreateWindow(xd, root, 0, 0, w, h, 0, CopyFromParent, InputOutput, CopyFromParent, CWEventMask, &swa);
  XStoreName(xd, win, title);
  XMapWindow(xd, win);
  xwin_ = win;

  EGLDisplay d = eglGetDisplay((EGLNativeDisplayType)xd);
  if (d == EGL_NO_DISPLAY) { std::fprintf(stderr, "eglGetDisplay failed\n"); return false; }
  if (!eglInitialize(d, nullptr, nullptr)) { std::fprintf(stderr, "eglInitialize failed\n"); return false; }
  eglBindAPI(EGL_OPENGL_ES_API);
  EGLConfig cfg = chooseConfig(d, true);
  if (!cfg) return false;
  EGLSurface s = eglCreateWindowSurface(d, cfg, (EGLNativeWindowType)win, nullptr);
  if (s == EGL_NO_SURFACE) { std::fprintf(stderr, "eglCreateWindowSurface failed\n"); return false; }
  EGLint ctxAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  EGLContext c = eglCreateContext(d, cfg, EGL_NO_CONTEXT, ctxAttrs);
  if (c == EGL_NO_CONTEXT) { std::fprintf(stderr, "eglCreateContext failed\n"); return false; }
  if (!eglMakeCurrent(d, s, s, c)) { std::fprintf(stderr, "eglMakeCurrent failed\n"); return false; }
  disp_ = d; surf_ = s; ctx_ = c;
  std::printf("GL: %s | %s\n", (const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION));
  return true;
}

bool Platform::initOffscreen(int w, int h) {
  w_ = w; h_ = h; offscreen_ = true;
  EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (d == EGL_NO_DISPLAY || !eglInitialize(d, nullptr, nullptr)) {
    // fall back to the X display
    Display* xd = XOpenDisplay(nullptr);
    if (!xd) { std::fprintf(stderr, "offscreen: no EGL display\n"); return false; }
    xdisp_ = xd;
    d = eglGetDisplay((EGLNativeDisplayType)xd);
    if (!eglInitialize(d, nullptr, nullptr)) { std::fprintf(stderr, "offscreen: eglInitialize failed\n"); return false; }
  }
  eglBindAPI(EGL_OPENGL_ES_API);
  EGLConfig cfg = chooseConfig(d, false);
  if (!cfg) return false;
  EGLint pb[] = {EGL_WIDTH, w, EGL_HEIGHT, h, EGL_NONE};
  EGLSurface s = eglCreatePbufferSurface(d, cfg, pb);
  if (s == EGL_NO_SURFACE) { std::fprintf(stderr, "eglCreatePbufferSurface failed\n"); return false; }
  EGLint ctxAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  EGLContext c = eglCreateContext(d, cfg, EGL_NO_CONTEXT, ctxAttrs);
  if (c == EGL_NO_CONTEXT) { std::fprintf(stderr, "eglCreateContext failed\n"); return false; }
  if (!eglMakeCurrent(d, s, s, c)) { std::fprintf(stderr, "eglMakeCurrent failed\n"); return false; }
  disp_ = d; surf_ = s; ctx_ = c;
  std::printf("GL(offscreen): %s | %s\n", (const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION));
  return true;
}

static void setKey(Keys& s, KeySym ks, bool down) {
  switch (ks) {
    case XK_Left: s.left = down; break;
    case XK_Right: s.right = down; break;
    case XK_Up: s.up = down; break;
    case XK_Down: s.down = down; break;
    case XK_w: case XK_W: s.thrUp = down; break;
    case XK_s: case XK_S: s.thrDn = down; break;
    case XK_a: case XK_A: s.strafeL = down; break;
    case XK_d: case XK_D: s.strafeR = down; break;
    case XK_space: s.boost = down; break;
    case XK_b: case XK_B: case XK_Shift_L: case XK_Shift_R: s.brake = down; break;
    case XK_r: case XK_R: s.respawn = down; break;
    case XK_t: case XK_T: if (down) s.nextTime = true; break;
    case XK_x: case XK_X: if (down) s.nextWx = true; break;
    case XK_n: case XK_N: if (down) s.nextWind = true; break;
    case XK_Escape: if (down) s.quit = true; break;
    default: break;
  }
}

bool Platform::poll(Keys& k) {
  if (!offscreen_ && xdisp_) {
    Display* xd = (Display*)xdisp_;
    while (XPending(xd)) {
      XEvent ev; XNextEvent(xd, &ev);
      if (ev.type == KeyPress || ev.type == KeyRelease) {
        KeySym ks = XLookupKeysym(&ev.xkey, 0);
        setKey(state_, ks, ev.type == KeyPress);
      } else if (ev.type == ConfigureNotify) {
        w_ = ev.xconfigure.width; h_ = ev.xconfigure.height;
      }
    }
  }
  k = state_;
  state_.respawn = false;        // edge-ish: consume one-shot
  state_.nextTime = state_.nextWx = state_.nextWind = false;
  return !state_.quit;
}

void Platform::swap() { if (disp_ && surf_) eglSwapBuffers((EGLDisplay)disp_, (EGLSurface)surf_); }

bool Platform::savePPM(const char* path) {
  std::vector<unsigned char> px((size_t)w_ * h_ * 4);
  glReadPixels(0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  std::fprintf(f, "P6\n%d %d\n255\n", w_, h_);
  std::vector<unsigned char> row(w_ * 3);
  for (int y = h_ - 1; y >= 0; y--) {                 // GL origin bottom-left -> flip
    for (int x = 0; x < w_; x++) {
      size_t i = ((size_t)y * w_ + x) * 4;
      row[x * 3 + 0] = px[i + 0]; row[x * 3 + 1] = px[i + 1]; row[x * 3 + 2] = px[i + 2];
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }
  std::fclose(f);
  return true;
}

void Platform::shutdown() {
  if (disp_) {
    eglMakeCurrent((EGLDisplay)disp_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (ctx_) eglDestroyContext((EGLDisplay)disp_, (EGLContext)ctx_);
    if (surf_) eglDestroySurface((EGLDisplay)disp_, (EGLSurface)surf_);
    eglTerminate((EGLDisplay)disp_);
  }
  if (xdisp_) { if (xwin_) XDestroyWindow((Display*)xdisp_, xwin_); XCloseDisplay((Display*)xdisp_); }
  disp_ = surf_ = ctx_ = xdisp_ = nullptr; xwin_ = 0;
}

} // namespace fohn
