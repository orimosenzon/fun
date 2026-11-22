// PS4 platform (OpenOrbis homebrew): video-out + Piglet (GLES2) + scePad.
//
// SCAFFOLD. This file is built only with -DPS4=ON under the OpenOrbis toolchain
// (see ../../toolchain/NOTES.md). It mirrors the PC platform's interface so main.cpp,
// the renderer and the whole simulation are shared unchanged. The GLES2 calls in the
// renderer run on the PS4 GPU through Piglet exactly as they run through Mesa/NVIDIA on PC.
//
// What still has to be wired against the installed SDK (left as TODO, because it cannot be
// compiled or tested without a jailbroken console on firmware 13.50):
//   * loading the system modules (libSceVideoOut, Piglet libScePigletv2VSH, the shader
//     compiler libSceShccVSH) — on some exploited consoles the runtime shader compiler is
//     missing, so shaders may need to be precompiled (see the PS4GL_GNM route in NOTES.md);
//   * the exact VideoOut -> EGLNativeWindow handoff (OpenOrbis ships a sample for this);
//   * mapping the DualShock 4 analog sticks to the flight inputs (here they are thresholded
//     into the shared digital Keys; a later pass should feed the sticks straight into
//     fohn::Input for proper analog control).
#if defined(__ORBIS__) || defined(PS4_BUILD)

#include "platform.hpp"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <cstdio>
#include <cstring>

// OpenOrbis headers (present in the toolchain's include path)
#include <orbis/libkernel.h>
#include <orbis/VideoOut.h>
#include <orbis/Pad.h>
#include <orbis/UserService.h>

namespace fohn {

static int s_padHandle = -1;

bool Platform::initWindow(int w, int h, const char* title) {
  (void)title;
  w_ = w; h_ = h; offscreen_ = false;

  // --- user + pad ---
  OrbisUserServiceInitializeParams up; std::memset(&up, 0, sizeof(up)); up.priority = ORBIS_KERNEL_PRIO_FIFO_LOWEST;
  sceUserServiceInitialize(&up);
  int userId = -1; sceUserServiceGetInitialUser(&userId);
  scePadInit();
  s_padHandle = scePadOpen(userId, ORBIS_PAD_PORT_TYPE_STANDARD, 0, nullptr);

  // --- video out + EGL (Piglet) ---
  // TODO: create the VideoOut handle and flip chain, obtain the EGLNativeWindow, then:
  EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (d == EGL_NO_DISPLAY || !eglInitialize(d, nullptr, nullptr)) { std::fprintf(stderr, "PS4: EGL init failed\n"); return false; }
  eglBindAPI(EGL_OPENGL_ES_API);
  EGLint attrs[] = {
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE};
  EGLConfig cfg; EGLint n = 0;
  if (!eglChooseConfig(d, attrs, &cfg, 1, &n) || n < 1) { std::fprintf(stderr, "PS4: no EGL config\n"); return false; }
  // EGLNativeWindowType from the VideoOut surface goes here:
  EGLNativeWindowType nativeWin = (EGLNativeWindowType)0; // TODO: fill from sceVideoOut
  EGLSurface s = eglCreateWindowSurface(d, cfg, nativeWin, nullptr);
  EGLint ctxAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  EGLContext c = eglCreateContext(d, cfg, EGL_NO_CONTEXT, ctxAttrs);
  if (!eglMakeCurrent(d, s, s, c)) { std::fprintf(stderr, "PS4: eglMakeCurrent failed\n"); return false; }
  disp_ = d; surf_ = s; ctx_ = c;
  return true;
}

bool Platform::initOffscreen(int w, int h) { return initWindow(w, h, "offscreen"); }

bool Platform::poll(Keys& k) {
  state_.respawn = false;
  if (s_padHandle >= 0) {
    OrbisPadData pad; std::memset(&pad, 0, sizeof(pad));
    if (scePadReadState(s_padHandle, &pad) == 0) {
      // analog sticks (0..255, 128 centre) -> thresholded digital for the shared Keys
      auto lo = [](unsigned char v) { return v < 90; };
      auto hi = [](unsigned char v) { return v > 166; };
      state_.left = lo(pad.leftStick.x);  state_.right = hi(pad.leftStick.x);
      state_.up = lo(pad.leftStick.y);    state_.down = hi(pad.leftStick.y);     // stick forward = smaller y; pulled back (down) climbs
      // as the web build maps a pad: right stick sideways = side-step, triggers = throttle, cross = overdrive, circle = brake
      state_.strafeL = lo(pad.rightStick.x); state_.strafeR = hi(pad.rightStick.x);
      state_.thrUp = (pad.buttons & ORBIS_PAD_BUTTON_R2);
      state_.thrDn = (pad.buttons & ORBIS_PAD_BUTTON_L2);
      state_.boost = (pad.buttons & ORBIS_PAD_BUTTON_CROSS);
      state_.brake = (pad.buttons & ORBIS_PAD_BUTTON_CIRCLE);
      state_.respawn = (pad.buttons & ORBIS_PAD_BUTTON_TRIANGLE);
      // d-pad: the hour, the weather, the wind (one step per press)
      static unsigned prevButtons = 0;
      unsigned pressed = pad.buttons & ~prevButtons; prevButtons = pad.buttons;
      state_.nextTime = (pressed & ORBIS_PAD_BUTTON_LEFT); state_.nextWx = (pressed & ORBIS_PAD_BUTTON_RIGHT); state_.nextWind = (pressed & ORBIS_PAD_BUTTON_UP);
      if (pad.buttons & ORBIS_PAD_BUTTON_OPTIONS) state_.quit = true;
    }
  }
  k = state_;
  return !state_.quit;
}

void Platform::swap() { if (disp_ && surf_) eglSwapBuffers((EGLDisplay)disp_, (EGLSurface)surf_); }

bool Platform::savePPM(const char*) { return false; }   // not used on the console

void Platform::shutdown() {
  if (disp_) {
    eglMakeCurrent((EGLDisplay)disp_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (ctx_) eglDestroyContext((EGLDisplay)disp_, (EGLContext)ctx_);
    if (surf_) eglDestroySurface((EGLDisplay)disp_, (EGLSurface)surf_);
    eglTerminate((EGLDisplay)disp_);
  }
  disp_ = surf_ = ctx_ = nullptr;
}

} // namespace fohn

#endif // __ORBIS__
