// Window + GLES2 context + input. The PC build uses X11 + EGL; the PS4 build (platform_ps4.cpp)
// uses OpenOrbis video-out + Piglet with the same interface.
#pragma once

namespace fohn {

struct Keys {
  // the stick: left / right turn, and the stick pulled BACK (down) climbs, pushed forward (up)
  // dives, as on the web build
  bool left = false, right = false, up = false, down = false;
  bool thrUp = false, thrDn = false;                            // W/S: the throttle lever
  bool strafeL = false, strafeR = false;                        // A/D: side-step in the hover, side kick at speed
  bool boost = false, brake = false;                            // Space / Shift or B
  bool respawn = false, quit = false;                           // R / Esc
  bool nextTime = false, nextWx = false, nextWind = false;      // T / X / N: the hour, the weather, the wind (one press, one step)
};

class Platform {
 public:
  bool initWindow(int w, int h, const char* title);
  bool initOffscreen(int w, int h);          // EGL pbuffer, no visible window (for screenshots)
  bool poll(Keys& k);                        // false if the window asked to close
  void swap();
  void shutdown();
  bool savePPM(const char* path);            // grab the current framebuffer as a P6 image

  int width() const { return w_; }
  int height() const { return h_; }

 private:
  int w_ = 1280, h_ = 720;
  void* disp_ = nullptr;      // EGLDisplay
  void* surf_ = nullptr;      // EGLSurface
  void* ctx_ = nullptr;       // EGLContext
  void* xdisp_ = nullptr;     // Display*
  unsigned long xwin_ = 0;    // Window
  bool offscreen_ = false;
  Keys state_{};
};

} // namespace fohn
