# דרך קיצור לאנדרואיד

The Android app is a Capacitor shell around the web app. It does not carry its
own copy of `web/`: it loads the live site from GitHub Pages
(`capacitor.config.json` → `server.url`), so every push reaches installed apps
at once. Only a change on the native side needs a new APK.

What the app adds over the site (all of it in `web/native.js`, a no-op in a
browser):

- **Walk recording with the screen off.** A native watcher
  (`@capacitor-community/background-geolocation`) runs as a foreground service
  behind a "דרך קיצור מקליטה שביל" notification. The site cannot do this,
  because a locked screen stops a page from receiving GPS.
- **Precise location asked by the app itself**, so Android's own dialog
  offers "precise" by default. In Chrome the choice belongs to Chrome, and
  "approximate" there broke walk recording (11/10/2026).
- **The back button** closes whatever is open, the way Escape does. During a
  recording it only moves the app to the background. From the bare map it
  goes Home.
- Full screen, its own icon, an offline page (`www/offline.html`).

## Building

Needs JDK 21 (`~/.jdks/jdk-21*`, picked up by the script) and the Android SDK
at `~/android-sdk` (`android/local.properties`).

    npm install
    ./build.sh          # the real app, loads the live site
    ./build.sh dev      # loads http://127.0.0.1:8790 to try unpushed web changes

The APK is at `android/app/build/outputs/apk/debug/app-debug.apk`. Install
with `adb install -r <apk>`, or copy it to the phone and open it.

`python3 make_icons.py` redraws the launcher icons and splash from
`web/icon-512.png`.

## Testing on the emulator

An AVD named `dk_test` (Pixel 6, Android 15) is set up:

    ~/android-sdk/emulator/emulator -avd dk_test -no-window -no-audio -gpu swiftshader_indirect
    adb reverse tcp:8790 tcp:8790          # for a dev build
    adb emu geo fix <lng> <lat>            # move the fake GPS

Debug builds expose the WebView to DevTools (`chrome://inspect`, or
`adb forward tcp:9333 localabstract:webview_devtools_remote_<pid>` and raw
CDP). Playwright's `connect_over_cdp` does not work against a WebView.

## Not done yet

- A release signing key. The first APK given out sets the key for every later
  update, and losing it means people have to uninstall and reinstall. So the
  key should be made once and kept somewhere safe.
- Opening GPX/KML files from WhatsApp straight into the app.
- Google Play: developer account, privacy policy, data safety form, closed
  test (12 testers × 14 days for a new personal account).
