package il.derechkitzur.app;

import android.app.ActivityManager;
import android.app.ApplicationExitInfo;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.location.Location;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.os.Process;
import android.os.SystemClock;

import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import java.io.File;
import java.io.FileWriter;
import java.io.RandomAccessFile;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * The walk log, kept by the app rather than the page.
 *
 * A walk recorded with the screen off goes through three things that can each
 * stop on their own: the location service, the WebView that receives its
 * fixes, and the process that holds both. The page's own log (localStorage)
 * dies with the page, so it cannot say which one stopped. This file is written
 * from Java, survives a reload and a killed process, and the page appends its
 * own lines to it too, so one file holds the whole timeline:
 *
 *   process start, and why Android ended the previous one (API 30+)
 *   activity create/start/resume/pause/stop/destroy
 *   screen on/off, memory pressure, the WebView renderer dying
 *   every location the service hands out, summed every 30 s, and gaps
 *   a heartbeat every minute, to tell a sleeping process from a dead one
 *
 * It stays on the phone. The page shows it under ☰ → לוג הקלטה, and nothing
 * sends it anywhere unless the walker presses share.
 */
final class DkLog {

    private static final String FILE = "dk-walk.log";
    private static final long MAX_BYTES = 600_000;   // trimmed to half past this
    private static final String GEO_BROADCAST =
            "com.equimaps.capacitor_background_geolocation.broadcast";

    private static File file;
    private static boolean started;

    // Location summary, written every SUMMARY_MS.
    private static final long SUMMARY_MS = 30_000;
    private static final long GAP_MS = 10_000;
    private static int fixes;
    private static long lastFixAt;          // elapsedRealtime of the last fix
    private static long lastSummaryAt;
    private static Location lastFix;

    private DkLog() {}

    static synchronized void w(String... parts) {
        if (file == null) return;
        String t = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
        StringBuilder sb = new StringBuilder(t).append(" N ");
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(parts[i]);
        }
        append(sb.append('\n').toString());
    }

    /** A line from the page, already stamped by it. */
    static synchronized void page(String line) {
        if (file == null) return;
        append(line.replace('\n', ' ') + "\n");
    }

    private static void append(String s) {
        try (FileWriter fw = new FileWriter(file, true)) {
            fw.write(s);
        } catch (Exception ignore) {
            return;
        }
        if (file.length() > MAX_BYTES) trim();
    }

    private static void trim() {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            long keep = MAX_BYTES / 2;
            raf.seek(raf.length() - keep);
            byte[] buf = new byte[(int) keep];
            raf.readFully(buf);
            String tail = new String(buf, "UTF-8");
            int nl = tail.indexOf('\n');
            tail = nl >= 0 ? tail.substring(nl + 1) : tail;
            try (FileWriter fw = new FileWriter(file, false)) { fw.write(tail); }
        } catch (Exception ignore) {}
    }

    static synchronized String read() {
        if (file == null || !file.exists()) return "";
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte[] buf = new byte[(int) raf.length()];
            raf.readFully(buf);
            return new String(buf, "UTF-8");
        } catch (Exception e) {
            return "(read failed: " + e + ")";
        }
    }

    static synchronized void clear() {
        if (file != null) file.delete();
        w("log cleared");
    }

    /** Once per process. */
    static void start(Context any) {
        if (started) return;
        started = true;
        final Context ctx = any.getApplicationContext();
        file = new File(ctx.getFilesDir(), FILE);

        PowerManager pm = (PowerManager) ctx.getSystemService(Context.POWER_SERVICE);
        w("==== process start", "pid=" + Process.myPid(),
                "android=" + Build.VERSION.RELEASE + "/" + Build.VERSION.SDK_INT,
                Build.MANUFACTURER + " " + Build.MODEL,
                "app=" + version(ctx),
                "batteryOptExempt=" + (pm != null && pm.isIgnoringBatteryOptimizations(ctx.getPackageName())),
                "powerSave=" + (pm != null && pm.isPowerSaveMode()));
        lastExit(ctx);

        // Screen on/off: the moments the walker reports.
        BroadcastReceiver screen = new BroadcastReceiver() {
            @Override public void onReceive(Context c, Intent i) {
                String a = i.getAction();
                PowerManager p = (PowerManager) c.getSystemService(Context.POWER_SERVICE);
                w(Intent.ACTION_SCREEN_ON.equals(a) ? "screen on"
                        : Intent.ACTION_SCREEN_OFF.equals(a) ? "screen off" : "unlocked",
                        "idle=" + (p != null && Build.VERSION.SDK_INT >= 23 && p.isDeviceIdleMode()));
            }
        };
        IntentFilter sf = new IntentFilter();
        sf.addAction(Intent.ACTION_SCREEN_ON);
        sf.addAction(Intent.ACTION_SCREEN_OFF);
        sf.addAction(Intent.ACTION_USER_PRESENT);
        ContextCompat.registerReceiver(ctx, screen, sf, ContextCompat.RECEIVER_NOT_EXPORTED);

        // Every fix the location service sends out, whether or not the page is
        // there to hear it. Same broadcast the plugin itself listens to.
        LocalBroadcastManager.getInstance(ctx).registerReceiver(new BroadcastReceiver() {
            @Override public void onReceive(Context c, Intent i) {
                Location loc = i.getParcelableExtra("location");
                if (loc == null) return;
                onFix(loc, i.getStringExtra("id"));
            }
        }, new IntentFilter(GEO_BROADCAST));

        // Heartbeat on the uptime clock: it stops while the CPU sleeps, so a
        // long silence followed by more beats means the process slept, and no
        // more beats with a "process start" after them means it was killed.
        final Handler h = new Handler(Looper.getMainLooper());
        h.postDelayed(new Runnable() {
            @Override public void run() {
                long since = lastFixAt == 0 ? -1 : (SystemClock.elapsedRealtime() - lastFixAt) / 1000;
                w("beat", "lastFix=" + (since < 0 ? "never" : since + "s ago"));
                h.postDelayed(this, 60_000);
            }
        }, 60_000);
    }

    private static String version(Context ctx) {
        try {
            return ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0).versionName;
        } catch (Exception e) {
            return "?";
        }
    }

    private static synchronized void onFix(Location loc, String watcher) {
        long now = SystemClock.elapsedRealtime();
        if (lastFixAt != 0 && now - lastFixAt > GAP_MS) {
            w("fix gap", ((now - lastFixAt) / 1000) + "s", "watcher=" + watcher);
        }
        if (lastFixAt == 0) w("first fix", "watcher=" + watcher, fmt(loc));
        lastFixAt = now;
        lastFix = loc;
        fixes++;
        if (now - lastSummaryAt >= SUMMARY_MS) {
            w("fixes", "n=" + fixes, "watcher=" + watcher, fmt(loc));
            fixes = 0;
            lastSummaryAt = now;
        }
    }

    private static String fmt(Location l) {
        return String.format(Locale.US, "acc=%d %.5f,%.5f age=%dms",
                Math.round(l.getAccuracy()), l.getLatitude(), l.getLongitude(),
                System.currentTimeMillis() - l.getTime());
    }

    /** Why Android ended this app's previous process, as Android itself says. */
    private static void lastExit(Context ctx) {
        if (Build.VERSION.SDK_INT < 30) return;
        try {
            ActivityManager am = (ActivityManager) ctx.getSystemService(Context.ACTIVITY_SERVICE);
            List<ApplicationExitInfo> list = am.getHistoricalProcessExitReasons(null, 0, 3);
            SimpleDateFormat f = new SimpleDateFormat("dd/MM HH:mm:ss", Locale.US);
            for (ApplicationExitInfo e : list) {
                w("previous exit", f.format(new Date(e.getTimestamp())),
                        "reason=" + reason(e.getReason()),
                        "importance=" + e.getImportance(),
                        "pid=" + e.getPid(),
                        "desc=" + e.getDescription());
            }
        } catch (Exception e) {
            w("previous exit unavailable", e.toString());
        }
    }

    private static String reason(int r) {
        switch (r) {
            case ApplicationExitInfo.REASON_ANR: return "ANR";
            case ApplicationExitInfo.REASON_CRASH: return "CRASH";
            case ApplicationExitInfo.REASON_CRASH_NATIVE: return "CRASH_NATIVE";
            case ApplicationExitInfo.REASON_DEPENDENCY_DIED: return "DEPENDENCY_DIED";
            case ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE: return "EXCESSIVE_RESOURCE_USAGE";
            case ApplicationExitInfo.REASON_EXIT_SELF: return "EXIT_SELF";
            case ApplicationExitInfo.REASON_INITIALIZATION_FAILURE: return "INIT_FAILURE";
            case ApplicationExitInfo.REASON_LOW_MEMORY: return "LOW_MEMORY";
            case ApplicationExitInfo.REASON_OTHER: return "OTHER";
            case ApplicationExitInfo.REASON_PERMISSION_CHANGE: return "PERMISSION_CHANGE";
            case ApplicationExitInfo.REASON_SIGNALED: return "SIGNALED";
            case ApplicationExitInfo.REASON_USER_REQUESTED: return "USER_REQUESTED";
            case ApplicationExitInfo.REASON_USER_STOPPED: return "USER_STOPPED";
            case ApplicationExitInfo.REASON_FREEZER: return "FREEZER";
            default: return String.valueOf(r);
        }
    }
}
