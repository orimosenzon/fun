package il.derechkitzur.app;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.location.Location;
import android.media.AudioAttributes;
import android.media.RingtoneManager;
import android.os.Build;
import android.os.SystemClock;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

/**
 * "Did you forget the recording?"
 *
 * A walk recorder left on after the walk keeps recording the walker's sofa,
 * and the trail sent in afterwards ends in their living room. Ori asked
 * (21/11/2026) for the phone to say so: standing in one place for ten minutes
 * with the recorder on, it rings and asks.
 *
 * This lives on the native side on purpose. With the screen off the page may
 * be asleep (the five-minute freeze of 21/11 was the fixes it had missed,
 * arriving together when it woke), so a timer in the page cannot be trusted
 * to go off. The location service keeps sending fixes, and this hears every
 * one of them on the same broadcast the plugin and DkLog listen to.
 *
 * "In one place" means: no fix precise enough to trust (GOOD_ACC_M) has been
 * further than MOVE_M from where the walker was when they last moved. Vague
 * fixes (indoors, a pocket) never count as moving, so a recorder forgotten at
 * home still rings. It asks again AGAIN_MS later, MAX_TELLS times at most,
 * and moving off resets it. A silence in the fixes longer than FRESH_MS is
 * taken as a new recording, so a stopped walk does not carry over.
 *
 * Tapping the notification opens the app, where the walk's bar has the
 * finish button. A debug build uses short times, for trying it at a desk.
 */
final class StillAlarm {
    static long STILL_MS = 10 * 60_000;
    static long AGAIN_MS = 15 * 60_000;
    static final int MAX_TELLS = 3;
    static final float MOVE_M = 40f;
    static final float GOOD_ACC_M = 30f;
    static final long FRESH_MS = 2 * 60_000;

    private static final String GEO_BROADCAST =
            "com.equimaps.capacitor_background_geolocation.broadcast";
    // A channel's sound and vibration are fixed once it is created, so
    // changing them means a new id (v2, v3...).
    private static final String CHANNEL = "dk-still-v1";
    private static final int NOTE_ID = 4711;

    private static boolean started;
    private static Context app;
    private static Location anchor;        // where the walker last was moving
    private static long lastAt;            // elapsedRealtime of the last fix
    private static long nextTellAt;
    private static int tells;

    private StillAlarm() {}

    static void start(Context any) {
        if (started) return;
        started = true;
        app = any.getApplicationContext();
        if ((app.getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0) {
            STILL_MS = 45_000;
            AGAIN_MS = 30_000;
        }
        channel();
        LocalBroadcastManager.getInstance(app).registerReceiver(new BroadcastReceiver() {
            @Override public void onReceive(Context c, Intent i) {
                Location loc = i.getParcelableExtra("location");
                if (loc != null) onFix(loc);
            }
        }, new IntentFilter(GEO_BROADCAST));
        DkLog.w("still alarm on", "after=" + STILL_MS / 1000 + "s", "move=" + (int) MOVE_M + "m");
    }

    private static synchronized void onFix(Location loc) {
        long now = SystemClock.elapsedRealtime();
        boolean fresh = anchor == null || now - lastAt > FRESH_MS;
        lastAt = now;
        if (fresh) { moved(loc, now); return; }
        if (loc.getAccuracy() <= GOOD_ACC_M && loc.distanceTo(anchor) > MOVE_M) {
            if (tells > 0) {
                DkLog.w("still alarm: moving again");
                NotificationManagerCompat.from(app).cancel(NOTE_ID);
            }
            moved(loc, now);
            return;
        }
        if (tells < MAX_TELLS && now >= nextTellAt) {
            tells++;
            nextTellAt = now + AGAIN_MS;
            tell(loc);
        }
    }

    private static void moved(Location loc, long now) {
        anchor = loc;
        nextTellAt = now + STILL_MS;
        tells = 0;
    }

    private static void tell(Location loc) {
        long minutes = Math.max(1, (STILL_MS + (tells - 1) * AGAIN_MS) / 60_000);
        DkLog.w("still alarm RING", "tell=" + tells, "still~" + minutes + "min",
                String.format(java.util.Locale.US, "acc=%d", Math.round(loc.getAccuracy())));
        Intent open = new Intent(app, MainActivity.class)
                .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent pi = PendingIntent.getActivity(app, 0, open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        NotificationCompat.Builder b = new NotificationCompat.Builder(app, CHANNEL)
                .setSmallIcon(R.drawable.ic_walk_note)
                .setContentTitle("ההקלטה עדיין פועלת")
                .setContentText("לא זזת כבר " + minutes + " דקות. סיימת ללכת?")
                .setStyle(new NotificationCompat.BigTextStyle().bigText(
                        "לא זזת כבר " + minutes + " דקות והקלטת השביל עדיין פועלת. "
                        + "אם סיימת, לחץ כאן ואז על סיום. אם אתה רק נח, אפשר להמשיך כרגיל."))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_REMINDER)
                .setDefaults(NotificationCompat.DEFAULT_ALL)
                .setAutoCancel(true)
                .setOnlyAlertOnce(false)
                .setContentIntent(pi);
        try {
            NotificationManagerCompat.from(app).notify(NOTE_ID, b.build());
        } catch (SecurityException e) {
            // Notifications not allowed (Android 13+): the log says so.
            DkLog.w("still alarm: notifications not allowed");
        }
    }

    private static void channel() {
        if (Build.VERSION.SDK_INT < 26) return;
        NotificationManager nm = app.getSystemService(NotificationManager.class);
        if (nm == null || nm.getNotificationChannel(CHANNEL) != null) return;
        NotificationChannel ch = new NotificationChannel(CHANNEL,
                "תזכורת: הקלטה שנשכחה", NotificationManager.IMPORTANCE_HIGH);
        ch.setDescription("צלצול כשההקלטה פועלת ולא זזת כבר עשר דקות");
        ch.enableVibration(true);
        ch.setVibrationPattern(new long[] { 0, 400, 200, 400, 200, 400 });
        ch.setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION),
                new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_NOTIFICATION_EVENT)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build());
        nm.createNotificationChannel(ch);
    }
}
