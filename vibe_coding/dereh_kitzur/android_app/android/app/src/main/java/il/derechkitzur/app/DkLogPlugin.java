package il.derechkitzur.app;

import android.content.Intent;
import android.net.Uri;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileWriter;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

/** The page's way into DkLog: add a line, read the file, clear it. */
@CapacitorPlugin(name = "DkLog")
public class DkLogPlugin extends Plugin {

    @PluginMethod
    public void log(PluginCall call) {
        DkLog.page(call.getString("line", ""));
        call.resolve();
    }

    @PluginMethod
    public void read(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("text", DkLog.read());
        call.resolve(ret);
    }

    /** Android's share sheet with the log as an attached text file. The
     *  WebView has no navigator.share, and the log is too long to go as the
     *  text of an intent (Android caps what one intent may carry). */
    @PluginMethod
    public void share(PluginCall call) {
        try {
            File out = new File(getContext().getCacheDir(), "dk-walk-log.txt");
            try (FileWriter fw = new FileWriter(out, false)) { fw.write(DkLog.read()); }
            Uri uri = FileProvider.getUriForFile(getContext(),
                    getContext().getPackageName() + ".fileprovider", out);
            Intent send = new Intent(Intent.ACTION_SEND);
            send.setType("text/plain");
            send.putExtra(Intent.EXTRA_STREAM, uri);
            send.putExtra(Intent.EXTRA_SUBJECT, "לוג הקלטה - דרך קיצור");
            send.putExtra(Intent.EXTRA_TEXT, "הלוג המלא בקובץ המצורף.");
            send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            Intent chooser = Intent.createChooser(send, "שיתוף הלוג");
            chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(chooser);
            DkLog.w("log shared");
            call.resolve();
        } catch (Exception e) {
            DkLog.w("log share failed", e.toString());
            call.reject(e.toString());
        }
    }

    @PluginMethod
    public void clear(PluginCall call) {
        DkLog.clear();
        call.resolve();
    }
}
