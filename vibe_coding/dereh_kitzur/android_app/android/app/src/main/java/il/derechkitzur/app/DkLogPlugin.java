package il.derechkitzur.app;

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

    @PluginMethod
    public void clear(PluginCall call) {
        DkLog.clear();
        call.resolve();
    }
}
