package il.derechkitzur.app;

import android.os.Bundle;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebView;

import com.getcapacitor.BridgeActivity;
import com.getcapacitor.WebViewListener;

/** Capacitor's activity, plus the walk log (DkLog): each lifecycle step, and
 *  the WebView loading a page or losing its renderer, goes into the file. */
public class MainActivity extends BridgeActivity {

    @Override
    public void onCreate(Bundle saved) {
        DkLog.start(this);
        StillAlarm.start(this);
        registerPlugin(DkLogPlugin.class);
        super.onCreate(saved);
        DkLog.w("activity create", saved != null ? "restored-from-state" : "fresh");
        bridge.addWebViewListener(new WebViewListener() {
            @Override public void onPageStarted(WebView v) { DkLog.w("webview page started"); }
            @Override public void onPageLoaded(WebView v) { DkLog.w("webview page loaded"); }
            @Override public boolean onRenderProcessGone(WebView v, RenderProcessGoneDetail d) {
                DkLog.w("webview RENDERER GONE", "crash=" + d.didCrash(),
                        "priority=" + d.rendererPriorityAtExit());
                return false;
            }
        });
    }

    @Override public void onStart() { super.onStart(); DkLog.w("activity start"); }
    @Override public void onResume() { super.onResume(); DkLog.w("activity resume"); }
    @Override public void onPause() { DkLog.w("activity pause"); super.onPause(); }
    @Override public void onStop() { DkLog.w("activity stop"); super.onStop(); }

    @Override
    public void onDestroy() {
        DkLog.w("activity DESTROY", "finishing=" + isFinishing(),
                "config=" + isChangingConfigurations());
        super.onDestroy();
    }

    @Override
    public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        DkLog.w("trim memory", "level=" + level);
    }

    @Override
    public void onLowMemory() {
        super.onLowMemory();
        DkLog.w("LOW MEMORY");
    }
}
