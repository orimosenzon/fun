import java.util.Properties

plugins {
    id("com.android.application")
}

// The app is the website itself, opened full screen in Chrome (a Trusted Web Activity). It holds
// no code of its own: everything new on the network reaches the app the moment it is deployed.
// A new APK is needed only when something here changes, and then versionCode goes up by one.
val host = "free-network.orimosenzon.workers.dev"

// The signing key lives outside the public repository. Losing it means members cannot update the
// app in place, and changing it means /.well-known/assetlinks.json must list the new fingerprint.
val signing = Properties().apply {
    val file = File(System.getProperty("user.home"), ".config/free-network/android-signing.properties")
    if (file.exists()) file.inputStream().use { load(it) }
}

android {
    namespace = "il.pardes.freenetwork"
    compileSdk = 36

    defaultConfig {
        applicationId = "il.pardes.freenetwork"
        minSdk = 23
        targetSdk = 36
        versionCode = 2
        versionName = "1.1"
        manifestPlaceholders["host"] = host
    }

    signingConfigs {
        create("release") {
            if (signing.isNotEmpty()) {
                storeFile = File(signing.getProperty("storeFile"))
                storePassword = signing.getProperty("storePassword")
                keyAlias = signing.getProperty("keyAlias")
                keyPassword = signing.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("com.google.androidbrowserhelper:androidbrowserhelper:2.6.2")
}
