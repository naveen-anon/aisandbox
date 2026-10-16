plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.naveen.aisandbox"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.naveen.aisandbox"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // No INTERNET permission is declared anywhere in this app on purpose.
        // The training sandbox process physically cannot reach the network
        // even if a dependency tries, because isolatedProcess services are
        // denied network sockets by the Android kernel/SELinux policy
        // regardless of manifest permissions.
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    // Keep the on-device training model(s) uncompressed so TFLite can mmap them.
    androidResources {
        noCompress += listOf("tflite")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.work:work-runtime-ktx:2.9.0")

    // On-device training + inference runtime only. No tensorflow-lite-gpu-delegate
    // networked model-download helpers are used — model ships in assets/.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}
