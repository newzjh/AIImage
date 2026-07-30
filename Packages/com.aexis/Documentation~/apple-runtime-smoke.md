# macOS and iOS runtime smoke

Use this guide on a Mac that has the AIImage repository, Unity 6000.2.7f2 with the required macOS/iOS PlaybackEngine, the licensed model files, and an iPhone or iPad for the iOS run. The test never changes `Assets/StreamingAssets`. It copies `ref/02.png` and `ref/03.jpg` only into the generated Player output as `aiimage-smoke/face.png` and `aiimage-smoke/scene.jpg`.

The default runner set is CodeFormer, Real-ESRGAN, GFPGAN, YOLOv8 person segmentation, DeepFillV2, Matting, CLIP MobileCLIP S0, and Qwen3.5 mobile Q4. Each runner uses its strict texture configuration. The report records the active graphics API, hardware identity, input paths, elapsed time, output dimensions, person count, mask coverage, and Qwen text result.

Do not use `-nographics`. The Player needs a real Metal graphics device.

## Prerequisites

Set the repository path and Unity executable. Adjust the Unity version path only when the installed editor differs.

```bash
export PROJECT_PATH="$HOME/Projects/AIImage"
export UNITY_EXECUTABLE="/Applications/Unity/Hub/Editor/6000.2.7f2/Unity.app/Contents/MacOS/Unity"
cd "$PROJECT_PATH"
test -x "$UNITY_EXECUTABLE"
test -f ref/02.png
test -f ref/03.jpg
```

The commands below use a 600-second watchdog for each external process.

```bash
run_with_10m_timeout() {
  "$@" &
  local child_pid=$!
  ( sleep 600; kill -TERM "$child_pid" 2>/dev/null || true ) &
  local watchdog_pid=$!
  wait "$child_pid"
  local exit_code=$?
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  return "$exit_code"
}
```

If a Qwen run legitimately exceeds ten minutes on a target device, preserve the partial report and log, then rerun only after changing the local watchdog value and recording that change in the returned log.

## macOS Metal Player

Build a dedicated development Player. The build report is written even when the build fails.

```bash
export AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_OUTPUT="$PROJECT_PATH/Builds/AexisRuntimeSmoke/macos/AexisRuntimeSmoke.app"
export AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_REPORT="$PROJECT_PATH/Reports/AIImage_MacOS_RuntimeSmoke_Build.json"
mkdir -p "$PROJECT_PATH/Reports" "$PROJECT_PATH/Logs"

run_with_10m_timeout "$UNITY_EXECUTABLE" \
  -batchmode -quit \
  -projectPath "$PROJECT_PATH" \
  -executeMethod AIImageAppleRuntimeSmokeTest.BuildMacOSMetalRuntimeSmokeBatch \
  -logFile "$PROJECT_PATH/Logs/AIImage_MacOS_RuntimeSmoke_Build.log"
```

Run the built Player. The command-line flag starts the report only for this launch, and the quit flag closes the Player after the complete report has been written.

```bash
PLAYER_EXECUTABLE="$(find "$AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_OUTPUT/Contents/MacOS" -type f -perm -111 | head -n 1)"
export MACOS_RUNNER_REPORT="$PROJECT_PATH/Reports/AIImage_MacOS_RuntimeSmoke_Runners.json"
test -n "$PLAYER_EXECUTABLE"

run_with_10m_timeout "$PLAYER_EXECUTABLE" \
  -aiimage_runner_smoke \
  -aiimage_runner_qwen_device_qualification \
  -aiimage_runner_smoke_report "$MACOS_RUNNER_REPORT" \
  -aiimage_runner_smoke_quit_when_done \
  -logFile "$PROJECT_PATH/Logs/AIImage_MacOS_RuntimeSmoke_Player.log"

test -s "$MACOS_RUNNER_REPORT"
cat "$MACOS_RUNNER_REPORT"
```

Send these files after the run:

- `Reports/AIImage_MacOS_RuntimeSmoke_Build.json`
- `Reports/AIImage_MacOS_RuntimeSmoke_Runners.json`
- `Logs/AIImage_MacOS_RuntimeSmoke_Player.log`

## iOS Metal device

Build the Xcode project. This temporary test build compiles the `AEXIS_IOS_RUNTIME_SMOKE_AUTORUN` flag, so the report starts once when the app launches on a physical Metal device. The project define is restored after the Unity build completes.

```bash
export AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_OUTPUT="$PROJECT_PATH/Builds/AexisRuntimeSmoke/ios/AexisRuntimeSmokeXcode"
export AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_REPORT="$PROJECT_PATH/Reports/AIImage_iOS_RuntimeSmoke_Build.json"
mkdir -p "$PROJECT_PATH/Reports" "$PROJECT_PATH/Logs"

run_with_10m_timeout "$UNITY_EXECUTABLE" \
  -batchmode -quit \
  -projectPath "$PROJECT_PATH" \
  -executeMethod AIImageAppleRuntimeSmokeTest.BuildIosMetalRuntimeSmokeBatch \
  -logFile "$PROJECT_PATH/Logs/AIImage_iOS_RuntimeSmoke_Build.log"
```

Build, install, and stream the physical-device console with Xcode 15 or newer. Set `IOS_UDID` to the connected device identifier and `IOS_BUNDLE_ID` to the application identifier configured for this build. The Player remains open after the report; stop it normally after the `runner-report=` marker appears.

```bash
export IOS_UDID="replace-with-device-udid"
export IOS_BUNDLE_ID="replace-with-ios-bundle-id"
export IOS_DERIVED_DATA="$PROJECT_PATH/Builds/AexisRuntimeSmoke/ios/DerivedData"

run_with_10m_timeout xcodebuild \
  -project "$AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_OUTPUT/Unity-iPhone.xcodeproj" \
  -scheme Unity-iPhone \
  -configuration Debug \
  -destination "id=$IOS_UDID" \
  -derivedDataPath "$IOS_DERIVED_DATA" \
  build

IOS_APP="$(find "$IOS_DERIVED_DATA/Build/Products/Debug-iphoneos" -maxdepth 1 -name '*.app' -type d | head -n 1)"
test -n "$IOS_APP"
run_with_10m_timeout xcrun devicectl device install app --device "$IOS_UDID" "$IOS_APP"
run_with_10m_timeout xcrun devicectl device process launch \
  --device "$IOS_UDID" \
  --console \
  --terminate-existing \
  "$IOS_BUNDLE_ID" \
  | tee "$PROJECT_PATH/Reports/AIImage_iOS_RuntimeSmoke_DeviceConsole.log"
```

The console contains one compact JSON line prefixed with `[AEXIS_RUNTIME_SMOKE] runner-report=` after every runner has finished. The same pretty-printed JSON is written to `aiimage_apple_runner_smoke.json` under the app's `Application.persistentDataPath`; its path is included in the console report.

Send these files after the run:

- `Reports/AIImage_iOS_RuntimeSmoke_Build.json`
- `Reports/AIImage_iOS_RuntimeSmoke_DeviceConsole.log`
- The `runner-report=` JSON line, if the console process was stopped after it appeared

## Reporting rules

A valid report has `"status": "passed"` and `"valid": true`. Do not replace a failure with a manual timing. Return the JSON and log unchanged when a runner fails, the graphics API is not Metal, an input is missing, or the watchdog expires. The documentation will record the device model, Unity version, graphics API, runner configuration, and timing exactly as reported.
