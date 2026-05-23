#!/bin/bash
set -euo pipefail

native_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$native_dir/build-macos"
ncnn_root="$native_dir/ncnn-20260113-macos-vulkan"

cmake -S "$native_dir" -B "$build_dir" -G Xcode -DNCNN_ROOT="$ncnn_root"
cmake --build "$build_dir" --config Release

project_root="$native_dir"
for _ in 1 2 3 4 5 6 7 8; do
  if [ -d "$project_root/Assets" ]; then
    break
  fi
  project_root="$(cd "$project_root/.." && pwd)"
done
if [ ! -d "$project_root/Assets" ]; then
  echo "Unity project root not found from: $native_dir" >&2
  exit 1
fi

plugins_dir="$project_root/Assets/Plugins"
dst_dir="$plugins_dir/macOS"
mkdir -p "$dst_dir"

cp -f "$build_dir/Release/realesrgan_unity.dylib" "$dst_dir/realesrgan_unity.dylib"
cp -Rf "$ncnn_root/ncnn.framework" "$dst_dir/ncnn.framework"
cp -Rf "$ncnn_root/glslang.framework" "$dst_dir/glslang.framework"
cp -Rf "$ncnn_root/openmp.framework" "$dst_dir/openmp.framework"
