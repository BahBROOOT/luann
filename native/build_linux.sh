#!/usr/bin/env sh
set -eu

LUA_ABI="${LUA_ABI:-lua54}"
OUT_DIR="${OUT_DIR:-bin/linux-x64}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TARGET_DIR="$SCRIPT_DIR/$OUT_DIR/$LUA_ABI"
mkdir -p "$TARGET_DIR"

cc -O2 -fPIC -shared \
  "$SCRIPT_DIR/opencl_bridge.c" \
  -o "$TARGET_DIR/luann_opencl.so" \
  -ldl

printf 'Built %s\n' "$TARGET_DIR/luann_opencl.so"
