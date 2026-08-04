#!/usr/bin/env bash
set -euo pipefail

REPO="UnlikedOne/diffuse"
INSTALL_DIR="${HOME}/.diffuse"
WORKER_DIR="${INSTALL_DIR}/worker"
BIN_DIR="${HOME}/.local/bin"

echo "Diffuse installer"
echo

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}/${ARCH}" in
  Linux/x86_64)          BINARY_NAME="diffuse-linux-x86_64" ;;
  Linux/aarch64|Linux/arm64) BINARY_NAME="diffuse-linux-aarch64" ;;
  Darwin/x86_64)         BINARY_NAME="diffuse-macos-x86_64" ;;
  Darwin/arm64)          BINARY_NAME="diffuse-macos-aarch64" ;;
  *)
    echo "No prebuilt binary for ${OS} ${ARCH}."
    echo "Build from source instead:"
    echo "    git clone https://github.com/${REPO}.git && cd diffuse && cargo build --release"
    exit 1
    ;;
esac

echo "Platform: ${OS} ${ARCH} -> ${BINARY_NAME}"

for cmd in curl python3 git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
done

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

echo "Downloading the latest diffuse binary..."
BASE="https://github.com/${REPO}/releases/latest/download"
curl -fsSL "${BASE}/${BINARY_NAME}" -o "${BIN_DIR}/diffuse"
chmod +x "${BIN_DIR}/diffuse"

if curl -fsSL "${BASE}/${BINARY_NAME}.sha256" -o "${INSTALL_DIR}/diffuse.sha256" 2>/dev/null; then
  EXPECTED="$(awk '{print $1}' "${INSTALL_DIR}/diffuse.sha256")"
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL="$(sha256sum "${BIN_DIR}/diffuse" | awk '{print $1}')"
  else
    ACTUAL="$(shasum -a 256 "${BIN_DIR}/diffuse" | awk '{print $1}')"
  fi
  if [ "$EXPECTED" != "$ACTUAL" ]; then
    echo "Checksum mismatch: the download does not match the published hash."
    rm -f "${BIN_DIR}/diffuse"
    exit 1
  fi
  echo "Checksum verified."
  rm -f "${INSTALL_DIR}/diffuse.sha256"
fi

if [ "$OS" = "Darwin" ]; then
  # Gatekeeper quarantines anything downloaded without a notarised signature.
  xattr -d com.apple.quarantine "${BIN_DIR}/diffuse" 2>/dev/null || true
fi

echo "Binary installed to ${BIN_DIR}/diffuse"

echo "Fetching the source..."
TMP="$(mktemp -d)"
git clone --depth 1 "https://github.com/${REPO}.git" "$TMP/diffuse" >/dev/null 2>&1

echo "Installing the worker to ${WORKER_DIR}..."
rm -rf "$WORKER_DIR"
cp -r "$TMP/diffuse/worker" "$WORKER_DIR"
# proto/ is needed by gen_proto.sh, which looks two levels up from worker/scripts
cp -r "$TMP/diffuse/proto" "${INSTALL_DIR}/proto"
rm -rf "$TMP"

echo "Setting up the worker environment (downloads PyTorch, may take a few minutes)..."
python3 -m venv "${WORKER_DIR}/.venv"
# shellcheck disable=SC1091
source "${WORKER_DIR}/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet "setuptools<82"
if [ "$OS" = "Darwin" ]; then
  pip install --quiet torch
else
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install --quiet transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
echo "Installing the media and diffusion extras..."
pip install --quiet -e "${WORKER_DIR}[multimodal,diffusion]"

echo "Generating protobuf stubs..."
( cd "$WORKER_DIR" && bash scripts/gen_proto.sh )
deactivate

# proto/ is only needed at generation time
rm -rf "${INSTALL_DIR}/proto"

echo
echo "Diffuse is installed."
echo

case ":${PATH}:" in
  *":${BIN_DIR}:"*) : ;;
  *)
    echo "NOTE: ${BIN_DIR} is not in your PATH."
    echo "Add this line to your ~/.bashrc or ~/.zshrc, then restart your shell:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo
    ;;
esac

echo "Try it:"
echo "    diffuse chat"
echo
