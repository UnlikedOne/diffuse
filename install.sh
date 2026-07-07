#!/usr/bin/env bash
set -euo pipefail

REPO="UnlikedOne/diffuse"
INSTALL_DIR="${HOME}/.diffuse"
WORKER_DIR="${INSTALL_DIR}/worker"
BIN_DIR="${HOME}/.local/bin"
BINARY_NAME="diffuse-linux-x86_64"

echo "Diffuse installer"
echo

# 1. Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
  echo "This installer currently supports Linux x86_64 only."
  echo "On other platforms, build from source: cargo build --release"
  exit 1
fi

# 2. Check prerequisites
for cmd in curl python3 git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    echo "Please install it and re-run."
    exit 1
  fi
done

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

# 3. Download the latest release binary
echo "Downloading the latest diffuse binary..."
LATEST_URL="https://github.com/${REPO}/releases/latest/download/${BINARY_NAME}"
curl -fsSL "$LATEST_URL" -o "${BIN_DIR}/diffuse"
chmod +x "${BIN_DIR}/diffuse"
echo "Binary installed to ${BIN_DIR}/diffuse"

# 4. Fetch the worker source
echo "Fetching the Python worker..."
TMP="$(mktemp -d)"
git clone --depth 1 "https://github.com/${REPO}.git" "$TMP/diffuse" >/dev/null 2>&1
rm -rf "$WORKER_DIR"
cp -r "$TMP/diffuse/worker" "$WORKER_DIR"
rm -rf "$TMP"

# 5. Set up the worker virtualenv
echo "Setting up the worker environment (this downloads PyTorch, may take a few minutes)..."
python3 -m venv "${WORKER_DIR}/.venv"
# shellcheck disable=SC1091
source "${WORKER_DIR}/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
pip install --quiet -e "$WORKER_DIR"

# 6. Generate protobuf stubs
( cd "$WORKER_DIR" && bash scripts/gen_proto.sh )
deactivate

echo
echo "Diffuse is installed."
echo

# 7. PATH hint
case ":${PATH}:" in
  *":${BIN_DIR}:"*) : ;;
  *)
    echo "NOTE: ${BIN_DIR} is not in your PATH."
    echo "Add this line to your ~/.bashrc or ~/.zshrc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo
    ;;
esac

echo "Try it:"
echo "    diffuse chat"
echo
