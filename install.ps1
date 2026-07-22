$ErrorActionPreference = "Stop"

Write-Host "Diffuse installer (Windows, build from source)"
Write-Host ""

$repo = "https://github.com/UnlikedOne/diffuse.git"
$installDir = Join-Path $HOME ".diffuse"
$srcDir = Join-Path $installDir "src"
$workerDir = Join-Path $installDir "worker"
$binDir = Join-Path $HOME ".local\bin"

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "Missing required command: $name"
        Write-Host $hint
        exit 1
    }
}

Require-Command "git" "Install Git: https://git-scm.com/download/win"
Require-Command "cargo" "Install Rust: https://rustup.rs"
Require-Command "python" "Install Python 3: https://www.python.org/downloads/"

New-Item -ItemType Directory -Force -Path $installDir, $binDir | Out-Null

Write-Host "Fetching the source..."
if (Test-Path $srcDir) { Remove-Item -Recurse -Force $srcDir }
git clone --depth 1 $repo $srcDir | Out-Null

Write-Host "Building the diffuse binary (this can take a few minutes)..."
Push-Location $srcDir
cargo build --release
Pop-Location

Copy-Item (Join-Path $srcDir "target\release\diffuse.exe") (Join-Path $binDir "diffuse.exe") -Force
Write-Host "Binary installed to $binDir\diffuse.exe"

Write-Host "Installing the worker to $workerDir..."
if (Test-Path $workerDir) { Remove-Item -Recurse -Force $workerDir }
Copy-Item -Recurse (Join-Path $srcDir "worker") $workerDir

Write-Host "Setting up the worker environment (downloads PyTorch, may take a few minutes)..."
python -m venv (Join-Path $workerDir ".venv")
$py = Join-Path $workerDir ".venv\Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet torch transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
& $py -m pip install --quiet -e $workerDir

Write-Host "Generating protobuf stubs..."
Push-Location $workerDir
bash scripts/gen_proto.sh
Pop-Location

[Environment]::SetEnvironmentVariable("DIFFUSE_WORKER_DIR", $workerDir, "User")

Write-Host ""
Write-Host "Diffuse is installed."
Write-Host "Add $binDir to your PATH if it is not already, then run: diffuse chat"
