$ErrorActionPreference = "Stop"

Write-Host "Diffuse installer (Windows)"
Write-Host ""

$repo = "UnlikedOne/diffuse"
$installDir = Join-Path $HOME ".diffuse"
$srcDir = Join-Path $installDir "src"
$workerDir = Join-Path $installDir "worker"
$binDir = Join-Path $HOME ".local\bin"
$asset = "diffuse-windows-x86_64.exe"

if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne "X64") {
    Write-Host "No prebuilt binary for this architecture."
    Write-Host "Build from source: git clone https://github.com/$repo.git; cargo build --release"
    exit 1
}

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "Missing required command: $name"
        Write-Host $hint
        exit 1
    }
}

Require-Command "git" "Install Git: https://git-scm.com/download/win"
Require-Command "python" "Install Python 3: https://www.python.org/downloads/"

New-Item -ItemType Directory -Force -Path $installDir, $binDir | Out-Null

Write-Host "Downloading the latest diffuse binary..."
$base = "https://github.com/$repo/releases/latest/download"
$exe = Join-Path $binDir "diffuse.exe"
Invoke-WebRequest -Uri "$base/$asset" -OutFile $exe -UseBasicParsing

try {
    $sumFile = Join-Path $installDir "diffuse.sha256"
    Invoke-WebRequest -Uri "$base/$asset.sha256" -OutFile $sumFile -UseBasicParsing
    $expected = ((Get-Content $sumFile -Raw) -split '\s+')[0]
    $actual = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) {
        Write-Host "Checksum mismatch: the download does not match the published hash."
        Remove-Item $exe -Force
        exit 1
    }
    Write-Host "Checksum verified."
    Remove-Item $sumFile -Force
} catch {
    Write-Host "Could not verify the checksum, continuing."
}

Write-Host "Binary installed to $exe"

Write-Host "Fetching the source..."
if (Test-Path $srcDir) { Remove-Item -Recurse -Force $srcDir }
git clone --depth 1 "https://github.com/$repo.git" $srcDir | Out-Null

Write-Host "Installing the worker to $workerDir..."
if (Test-Path $workerDir) { Remove-Item -Recurse -Force $workerDir }
Copy-Item -Recurse (Join-Path $srcDir "worker") $workerDir

Write-Host "Setting up the worker environment (downloads PyTorch, may take a few minutes)..."
python -m venv (Join-Path $workerDir ".venv")
$py = Join-Path $workerDir ".venv\Scripts\python.exe"
& $py -m pip install --quiet --upgrade pip
& $py -m pip install --quiet "setuptools<82"
& $py -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install --quiet transformers safetensors grpcio grpcio-tools protobuf numpy psutil huggingface_hub
Write-Host "Installing the media and diffusion extras..."
& $py -m pip install --quiet -e "$workerDir[multimodal,diffusion]"

# gen_proto.sh is a shell script and Windows has no shell to run it, so its two
# steps are done here directly: generate the stubs, then point the grpc stub at
# the package rather than at a bare module name.
Write-Host "Generating protobuf stubs..."
$protoDir = Join-Path $srcDir "proto"
$outDir = Join-Path $workerDir "diffuse_worker"
& $py -m grpc_tools.protoc -I $protoDir "--python_out=$outDir" "--grpc_python_out=$outDir" (Join-Path $protoDir "data.proto")
$grpcFile = Join-Path $outDir "data_pb2_grpc.py"
(Get-Content $grpcFile) -replace '^import data_pb2 as data__pb2', 'from diffuse_worker import data_pb2 as data__pb2' | Set-Content $grpcFile

Remove-Item -Recurse -Force $srcDir

[Environment]::SetEnvironmentVariable("DIFFUSE_WORKER_DIR", $workerDir, "User")

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$binDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$binDir", "User")
    Write-Host "Added $binDir to your PATH (restart your terminal to pick it up)."
}

Write-Host ""
Write-Host "Diffuse is installed."
Write-Host "Open a new terminal and run: diffuse chat"
