#!/usr/bin/env bash
# Throwaway probe — answers three questions:
#   1. Does `pip install torchcodec` succeed on aarch64 / torch 2.9.1 cu130? What version?
#   2. Does `import torchcodec` succeed?
#   3. Does `from TTS.api import TTS` succeed once torchcodec stub is present?
# Also shows the exact TTS/__init__.py:46 codec-check source so the sed patch can be authored.
set -euxo pipefail
LOG=/home/msr8109/projects/voice_tts/logs/probe_torchcodec.log
exec > >(tee -a "$LOG") 2>&1
echo "=== probe start $(date -Iseconds) ==="

docker run --rm nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 bash -c '
set -e
echo "=== arch / kernel ==="
uname -a
echo "=== apt install ==="
apt-get update -qq
apt-get install -qq -y python3.12 python3.12-venv python3.12-dev build-essential >/dev/null
python3.12 -m venv /opt/v
/opt/v/bin/pip install -q --upgrade pip

echo "=== install torch trio cu130 ==="
/opt/v/bin/pip install -q torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu130

echo "=== install coqui-tts + transformers<5 ==="
/opt/v/bin/pip install -q "transformers>=4.40,<5"
/opt/v/bin/pip install -q coqui-tts==0.27.5

echo "=== Q0: baseline TTS import BEFORE torchcodec (expected: TORCHCODEC ImportError) ==="
/opt/v/bin/python -c "from TTS.api import TTS" 2>&1 | tail -3 || true

echo "=== show TTS/__init__.py lines 1-60 (so we can author a sed patch) ==="
/opt/v/bin/python -c "import TTS, pathlib; print(pathlib.Path(TTS.__file__).read_text().splitlines()[:60])" 2>&1 | sed "s/, /\n  /g" | head -80 || true
echo "--- raw head of TTS/__init__.py ---"
head -60 /opt/v/lib/python3.12/site-packages/TTS/__init__.py

echo "=== Q1: pip install torchcodec ==="
/opt/v/bin/pip install torchcodec 2>&1 | tail -15 || true

echo "=== Q1b: which torchcodec version landed? ==="
/opt/v/bin/pip show torchcodec 2>&1 | head -10 || true

echo "=== Q2: import torchcodec ==="
/opt/v/bin/python -c "import torchcodec; print(\"torchcodec.__version__\", torchcodec.__version__)" 2>&1 | tail -5 || true
/opt/v/bin/python -c "import torchcodec; print(\"torchcodec.__file__\", torchcodec.__file__)" 2>&1 | tail -5 || true

echo "=== Q3: from TTS.api import TTS (with torchcodec stub present) ==="
/opt/v/bin/python -c "from TTS.api import TTS; print(\"TTS api OK\")" 2>&1 | tail -5 || true

echo "=== also try coqui-tts[codec] extra resolution (just resolve, dont reinstall) ==="
/opt/v/bin/pip install --dry-run coqui-tts[codec]==0.27.5 2>&1 | tail -20 || true

echo "=== probe done ==="
'
echo "=== probe complete exit=$? $(date -Iseconds) ==="
