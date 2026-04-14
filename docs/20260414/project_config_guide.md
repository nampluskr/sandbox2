# 프로젝트 설정 관리 가이드

## 1. 설정 관리 전략 개요

### 경로 성격에 따른 관리 방법

```
프로젝트 내부 경로  →  __file__ / os.getcwd() 로 자동 계산
프로젝트 외부 경로  →  .env 로 관리
모델/학습 설정      →  YAML 파일로 관리
```

### 관리 방법별 용도

| 항목 | 관리 방법 |
|---|---|
| SOURCE_DIR (src/ 경로) | `__file__` / `os.getcwd()` |
| pretrained backbone 경로 | `.env` |
| checkpoint 저장 경로 | `.env` |
| 데이터셋 경로 | `.env` |
| 모델 구조 설정 | YAML |
| 학습 하이퍼파라미터 | YAML |
| API 키, 토큰 | `.env` |

---

## 2. SOURCE_DIR 설정

### 개념 비교

| | `__file__` | `os.getcwd()` |
|---|---|---|
| 의미 | 현재 **파일**의 절대경로 | 현재 **실행 위치** |
| `.py` 사용 | ✅ | ⚠️ 실행 위치에 따라 변동 |
| `.ipynb` 사용 | ❌ 정의되지 않음 | ✅ |

### `os.path.normpath` 역할

```python
os.path.normpath("C:/myrepo/experiments/../src")
# → "C:\myrepo\src"   (Windows)
# → "C:/myrepo/src"   (Mac/Linux)
```

`..` 를 실제 경로로 변환하고 구분자를 정규화합니다.

---

### 위치별 SOURCE_DIR 설정 코드

#### `experiments/*.py`

```python
import sys
import os

# __file__ = "C:\myrepo\experiments\train.py"
SOURCE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",      # experiments/ → myrepo/
        "src"      # myrepo/src
    )
)
# → C:\myrepo\src

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from myrepo.models.resnet import ResNet
```

#### `notebooks/*.ipynb` 첫 번째 셀 (태그: `remove-cell`)

```python
import sys
import os

# os.getcwd() = "C:\myrepo\notebooks"
SOURCE_DIR = os.path.normpath(
    os.path.join(
        os.getcwd(),
        "..",      # notebooks/ → myrepo/
        "src"      # myrepo/src
    )
)
# → C:\myrepo\src

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
```

#### `docs/chXX/*.ipynb` 첫 번째 셀 (태그: `remove-cell`)

```python
import sys
import os

# os.getcwd() = "C:\myrepo\docs\ch03_models"
SOURCE_DIR = os.path.normpath(
    os.path.join(
        os.getcwd(),
        "..",      # ch03_models/ → docs/
        "..",      # docs/        → myrepo/
        "src"      # myrepo/src
    )
)
# → C:\myrepo\src

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
```

### 위치별 `..` 횟수 정리

| 파일 위치 | 기준 | `..` 횟수 | 결과 |
|---|---|---|---|
| `experiments/*.py` | `__file__` | 1번 | `myrepo/src` |
| `notebooks/*.ipynb` | `os.getcwd()` | 1번 | `myrepo/src` |
| `docs/chXX/*.ipynb` | `os.getcwd()` | 2번 | `myrepo/src` |

---

## 3. `.env` — 외부 경로 관리

### 설치

```bash
pip install python-dotenv
```

### `.env` 파일 위치

프로젝트 루트에 저장합니다.

```
myrepo/
├── .env          ← 여기 (gitignore)
├── .env.example  ← 템플릿 (git 추적)
└── ...
```

`load_dotenv()` 는 실행 위치에서 상위로 올라가며 `.env` 를 자동으로 찾습니다.

### `.env.example` 내용

```bash
# ================================================================
# .env.example
# 이 파일을 복사해서 .env 로 저장 후 경로를 설정하세요.
# cp .env.example .env   (Mac/Linux)
# copy .env.example .env (Windows)
# ================================================================

# ----------------------------------------------------------------
# Pretrained Backbone 저장 경로
# 예) resnet50.pth, vit_base.pth 등이 저장된 폴더
# ----------------------------------------------------------------
BACKBONE_DIR=

# ----------------------------------------------------------------
# Checkpoint 저장 경로
# 학습 중 저장되는 모델 가중치 폴더
# ----------------------------------------------------------------
CHECKPOINT_DIR=

# ----------------------------------------------------------------
# Dataset 저장 경로
# 학습/검증/테스트 데이터셋 루트 폴더
# ----------------------------------------------------------------
DATASET_DIR=

# ----------------------------------------------------------------
# Log 저장 경로
# TensorBoard 등 학습 로그 저장 폴더
# ----------------------------------------------------------------
LOG_DIR=
```

### 실제 `.env` 작성 예시

```bash
# Windows
BACKBONE_DIR="C:\Users\xxx\pretrained\backbones"
CHECKPOINT_DIR="C:\Users\xxx\checkpoints"
DATASET_DIR="C:\Users\xxx\datasets"
LOG_DIR="C:\Users\xxx\logs"

# Mac/Linux
BACKBONE_DIR=/home/xxx/pretrained/backbones
CHECKPOINT_DIR=/home/xxx/checkpoints
DATASET_DIR=/home/xxx/datasets
LOG_DIR=/home/xxx/logs
```

> **주의:** 경로에 빈칸이 있으면 반드시 큰따옴표로 감쌉니다.
> `python-dotenv` 는 따옴표를 자동으로 제거합니다.

### `.env` 로드 — 명시적 경로 지정

#### `experiments/*.py`

```python
from dotenv import load_dotenv
import os

ENV_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        ".env"
    )
)
load_dotenv(ENV_PATH)

BACKBONE_DIR   = os.environ["BACKBONE_DIR"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]
DATASET_DIR    = os.environ["DATASET_DIR"]
```

#### `notebooks/*.ipynb`

```python
from dotenv import load_dotenv
import os

ENV_PATH = os.path.normpath(
    os.path.join(os.getcwd(), "..", ".env")
)
load_dotenv(ENV_PATH)

BACKBONE_DIR   = os.environ["BACKBONE_DIR"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]
DATASET_DIR    = os.environ["DATASET_DIR"]
```

---

## 4. YAML — 모델/학습 설정 관리

### 설치

```bash
pip install pyyaml
```

### `experiments/configs/resnet_config.yaml`

```yaml
model:
  backbone: resnet50
  num_classes: 10
  pretrained: true

paths:
  checkpoint_dir: "checkpoints/resnet50"
  log_dir: "logs/resnet50"

train:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001

data:
  image_size: 224
  num_workers: 4
```

### YAML 로드

```python
import yaml

with open("experiments/configs/resnet_config.yaml", "r") as f:
    config = yaml.safe_load(f)

backbone    = config["model"]["backbone"]     # resnet50
num_classes = config["model"]["num_classes"]  # 10
lr          = config["train"]["learning_rate"] # 0.001
```

---

## 5. 전체 통합 패턴

`experiments/train.py` 에서 모든 설정을 통합합니다.

```python
import sys
import os
import yaml
from dotenv import load_dotenv

# ── 1. SOURCE_DIR 설정 ──────────────────────────
SOURCE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "src"
    )
)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

# ── 2. 외부 경로 로드 (.env) ────────────────────
ENV_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", ".env"
    )
)
load_dotenv(ENV_PATH)

BACKBONE_DIR   = os.environ["BACKBONE_DIR"]
CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]
DATASET_DIR    = os.environ["DATASET_DIR"]

# ── 3. 모델/학습 설정 로드 (YAML) ───────────────
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs", "resnet_config.yaml"
)
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

backbone    = config["model"]["backbone"]
num_classes = config["model"]["num_classes"]

# ── 4. 실제 경로 조합 ────────────────────────────
backbone_path   = os.path.join(BACKBONE_DIR, f"{backbone}.pth")
checkpoint_path = os.path.join(CHECKPOINT_DIR, backbone)

print(backbone_path)
# → C:\Users\xxx\pretrained\backbones\resnet50.pth

# ── 5. 모듈 import ───────────────────────────────
from myrepo.models.resnet import ResNet
from myrepo.datasets.custom_dataset import CustomDataset
```

---

## 6. VSCode 설정

### `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "notebook.cellToolbarLocation": {
    "default": "right"
  },
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

### `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Tag First Cell (remove-cell)",
      "type": "shell",
      "command": "python tools/tag_first_cell.py",
      "group": "build"
    },
    {
      "label": "Jupyter Book Build",
      "type": "shell",
      "command": "jupyter-book build docs/",
      "group": "build"
    }
  ]
}
```

### `.vscode/extensions.json`

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-renderers"
  ]
}
```

---

## 7. `.gitignore` 전체

```gitignore
# 환경변수
.env

# Jupyter Book 빌드
docs/_build/

# 노트북 캐시
.jupyter_cache/
.ipynb_checkpoints/

# 모델 가중치
*.pth
*.pt
*.ckpt

# 데이터
data/

# Python
__pycache__/
*.pyc
.venv/
.env
```

---

## 8. `requirements.txt`

```text
jupyter-book<2
ghp-import
torch
torchvision
numpy
matplotlib
pyyaml
python-dotenv
```
