# Jupyter Book 완전 가이드

## 1. Jupyter Book 이란?

Jupyter Notebook (.ipynb) 과 Markdown (.md) 파일을 이용해
웹 기반 문서/책을 자동으로 생성해주는 도구입니다.

- 코드 실행 결과, 수식(LaTeX), 인터랙티브 위젯 포함 가능
- GitHub Pages 로 무료 배포 가능
- 딥러닝 튜토리얼, 기술 문서에 적합

---

## 2. 설치

```bash
# Jupyter Book 1.x 설치 (Node.js 불필요)
pip install "jupyter-book<2"
pip install ghp-import

# 설치 확인
jupyter-book --version
```

> **주의:** `jupyter-book>=2` 는 Node.js 가 필요합니다.
> Node.js 없이 사용하려면 반드시 `jupyter-book<2` 로 설치하세요.

---

## 3. 프로젝트 구조 생성

```bash
# docs 폴더에 jupyter-book 초기화
jupyter-book create docs

# 생성된 기본 구조
docs/
├── _config.yml      # 책 전체 설정
├── _toc.yml         # 목차 구조
├── intro.md         # 랜딩 페이지
└── logo.png
```

---

## 4. 최종 저장소 구조

```
myrepo/
├── .github/
│   └── workflows/
│       └── deploy-book.yml       # GitHub Actions 자동 배포
├── src/
│   └── myrepo/                   # 핵심 소스코드
│       ├── __init__.py
│       ├── models/
│       ├── datasets/
│       └── utils/
├── experiments/                  # 실행 스크립트
│   ├── train.py
│   ├── evaluate.py
│   └── configs/
├── notebooks/                    # 원본 실험용 노트북
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── docs/                         # Jupyter Book 루트
│   ├── _config.yml
│   ├── _toc.yml
│   ├── intro.md
│   ├── ch01_overview/
│   │   ├── intro.md
│   │   ├── sec01_background.md
│   │   └── sec02_installation.md
│   ├── ch02_data/
│   │   ├── intro.md
│   │   ├── sec01_dataset.md
│   │   └── sec02_preprocessing.ipynb
│   ├── ch03_models/
│   │   ├── intro.md
│   │   ├── sec01_architecture.md
│   │   └── sec02_training.ipynb
│   ├── ch04_experiments/
│   │   ├── intro.md
│   │   ├── sec01_evaluation.ipynb
│   │   └── sec02_visualization.ipynb
│   └── _build/                   # 빌드 결과물 (gitignore)
│       ├── html/                 # 배포 대상
│       ├── jupyter_execute/      # 노트북 실행 결과
│       └── .doctrees/
├── .env                          # 환경변수 (gitignore)
├── .env.example                  # 환경변수 템플릿
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. `_config.yml` 설정

```yaml
title: myrepo
author: Your Name
logo: logo.png

execute:
  execute_notebooks: "cache"      # 실행 결과 캐시 저장
  cache: "../.jupyter_cache"
  timeout: 120

parse:
  myst_enable_extensions:
    - dollarmath                  # $...$ LaTeX 수식 지원
    - colon_fence
    - html_image

html:
  use_repository_button: true
  use_edit_page_button: true
  use_issues_button: true

repository:
  url: https://github.com/yourname/myrepo
  path_to_book: docs
  branch: main
```

---

## 6. `_toc.yml` 목차 구조

### Part / Chapter / Section / Subsection 계층

```yaml
format: jb-book
root: intro

parts:
  - caption: "Part 1. Deep Learning with PyTorch"
    chapters:

      - file: ch01_overview/intro
        sections:
          - file: ch01_overview/sec01_background
          - file: ch01_overview/sec02_installation

      - file: ch02_data/intro
        sections:
          - file: ch02_data/sec01_dataset
          - file: ch02_data/sec02_preprocessing

      - file: ch03_models/intro
        sections:
          - file: ch03_models/sec01_architecture
          - file: ch03_models/sec02_training

      - file: ch04_experiments/intro
        sections:
          - file: ch04_experiments/sec01_evaluation
          - file: ch04_experiments/sec02_visualization
```

### 계층 구조 규칙

| 레벨 | `_toc.yml` 키 | 파일 필요 | 사이드바 클릭 |
|---|---|---|---|
| Part | `caption` | ❌ | ❌ 레이블만 |
| Chapter | `chapters - file` | ✅ | ✅ |
| Section | `sections - file` | ✅ | ✅ |
| Subsection | `subsections - file` | ✅ | ✅ |

> `_toc.yml` 에서 확장자 (`.md`, `.ipynb`) 는 생략합니다.

---

## 7. 문서 파일 작성 규칙

### 헤드라인 구조

각 파일은 `#` 하나로 시작하고 순서대로 작성합니다.

```markdown
# 페이지 제목          ← 사이드바에 표시되는 이름 (파일당 1개)

## 소제목              ← 우측 내부 목차에 표시

### 소소제목           ← 우측 내부 목차에 표시 (들여쓰기)

#### 그 이하           ← 본문 강조용 (내부 목차 미표시)
```

### 사이드바 vs 내부 목차 역할

```
좌측 사이드바                우측 내부 목차 (On This Page)
─────────────────────        ──────────────────────────────
_toc.yml 이 결정             파일 내 ## / ### 이 결정
```

### 헤드라인 작성 주의사항

```markdown
✅ 올바른 예
# Model Architecture
## Overview
### ResNet

❌ 잘못된 예 — # 이 두 개
# Model Architecture
# Overview

❌ 잘못된 예 — ## 를 건너뜀
# Model Architecture
### ResNet
```

---

## 8. 소스코드 참조 방법

### `literalinclude` — 코드 설명/튜토리얼용

```markdown
<!-- 특정 클래스 삽입 -->
```{literalinclude} ../../src/myrepo/models/resnet.py
:language: python
:pyobject: ResNet
```

<!-- 특정 함수 삽입 -->
```{literalinclude} ../../src/myrepo/utils/transforms.py
:language: python
:pyobject: get_transforms
```

<!-- 특정 라인 강조 -->
```{literalinclude} ../../src/myrepo/models/resnet.py
:language: python
:pyobject: ResNet
:emphasize-lines: 3,5
```
```

### `autodoc` — API 레퍼런스용

`_config.yml` 에 확장 추가:

```yaml
sphinx:
  extra_extensions:
    - sphinx.ext.autodoc
    - sphinx.ext.napoleon
  config:
    autodoc_mock_imports:
      - torch
      - torchvision
```

md 파일에서 사용:

```markdown
```{eval-rst}
.. autoclass:: myrepo.models.resnet.ResNet
   :members:
   :undoc-members:
```

```{eval-rst}
.. autofunction:: myrepo.utils.transforms.get_transforms
```
```

---

## 9. 노트북 셀 태그 설정 (remove-cell)

### VSCode 에서 태그 추가

```
셀 클릭
→ 셀 우측 상단 [...] 버튼 클릭
→ "Add Cell Tag" 클릭
→ "remove-cell" 입력 후 Enter
```

### 태그 종류

| 태그 | 효과 |
|---|---|
| `remove-cell` | 코드 + 출력 모두 제거 |
| `hide-cell` | 토글로 숨김 |
| `remove-input` | 코드만 제거, 출력 표시 |
| `hide-input` | 코드만 토글 숨김 |
| `remove-output` | 출력만 제거 |

### 일괄 태깅 스크립트

```python
# tools/tag_first_cell.py
import json
import glob

targets = ["notebooks/*.ipynb", "docs/**/*.ipynb"]

for pattern in targets:
    for path in glob.glob(pattern, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        if not nb["cells"]:
            continue
        first_cell = nb["cells"][0]
        if "metadata" not in first_cell:
            first_cell["metadata"] = {}
        if "tags" not in first_cell["metadata"]:
            first_cell["metadata"]["tags"] = []
        if "remove-cell" not in first_cell["metadata"]["tags"]:
            first_cell["metadata"]["tags"].append("remove-cell")
            print(f"Tagged : {path}")
        else:
            print(f"Skip   : {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
```

```bash
python tools/tag_first_cell.py
```

---

## 10. 빌드 및 미리보기

```bash
# HTML 빌드
jupyter-book build docs/

# 전체 재빌드 (캐시 무시)
jupyter-book build --all docs/

# 브라우저로 확인
open docs/_build/html/index.html        # Mac
start docs/_build/html/index.html       # Windows
```

### 빌드 결과 구조

```
docs/_build/
├── html/                  # 배포 대상
│   ├── index.html
│   ├── _static/
│   └── .nojekyll
├── jupyter_execute/       # 노트북 실행 결과
└── .doctrees/             # Sphinx 내부 캐시
```

---

## 11. GitHub Pages 배포

### 수동 배포

```bash
ghp-import -n -p -f docs/_build/html
```

GitHub 저장소 Settings → Pages → Source → `gh-pages` 브랜치 선택

### GitHub Actions 자동 배포

`.github/workflows/deploy-book.yml`:

```yaml
name: Deploy Jupyter Book

on:
  push:
    branches: [main]

jobs:
  deploy-book:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install "jupyter-book<2" ghp-import

      - name: Build Jupyter Book
        run: jupyter-book build docs/

      - name: Deploy to GitHub Pages
        run: ghp-import -n -p -f docs/_build/html
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 12. `.gitignore`

```gitignore
# Jupyter Book 빌드 결과물
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
.env
```

---

## 13. 전체 작업 순서 요약

```
Step  1. pip install "jupyter-book<2" ghp-import
Step  2. git init
Step  3. 폴더 구조 생성
Step  4. jupyter-book create docs/
Step  5. 불필요한 기본 파일 제거
Step  6. _config.yml 작성
Step  7. _toc.yml 작성
Step  8. 챕터/섹션 md, ipynb 파일 작성
Step  9. 노트북 첫 셀 remove-cell 태그 추가
Step 10. jupyter-book build docs/ 로 로컬 확인
Step 11. .github/workflows/deploy-book.yml 작성
Step 12. git push → GitHub Pages 자동 배포
```
