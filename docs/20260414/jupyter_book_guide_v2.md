# Jupyter Book을 사용한 문서 작성 및 GitHub Pages 배포 가이드  

본 문서는 `repo1`, `repo2`, `repo3`와 같이 **코드 개발을 주목적으로 하는 저장소**에서, `docs/` 폴더 내에 위치한 Jupyter Book 형식의 문서를  
- `https://your-username.github.io/repo1`
- `https://your-username.github.io/repo2`
- `https://your-username.github.io/repo3`
  
형태로 **독립된 페이지로 배포**하는 과정을 안내합니다.  

각 저장소는 `src`, `experiments`, `notebooks`, `docs` 폴더로 구성되어 있으며, 문서는 `docs` 폴더 내에서 관리됩니다.

---

## 목표

- `repo1`, `repo2`, `repo3` 저장소는 코드 및 실험 중심의 개발 저장소입니다.
- 각 저장소의 `docs/` 폴더에 Jupyter Book 형식의 문서를 작성합니다.
- `jupyter-book build` 명령어로 문서를 빌드하고, GitHub Pages를 통해 다음 주소로 배포합니다:
  - `https://your-username.github.io/repo1`
  - `https://your-username.github.io/repo2`
  - `https://your-username.github.io/repo3`
- 문서는 저장소의 일부로 관리되며, 코드 변경과 함께 문서도 함께 유지보수됩니다.

---

## 1. 라이브러리 설치

로컬 환경에 Jupyter Book을 설치합니다. 명령줄에서 다음 명령어를 실행합니다.

```bash
pip install jupyter-book
```

설치 완료 후, 버전 확인을 통해 정상 설치 여부를 검증합니다.

```bash
jupyter-book --version
```

---

## 2. 저장소 및 폴더 구조 생성

각 저장소를 로컬에 생성하고, `docs` 폴더 내에 Jupyter Book을 초기화합니다.

예시: `repo1` 생성

```bash
mkdir ~/Documents/repo1
cd ~/Documents/repo1
```

하위 폴더 생성:

```bash
mkdir src experiments notebooks docs
```

`docs` 폴더로 이동하여 Jupyter Book 초기화:

```bash
cd docs
jupyter-book create ./
```

> `jupyter-book create`가 지원되지 않을 경우, 수동으로 `_toc.yml`, `_config.yml`, `intro.md` 생성 가능

---

## 3. 문서 작성

`docs/intro.md` 파일을 편집기로 열고 내용을 작성합니다.

예시 (`docs/intro.md`):

```markdown
# repo1 문서 시작

이 문서는 repo1 프로젝트의 개발 기록과 사용법을 안내합니다.

## 코드 예제

```python
from src.module import train
train.run()
```
```

목차 파일 (`docs/_toc.yml`) 예시:

```yaml
format: jb-book
root: intro
chapters:
  - file: intro
```

---

## 4. 로컬 미리보기

문서를 빌드하여 로컬에서 확인합니다.

```bash
jupyter-book build ./
```

빌드 결과는 `docs/_build/html/`에 생성됩니다.  
브라우저에서 `docs/_build/html/index.html` 파일을 열어 확인합니다.

---

## 5. GitHub 저장소 생성

각 저장소에 대해 GitHub에 저장소를 생성합니다.

- https://github.com/your-username/repo1
- https://github.com/your-username/repo2
- https://github.com/your-username/repo3

생성 시 다음을 확인합니다:
- Public 설정
- README, .gitignore 등은 수동 추가

---

## 6. 저장소 초기화 및 푸시

로컬 저장소를 초기화하고 원격 저장소에 연결합니다.

예시 (`repo1`):

```bash
cd ~/Documents/repo1
git init
git add .
git commit -m "Initialize repo with src, experiments, notebooks, docs"
git branch -M main
git remote add origin https://github.com/your-username/repo1.git
git push -u origin main
```

`your-username`은 본인의 GitHub 사용자 이름으로 대체합니다.

---

## 7. GitHub Actions 자동 배포 설정

각 저장소에 대해 `docs` 폴더 내 문서를 자동으로 배포하는 워크플로우를 설정합니다.

### `.github/workflows/deploy.yml` 파일 생성

경로: `repo1/.github/workflows/deploy.yml`

내용:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Jupyter Book
        run: |
          pip install jupyter-book

      - name: Build documentation
        run: |
          cd docs
          jupyter-book build ./

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build/html
          publish_branch: gh-pages
```

이 파일을 커밋하고 푸시합니다.

```bash
git add .github/workflows/deploy.yml
git commit -m "Add documentation deployment workflow"
git push
```

---

## 8. GitHub Pages 활성화

각 저장소에서 다음 절차를 수행합니다.

1. 저장소 방문 → **Settings** 탭 선택
2. 왼쪽 메뉴에서 **Pages** 클릭
3. **Source** 섹션에서:
   - Branch: `gh-pages`
   - Folder: `/ (root)`
4. **Save** 클릭

저장 후, 아래 메시지가 표시됩니다:
> "Your site is published at https://your-username.github.io/repo1"

---

## 9. 배포 확인

다음 URL에서 문서 확인이 가능합니다.

- https://your-username.github.io/repo1
- https://your-username.github.io/repo2
- https://your-username.github.io/repo3

각 문서는 해당 저장소의 `docs` 폴더에 있는 Jupyter Book 기반 콘텐츠입니다.

---

## 문제 해결

| 현상 | 원인 및 해결 방법 |
|------|------------------|
| `jupyter-book` 명령어를 인식하지 못함 | Python 설치 후 PATH 등록 확인. 터미널 재시작 |
| GitHub 푸시 실패 | 저장소 이름, 사용자 이름 정확히 확인. 2단계 인증 사용 시 Personal Access Token 필요 |
| 문서 변경 사항 미반영 | GitHub Actions 실행 상태 확인 (Actions 탭). 빌드 실패 여부 점검 |
| 이미지 또는 링크 깨짐 | 상대 경로 사용 확인. `_config.yml`에 `site.baseurl` 설정 필요 시 추가 |

---

## 추가 설정 (선택 사항)

### 기본 URL 설정

문서 내부 링크가 `/repo1` 기준으로 동작하도록 `_config.yml`에 설정 추가:

```yaml
site:
  baseurl: "/repo1"
```

이 설정은 CSS, 이미지, 내부 링크의 경로를 올바르게 유지하는 데 도움이 됩니다.

---

## 요약

본 가이드를 통해 다음을 달성할 수 있습니다:

- `repo1`, `repo2`, `repo3`와 같은 개발 중심 저장소에서 문서를 병행 관리
- `docs/` 폴더 내에 Jupyter Book 형식의 문서 작성
- GitHub Actions를 통해 자동으로 `github.io/repoX` 형태로 배포
- 코드와 문서의 통합된 버전 관리

이 방식은 프로젝트 개발 과정에서 산출되는 지식을 체계적으로 기록하고 공유하는 데 적합합니다.
