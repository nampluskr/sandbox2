# Jupyter Book을 사용한 문서 작성 및 GitHub Pages 배포 가이드

본 문서는 Jupyter Book을 활용하여 기술 문서를 작성하고, GitHub Pages를 통해

- `https://your-username.github.io/mybook1`
- `https://your-username.github.io/mybook2`
- `https://your-username.github.io/mybook3`

형태로 세 개의 독립된 페이지로 배포하는 전체 과정을 안내합니다. 초보자를 대상으로 단계별로 설명합니다.

---

## 1. 라이브러리 설치

로컬 환경에 Jupyter Book을 설치합니다. 명령줄에서 다음 명령어를 실행합니다.

```bash
pip install jupyter-book
```

설치 완료 후, 버전 확인을 통해 정상 설치 여부를 검증할 수 있습니다.

```bash
jupyter-book --version
```

---

## 2. 로컬 폴더 생성 및 초기화

각 문서를 위한 별도의 폴더를 생성합니다.

```bash
mkdir ~/Documents/mybook1
mkdir ~/Documents/mybook2
mkdir ~/Documents/mybook3
```

각 폴더로 이동하여 Jupyter Book 프로젝트를 초기화합니다.

```bash
cd ~/Documents/mybook1
jupyter-book create ./
```

> `jupyter-book create` 명령어가 지원되지 않을 경우, 수동으로 `_toc.yml`, `_config.yml`, `intro.md` 파일을 생성할 수 있습니다.

---

## 3. 문서 작성

각 폴더 내의 `intro.md` 파일을 편집기로 열고 내용을 작성합니다.

예시 (`mybook1/intro.md`):

```markdown
# 소개

이 문서는 mybook1 프로젝트의 시작 페이지입니다.

다음은 Python 코드 예제입니다.

```python
print("Hello, World!")
```
```

필요 시, 추가 챕터를 작성하고 `_toc.yml` 파일에 목차를 업데이트합니다.

예시 `_toc.yml`:

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

빌드 완료 후, 생성된 HTML 파일을 브라우저에서 열어 확인합니다.

```
./_build/html/index.html
```

---

## 5. GitHub 저장소 생성

각 문서에 대해 별도의 저장소를 생성합니다.

- https://github.com/your-username/mybook1
- https://github.com/your-username/mybook2
- https://github.com/your-username/mybook3

저장소 생성 시 다음을 확인합니다:
- Public 설정
- README, .gitignore, 라이선스 파일은 추가하지 않음

---

## 6. 로컬 프로젝트를 GitHub 저장소에 연결

각 폴더에서 Git 초기화 및 원격 저장소 연결을 수행합니다.

예시 (mybook1):

```bash
cd ~/Documents/mybook1
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/mybook1.git
git push -u origin main
```

`your-username`은 본인의 GitHub 사용자 이름으로 대체합니다.

---

## 7. GitHub Actions를 통한 자동 배포 설정

각 저장소에 자동 빌드 및 배포 파이프라인을 설정합니다.

`.github/workflows/deploy.yml` 파일을 생성합니다.

경로: `mybook1/.github/workflows/deploy.yml`

내용:

```yaml
name: Deploy Jupyter Book

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

      - name: Build the book
        run: |
          jupyter-book build ./

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./_build/html
          publish_branch: gh-pages
```

이 파일을 커밋하고 푸시합니다.

```bash
git add .github/workflows/deploy.yml
git commit -m "Add GitHub Actions workflow"
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
> "Your site is published at https://your-username.github.io/mybook1"

---

## 9. 배포 확인

다음 URL에서 문서 확인이 가능합니다.

- https://your-username.github.io/mybook1
- https://your-username.github.io/mybook2
- https://your-username.github.io/mybook3

각 문서는 독립된 저장소에서 관리되며, 별도로 수정 및 배포됩니다.

---

## 문제 해결

| 현상 | 원인 및 해결 방법 |
|------|------------------|
| `jupyter-book` 명령어를 인식하지 못함 | Python 스크립트 경로가 PATH에 없을 수 있음. 재설치 후 터미널 재시작 |
| GitHub 푸시 실패 | 사용자 이름, 저장소 이름 오기입 여부 확인. 2단계 인증 사용 시 Personal Access Token 필요 |
| 변경 사항이 반영되지 않음 | GitHub Actions가 정상 실행되었는지 **Actions** 탭에서 확인 |
| 이미지 또는 링크 오류 | 상대 경로 사용 확인. 필요 시 `_config.yml`에 `baseurl` 설정 추가 |

---

## 추가 설정 (선택 사항)

### 기본 URL 설정

문서 내부 링크가 `/mybook1` 기준으로 동작하도록 `_config.yml`에 설정 추가:

```yaml
site:
  baseurl: "/mybook1"
```

---

## 요약

본 가이드를 통해 다음을 수행할 수 있습니다:

- Jupyter Book을 사용한 문서 작성
- 로컬 빌드 및 미리보기
- GitHub 저장소 생성 및 연결
- GitHub Actions를 통한 자동 배포
- 세 개의 독립된 문서 페이지 운영

이 방식은 기술 문서, 내부 가이드, 교육 자료 등 다양한 용도로 활용 가능합니다.

--- 

문서 작성자: OLEDi (삼성디스플레이 사내 AI 어시스턴트)  
최종 업데이트: 2026-04-14
