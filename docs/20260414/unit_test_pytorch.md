# PyTorch 딥러닝 프로젝트 pytest 유닛테스트 매뉴얼

> 대상 독자: PyTorch 기반 딥러닝 프로젝트를 시작하는 초보자  
> 사용 도구: Python 3.9+, PyTorch, torchvision, pytest

---

## 1. 왜 유닛테스트가 필요한가?

딥러닝 프로젝트는 다음과 같은 이유로 버그를 찾기 어렵습니다.

- 모델이 학습은 되지만 결과가 이상한 경우 (silent bug)
- Tensor shape 불일치로 인한 런타임 에러
- Augmentation 후 bounding box / mask 가 어긋나는 경우
- GPU/CPU 간 데이터 이동 누락

유닛테스트를 작성하면 **코드 변경 시 즉시 문제를 감지**할 수 있습니다.

---

## 2. 환경 설치

```bash
pip install pytest pytest-cov
```

프로젝트에서 사용하는 경우 `requirements.txt` 에 추가합니다.

```text
torch
torchvision
pytest
pytest-cov
```

---

## 3. 폴더 구조

```
project/
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Dataset, DataLoader
│   ├── transforms.py     # Augmentation
│   ├── model.py          # 모델 정의
│   └── trainer.py        # 학습 루프
├── tests/
│   ├── __init__.py
│   ├── conftest.py       # 공통 fixture
│   ├── phase01/          # Dataset / Transform 테스트
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   └── test_dataset.py
│   ├── phase02/          # Model 테스트
│   │   ├── __init__.py
│   │   └── test_model.py
│   └── phase03/          # 학습 루프 테스트
│       ├── __init__.py
│       └── test_trainer.py
├── pytest.ini
└── requirements.txt
```

---

## 4. pytest 기본 설정 (`pytest.ini`)

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- `testpaths` : pytest 가 테스트를 탐색할 폴더
- `pythonpath` : `from src.xxx import ...` 가 동작하려면 반드시 필요

---

## 5. 네이밍 규칙

pytest 가 테스트를 자동으로 인식하려면 아래 규칙을 반드시 따릅니다.

| 대상 | 규칙 | 예시 |
|---|---|---|
| 파일 | `test_` 로 시작 | `test_dataset.py` |
| 함수 | `test_` 로 시작 | `test_output_shape()` |
| 클래스 | `Test` 로 시작 | `TestImageDataset` |
| 메서드 | `test_` 로 시작 | `def test_length(self)` |

### 권장 함수 네이밍 패턴

```
test_[대상]_[조건]_[기대결과]

예)
test_model_output_shape_is_correct
test_dataloader_batch_size_is_eight
test_transform_boxes_within_image_bounds
test_divide_by_zero_raises_value_error
```

---

## 6. fixture 사용법

fixture 는 테스트에 필요한 **공통 데이터나 객체를 준비**하는 함수입니다.  
테스트 함수의 인자로 넘기면 pytest 가 자동으로 주입해 줍니다.

```python
import pytest
import torch

@pytest.fixture
def sample_images():
    return torch.randn(8, 3, 64, 64)

def test_image_shape(sample_images):
    assert sample_images.shape == (8, 3, 64, 64)
```

### fixture scope — 딥러닝에서 중요

무거운 모델을 매 테스트마다 로드하면 속도가 매우 느려집니다.  
`scope` 옵션으로 fixture 의 생존 범위를 조절할 수 있습니다.

| scope | 생존 범위 | 권장 대상 |
|---|---|---|
| `function` (기본) | 테스트 함수마다 | 가벼운 Tensor, Dataset |
| `class` | 클래스 전체 | 중간 크기 객체 |
| `module` | 파일 전체 | 모델 로드, 대용량 데이터 |
| `session` | 전체 테스트 세션 | pretrained 모델 |

```python
@pytest.fixture(scope="module")
def pretrained_model():
    model = MyModel()
    model.load_state_dict(torch.load("weights.pth"))
    model.eval()
    return model
```

### 공통 fixture — `conftest.py`

`conftest.py` 에 작성한 fixture 는 **같은 폴더 및 하위 폴더** 전체에서  
import 없이 사용할 수 있습니다.

```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 7. 테스트 함수 작성법

가장 기본적인 형태로, `test_` 로 시작하는 독립적인 함수를 작성합니다.  
단순하고 독립적인 테스트에 적합합니다.

### 기본 구조

```python
# 형태: test_[대상]_[조건]_[기대결과]
def test_add_positive_numbers_returns_sum():
    assert add(1, 2) == 3

def test_model_output_shape_is_correct():
    model = MyModel().eval()
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == torch.Size([2, 10])
```

### fixture 를 인자로 받기

```python
@pytest.fixture
def model():
    return MyModel().eval()

# fixture 이름을 인자로 선언하면 pytest 가 자동 주입
def test_output_no_nan(model):
    x = torch.randn(2, 3, 224, 224)
    assert not torch.isnan(model(x)).any()
```

### 파라미터화 함수

여러 입력 케이스를 하나의 함수로 테스트합니다.

```python
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_model_handles_various_batch_sizes(batch_size):
    model = MyModel().eval()
    x = torch.randn(batch_size, 3, 224, 224)
    assert model(x).shape[0] == batch_size
```

### 예외 검증 함수

```python
def test_invalid_input_raises_value_error():
    with pytest.raises(ValueError, match="Invalid input"):
        model(None)
```

### 언제 함수를 사용하나?

- 하나의 기능을 빠르게 검증할 때
- 다른 테스트와 상태를 공유하지 않을 때
- 파라미터화로 단순 반복 테스트를 처리할 때

---

## 8. 테스트 클래스 작성법

관련 테스트를 논리적으로 묶을 때 사용합니다.  
클래스명은 반드시 `Test` 로 시작하고, `__init__` 메서드는 작성하지 않습니다.

### 기본 구조

```python
class TestMyModel:

    def test_output_shape(self):
        model = MyModel().eval()
        x = torch.randn(2, 3, 224, 224)
        assert model(x).shape == torch.Size([2, 10])

    def test_output_no_nan(self):
        model = MyModel().eval()
        x = torch.randn(2, 3, 224, 224)
        assert not torch.isnan(model(x)).any()
```

### 클래스 내 fixture 사용

```python
class TestMyModel:

    # autouse=True: 클래스 내 모든 테스트에 자동 적용
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = MyModel().eval()
        self.x = torch.randn(2, 3, 224, 224)

    def test_output_shape(self):
        assert self.model(self.x).shape == torch.Size([2, 10])

    def test_output_no_nan(self):
        assert not torch.isnan(self.model(self.x)).any()

    def test_output_dtype(self):
        assert self.model(self.x).dtype == torch.float32
```

### 클래스 내 파라미터화

```python
class TestBackbone:

    @pytest.mark.parametrize("backbone,feat_dim", [
        ("resnet18",  512),
        ("resnet50",  2048),
    ])
    def test_feature_dim(self, backbone, feat_dim):
        model = build_backbone(backbone).eval()
        x = torch.randn(1, 3, 224, 224)
        assert model(x).shape[1] == feat_dim
```

### 중첩 클래스로 조건별 그룹화

조건이 많아질수록 중첩 클래스로 세분화하면 가독성이 높아집니다.

```python
class TestDetectionTransform:

    class WhenHorizontalFlip:
        @pytest.fixture
        def transform(self):
            return DetectionTransform(v2.RandomHorizontalFlip(p=1.0))

        def test_boxes_x_coords_flipped(self, transform, sample):
            _, out_boxes, _, _ = transform(*sample)
            assert torch.allclose(out_boxes[:, 0], W - sample[1][:, 2])

        def test_mask_is_mirrored(self, transform, sample):
            _, _, out_masks, _ = transform(*sample)
            assert torch.equal(out_masks, sample[2].flip(-1))

    class WhenResize:
        @pytest.fixture
        def transform(self):
            return DetectionTransform(v2.Resize((64, 64)))

        def test_image_shape(self, transform, sample):
            out_image, _, _, _ = transform(*sample)
            assert out_image.shape == torch.Size([3, 64, 64])

        def test_boxes_within_bounds(self, transform, sample):
            _, out_boxes, _, _ = transform(*sample)
            assert (out_boxes[:, 2] <= 64).all()
```

### 언제 클래스를 사용하나?

- 같은 대상(모델, Dataset 등)을 여러 관점에서 테스트할 때
- 조건(정상 입력 / 비정상 입력 / 경계값)별로 묶어서 관리할 때
- `self` 로 상태를 공유하며 setup 코드를 줄이고 싶을 때

---

## 9. 함수 vs 클래스 비교

| | 테스트 함수 | 테스트 클래스 |
|---|---|---|
| 네이밍 | `test_xxx` | `TestXxx` + `test_xxx` |
| 상태 공유 | fixture 만 가능 | `self` 또는 fixture |
| 코드 재사용 | 제한적 | `setup` fixture 로 공유 |
| 그룹화 | 불가 | 클래스 / 중첩 클래스 |
| 파라미터화 | `@pytest.mark.parametrize` | 동일하게 적용 가능 |
| 권장 상황 | 독립적인 단일 검증 | 관련 테스트 묶음 관리 |

---

## 10. 주요 assert 패턴

### 7-1. Tensor shape 검증

```python
def test_output_shape(model):
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([2, 1000])
```

### 7-2. Tensor 값 범위 검증

```python
def test_output_range():
    out = torch.sigmoid(torch.randn(4, 1))
    assert (out >= 0).all() and (out <= 1).all()
```

### 7-3. 부동소수점 비교

```python
def test_loss_value():
    loss = compute_loss(pred, target)
    assert loss.item() == pytest.approx(0.693, rel=1e-2)
```

### 7-4. 예외 발생 검증

```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="Invalid input"):
        model(None)
```

### 7-5. dtype / device 검증

```python
def test_output_dtype_and_device(model, device):
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    assert out.dtype == torch.float32
    assert out.device.type == device.type
```

---

## 11. 파라미터화 테스트

여러 설정을 한 번에 테스트할 때 사용합니다.

```python
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_model_handles_various_batch_sizes(batch_size):
    model = MyModel().eval()
    x = torch.randn(batch_size, 3, 224, 224)
    out = model(x)
    assert out.shape[0] == batch_size


@pytest.mark.parametrize("backbone,expected_dim", [
    ("resnet18",  512),
    ("resnet50",  2048),
    ("resnet101", 2048),
])
def test_backbone_feature_dim(backbone, expected_dim):
    model = build_backbone(backbone)
    x = torch.randn(1, 3, 224, 224)
    feat = model(x)
    assert feat.shape[1] == expected_dim
```

---

## 12. 조건부 스킵

GPU 가 없는 환경에서 GPU 테스트가 실패하지 않도록 skip 처리합니다.

```python
import pytest
import torch

# GPU 없을 때 스킵
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU를 사용할 수 없는 환경입니다."
)
def test_model_on_gpu():
    model = MyModel().cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    out = model(x)
    assert out.device.type == "cuda"
```

---

## 13. Dataset / DataLoader 테스트 예시

```python
# tests/phase01/test_dataset.py
import pytest
import torch
from torch.utils.data import DataLoader
from src.dataset import ImageDataset

@pytest.fixture
def dataset():
    images = torch.randn(100, 3, 64, 64)
    labels = torch.randint(0, 10, (100,))
    return ImageDataset(images, labels)

class TestImageDataset:

    def test_length_is_correct(self, dataset):
        assert len(dataset) == 100

    def test_getitem_returns_image_and_label(self, dataset):
        image, label = dataset[0]
        assert image.shape == torch.Size([3, 64, 64])
        assert label.ndim == 0   # scalar tensor

    def test_image_dtype_is_float(self, dataset):
        image, _ = dataset[0]
        assert image.dtype == torch.float32


class TestImageDataLoader:

    @pytest.fixture
    def loader(self, dataset):
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def test_batch_image_shape(self, loader):
        images, _ = next(iter(loader))
        assert images.shape == torch.Size([8, 3, 64, 64])

    def test_total_samples_count(self, loader):
        total = sum(imgs.shape[0] for imgs, _ in loader)
        assert total == 100
```

---

## 14. Model 테스트 예시

```python
# tests/phase02/test_model.py
import pytest
import torch
from src.model import MyModel

@pytest.fixture(scope="module")
def model():
    return MyModel().eval()

class TestMyModel:

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == torch.Size([2, 10])

    def test_output_is_probability(self, model):
        x = torch.randn(4, 3, 224, 224)
        out = torch.softmax(model(x), dim=1)
        assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_no_nan_in_output(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        model = MyModel()  # eval() 없이 학습 모드
        x = torch.randn(2, 3, 224, 224)
        out = model(x).sum()
        out.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 의 gradient 가 None 입니다."
```

---

## 12. Augmentation (torchvision v2) 테스트 예시

```python
# tests/phase01/test_transforms.py
import pytest
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from src.transforms import DetectionTransform

H, W, M = 128, 128, 3

@pytest.fixture
def sample():
    image  = torch.randn(3, H, W)
    boxes  = torch.tensor([[10,10,50,50],[20,20,80,80],[30,30,90,90]],
                           dtype=torch.float32)
    masks  = torch.randint(0, 2, (M, H, W), dtype=torch.bool)
    labels = torch.tensor([0, 1, 2])
    return image, boxes, masks, labels

class TestHorizontalFlip:

    @pytest.fixture
    def transform(self):
        return DetectionTransform(v2.RandomHorizontalFlip(p=1.0))

    def test_boxes_x_coords_are_flipped(self, sample, transform):
        image, boxes, masks, labels = sample
        _, out_boxes, _, _ = transform(image, boxes, masks, labels)
        expected_x1 = W - boxes[:, 2]
        assert torch.allclose(out_boxes[:, 0], expected_x1)

    def test_mask_is_mirrored(self, sample, transform):
        image, boxes, masks, labels = sample
        _, _, out_masks, _ = transform(image, boxes, masks, labels)
        assert torch.equal(out_masks, masks.flip(-1))

    def test_boxes_count_is_preserved(self, sample, transform):
        image, boxes, masks, labels = sample
        _, out_boxes, _, _ = transform(image, boxes, masks, labels)
        assert out_boxes.shape[0] == M

class TestResize:

    @pytest.fixture
    def transform(self):
        return DetectionTransform(v2.Resize((64, 64)))

    def test_image_resized_correctly(self, sample, transform):
        image, boxes, masks, labels = sample
        out_image, _, _, _ = transform(image, boxes, masks, labels)
        assert out_image.shape == torch.Size([3, 64, 64])

    def test_masks_resized_correctly(self, sample, transform):
        image, boxes, masks, labels = sample
        _, _, out_masks, _ = transform(image, boxes, masks, labels)
        assert out_masks.shape == torch.Size([M, 64, 64])

    def test_boxes_within_resized_bounds(self, sample, transform):
        image, boxes, masks, labels = sample
        _, out_boxes, _, _ = transform(image, boxes, masks, labels)
        assert (out_boxes[:, 0] >= 0).all()
        assert (out_boxes[:, 2] <= 64).all()
```

---

## 13. 테스트 실행 방법

```bash
# 전체 테스트 실행
pytest

# 상세 출력
pytest -v

# 특정 phase 만 실행
pytest tests/phase01/ -v

# 특정 클래스만 실행
pytest tests/phase02/test_model.py::TestMyModel -v

# 특정 함수만 실행
pytest tests/phase02/test_model.py::TestMyModel::test_output_shape

# 실패 시 즉시 중단
pytest -x

# 코드 커버리지 확인
pytest --cov=src --cov-report=term-missing
```

---

## 14. 자주 하는 실수 & 해결법

| 실수 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: src` | pythonpath 미설정 | `pytest.ini` 에 `pythonpath = .` 추가 |
| fixture 가 인식 안 됨 | `conftest.py` 위치 잘못됨 | 테스트 파일과 같은 폴더 또는 상위 폴더에 위치 |
| 모델 로드가 느림 | fixture scope 가 `function` | `scope="module"` 또는 `scope="session"` 으로 변경 |
| GPU 테스트 실패 | GPU 없는 환경 | `@pytest.mark.skipif` 로 조건부 스킵 |
| `tensor.shape` 비교 실패 | `tuple` 과 `torch.Size` 비교 | `torch.Size([...])` 또는 `tuple` 로 통일 |

---

## 15. 체크리스트 — 테스트 작성 전 확인사항

- [ ] `pytest.ini` 에 `pythonpath = .` 설정 완료
- [ ] 테스트 파일명이 `test_` 로 시작하는지 확인
- [ ] 공통 fixture 는 `conftest.py` 에 작성
- [ ] 무거운 모델은 fixture `scope="module"` 사용
- [ ] GPU 테스트는 `skipif` 로 조건부 처리
- [ ] Tensor shape 는 `torch.Size([...])` 로 비교
- [ ] NaN 체크 (`torch.isnan`) 포함
- [ ] gradient 흐름 테스트 포함 (학습 가능 여부 확인)
