# Safety_Monitor
# 제조 현장 안전 모니터

영상 또는 화면을 캡처해서 사람을 검출하고, 사용자가 지정한 위험 구역에 사람이 들어오거나 접근하면 화면 표시와 알람으로 알려주는 Python/PySide6 기반 모니터링 앱입니다.

## 기획 배경

현장 안전 모니터링 시스템은 전문 업체에 요청하면 카메라, 서버, 모델, 알람 장비까지 포함해서 비교적 큰 단위로 셋업되는 경우가 많습니다. 하지만 실제로 비용과 시간을 들여 정식 구축을 하기 전에는, 이런 방식의 모니터링이 우리 작업 환경에 얼마나 잘 맞는지 판단하기 어렵습니다.

특히 확인하고 싶은 것은 단순히 “사람을 검출할 수 있는가”가 아니라, 현재 사용하는 영상이나 모니터링 화면에서 위험 구역을 직접 지정했을 때 사람이 구역 안에 들어오거나 접근하는 상황을 충분히 알아차릴 수 있는지입니다. 영상 각도, 작업자 크기, 화면 해상도, 조명, 가림, 장비 배치 같은 조건에 따라 실제 유용성이 크게 달라질 수 있기 때문입니다.

이 프로젝트는 그런 판단을 빠르게 해보기 위한 데모입니다. 작은 YOLO 모델을 사용해서 가볍게 실행할 수 있게 만들고, 기존 모니터링 화면이나 플레이어 창을 그대로 캡처해서 테스트할 수 있도록 했습니다. 이를 통해 전문 업체에 본격적으로 의뢰하기 전에 현재 환경에서 어떤 수준까지 동작하는지, 어떤 한계가 있는지, 알람 기준이나 위험 구역 설정이 실무적으로 의미가 있는지 먼저 확인하는 것이 목적입니다.

즉, 이 프로그램은 완성된 상용 안전 시스템을 대체하려는 것이 아니라, 도입 전에 우리 상황에 맞는지 검증하기 위한 실험용 프로토타입입니다.

## 주요 기능

- YOLO11s ONNX 모델을 이용한 사람 검출
- 모니터 전체 화면 또는 특정 앱/플레이어 창 캡처
- 영상 위에 위험 구역 다각형 그리기 및 편집
- 위험 구역 진입/접근 상태 표시
- 구역 안에 들어간 detection 박스 빨간색 표시
- 알람 사운드 및 이벤트 로그 기록
- 캡처 FPS와 YOLO 추론 FPS 분리 설정

## 데모 영상

[![제조 현장 안전 모니터 데모 영상](https://img.youtube.com/vi/Sia39VRvavQ/0.jpg)](https://www.youtube.com/embed/Sia39VRvavQ?autoplay=1)

[데모 영상 바로 재생](https://www.youtube.com/embed/Sia39VRvavQ?autoplay=1)

## 요구 사항

- Python 3.10 이상
- macOS에서 창 단위 캡처를 사용하려면 화면 녹화 권한이 필요할 수 있습니다.
- 창 단위 캡처는 `pyobjc-framework-Quartz`를 사용하므로 현재 macOS 환경을 기준으로 동작합니다.

## 설치 파일

현재 제공되는 설치 파일은 macOS용 DMG 파일만 있습니다.

- macOS: [`safety_monitor.dmg` 다운로드](https://drive.google.com/drive/folders/14DpEDMAutoBFKsIMnM9rql28E9aYuPMm?usp=sharing)
- Windows/Linux: 개발 중

DMG 파일을 받은 사용자는 파일을 열고, 안에 있는 앱을 `Applications` 폴더로 드래그해서 설치하면 됩니다.

## 설치

```bash
make install
```

위 명령은 `.venv` 가상환경을 만들고 `requirements.txt` 의존성을 설치합니다.

## 모델 다운로드

```bash
make download-model
```

`models/yolo11s.onnx`가 없으면 Ultralytics 공식 assets 릴리스에서 YOLO11s 체크포인트를 다운로드한 뒤 ONNX로 변환합니다.

## 실행

```bash
make run
```

직접 실행하려면 다음 명령도 사용할 수 있습니다.

```bash
.venv/bin/python -m safety_monitor
```

## 사용 방법

1. 앱을 실행합니다.
2. 툴바의 `캡처 소스`에서 `모니터 사용` 또는 캡처할 앱/플레이어 창을 선택합니다.
3. 창 목록이 바뀌었으면 `창 새로고침`을 누릅니다.
4. `구역 설정 시작`을 눌러 영상을 표시합니다.
5. `다각형 그리기`로 위험 구역을 찍고, 우클릭 또는 Enter로 구역을 확정합니다.
6. `모니터링 시작`을 누르면 사람 검출과 위험 구역 판정이 시작됩니다.

## 설정

주요 설정은 `config.json`에서 관리합니다.

```json
{
  "model": {
    "path": "models/yolo11s.onnx",
    "input_size": 640,
    "confidence_threshold": 0.35,
    "use_demo_detector": false
  },
  "capture": {
    "mode": "window",
    "capture_max_fps": 3
  },
  "pipeline": {
    "inference_max_fps": 3
  }
}
```

- `model.path`: 사용할 ONNX 모델 경로
- `model.input_size`: 모델 입력 크기
- `model.confidence_threshold`: 검출 신뢰도 기준
- `capture.mode`: `monitor`, `region`, `window` 중 하나
- `capture.capture_max_fps`: 화면/창 캡처 최대 FPS
- `pipeline.inference_max_fps`: YOLO 추론 최대 FPS

캡처 FPS는 화면을 얼마나 자주 가져올지, 추론 FPS는 YOLO를 얼마나 자주 실행할지를 의미합니다. 예를 들어 캡처가 15fps이고 추론이 3fps이면 화면은 부드럽게 갱신하되 YOLO는 초당 3번만 실행하고 중간 프레임에서는 직전 검출 결과를 재사용합니다.

## Make 명령

```bash
make help
make install
make download-model
make run
```

- `make help`: 사용 가능한 명령 출력
- `make install`: 가상환경 생성 및 의존성 설치
- `make download-model`: YOLO11s 체크포인트 다운로드 및 ONNX 변환
- `make run`: 앱 실행
- `make clean-venv`: `.venv` 가상환경 삭제

## macOS 권한 및 배포 주의사항

- 창 또는 화면 캡처를 사용하려면 macOS의 `시스템 설정 > 개인정보 보호 및 보안 > 화면 및 시스템 오디오 녹음`에서 앱 권한을 허용해야 할 수 있습니다.
- 현재 빌드는 Apple Developer ID 서명과 공증을 하지 않습니다. 다른 Mac에서 처음 실행할 때 Gatekeeper 경고가 표시될 수 있습니다.
- 아이콘 파일은 아직 포함하지 않았기 때문에 기본 앱 아이콘으로 표시됩니다.
- 앱으로 실행할 때 사용자 설정 저장 위치는 `~/Library/Application Support/제조 현장 안전 모니터/config.json`입니다.

## 참고 사항

- DRM 보호 영상이나 최소화된 창은 창 캡처가 검은 화면으로 보일 수 있습니다.
- 단일 모니터 환경에서는 디스플레이 전체 캡처보다 플레이어/브라우저 창 캡처를 사용하는 것이 앱 창과 영상이 겹치는 문제를 줄이는 데 유리합니다.
