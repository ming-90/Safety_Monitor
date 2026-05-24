.PHONY: help check-python venv install run download-model clean-venv

PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null || command -v python3)
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

help:
	@echo "사용 가능한 명령:"
	@echo "  make venv            가상환경 생성"
	@echo "  make install         의존성 설치"
	@echo "  make download-model  YOLO11n ONNX 모델 다운로드"
	@echo "  make run             앱 실행"
	@echo "  make clean-venv      가상환경 삭제"

check-python:
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "Python 3.10 이상이 필요합니다. 예: make clean-venv && make PYTHON=python3.11 install")'

venv: check-python
	$(PYTHON) -m venv $(VENV)
	@$(VENV_PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else ".venv가 Python 3.10 미만으로 만들어져 있습니다. make clean-venv 후 다시 실행하세요.")'

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

download-model:
	$(VENV_PYTHON) scripts/download_yolo11n_onnx.py

run:
	$(VENV_PYTHON) -m safety_monitor

clean-venv:
	rm -rf $(VENV)
