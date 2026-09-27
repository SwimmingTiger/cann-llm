BACKEND ?= cann
PY ?= python3
export PYTHONPATH := src

.PHONY: test test-v test-unittest lint chat server clean

# 跑测试用的解释器：依次尝试，取第一个【真能收集整套测试】的。
# ★ 只看 `import pytest` 不够 —— 有的解释器 import 得动、收集阶段却会崩，
#   所以这里用 --collect-only 实跑一遍来判定。
#   指定解释器：make test TEST_PY=/path/to/python
TEST_PY ?=
TEST_CANDIDATES = $(if $(TEST_PY),$(TEST_PY),$(PY) python3 /data/service/hnp/bin/python3 python)

test:            ## 跑单元测试（pytest；自动挑一个能跑的解释器）
	@for p in $(TEST_CANDIDATES); do \
	  if command -v $$p >/dev/null 2>&1 || [ -x $$p ]; then \
	    if $$p -m pytest -q --collect-only >/dev/null 2>&1; then \
	      echo "用 $$p 跑测试"; exec $$p -m pytest -q; \
	    fi; \
	  fi; \
	done; \
	echo "找不到能跑 pytest 的解释器：用 TEST_PY=/path/to/python 指定" >&2; exit 1

test-v:          ## 跑单元测试（pytest，详细）
	@for p in $(TEST_CANDIDATES); do \
	  if command -v $$p >/dev/null 2>&1 || [ -x $$p ]; then \
	    if $$p -m pytest -q --collect-only >/dev/null 2>&1; then \
	      echo "用 $$p 跑测试"; exec $$p -m pytest -v; \
	    fi; \
	  fi; \
	done; \
	echo "找不到能跑 pytest 的解释器：用 TEST_PY=/path/to/python 指定" >&2; exit 1

test-unittest:   ## 跑单元测试（stdlib unittest，无需第三方依赖；收集不到全部用例）
	$(PY) -m unittest discover -s tests -t . -q

lint:            ## 需要 ruff：python3 -m pip install ruff
	ruff check src tests

chat:            ## 交互式对话：make chat MODEL=/path/to/model_dir [BACKEND=cann|hiai]
	$(PY) -m cann_llm.cli.chat -b $(BACKEND) -d $(MODEL)

server:          ## 启动服务：make server MODEL=/path PORT=8000 [BACKEND=cann|hiai]
	$(PY) -m cann_llm.api.server -b $(BACKEND) -d $(MODEL) --port $(PORT)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
