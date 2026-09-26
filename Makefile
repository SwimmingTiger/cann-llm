PY ?= python3
export PYTHONPATH := src

.PHONY: test test-v lint chat server clean

test:            ## 跑单元测试（stdlib unittest，无需 pytest）
	$(PY) -m unittest discover -s tests -t . -q

test-v:          ## 跑单元测试（详细）
	$(PY) -m unittest discover -s tests -t . -v

lint:            ## 需要 ruff：pip install ruff
	ruff check src tests

chat:            ## 交互式对话：make chat MODEL=/path/to/model_dir
	$(PY) -m cann_llm.cli.chat -d $(MODEL)

server:          ## 启动 OpenAI 兼容服务：make server MODEL=/path PORT=8000
	$(PY) -m cann_llm.api.server -d $(MODEL) --port $(PORT)

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
