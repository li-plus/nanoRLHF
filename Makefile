.PHONY: train chat test lint

train:
	python3 train_rlhf.py

chat:
	python3 chat_rlhf.py

test:
	pytest -s

lint:
	isort *.py
	black *.py --line-length 120 --verbose
