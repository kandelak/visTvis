SHELL := /bin/bash


YELLOW := "\e[1;33m"
NC := "\e[0m"

# Logger function
INFO := @bash -c '\
  printf $(YELLOW); \
  echo "=> $$1"; \
  printf $(NC)' SOME_VALUE

.venv:  # creates .venv folder if does not exist
	python3.10 -m venv .venv


.venv/bin/uv: .venv # installs latest pip
	.venv/bin/pip install -U uv
	.venv/bin/uv pip install setuptools wheel twine artifacts-keyring

install: .venv/bin/uv 
	.venv/bin/python3 -m uv pip install -r requirements.txt

build-package:
	.venv/bin/python3 -m build --sdist

upload-package:
	.venv/bin/python3 -m twine upload dist/*