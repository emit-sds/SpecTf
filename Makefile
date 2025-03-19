NAME := spectf
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PYTHON_COMMAND := python3
else ifeq ($(UNAME_S),Darwin)
    PYTHON_COMMAND := python3
else # Naively assume Windows
    PYTHON_COMMAND := py
endif

GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
CYAN   := $(shell tput -Txterm setaf 6)
RESET  := $(shell tput -Txterm sgr0)

echo:
	@echo $(MAKEFILE_LIST)

## Help:
help: ## Show this help.
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "    ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2} \
		else if (/^## .*$$/) {printf "  ${CYAN}%s${RESET}\n", substr($$1,4)} \
		}' $(MAKEFILE_LIST)

## Building:
build: ## Build project artifact
	$(PYTHON_COMMAND) -m pip install --upgrade build
	$(PYTHON_COMMAND) -m build --wheel --sdist .

clean: ## Clean project
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

dev-install: ## Locally install project in development mode
	$(PYTHON_COMMAND) -m pip uninstall $(NAME)
	$(PYTHON_COMMAND) -m pip install -e .

test: ## Quickly run unit tests
	$(PYTHON_COMMAND) -m unittest discover -s tests

upload: ## Upload SpecTf python build to PyPI
	$(PYTHON_COMMAND) -m pip install --upgrade twine
	twine upload dist/*

upload-test: ## Upload SpecTf python build to TestPyPI
	$(PYTHON_COMMAND) -m pip install --upgrade twine
	twine upload -r testpypi dist/*