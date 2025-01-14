UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PYTHON_COMMAND := python3
else ifeq ($(UNAME_S),Darwin)
    PYTHON_COMMAND := python3
else # Naively assume Windows
    PYTHON_COMMAND := py
endif

## Building:
build: ## Build project artifact
	$(PYTHON_COMMAND) -m pip install --upgrade build
	$(PYTHON_COMMAND) -m build
