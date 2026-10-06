# Static checks only: the scripts drive remote GPU hosts and are never executed here.
SHELL := /bin/bash
SH_FILES := $(shell find deploy ops bench -type f -name '*.sh' | sort)
PY_FILES := $(shell find bench -type f -name '*.py' | sort)

.PHONY: check
check:
	@for f in $(SH_FILES); do bash -n "$$f" || exit 1; done
	@python3 -c 'import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' $(PY_FILES)
	@echo "OK: $(words $(SH_FILES)) shell scripts, $(words $(PY_FILES)) Python files"
