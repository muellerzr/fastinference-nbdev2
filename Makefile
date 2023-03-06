.PHONY: quality style test docs

check_dirs := src

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	ruff $(check_dirs)
	doc-builder style src/fastinference --max_len 119 --check_only

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	black $(check_dirs)
	ruff $(check_dirs) --fix
	doc-builder style src/fastinference --max_len 119