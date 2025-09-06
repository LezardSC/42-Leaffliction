need version of python 3.12 or lower. 3.13 isn't supported.
just do `python3.12 -m venv venv` to create an environment with the right version of python.

if error: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.2 as it may crash.`

try:

```
	pip uninstall numpy

	pip install numpy==1.26.4
```

