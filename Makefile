lint: clean
	flake8 zhtools --format=pylint || true


test: lint
	py.test -vvv --cov zhtools --cov-report term-missing --cov-report xml:cobertura.xml --junitxml=testresult.xml tests


lock-requirements:
	- pip install pip-tools
	- pip-compile --output-file requirements.txt requirements.in


deps: lock-requirements
	- pip install -r requirements.txt

clean:
	- find . -iname "*__pycache__" | xargs rm -rf
	- find . -iname "*.pyc" | xargs rm -rf
	- rm cobertura.xml -f
	- rm testresult.xml -f
	- rm .coverage -f
	- rm .pytest_cache -rf
