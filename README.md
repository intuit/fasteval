# Python Package: `fasteval`

![Supported Python versions](https://shields.io/badge/python-3.10_|_3.11_|_3.12_|_3.13_|_3.14-green?logo=python)
[![Artifactory Releases](https://img.shields.io/badge/Artifactory-Releases-41BF47.svg?logo=JFrog)](https://artifact.intuit.com/artifactory/api/pypi/pypi-intuit/simple/intlgntsys-mlservices.fasteval.fasteval/)
[![Build Status](https://build.intuit.com/tech-ai/buildStatus/buildIcon?job=intlgntsys-mlservices/fasteval/fasteval/master)](https://build.intuit.com/tech-ai/job/intlgntsys-mlservices/job/fasteval/job/fasteval/job/master/)
[![Code Coverage](https://build.intuit.com/tech-ai/buildStatus/coverageIcon?job=intlgntsys-mlservices/fasteval/fasteval/master)](https://build.intuit.com/tech-ai/job/intlgntsys-mlservices/job/fasteval/job/fasteval/job/master/)

This Python package was created from Intuit's Library Paved Road.  
For support with the paved road:
* Check [StackOverflow tag `psk`](https://stackoverflow.intuit.com/posts/tagged/4803)
* Ask in Slack channel [#psk-support](https://intuit-teams.slack.com/archives/C04AR7RF97G)

For support with this package, use Slack channel [#REPLACEME](https://intuit-teams.slack.com/archives/REPLACEME).

## Installation

### uv

PSK recommends using [uv](https://docs.astral.sh/uv/) to manage Python dependencies.  
Run the following to update `pyproject.toml`:

```shell
uv add intlgntsys-mlservices.fasteval.fasteval
```

Make sure you [set uv to use Intuit Artifactory](https://stackoverflow.intuit.com/a/37136/1539) 
for both packages and [Python standalone installations](https://docs.astral.sh/uv/guides/install-python/).

### Poetry

PSK previously recommended using [Poetry](https://python-poetry.org/) to manage Python dependencies.  
Run the following to update `pyproject.toml`:

```shell
poetry add intlgntsys-mlservices.fasteval.fasteval
```

Make sure you [set Poetry to use Intuit's PyPI](https://stackoverflow.intuit.com/a/24365/1539).

### Virtual Environment with pip

If you prefer to keep it simple, 
you can create a [virtual environment](https://docs.python.org/3/library/venv.html) and then [install using `pip`](https://pip.pypa.io/en/stable/user_guide/#installing-packages):

```shell
python -m venv venv
venv/bin/python -m pip install intlgntsys-mlservices.fasteval.fasteval
```

Make sure to [configure pip to read from Intuit's PyPI in Artifactory](https://wiki.intuit.com/display/CFTKB/Intuit+PyPI+Registry#IntuitPyPIRegistry-ProjectSetupforInstalling).


## Usage

ASSET OWNER: This section should describe the project's functionality from an end user's point of view. What are the top features for users? Screenshots are recommended.

## Local Development

### uv

This library uses [uv to manage Python dependencies](https://docs.astral.sh/uv/getting-started/features/#projects).  
`brew` is The easiest way to install on macOS:

```shell
brew install uv
```

For additional installation options (e.g. setting the PATH, installing a specific version, etc),
see the installation docs:  
https://docs.astral.sh/uv/getting-started/installation/

## Python Versions Supported

### Increase Minimum Supported Version

At present, the library is using Python 3.13 for packaging purposes.

To modify which Python versions are supported and tested by this library:
- Update "envlist" in "tox" section of [tox.ini](/tox.ini)
- Update "Supported Python versions" badge in [README.md](/README.md)
- Update "project.requires-python" in [pyproject.toml](/pyproject.toml) (if needed)
- Update "tool.black.target-version" in [pyproject.toml](/pyproject.toml) (optional)
- Update the "pythonBaseVersion" value in [msaas-config.yaml](/msaas-config.yaml) if that version is no longer supported

### Virtual Environment

Create by running:

```shell
uv sync --all-extras
```

Run a command from the virtual environment, like code formatting:

```shell
uv run black .
```

To activate the virtual environment:

```shell
source .venv/bin/activate
```

For more information, refer to [uv's documentation](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment).

### Type checking (mypy)

This library supports Python type [annotation](https://peps.python.org/pep-0484/).
Types will be checked as part of the test suite (see below).
For more information, see the [mypy documentation](https://mypy.readthedocs.io/en/stable/getting_started.html#dynamic-vs-static-typing).

### Testing

To run the test suite locally:
```shell
uv run tox
```

## Publishing

Versions are published by opening a PR, 
adding a `major`/`minor`/`patch` label, 
waiting for the checks to pass, 
then merging the PR.  
See https://stackoverflow.intuit.com/a/26011/1539 for details.

Click on the "Artifactory Releases" badge at the top to see all versions.

## Contributing

See [Contribution Guidelines](./CONTRIBUTING.md)

## Support

Please use [#REPLACEME](https://slack.com/app_redirect?team=T2G8RTHAM&channel=REPLACEME)
