# 12 Comp Sci CCA

A Flask web application for the Year 12 Computer Science CCA project.

## Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [Docker](https://www.docker.com)
- Git

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ShrimpyStuff/12-Comp-Sci-CCA.git
cd 12-Comp-Sci-CCA
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

> If `python` is not recognised, use `python3` instead.

### 3. Activate the virtual environment

| Shell           | Command                          |
| --------------- | -------------------------------- |
| Command Prompt  | `.venv\Scripts\activate`         |
| PowerShell      | `.\.venv\Scripts\Activate.ps1`   |
| macOS / Linux   | `source .venv/bin/activate`      |

To deactivate the virtual environment at any time:

```bash
deactivate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running with Docker

From the project root, build the image:

```bash
docker build -t my-image-name .
```

Then start the container:

```bash
docker run -d -p 8080:80 my-image-name
```

The app will be available at [http://localhost:8080](http://localhost:8080).
