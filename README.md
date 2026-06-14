# 12 Comp Sci CCA

A Tkinter app for the Year 12 Computer Science CCA project.

## Prerequisites

- [Python 3](https://www.python.org/downloads/)
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


### 5. Run GUI.py

```bash
python src/gui.py
```