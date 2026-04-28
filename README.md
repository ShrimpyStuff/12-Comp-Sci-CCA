1. To start open your terminal/powershell, paste, and run the following command: 

git clone https://github.com/ShrimpyStuff/12-Comp-Sci-CCA.git

2. Create the virtual environment:

python -m venv .venv

OR

python3 -m venv .venv

To run virtual environment:

If using Command Prompt: 

.venv\Scripts\activate

If using Powershell:

.\.venv\Scripts\Activate.ps1

If using Terminal:

source .venv/bin/activate

(Just in case) To stop running venv:

deactivate

3. Download Docker from the following link:

https://www.docker.com

Build the docker image (be in the same directory):

docker build -t my-image-name .

Run the docker container:

docker run -d -p 8080:80 my-image-name

