#create virtual env
python3 -m venv .venv

#activate virtual env
source .venv/bin/activate

#install requirements
pip install -r requirements.txt

# adds the project root to the python path, in any python 3 installation
echo $PWD > $VIRTUAL_ENV/lib/python3*/site-packages/project.pth
