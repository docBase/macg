cd ~/projects
git clone git@github.com:docBase/macg.git
cd macg
source activate root
echo "`pwd`/src" > $CONDA_ENV_PATH/lib/python3.5/site-packages/macg.pth
cd src/macg
jupyter notebook
# or, for instance
./__init__.py GCI.C.12.01 GCI.C.12.06 2015-2016