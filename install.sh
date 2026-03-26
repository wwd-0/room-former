pip install -r requirements.txt 
cd models/ops
bash make.sh
cd ../../diff_ras
python setup.py build develop
