pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd models/ops
bash make.sh
cd ../../diff_ras
python setup.py build develop
cd ..
python main.py  --dataset_root /content/dataset_v1 --epochs 500 --batch_size 1 --lr 2e-4 --lr_drop 400 --num_queries 800 --num_polys 20 --use_pano --pano_backproject_bias --output_dir ../output
