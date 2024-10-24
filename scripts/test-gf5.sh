echo 'gf5-x2' &&
python test.py --config ano_configs/test/test-gf5-2.yaml --model $1 --gpu $2 &&
echo 'gf5-x3' &&
python test.py --config ano_configs/test/test-gf5-3.yaml --model $1 --gpu $2 &&
echo 'gf5-x4' &&
python test.py --config ano_configs/test/test-gf5-4.yaml --model $1 --gpu $2 &&
echo 'gf5-x6' &&
python test.py --config ano_configs/test/test-gf5-6.yaml --model $1 --gpu $2 &&
echo 'gf5-x8' &&
python test.py --config ano_configs/test/test-gf5-8.yaml --model $1 --gpu $2 &&
echo 'gf5-x16' &&
python test.py --config ano_configs/test/test-gf5-16.yaml --model $1 --gpu $2 &&



true
