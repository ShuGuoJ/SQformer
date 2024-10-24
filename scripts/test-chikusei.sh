echo 'chikusei-x2' &&
python test.py --config ano_configs/test/test-chikusei-2.yaml --model $1 --gpu $2 &&
echo 'chikusei-x3' &&
python test.py --config ano_configs/test/test-chikusei-3.yaml --model $1 --gpu $2 &&
echo 'chikusei-x4' &&
python test.py --config ano_configs/test/test-chikusei-4.yaml --model $1 --gpu $2 &&
echo 'chikusei-x6' &&
python test.py --config ano_configs/test/test-chikusei-6.yaml --model $1 --gpu $2 &&
echo 'chikusei-x8' &&
python test.py --config ano_configs/test/test-chikusei-8.yaml --model $1 --gpu $2 &&
echo 'chikusei-x16' &&
python test.py --config ano_configs/test/test-chikusei-16.yaml --model $1 --gpu $2 &&



true
