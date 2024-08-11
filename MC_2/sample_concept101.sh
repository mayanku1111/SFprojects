# bash sample_concept101.sh <CUDA_VISIBLE_DEVICES> <composition_num:start> <composition_num:end>
# bash sample_concept101.sh 1 0 22

export CUDA_VISIBLE_DEVICES=$1

for ((i=$2; i<=$3; i++))
do
# echo $i
python run_a_composition.py --composition_num $i
done
