cd `dirname $0`

echo $1 $2
echo $(pwd)
out_dir=../src/experiments/exp_$1
echo ${out_dir}

if [ $2 = f ]; then 
    echo フルトレーニングモード
    nohup sh ${out_dir}/run.sh $2 > ${out_dir}/out.log 2> ${out_dir}/error.log &
    
elif [ $2 = d ]; then
    echo デバッグモード
    sh ${out_dir}/run.sh $2
else
    echo そのた
fi