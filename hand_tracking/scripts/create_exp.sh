cd src/experiments 

dnum=$(ls -l | grep ^d | wc -l)
dnum=$(($dnum - 1))
#dnum=${dnum//[[:blank:]]}
dsize=${#dnum}

if [ $dsize = 1 ]; then
    dnum=00$dnum
elif [ $dsize = 2 ]; then
    dnum=0$dnum
fi

dname=exp_$dnum
cp -r _template ${dname}
cp ../configs/exp/_template.yaml ../configs/exp/exp_${dnum}.yaml