cd `dirname $0`
ls ${out_dir} |  grep -v -E 'run.sh|out.log|error.log' | xargs rm -rf # 実行スクリプト以外の結果を消す

if [ $1 = f ]; then
    echo full training
    debug=False
elif [ $1 = d ]; then
    echo debug
    debug=True
else
    echo invalid argment, d debug, f full training
    exit 1
fi
root=$(dirname $(dirname $(dirname $(pwd))))
name=$(basename $(pwd))
python ../../main.py \
WORK_DIR=$(pwd) \
ROOT=${root} \
exp_name=${name} \
debug=${debug} \
+exp=${name} \
+optimizer=Adam \
+scheduler=MStepLR \
+augmentation=normal \
