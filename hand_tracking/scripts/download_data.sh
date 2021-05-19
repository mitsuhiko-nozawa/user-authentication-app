cd input

echo download LFW dataset...
curl http://vis-www.cs.umass.edu/lfw/lfw.tgz -O
tar -xf lfw.tgz
rm lfw.tgz

cd ../
echo finish!