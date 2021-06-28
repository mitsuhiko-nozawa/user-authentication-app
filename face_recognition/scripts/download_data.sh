cd `dirname $0`
cd ../input

echo download LFW dataset...
curl http://vis-www.cs.umass.edu/lfw/lfw.tgz -O
tar -xf lfw.tgz
rm lfw.tgz

# WebFace https://drive.google.com/u/1/uc?export=download&confirm=Aew7&id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz

cd ../
echo finish!