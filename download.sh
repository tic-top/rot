#\bin\bash
apt-get install unzip
mkdir test
git clone https://github.com/koishi70/Landscape-Dataset
cat ./Landscape-Dataset/landscape_dataset.zip.* > combined.zip
unzip -q combined.zip -d ./google-street-view
rm -r ./Landscape-Dataset
rm -r ./combined.zip
# for i in {1..10}
# do
#    while true; do wget -T 10 -c https://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/zipped%20images/part$i.zip && break; done
#    unzip -q part$i.zip -d ./dd/street
#    rm -r part$i.zip
# done
# cd ./dd/street
# rm -r *_5.jpg
