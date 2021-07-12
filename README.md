# Transferable-Query-Selection (Editing)
Code Release for "Transferable Query Selection for Active Domain Adaptation"(CVPR2021)  

Waiting for code update and document.

The adversarial-examples refs to https://github.com/sarathknv/adversarial-examples-pytorch

* **Dataset Download** 

Dataset download：<br />
Office-31: http://people.eecs.berkeley.edu/~jhoffman/domainadapt/ <br />
Office-Home: http://hemanthdv.org/OfficeHome-Dataset/ <br />
VisDA: https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification <br />
Download the dataset according to the instructions on the above, and update the path in each file in the 'data' folder

* **Specification of dependencies** 

We use the following libraries: pytorch 1.7, torchvision 0.6, numpy 1.18 and matplotlib 3.2. <br />
Pre-trained models resnet-50 can be automatically downloaded from the pytorch community. <br />

* **Command** 

For Office-31 command: <br />
python3 main.py --gpu 0 --lr 0.1 --batch-size 32 --epochs 50 --source data/office/amazon.txt --source-val data/office/amazon.txt --target data/office/dslr.txt --target-val data/office/dslr.txt --class-num 31 | tee "A_D.log"

For Office-Home command: <br />
python3 main.py --gpu 0 --lr 0.1 --epochs 40 --batch-size 32 --source data/office-home/Art.txt --target data/office-home/Clipart.txt --target-val data/office-home/Clipart.txt --class-num 65 | tee "A_C.log"

For VisDA command: <br />
python3 main.py --gpu 0 --lr 0.1 --batch-size 32 --epochs 20 --source data/visda2017/train_list.txt --target data/visda2017/validation_list.txt --target-val data/visda2017/validation_list.txt --class-num 12 | tee "vis.log"

