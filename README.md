Objectives for multi-task learning approach :

Primary task - to detect faces that have their masks worn correctly or incorrectly

Secondary task - to detect faces that have their mask only covering the nose and mouth; masks only covering mouth and chin and mask under the mouth (i.e three cases of mask incorrectly worn)

### RUN
Models for Face mask detection :

1. MobileNet

         Facemask-detection-task1.ipynb - to train MobileNet for primary task 
        
         Facemask-detection-task2.ipynb - to train MobileNet for secondary task
        
2. BKNet
 
         Evalsingletask.ipynb - to train BKNet for primary task - 
        
         Training BKNet for both primary and secondary tasks:
        
                    Training - BKNetMultitask/BKNet_multitask_train.ipynb
                    
                    Evaluation - BKNetMultitask/BKNet_multitask_evaluate.ipynb
                    
                    Model implementation - BKNetMultitask/BKNetStyle.py

![mask](https://user-images.githubusercontent.com/56112545/189886063-7847274c-c344-4454-a322-9619587f302b.png)

## References
The following papers and code were used for this project

Sang, Dinh & Bao, Cuong. (2018). Effective Deep Multi-source Multi-task Learning Frameworks for Smile Detection, Emotion Recognition and Gender Classification. Informatica. 42. 10.31449/inf.v42i3.2301. 
https://github.com/truongnmt/multi-task-learning

Cabani et al., "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, Elsevier, 2020, 
