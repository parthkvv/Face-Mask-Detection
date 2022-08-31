# Face-Mask-detection using Multi task learning

## Overview
We use multi-task learning approach for the following objectives

i) The primary task to detect faces that have their masks worn correctly or incorrectly

ii) The secondary task to detect faces that have their mask only covering the nose and mouth; masks only covering mouth and chin and mask under the mouth (i.e three cases of mask incorrectly worn)

## Code Structure
This repository contains implementation of the below models for Face mask detection

1. MobileNet

        to train MobileNet for primary task - Facemask-detection-task1.ipynb
        
        to train MobileNet for secondary task - Facemask-detection-task2.ipynb
        
2. BKNet
 
        to train BKNet for primary task - Evalsingletask.ipynb
        
        to train BKNet for both primary and secondary tasks:
        
                    Training - BKNetMultitask/BKNet_multitask_train.ipynb
                    
                    Evaluation - BKNetMultitask/BKNet_multitask_evaluate.ipynb
                    
                    Model implementation - BKNetMultitask/BKNetStyle.py
                    
Code for data processing: Data Pre-processing.ipynb

## Dataset
MaskedFace-Net that consists of 133,783 synthetically generated images belonging to below categories was used

1. Mask Correctly worn

3. Mask incorrectly worn

    i) Chin exposed
    
    ii) Nose exposed
    
    iii) Nose & mouth exposed
    
The data is accessible at https://github.com/cabani/MaskedFace-Net


## Environment requirements
The code in this repo is written in Python 3. BKNet models use Tensorflow 1.13.1 and MobileNet models use Tensorflow 2.x version

## Contributors
Swasthi Chittoor Shetty

Sanjana Vijay Ganesh

Samarth Varshney

Meghana Deepak

Isha Dilipkumar Shah

This work was done as part of CS6220 Big Data Systems and Analytics project requirements at Georgia Tech

## References
The following papers and code were used for this project

Sang, Dinh & Bao, Cuong. (2018). Effective Deep Multi-source Multi-task Learning Frameworks for Smile Detection, Emotion Recognition and Gender Classification. Informatica. 42. 10.31449/inf.v42i3.2301. 
https://github.com/truongnmt/multi-task-learning

Cabani et al., "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, Elsevier, 2020, 
