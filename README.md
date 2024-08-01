# Mushroom Identifier

Mushroom Identifier is an AI that can categorize a mushroom in one of nine broad genuses.

![image of a mushroom](https://github.com/user-attachments/assets/54764509-068e-4333-bed8-c1ea40863caf)

## The Algorithm

The model is retrained from the resnet-18 model, using a dataset of over 5000 images ([Found here](https://www.kaggle.com/datasets/lizhecheng/mushroom-classification)). The model was retrained using jetson inference's built-in retraining system.
The python file utilizes the .onnx file to recognize an image that the user inputs. A few test images are also included.

This mushroom genuses this model can detect are:
* Agaricus
* Amanita
* Boletus
* Cortinarius
* Entoloma
* Hygrocybe
* Russula
* Sullius
* Lactarius

## Running this project

[View a video explanation here](https://drive.google.com/file/d/1gGGx5MgdahSwqICQyDdbqE0siOVc_TmI/view?usp=sharing)

1: Be sure the jetson inference library is installed on your system

2: Clone the repository by running this command
```sh
git clone https://github.com/Speed-Type/MushroomIdentifier
```

3: Move into the project folder
```sh
cd MushroomIdentifier
```

4: Run the python script
```sh
python3 mushroom.py path/to/file/here
```
