# StarGAN-tensorflow

Tensorflow implementation of [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) <br>
The author's torch code can be found [here](https://github.com/yunjey/StarGAN)
<br><br>

## Prerequisites
* Python 3.5
* Tensorflow 1.3.0
* Scipy

## Usage
So far I'm only using celebA datasets <br>
First, download dataset with:
```
$ python download.py
```
To train a model:
```
$ python main.py --phase=train --image_size=64 --batch_size=16
```
To test the model:
```
$ python main.py --phase=test --image_size=64 --binary_attrs=1001110
```
Bianry attributes are now set up with the following sequence:
```
'Black_Hair','Blond_Hair','Brown_Hair', 'Male', 'Young','Mustache','Pale_Skin'
```
All the available attributes are:
```
'5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
```
which are the attributes provided in celebA dataset. you can change the attribute set in model.py <br>
if you change the attribute set, you should also change the n_label argument. for example:
```
$ python main.py --phase=train --n_label=10
```

## Result

## Training details
Details of the loss of Discriminator and Generator
