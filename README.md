# Counting Coin by Object Detection

Using computer vision to counting japanese coins

Technique: object detection
Consider each coin is an object that need to be detect.

Model to train: RetinaNet

Japanese coin image dataset: https://drive.google.com/drive/folders/1tR5S4su0RXdCuyNFc6CDHXaI4zxxdYl7?usp=sharing

Model weight file: https://drive.google.com/drive/folders/1cL-QQc4Oc2R-cz-cY10eYWLDAlmvXxs1?usp=sharing

Test some images after training

![myimage-alt-tag](https://github.com/oattao/japan_coin/blob/master/show/test1.png?raw=true)

![myimage-alt-tag](https://github.com/oattao/japan_coin/blob/master/show/test2.png?raw=true)

![myimage-alt-tag](https://github.com/oattao/japan_coin/blob/master/show/test4.png?raw=true)

To count coins in an image run this command:

```console
python count.py --model_dir=path_to_trained_model --image_dir=path_to_image_file
```
