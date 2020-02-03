# SinGAN Apps

This is an interface for [SinGAN](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf), designed to enable dynamic image augmentations using the original code.
Specifically it avoids saving or loading any type of data (models or images).

### Usage
The interface is simple: **Given a tensor image produce a list of *n* augmentations of it:**
```
from apps import singan_augment
images_augmented = singan_augment(img_tensor_CHW, n_aug)
```
In detail:
```
>>> from PIL import Image
>>> from apps import singan_augment, toimage, totensor
>>> img_path = 'Input/Images/mountains.jpg'
>>> img = Image.open(img_path)
>>> images_augmented = singan_augment(totensor(img), 5)
>>> toimage(images_augmented[0])
```
The result is a PIL image.

### Important considerations
Note that a call to `singan_augment` will take a while (20-60min) as it trains a GAN on your image. 
Once trained, generating augmentations is very fast. Therefore the number of desired augmentations has a negligible impact on the total run-time.
