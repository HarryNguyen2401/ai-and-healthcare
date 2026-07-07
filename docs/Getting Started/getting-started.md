---
title: AI In The Journey To Healthier Lungs
excerpt: >-
  This project explores how deep learning can improve lung segmentation in chest
  X-ray images to support faster and more consistent medical diagnosis. Using an
  Attention U-Net architecture implemented in TensorFlow/Keras, the system
  automatically identifies lung regions from chest radiographs. The project
  focuses on improving segmentation accuracy while maintaining computational
  efficiency, demonstrating the potential of AI-assisted medical imaging for
  future clinical applications.
hidden: false
---
# Table of Contents

***

- Problem Statement & Motivation
- U-Net Model
- Attention Integration in U-Net
- Our Method
- Code Demo
- Result & Conclusion
- Future Development

<br />

# Problem Statement & Motivation

***

Something we’ve all seen before in the healthcare industry — **chest X-rays**. They’re fast, inexpensive, and available almost everywhere. That’s why, in many developing countries where advanced scanners like CT or MRI are rare, chest X-rays are often the only imaging tool doctors can rely on.

Today, medical imaging stands at the core of modern healthcare. According to the _World Health Organization_, over 2 billion chest X-rays are taken every year. That makes them one of the most common diagnostic tests in the world — and one of the most powerful. From pneumonia and tuberculosis to lung cancer, these scans can literally make the difference between early treatment and a missed diagnosis.

![](https://files.readme.io/7d468dde11fbc295d2923c3ffbae8b4bb7c3cea116f2a09ba7d78b3b8663e85f-image.png)

<br />

But here’s the challenge: reading X-rays images accurately _isn’t an easy task_. Even when X-rays are available, the early signs of disease can be so faint, so easy to miss, that they quietly escape notice. As a result, when early signs go unseen, the consequences can be devastating.

Every year, nearly _four million people_ around the world lose their lives to lung diseases; many of which could have been prevented with early detection. Yet, in countless hospitals, radiologists face overwhelming workloads, while in some remote regions, there may be only one specialist for thousands of patients. The result is a silent crisis: delays in diagnosis, missed opportunities for treatment, and lives that could have been saved.

![](https://files.readme.io/adf180c29c38c529a49668a94f14f34d242c27f4a3f6bc8c689f5635c7fd6b8c-Screenshot_2026-06-30_153914.png)

<br />

The motivation behind this project comes from a simple question — _how can we make lung diagnosis faster, fairer, and more consistent?_

**AI provides an exciting answer.&#x20;**&#x49;n the face of these challenges, artificial intelligence has emerged as a powerful ally in transforming medical diagnostics. Rather than replacing radiologists, AI amplifies their capabilities, which makes healthcare faster, more consistent, and more accessible than ever before.

It can process large volumes of chest X-rays within minutes, dramatically reducing diagnostic delays. Once trained, the system is cost-effective, making it a practical tool for smaller hospitals and rural clinics that lack specialized staff.&#x20;

And most importantly, AI offers consistent accuracy: it highlights lung regions and subtle abnormalities that may go unnoticed by the human eye. Together, these strengths make AI not just a piece of technology, but a bridge connecting medical expertise to where it’s needed most.

<br />

# U-Net Model&#x20;

***

> _A deep learning architecture specifically designed for biomedical image segmentation. Introduced in 2015 by Olaf Ronneberger et al., it has become a foundational model in the field due to its effectiveness in delineating complex structures in medical images._

<br />

## Architecture Overview

- **Contracting path (Encoder)**: Extracts hierarchical features through convolution and downsampling.
- **Expanding path (Decoder)**: Restores image resolution for pixel-wise segmentation.
- **Skip connections**: Fuse low-level spatial information with high-level semantic features.

![](https://files.readme.io/878cecfedcaa4830241bad40f68890cf88c63e2af907a36f6ec6f1e1ad31d8e5-image.png)

<br />

### Why U-Net?

U-Net is designed to work well with relatively few training images. This is particularly important in the biomedical field, where labeled data can be scarce. The model uses data augmentation techniques to enhance training and improve generalization.

It's widely used for segmenting various biomedical images:

| Image Type            | Application                                                                      |
| --------------------- | -------------------------------------------------------------------------------- |
| MRI & CT Scans        | Identifying tumors, lesions, and organ boundaries                                |
| Histopathology images | Detecting cellular structures and classifying tissue types                       |
| Cell Segmentation     | Tasks like identifying individual cells and their components (Microscopy images) |

The architecture has been shown to achieve high accuracy and robustness in various segmentation tasks, often outperforming traditional methods. Its ability to produce precise pixel-wise predictions makes it a preferred choice for medical image analysis.

<br />

# Attention Integration in U-Net

***

_Attention U-Net_ is an extension of the U-Net architecture for image segmentation, distinguished by the incorporation of **attention gates** (AGs) into its skip connections.

![](https://files.readme.io/d629c2aad90d4b7db1a8d79b7a39f582072440109a475ff2b43f1ca6b644db89-attention_unet-compressed-2.jpeg)

<br />

An AG modulates the encoder features before fusion with the decoder features, supplying a context-driven mechanism that automatically emphasizes task-relevant spatial regions and suppresses background or distractor information. This design directly addreses issues such as:

1. High variability in organ morphology (e.g., pancreas, retinal vessels).&#x20;
2. Low contrast between anatomical structures and background.
3. The computational and design burden of multi-stage or cascaded segmentation pipelines.

<br />

### Why is attention needed in the U-Net?

- **Focuses on the most relevant regions.** Attention Gates guide the model toward important lung structures while reducing the influence of background noise.
- **Improves segmentation performance.** By emphasizing meaningful features, the model converges faster and generalizes better to unseen images.
- **Suppresses irrelevant information automatically.** The network learns to ignore non-essential regions without requiring additional localization modules.
- **Adapts to different anatomical structures.&#x20;**&#x41;ttention Gates effectively capture target regions of varying shapes and sizes in medical images.
- **Enhances U-Net with minimal overhead.** The attention mechanism can be integrated easily, improving sensitivity and accuracy with only a small increase in model parameters.

<br />

<Callout icon="ℹ️" theme="info">
  ### **More About Attention U-Net**

  If you want to delve deeper into attention U-Nets' applications in healthcare, check out this article: [_Attention U-Net: Learning Where to Look for the Pancreas_](https://arxiv.org/abs/1804.03999)
</Callout>

<br />

# Our Method

***

## Setup

- **Language**: Python
- **Environment**: Google Colab (GPU-enabled)
- **Frameworks**: TensorFlow / Keras
- **Loss & Metrics**: Binary Crossentropy, Dice Coefficient, Jaccard Index

## Pipeline

![](https://files.readme.io/d01a50527b175e1f7127f8e49b3bc9163d1f2c754ba50c2c84585d9dca892d45-Anh_chup_Man_hinh_2026-07-02_luc_16.38.45.png)

<br />

## Tested Dataset

<Cards>
  <Card title="Getting Started" href="#" icon="fa-rocket">
    New to our platform? Follow this guide to get started.
  </Card>

  <Card title="API Reference" href="#" icon="fa-code">
    Explore our interactive API reference.
  </Card>

  <Card title="Support & Community" href="#" icon="fa-comments" target="_blank">
    Join our community or checkout our FAQ.
  </Card>
</Cards>

<br />

> _This dataset consists of&#x20;_**_704 chest X-ray images_**_&#x20;that have been curated from two sources: the&#x20;_**_Montgomery County Chest X-ray Database_**_&#x20;(USA) and the&#x20;_**_Shenzhen Chest X-ray Database_**_&#x20;(China). The images are used for training and evaluating machine learning models for&#x20;_**_tuberculosis (TB) detection._**
>
> _The dataset contains both&#x20;_**_tuberculosis-positive_**_&#x20;and&#x20;_**_normal&#x20;_**_chest X-rays, along with demographic details such as&#x20;_**_gender, age_**_, and&#x20;_**_county_**_&#x20;of origin. The images are accompanied by&#x20;_**_lung segmentation masks_**_&#x20;and&#x20;_**_clinical metadata_**_, which makes the dataset highly suitable for deep learning applications in medical imaging._

![](https://files.readme.io/1d80197b5b4b8f336ae966ca928c47d5a55a9480d59812409c762be2d1d2882b-Anh_chup_Man_hinh_2026-07-07_luc_10.15.05.png)

<br />

# Code Demo

***

> ### Importing Libraries

```text
import kagglehub
iamtapendu_chest_x_ray_lungs_segmentation_path = kagglehub.dataset_download('iamtapendu/chest-x-ray-lungs-segmentation')
iamtapendu_lungs_segmentation_using_u_net_architecture_path = kagglehub.notebook_output_download('iamtapendu/lungs-segmentation-using-u-net-architecture')

print('Data source import complete.')

!pip install opendatasets --quiet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from google.colab import files, drive

import cv2
from cv2 import imread, resize
from scipy.ndimage import label, find_objects

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import os
import warnings
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, Add, Multiply
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_physical_devices('GPU'))
#Download dataset
import opendatasets as opendatasets
dataset_url = "https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation"
opendatasets.download(dataset_url, data_dir="/content")

#Set paths after download
dataset_dir = '/content/chest-x-ray-lungs-segmentation'
IMG_PATH = os.path.join(dataset_dir, 'Chest-X-Ray', 'Chest-X-Ray', 'image')
MSK_PATH = os.path.join(dataset_dir, 'Chest-X-Ray', 'Chest-X-Ray', 'mask')
METADATA_PATH = os.path.join(dataset_dir, 'MetaData.csv')

##PATHS
#IMG_PATH = '/kaggle/input/chest-x-ray-segmentation/Chest-X-Ray/Chest-X-Ray/image/'
#MSK_PATH = '/kaggle/input/chest-x-ray-segmentation/Chest-X-Ray/Chest-X-Ray/mask/'
```

###

> ### Building The Model

```text
# from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, Add, Multiply

def attention_gate(x, g, inter_channels):

    theta_x = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(x)
    theta_x = layers.BatchNormalization()(theta_x)
    phi_g = Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')(g)
    phi_g = layers.BatchNormalization()(phi_g)

    if theta_x.shape[1] != phi_g.shape[1] or theta_x.shape[2] != phi_g.shape[2]:
        phi_g = UpSampling2D(size=(theta_x.shape[1] // phi_g.shape[1],
                                   theta_x.shape[2] // phi_g.shape[2]),
                             interpolation='bilinear')(phi_g)

    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    psi = Conv2D(1, kernel_size=1, padding='same')(act_xg)
    psi = Activation('sigmoid')(psi)
    attn_out = Multiply()([x, psi])
    return attn_out

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(x)
    attn = attention_gate(skip, x, inter_channels=filters//2)
    x = layers.concatenate([x, attn])
    x = conv_block(x, filters)
    return x

    # Attention U-Net Model
def unet_with_attention(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)

    conv1 = conv_block(inputs, 32)
    pool1 = layers.MaxPooling2D((2,2))(conv1)

    conv2 = conv_block(pool1, 64)
    pool2 = layers.MaxPooling2D((2,2))(conv2)

    conv3 = conv_block(pool2, 128)
    pool3 = layers.MaxPooling2D((2,2))(conv3)

    conv4 = conv_block(pool3, 256)
    conv6 = decoder_block(conv4, conv3, 128)
    conv7 = decoder_block(conv6, conv2, 64)
    conv8 = decoder_block(conv7, conv1, 32)

    outputs = layers.Conv2D(num_classes, (1,1), activation='sigmoid')(conv8)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

    # Jaccard Index Metric
def jaccard_index(y_true, y_pred, smooth=100):
    """Calculates the Jaccard index (IoU), useful for evaluating the model's performance."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (total + smooth)

    # Dice Coefficient Metric
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

model = unet_with_attention(input_shape=(256, 256, 1))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',dice_coefficient,jaccard_index])
```

<br />

> ### Showcasing The Result

```text
imgs, msks  = val_data.__getitem__(1)

for img,msk in zip(imgs,msks):
    img = np.expand_dims(img, axis=0)
    pred = (np.squeeze(model.predict(img,verbose=0))*255).astype(np.uint8)
    img = (np.squeeze(img) * 255).astype(np.uint8)
    msk = (msk*255).astype(np.uint8)

    # Convert grayscale image to RGB
    img= cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2RGB)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12,4))

    plt.subplot(131)
    plt.imshow(img)
    plt.title('Image')
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.subplot(132)
    plt.imshow(get_colored_mask(img,msk))
    plt.title('Mask (Actual)')
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.subplot(133)
    plt.imshow(get_colored_mask(img,pred,color = [255,30,0]))
    plt.title('Mask (Prediction)')
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.tight_layout()
    plt.show()
```

<br />

# Result & Conclusion

***

> ### Plot Training & Validation Loss Values

![](https://files.readme.io/44c70187654532c300b9d08268ccd9ac0275df736688dd3535a6165c5df1c699-Anh_chup_Man_hinh_2026-07-02_luc_17.11.01.png)

The model demonstrates a strong capability in accurately segmenting lung regions from chest X-ray images. It achieves consistently high performance, with Dice coefficient, Jaccard index, and accuracy values of around 0.9, indicating close alignment between predicted and true lung areas.

<br />

> ### Model Predictions

![](https://files.readme.io/a4d11c9e14136960c99af33a80ce3ad4658b2e838faf6785f34dbcf8127f23ef-Anh_chup_Man_hinh_2026-07-02_luc_17.14.47.png)

![](https://files.readme.io/913fb88c2579e4f8365457abc997ce80530cb42b8da95f57baced647d87cce24-Anh_chup_Man_hinh_2026-07-02_luc_17.14.57.png)

![](https://files.readme.io/1b291b7ea2ebde084d54d6cd07e93d5f85437b26ea68051eff377dcd46fe28cf-Anh_chup_Man_hinh_2026-07-02_luc_17.14.39.png)

![](https://files.readme.io/86127714e1a056d8bcac3101aed31047a84695f5586d0c17fc51081f20308366-Anh_chup_Man_hinh_2026-07-02_luc_17.14.27.png)

While the segmentation results are generally precise, slight **oversegmentation** can be noticed along the lung boundaries, which suggests that the model occasionally includes small non-lung regions. This issue could be reduced through post-processing steps, such as morphological filtering or boundary refinement.

<br />

# Future Development

***

**Looking ahead, there are several promising directions to further strengthen this model.**

- Integrate more chest X-ray datasets to improve robustness
- Extend to multi-label classification & assessment
- Visual explanation methods: improve model transparency & clinician trust
- Model refinement to balance accuracy & efficiency

<br />

<br />

# Credits

***

### Author

Nguyen Phan Viet Hung

<br />

<br />
