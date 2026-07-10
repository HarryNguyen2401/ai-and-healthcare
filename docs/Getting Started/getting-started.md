---
title: AI In The Journey To Healthier Lungs
hidden: false
---
# :book: Table of Contents

***

- [About Project](<#About Project>)
- [Prerequisites](#Prerequisites)
- [Problem Statement & Motivation](<# Problem Statement--Motivation>)
- [U-Net Model](<#U-Net Model>)
- [Attention Integration in U-Net](<#Attention Integration in U-Net>)
- [Our Method](<#Our Method>)
- [Code Demo](<#Code Demo>)
- [Future Development](<#Future Development>)
- [Credits](#Credits)

![](https://files.readme.io/39818be11aebdd701f5116d37e77161c4d7451e2ecd43ede97632fae5d236cb7-Screenshot_2026-07-07_223319.png)

# :package:About Project

***

This project explores how deep learning can improve lung segmentation in chest X-ray images to support faster and more consistent medical diagnosis. Using an Attention U-Net architecture implemented in TensorFlow/Keras, the system automatically identifies lung regions from chest radiographs. The project focuses on improving segmentation accuracy while maintaining computational efficiency, demonstrating the potential of AI-assisted medical imaging for future clinical applications.

<Callout icon="far fa-circle-exclamation" theme="info">
  ### Note

  The proposed model is a prototype developed as part of the research program and **is not intended for clinical diagnosis or medical decision-making**.
</Callout>

![](https://files.readme.io/64b3f178d7ba2cc2c44ab2a113948af69dd96e8cbc2fdbe1afe9a1e7be92910d-gray_line.png)

# :file_folder: Prerequisites

***

<HTMLBlock>{`
<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
 <img src="https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
</p>
`}</HTMLBlock>

This project builds upon several outstanding open-source resources, including:

- TensorFlow & Keras
- Attention U-Net
- Montgomery County Chest X-ray Dataset
- Shenzhen Hospital Chest X-ray Dataset
- KaggleHub

![](https://files.readme.io/74e363cab33857cbf033f8b4fbb842e1f39902edf2feac41f5d072994fc6eaf8-Screenshot_2026-07-07_223319.png)

# :herb:Problem Statement & Motivation

***

Chest X-rays are among the most widely used medical imaging techniques due to their speed, affordability, and accessibility. According to the _World Health Organization (WHO)_, more tha&#x6E;**&#x20;2 billion chest X-rays** are performed every year, making them an essential tool for diagnosing diseases such as tuberculosis, pneumonia, and lung cancer.&#x20;

![](https://files.readme.io/7d468dde11fbc295d2923c3ffbae8b4bb7c3cea116f2a09ba7d78b3b8663e85f-image.png)

<br />

But here’s the challenge: reading X-rays images accurately _isn’t an easy task_. Even when X-rays are available, the early signs of disease can be so faint, so easy to miss, that they quietly escape notice. As a result, when early signs go unseen, the consequences can be devastating.

Every year, nearly _four million people_ around the world lose their lives to lung diseases; many of which could have been prevented with early detection. Yet, in countless hospitals, radiologists face overwhelming workloads, while in some remote regions, there may be only one specialist for thousands of patients. The result is a silent crisis: delays in diagnosis, missed opportunities for treatment, and lives that could have been saved.

![](https://files.readme.io/adf180c29c38c529a49668a94f14f34d242c27f4a3f6bc8c689f5635c7fd6b8c-Screenshot_2026-06-30_153914.png)

<br />

So, _how can we make lung diagnosis faster, fairer, and more consistent?_

**AI provides an exciting answer.&#x20;**&#x49;n the face of these challenges, artificial intelligence has emerged as a powerful ally in transforming medical diagnostics. Rather than replacing radiologists, AI amplifies their capabilities, which makes healthcare faster, more consistent, and more accessible than ever before.

It can process large volumes of chest X-rays within minutes, dramatically reducing diagnostic delays. Once trained, the system is cost-effective, making it a practical tool for smaller hospitals and rural clinics that lack specialized staff. And most importantly, AI offers consistent accuracy: it highlights lung regions and subtle abnormalities that may go unnoticed by the human eye. Together, these strengths make AI not just a piece of technology, but a bridge connecting medical expertise to where it’s needed most.

![](https://files.readme.io/f933ee27cee745109d73e67d6a3a3a5cdf3312697697ae1495e600682d74a65e-Screenshot_2026-07-07_223319.png)

# :robot: U-Net Model&#x20;

***

> _A deep learning architecture specifically designed for biomedical image segmentation. Introduced in 2015 by Olaf Ronneberger et al., it has become a foundational model in the field due to its effectiveness in delineating complex structures in medical images._

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

![](https://files.readme.io/cd1c3e08d62547a199bf8e28e83842421d7c6d2ae46cefaf471fea48b6dbdbae-Screenshot_2026-07-07_223319.png)

# :brain: Attention Integration in U-Net

***

_Attention U-Net_ is an extension of the U-Net architecture for image segmentation, distinguished by the incorporation of **attention gates** (AGs) into its skip connections.

![](https://files.readme.io/d629c2aad90d4b7db1a8d79b7a39f582072440109a475ff2b43f1ca6b644db89-attention_unet-compressed-2.jpeg)

<br />

An AG modulates the encoder features before fusion with the decoder features, supplying a context-driven mechanism that automatically emphasizes task-relevant spatial regions and suppresses background or distractor information. This design directly addreses issues such as:

1. High variability in organ morphology (e.g., pancreas, retinal vessels).&#x20;
2. Low contrast between anatomical structures and background.
3. The computational and design burden of multi-stage or cascaded segmentation pipelines.

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

![](https://files.readme.io/11fe18836b4fb5a876092a10574e4654e825b8a5d74ecbe53b03316d8b76f0ec-Screenshot_2026-07-07_223319.png)

# :wrench:Our Method

***

## Setup

- **Language**: Python
- **Environment**: Google Colab (GPU-enabled)
- **Frameworks**: TensorFlow / Keras
- **Loss & Metrics**: Binary Crossentropy, Dice Coefficient, Jaccard Index

<Cards>
  <Card title="Getting Started" icon="fa-rocket">
    New to our platform? Follow this guide to get started.
  </Card>

  <Card title="API Reference" icon="fa-code">
    Explore our interactive API reference.
  </Card>

  <Card title="Support & Community" icon="fa-comments" target="_blank">
    Join our community or checkout our FAQ.
  </Card>
</Cards>


## Pipeline

![](https://files.readme.io/d01a50527b175e1f7127f8e49b3bc9163d1f2c754ba50c2c84585d9dca892d45-Anh_chup_Man_hinh_2026-07-02_luc_16.38.45.png)

## Tested Dataset

<Callout icon="⚠️" theme="info">
  ### IMPORTANT!

  **This project uses a publicly available chest X-ray dataset obtained from Kaggle for  educational purposes only. All copyrights, ownership, and intellectual property rights remain with the original dataset creator and contributor.&#x20;**

  Please refer to the original dataset source before downloading or reusing the data: [Chest X-ray Dataset for Tuberculosis Segmentation](https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation)
</Callout>

> _This dataset consists of&#x20;_**_704 chest X-ray images_**_&#x20;that have been curated from two sources: the&#x20;_**_Montgomery County Chest X-ray Database_**_&#x20;(USA) and the&#x20;_**_Shenzhen Chest X-ray Database_**_&#x20;(China). The images are used for training and evaluating machine learning models for&#x20;_**_tuberculosis (TB) detection._**
>
> _The dataset contains both&#x20;_**_tuberculosis-positive_**_&#x20;and&#x20;_**_normal&#x20;_**_chest X-rays, along with demographic details such as&#x20;_**_gender, age_**_, and&#x20;_**_county_**_&#x20;of origin. The images are accompanied by&#x20;_**_lung segmentation masks_**_&#x20;and&#x20;_**_clinical metadata_**_, which makes the dataset highly suitable for deep learning applications in medical imaging._

![](https://files.readme.io/1d80197b5b4b8f336ae966ca928c47d5a55a9480d59812409c762be2d1d2882b-Anh_chup_Man_hinh_2026-07-07_luc_10.15.05.png)

<br />

![](https://files.readme.io/35cbd8e088f5c35f1293c2e28a2e670a7d77511ff6e9b350ad902789b70f5051-Screenshot_2026-07-07_223319.png)

# :computer:Code Demo

***

Rather than presenting the complete notebook, this section highlights several key components of the implementation to illustrate how the model was built and evaluated.

> ### Loading Dependencies & Dataset

```text
import kagglehub
import tensorflow as tf
from tensorflow import keras

dataset_path = kagglehub.dataset_download(
    "iamtapendu/chest-x-ray-lungs-segmentation"
)
```

The `kagglehub` package automatically downloads the dataset into the Colab environment, while `tensorflow` and `keras` provide the core framework for building and training the segmentation model.

![](https://files.readme.io/06bcdfc58a098eea9bf9a65112406b6ad6be25dc93a67bad7a7edb3b4a8f6977-Screenshot_2026-07-08_122629.png)

<br />

> ### Building The Model

### Attention Gate Structure

At the core of the project is the **Attention Gate**, which enables the network to focus on meaningful lung regions while suppressing irrelevant background features.

```text
theta_x = Conv2D(inter_channels, 1)(x)
phi_g   = Conv2D(inter_channels, 1)(g)

add_xg = Add()([theta_x, phi_g])
psi    = Activation("sigmoid")(psi)

attn_out = Multiply()([x, psi])
```

The encoder feature map `x` and decoder gating signal `g` are combined to generate an attention mask. This mask assigns higher weights to important anatomical structures and suppresses less relevant regions before feature fusion.

### Attention U-Net&#x20;

```text
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
```

The encoder progressively compresses the input image into rich feature representations, while the decoder restores spatial resolution to produce the final lung mask.&#x20;

The `decoder_block()` combines **skip connections** with **Attention Gates**, allowing the network to recover fine anatomical details while filtering out irrelevant background information.

![](https://files.readme.io/d25f18b6463683c394aa37cf9f9633fb3f2515630a8e2f697b68e982aafbd662-Screenshot_2026-07-08_122629.png)

<br />

> ### Running Inference

After training, the model predicts a binary lung mask for each chest X-ray image.

```text
pred = model.predict(img, verbose=0)

img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
```

The predicted segmentation mask is generated using `model.predict()`, then converted into a visualization-friendly format for comparison with the original image and the ground-truth annotation.

![](https://files.readme.io/38ab92d86b76f0ba966a9e6aacffaaaaeafdbe9a03266f25cb2942b7831e7a0a-Screenshot_2026-07-08_122629.png)

<br />

> ### Visualizing the Results

Finally, the original image, ground-truth mask, and predicted mask are displayed side by side for qualitative evaluation.

```text
plt.subplot(131)
plt.imshow(img)

plt.subplot(132)
plt.imshow(msk)

plt.subplot(133)
plt.imshow(pred)
```

![](https://files.readme.io/162749768ab6887df0b9ae923b7e1dd0cf0550d45b10c502afde4c3b81062ad5-Screenshot_2026-07-07_223319.png)

# :page_with_curl:Result & Evaluation

***

> ### Plot Training & Validation Loss Values

![](https://files.readme.io/55d57aa69c61267a69db7bfe8eb2a6ce3029d3f4eff2d78e6132fc60d880e2c7-image.png)

| Metrics             | Observation                                                                                                                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loss                | Training loss decreased steadily after the first few epochs, indicating successful optimization. Validation loss fluctuated considerably due to the relatively small dataset but gradually stabilized towards the end of training. |
| Accuracy            | Training accuracy remained consistently high (**≈0.98**), while validation accuracy improved progressively and reached approximately **0.85**, showing that the model learned meaningful segmentation features.                    |
| Dice Coefficient    | Converged around **0.94**, demonstrating strong overlap between predicted and ground-truth lung masks. Validation Dice improved during later epochs despite early instability.                                                     |
| Jaccard Index (IoU) | Similar trends were observed for IoU. Training performance remained stable (**\~0.90**), while validation scores increased after additional training epochs, suggesting better generalization.                                     |

### Performance Summary

![](https://files.readme.io/e8a6faf99ed340e5c26935fcadf3dff8d4ac410461f401ad47920cf717ec49f3-Screenshot_2026-07-10_102921.png)

**_⚠️ Validation metrics vary between epochs because the dataset is relatively small and contains images collected from different hospitals with varying resolutions and imaging conditions._**

![](https://files.readme.io/d5912ebf661d905f07939058e19c2a72ed6cbf7dc939dbabe2352762e07223cd-Screenshot_2026-07-08_122629.png)

<br />

> ### Model Predictions

![](https://files.readme.io/a4d11c9e14136960c99af33a80ce3ad4658b2e838faf6785f34dbcf8127f23ef-Anh_chup_Man_hinh_2026-07-02_luc_17.14.47.png)

![](https://files.readme.io/913fb88c2579e4f8365457abc997ce80530cb42b8da95f57baced647d87cce24-Anh_chup_Man_hinh_2026-07-02_luc_17.14.57.png)

![](https://files.readme.io/1b291b7ea2ebde084d54d6cd07e93d5f85437b26ea68051eff377dcd46fe28cf-Anh_chup_Man_hinh_2026-07-02_luc_17.14.39.png)

![](https://files.readme.io/86127714e1a056d8bcac3101aed31047a84695f5586d0c17fc51081f20308366-Anh_chup_Man_hinh_2026-07-02_luc_17.14.27.png)

### Key Observations

- **Overall Segmentation Quality**: The predicted masks closely follow the overall shape of the lungs.<br />Most anatomical structures are successfully identified with consistent left–right symmetry.
- **Boundary Preservation**: The model captures the upper, lower, and lateral lung contours reasonably well. Fine anatomical details are largely preserved through the encoder–decoder architecture and skip connections.
- **Attention Mechanism**: Attention Gates help the model concentrate on lung tissue while suppressing less relevant background structures. This enables cleaner segmentation compared with a standard U-Net in many challenging regions.

### Typical Prediction Errors

> Although the overall segmentation quality is strong, several limitations can still be observed.

- Slight over-segmentation appears around the lung boundaries, where nearby soft tissue or background pixels are occasionally included.
- Small regions near the diaphragm and mediastinum remain difficult to distinguish because their intensity is similar to surrounding anatomical structures.
- Prediction quality varies slightly across patients with different image resolutions or disease severity.

### **Interpretation**

The qualitative results demonstrate that the **Attention U-Net** successfully learns the global structure of the lungs while maintaining good pixel-level localization. The predicted masks generally align well with the expert annotations, indicating that the model captures the primary lung regions with high consistency.

Although minor boundary inaccuracies remain, these errors are relatively small compared with the overall segmented area. Additional post-processing techniques, larger training datasets, and longer training schedules could further refine lung boundaries and improve prediction accuracy.

![](https://files.readme.io/d1c9ccd547aa45d8633d57af9f08313f4b62a4b583754e6399233c050890e794-Screenshot_2026-07-07_223319.png)

# :rocket:Future Development

***

**Looking ahead, there are several promising directions to further strengthen this model:**

- [ ] Expand training with NIH ChestX-ray14, CheXpert, and MIMIC-CXR.
- [ ] Support multi-label thoracic disease classification.
- [ ] Integrate Explainable AI (Grad-CAM, attention visualization).
- [ ] Optimize Attention U-Net for faster inference.
- [ ] Validate performance on external clinical datasets.
- [ ] Deploy as a web-based clinical decision support system.
- [ ] Explore lightweight models for edge devices.

<br />

![](https://files.readme.io/01ac03ca2ba50b6bc3b1124b1e660ee3cbe8f41c349f024613ea309710678c2f-Screenshot_2026-07-07_223319.png)

# :handshake:Credits

***

### Author&#x20;

Nguyen Phan Viet Hung

<HTMLBlock>{`
<p>
  <a href="https://github.com/HarryNguyen2401">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/in/hungphanvietnguyen/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="mailto:viethung.mva@gmail.com">
    <img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
</p>
`}</HTMLBlock>

### Contributor

_This project was supervised by:_

> _Dr. Phan Duc Tri_ _(Nanyang Technological University, Singapore)_
>
> Email: [phanductribkhcm@gmail.com](phanductribkhcm@gmail.com "phanductribkhcm@gmail.com")

_&#x20;and several industry experts from the&#x20;_**_Global Science Journey (GSJ)_**_&#x20;program._

![](https://files.readme.io/6807b6411faf670967ce24baae4123fd88a308de08023d3a277ca8931ba52733-images_1.jpg)

<br />
