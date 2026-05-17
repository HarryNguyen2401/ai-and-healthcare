---
title: Welcome to AI in the journey to healthier lungs
hidden: false
---
> 📘 **Template:**  Delete this callout and edit this page with your content and links.

<Cards>
  {pK

## Recent Releases

AI In The Journey To Healthier Lungs

They’re fast, inexpensive, and available almost everywhere. That’s why, in many developing countries where advanced scanners like CT or MRI are rare, chest X-rays are often the only imaging tool doctors can rely on.

<Cards>

Problem Statement & Motivation

Today, medical imaging stands at the core of modern healthcare. According to the OECDAccording to the World Health Organization, over two billion chest X-rays are taken every year. That makes them one of the most common diagnostic tests in the world — and one of the most powerful. From pneumonia and tuberculosis to lung cancer, these scans can literally make the difference between early treatment and a missed diagnosis. 

But here’s the challenge: reading X-rays images accurately isn’t an easy task. Even when X-rays are available, the early signs of disease can be so faint, so easy to miss, that they quietly escape notice. As a result, when early signs go unseen, the consequences can be devastating.

Every year, nearly four million people around the world lose their lives to lung diseases; many of which could have been prevented with early detection. Yet, in countless hospitals, radiologists face overwhelming workloads, while in some remote regions, there may be only one specialist for thousands of patients. The result is a silent crisis: delays in diagnosis, missed opportunities for treatment, and lives that could have been saved.

 (2023), more than 3.6 billion imaging procedures are performed each year worldwide — a number that keeps rising as populations age and chronic diseases increase. However, the number of radiologists hasn’t grown at the same rate. This creates a widening gap between the number of scans that need to be read and the professionals available to interpret them.
In high-income countries, this already stretches resources thin. But in low- and middle-income nations, the shortage is much more severe. The U.S. has about 100 radiologists per million people, whereas countries in Southeast Asia, such as Vietnam, have fewer than 25 per million (WHO, 2023). This imbalance doesn’t just delay diagnoses, but rather can mean the difference between early treatment and missed opportunities for care. That’s why the integration of other diagnostic tools has become not just innovative, but essential: to support doctors, ensure consistency, and make timely healthcare accessible to all.

In the face of these challenges, artificial intelligence has emerged as a powerful ally in transforming medical diagnostics. Rather than replacing radiologists, AI amplifies their capabilities, which makes healthcare faster, more consistent, and more accessible than ever before.

<Cards>

Our Method

Now that we’ve explored the motivation behind this project, let’s move on to how the model was actually built.
In this section, I’ll walk you through the workflow — from data collection and preprocessing, to model design and training. We’ll see how each step was carefully designed to ensure the AI could not only recognize lung regions accurately, but also generalize well to real-world medical images.

To build and train the AI model, I used Python as the main programming language, running on Google Colab with GPU support to speed up computation. The model was developed using TensorFlow and Keras, two of the most widely used deep learning frameworks in medical imaging research. For the training process, I applied Binary Crossentropy as the loss function, along with Dice Coefficient and Jaccard Index as key evaluation metrics to measure segmentation accuracy.

The overall pipeline follows a clear flow:
Input images are first preprocessed, then passed through an encoder-decoder architecture — with attention gates in between — which help the model focus on the most relevant lung regions before generating the final segmentation mask predictions.

At the core of this project is the U-Net model, a convolutional neural network specifically designed for medical image segmentation.

The architecture follows an encoder–decoder structure.
In the encoder, the input X-ray image passes through several convolutional and pooling layers, which gradually compress and extract hierarchical features — from simple edges to complex lung textures.The decoder then performs the opposite process: it upsamples these compressed representations to reconstruct a full-resolution segmentation map. What makes U-Net unique is the use of skip connections, which directly transfer fine-grained spatial information from the encoder to the decoder. These connections help the model retain details such as the shape and edges of the lungs, which might otherwise be lost during downsampling. By combining global context from the encoder and local precision from the decoder, U-Net achieves a strong balance between accuracy and efficiency — making it highly suitable for detecting subtle lung regions in chest X-rays.

To further enhance the model’s focus, we integrated Attention Gates into the U-Net architecture. These attention modules act like a filter that helps the network concentrate on the most relevant regions while suppressing background noise such as ribs, heart shadows, or medical artifacts.

In practice, the attention mechanism dynamically weighs the importance of each spatial region, allowing the model to prioritize key features that contribute most to accurate segmentation. This not only improves precision but also helps the model generalize better across different image qualities and patient conditions. Overall, the addition of attention makes the model more clinically reliable, ensuring that the segmentation focuses on what truly matters: the lung fields and their subtle abnormalities.

For this project, I used the Chest X-ray Dataset for Tuberculosis Segmentation, which combines two well-known public datasets: Montgomery County and Shenzhen.

The Montgomery dataset includes 138 chest X-rays (80 normal and 58 showing tuberculosis) with high-resolution images around 4020x4892 pixels.

The Shenzhen dataset adds 662 images (326 normal and 336 tuberculosis cases) at approximately 3000x3000 pixels.
Together, these datasets provide a diverse and clinically meaningful collection of X-rays across different scanners, image qualities, and patient populations. This variety is crucial for training a model that generalizes well to real-world hospital data

<Cards>

Result

After training the model on both the Montgomery and Shenzhen datasets, I evaluated its ability to segment lung regions from chest X-rays based on various criteria.

Here, you can see the overall training and validation progress of the model.
The loss curve shows how the model gradually learned to distinguish lung regions more effectively through each epoch, while the validation trend confirms that the learning remained stable across unseen data.

On the screen, you can also see a few prediction masks compared to the ground truth.
The white X-ray image acts as a baseline for the segmentation test, while the pink parts in the middle image are the actual sections that the model needs to cover and the red image demonstrates how the AI analyzed the original X-ray picture.

<Cards>

Conclusion

In this final part, I’ll briefly highlight the main outcomes of the model, and discuss what could be done to further improve the current model to an applicable usage in healthcare.

The model demonstrates a strong capability in accurately segmenting lung regions from chest X-ray images. It achieves consistently high performance, with Dice coefficient, Jaccard index, and accuracy values of around 0.9, indicating close alignment between predicted and true lung areas. While the segmentation results are generally precise, slight oversegmentation can be noticed along the lung boundaries, which suggests that the model occasionally includes small non-lung regions. This issue could be reduced through post-processing steps, such as morphological filtering or boundary refinement. 

Overall, the model performs reliably across the dataset and effectively captures key lung structures. With further training, hyperparameter tuning, and optimization, it holds strong potential for real-world clinical and diagnostic applications.

<Cards>

Future Development

<br />

<br />

<br />

<br />

<br />
