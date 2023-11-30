# SSIRA
My URECA project on Self-Supervised-Image-Reconstruction-and-Aesthetics-Assessment

## Proposed Improvements for AestheticNet
Step 1: Integrate Global and Local Feature Extraction

* Inspiration: Adopt the multi-headed approach from TANet and Non-Local Blocks from BAID.
* Implementation: Incorporate a dual-pathway in AestheticNet where one path focuses on global features (using a non-local block) and the other on local features. This could enhance the model's ability to assess images based on both detailed and holistic characteristics.

Step 2: Adaptive Feature Integration
* Inspiration: Style transfer elements from BAID and CDF-based loss from TANet.
* Implementation: Implement an adaptive feature integration mechanism (similar to AdaIN) that aligns the features from the two pathways based on their relevance to the aesthetic task. However, instead of using style transfer directly, adapt the concept to align features based on aesthetic attributes.

Step 3: Self-Supervised Learning with Degradation and Reconstruction
* Inspiration: Degradation focus of BAID.
* Implementation: In the self-supervised phase, use manually degraded images (like in BAID) and task the network with reconstructing the original image. This phase would help the network learn robust features relevant to image quality and aesthetics.

Step 4: Supervised Fine-Tuning with Aesthetic Scores
* Inspiration: The use of detailed metrics in TANet.
* Implementation: After the self-supervised phase, fine-tune the network on a labeled dataset (like AVA) using aesthetic scores. Employ a loss function that accounts for both the distribution of scores (taking inspiration from TANet's CDF-based loss) and the accuracy of individual score predictions.

Step 5: Incorporate Attention Mechanisms
* Inspiration: Non-local blocks from BAID and the attention to detail in TANet.
* Implementation: Integrate attention mechanisms to allow the model to focus on salient regions of the image that are more likely to influence aesthetic judgments.

Step 6: Model Evaluation and Tuning
* Implementation: Use a combination of traditional metrics (like accuracy, MSE) and correlation-based metrics (like LCC, SRCC) for a comprehensive evaluation. Employ techniques like NAS (Neural Architecture Search) for optimizing the model architecture.

Step 7: Implementation of Efficient Training Strategies
* Implementation: Given the complexity of the proposed model, implement efficient training strategies like mixed precision training and progressive resizing of images.