II.b Segmentation
Methodology

The primary objective was to identify deforestation through the presence of agricultural areas, as forests are often cleared for farming. We had to manually tag a group of images because the dataset was not tagged for segmentation. Initially, we planned to maintain the same classes, such as forest coverage, agriculture coverage, water, and roads, etc. Given our specific interest in forest coverage, we simplified the classes into a binary system: forest and non-forest. Man-made objects like mines and agriculture were grouped under the non-forest class, as they all indicate deforestation. This allowed us to apply a binary mask to the segmentation training images, streamlining the training process.

To optimize the training process, we aimed to include images with as many distinct classes as possible, ensuring a balance of clear, partly-cloudy, and hazy images. This approach was intended to help the model learn to identify deforestation under various weather conditions. We selected 200 images to set aside for tagging based on these criteria. Out of the total number of training images, 153 had the highest number of tags (7-9 tags). We created a separate data frame for these images to analyze their class distribution. To achieve a balanced representation of all classes, we first evaluated the class distribution among these images. Since clear images were overrepresented, we randomly excluded some clear-tagged images, ensuring that images with tags that were severely underrepresented throughout the entire dataset (e.g., 'bare_ground', 'artisanal_mine', 'slash_burn', 'selective_logging', 'conventional_mine', 'blooming') were retained.

To build a dataset of around 200 images with balanced weather condition tags, we incorporated images from the pool with the next highest number of tags per image (6 tags). Out of the 996 images with 6 tags, we randomly selected those with underrepresented tags to ensure a balanced representation of approximately ⅓ clear, ⅓ partly-cloudy, and ⅓ hazy conditions. This approach ensured that our training dataset had a diverse and balanced representation of weather conditions, crucial for the model to accurately learn and identify deforestation.

A total of 47 training images and 42 validation images were manually tagged using LabelMe. During tagging, a Python function was created to take the image name as an argument and return the list of classes tagged in that image, aiding in accurate observation and tagging.

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure10.jpg" alt="Figure 10" style="width:50%">
  <figcaption>Figure 10: An example image showing the output of the project.</figcaption>
</figure>

<p></p>


**Limitations**

Some images were too obscure to tag accurately, particularly those labeled as "Haze" as shown above. We tried to include as many cloudy and hazy images as possible, but in some cases, it was impossible to discern where the forest started and stopped as shown in Figure 10.

For future research, the tagging process could involve capturing three examples of the same coordinates under different weather conditions: clear, cloudy, and hazy. However, this approach may face challenges if we need to detect deforestation day by day. For instance, if there is significant forest coverage loss over a week of clear weather, it would be difficult to use these clear images as ground truth when it becomes hazy. Additionally, using segmentation on non-clear images might be less effective. If clouds obscure areas where trees have been recently cut down, it would be impossible for both humans and machines to observe the changes, as all pixels in the area would appear white or light gray.

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure11.png" alt="Figure 11" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure11.png">Figure 11</a></figcaption>
</figure>

<p></p>

Figure 11 shows the tagging process on a given image. The main areas of forestation are captured, however it is hard to tell if there are other trees present in the untagged areas or if there are other types of vegetation.
Pretrained Model: YOLOv8n-Segmentation

The YOLOv8n-segmentation pretrained model algorithm involves a convolutional neural network designed for object detection and segmentation. It consists of multiple convolutional layers followed by batch normalization and activation functions. The network uses anchor boxes to predict bounding boxes and class probabilities for each object in the image. For segmentation, it incorporates additional layers to output pixel-wise masks for each detected object. The model is trained on a large, diverse dataset, enabling it to generalize well across various segmentation tasks. It optimizes the detection and segmentation outputs using a combination of loss functions, including classification, localization, and mask losses.







**Results**
<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure12.png" alt="Figure 12" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure12.png">Figure 12</a></figcaption>
</figure>

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure13.png" alt="Figure 13" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure13.png">Figure 13</a></figcaption>
</figure>

<p></p>

The BoxF1 curve evaluates an object detection model's performance based on bounding box predictions, measuring how well the model detects and localizes objects. It considers precision (correctly predicted positive bounding boxes to total predicted) and recall (correctly predicted positive bounding boxes to total ground truth). Figure 12 shows the BoxF1 curve with a maximum F1 score of around 0.36 at a confidence level of 0.239, indicating the best balance between precision and recall.

The performance of a segmentation model is assessed by the MaskF1 curve by measuring the F1 score based on pixel-wise predictions for object masks. Figure 13 shows the MaskF1 curve where the F1 score reaches a maximum of around 0.38 at a confidence level of 0.322. This indicates the best trade-off between precision (correctly predicted positive pixels to total predicted) and recall (correctly predicted positive pixels to total ground truth). An F1 score of 0.38 suggests moderate performance, indicating that while the model is capable of segmenting objects to some extent, there is room for improvement in both precision and recall. 

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure14.png" alt="Figure 14" style="width:70%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure14.png">Figure 14. Yolov8n-Seg Confusion Matrix</a></figcaption>
</figure>

<p></p>

Figure 14 shows the model yeilded a high rate of accuracy in detecting the 'background' class(100%) but struggles significantly with detecting the 'primary' class at only a 34% True positive rate. The model incorrectly identifies the 'primary' class as 'background' 66% of the time. The model's overall performance is heavily skewed towards correctly predicting 'background' instances, likely due to class imbalance or insufficient distinguishing features in the training data for the 'primary' class. Data augmentation, and further tuning of model hyperparameters can help with increasing the true positive rate. 

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure15.png" alt="Figure 15" style="width:80%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure15.png">Figure 15. Yolov8n-Seg Pretrained Model metrics</a></figcaption>
</figure>

<p></p>

The training and validation loss plots for the YOLOv8n-segmentation model indicate some instability during the training process. The training losses for box, segmentation, classification, and DFL all show a general downward trend, reflecting effective learning. The validation loss for segmentation makes a couple of significant drops early in the training process but then plateaus around 4%. This plateau suggests that while the model initially improves, it struggles to achieve further gains in segmentation performance. Performance metrics such as precision, recall, and mean Average Precision (mAP) for both bounding boxes (B) and masks (M) improve over time but display fluctuations, further indicating inconsistency. This instability in validation loss and performance metrics suggested that the training process may need additional refinement to stabilize and enhance the model's performance.
Custom Yolov8n-Segmentation Model:

To improve the performance of the YOLOv8n-segmentation model for identifying deforestation in satellite images, several modifications were made. The initial learning rate of 0.01, which caused unstable training, was adjusted to 0.001 for better convergence. The optimizer was switched to AdamW to enhance weight updates and generalization. The batch size was increased from 8 to 16 to stabilize gradient updates, and a dropout rate of 0.3 was introduced to prevent overfitting. A weight decay of 0.0005 was applied to penalize large weights and improve generalization. Data augmentation techniques such as random rotations, scaling, brightness/contrast adjustments, and Gaussian noise were applied using Albumentations to increase the model's robustness to variations in satellite imagery. These modifications aimed to enhance the model's accuracy and generalization capabilities in detecting deforestation by leading to a more stable and effective training process.

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure16.png" alt="Figure 16" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure16.png">Figure 16. BoxF1 curve</a></figcaption>
</figure>

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure17.png" alt="Figure 17" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure17.png">Figure 17. MaskF1 curve</a></figcaption>
</figure>

<p></p>

Comparing the pretrained and modified model results, we observe some differences. For the modified model, Figure 16 shows BoxF1 curve for all classes showed a slight improvement with an F1 score of 0.39 at a lower confidence level of 0.220. Figure17 shows the MaskF1 curve also improved slightly to an F1 score of 0.37 at a confidence level of 0.211. These changes suggest that the modifications to the model slightly enhanced its precision and recall trade-off, resulting in improved F1 scores at different confidence levels.

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure18.png" alt="Figure 18" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure18.png">Figure 18. Yolov8n-Seg Custom Model metrics</a></figcaption>
</figure>

<p></p>

The modified YOLOv8n-segmentation model shows improved stability in the training losses (box_loss, seg_loss, cls_loss, dfl_loss) with smoother declines, indicating better learning. Figure 18 displays the custom models training metrics where the precision and recall for both bounding boxes (B) and masks (M) exhibit a slight increase, reflecting more accurate detections and segmentations. The mAP50 and mAP50-95 values for both bounding boxes and masks also demonstrate an improvement, suggesting that the modifications enhanced the model's overall performance in identifying deforestation in satellite images.

Despite the modifications to the YOLOv8n-segmentation model, the normalized confusion matrices for the pretrained and modified models remained identical. Therefore, while the loss metrics and mean average precision (mAP) scores suggest some improvements in the model's training dynamics and performance, these did not translate into a noticeable change in the classification accuracy as represented by the confusion matrices.

<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure19.png" alt="Figure 19" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure19.png">Figure 19. Custom Yolov8n-Seg on Unseen Data</a></figcaption>
</figure>

<p></p>

The custom yolov8n-segmentation model performed very poorly on unseen data as pictured in Figure 19. For test images 4, 41, and 1, which were entirely forest coverage, the model failed to classify them as 100% primary, notably missing all primary in test image 1. In test image 7, which had only a small area of forest in the top right corner, the model over classified most of the image as primary. Test image 21 had accurate detection of forest areas but incorrectly classified much of the river as primary. In test image 26, the model struggled significantly, unable to clearly distinguish between primary and non-primary areas. This indicates that while the model can identify forest coverage, it requires further optimization to improve its accuracy and reduce false positives in mixed-content images.

Challenger: Custom UNet

In identifying deforestation in satellite images, our goal was purely segmentation. Given that Yolov8n-segmentation identifies objects and segments them, we hypothesized that UNet would be more accurate for our task. U-Net, a convolutional neural network architecture designed for biomedical image segmentation, has been successfully applied to other segmentation tasks. It features a contracting path (encoder) and an expansive path (decoder), allowing precise localization by combining high-level features with low-level features. UNet performs pixel-wise segmentation, essential for dense satellite images of forest coverage where distinct objects are not present. UNet was expected to perform better as it processes the entire image and produces a segmentation map assigning a class label to each pixel, unlike Mask R-CNN, which is designed for instance segmentation.

The custom U-Net model for satellite image segmentation uses an encoder-decoder architecture to classify each pixel. The contracting path captures context by downsampling the input, while the expansive path upscales and combines feature maps to generate a high-resolution segmentation mask. For our binary segmentation task, the model was trained with a binary cross-entropy loss function. The UNet model includes six downsampling and five upsampling blocks, uses dropout with a 50% probability before the final output, and incorporates batch normalization and ReLU activations for improved performance and stability. The final model uses a learning rate of 0.0001 with the Adam optimizer, a batch size of 4, and employs a Cosine Annealing learning rate scheduler. Early stopping with a patience of 10 epochs was also implemented. A small batch size of 4 was used for computational efficiency and to introduce noise in gradient calculation, promoting better generalization.

In subsequent tests, the batch size was increased to 8 to attempt to improve gradient estimation and stabilize training. However, a larger batch size led to worse model performance, as indicated by increased validation loss and decreased accuracy. After testing learning rates of 0.001 and 0.005, we found that a learning rate of 0.0001 and introducing a learning rate scheduler led to finer adjustments in the model parameters during training, improving overall stability and performance.

Data Transformations:

To preprocess the satellite images and corresponding segmentation masks for the U-Net model, we used the torchvision.transforms module. Images and masks were resized to 256x256 pixels to ensure uniform dimensions for batch processing, balancing computational efficiency and detail accuracy. The transforms.Resize((256, 256)) function was used for resizing, and transforms.ToTensor() was applied to convert the images and masks into PyTorch tensors, normalizing pixel values and adjusting the shape to the format expected by the U-Net model.

We initially incorporated random horizontal flip, random vertical flip, and random rotation (10 degrees) to help the model learn spatial invariance and improve generalization. However, these augmentations introduced excessive noise and unrealistic variations, destabilizing the model. Removing the additional transformations led to a better-calibrated model, shifting the optimal threshold for predictions from 0.7 to around 0.3, indicating improved probability outputs.


<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure20.png" alt="Figure 20" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure20.png">Figure 20. Custom U-Net Model Confusion Matrix</a></figcaption>
</figure>

<p></p>


Figure 20 shows that the final model correctly identified 73.83% of the primary class (forest coverage) and 75.12% of the background class. However, there were still misclassifications, with a false positive rate of 26.17% for the primary class and a false negative rate of 24.88% for the background class. These results suggest that while the model has improved significantly, there is still room for further enhancement, particularly in reducing false positives and negatives.


<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure21.png" alt="Figure 21" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure21.png">Figure 21. Custom U-Net Model Training Metrics</a></figcaption>
</figure>

<p></p>


Figure 21 shows the training and validation accuracy and loss over 34 epochs. The training accuracy steadily improves, reaching approximately 80%, while the validation accuracy fluctuates but stabilizes around 70-75%. The training loss consistently decreases, indicating effective learning, but the validation loss exhibits fluctuations, suggesting some instability and potential overfitting during certain epochs.

Through the various improvements described, including adjusting hyperparameters, implementing data augmentations, and utilizing a learning rate scheduler, the U-Net model's performance for forest coverage segmentation in satellite images has significantly improved. Initial models achieved validation accuracy of 58-72%. The final model achieved a validation accuracy of approximately 75% with the most stable training dynamics out of all runs. Additionally the final model performed the best on unseen data as shownin Figure 22.


<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure22.png" alt="Figure 22" style="width:50%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure22.png">Figure 22. Custom U-Net Model on Unseen Data</a></figcaption>
</figure>

<p></p>


Figure 22 shows that test images 41, 21, and 1 were nearly perfect in accuracy. The results for test image 7 are particularly interesting due to the difference between thresholds 0.2 and 0.3. While the threshold of 0.3 correctly identified the primary areas, the spots classified as primary at a threshold of 0.2 might be trees or something else, making it challenging to interpret. Test image 4 demonstrates that the model struggles with cloud coverage. As mentioned in the limitations, it is unclear what lies beneath the clouds, making it difficult to determine if the area is forested or not, especially for new data outside our current test set. Test image 1, which is clear and entirely primary, required a threshold of 0.7 for accurate classification. This likely occurs because the model was not trained on images that are purely primary, making it less confident in identifying forest coverage without a comparative context. Test image 26 performed fairly well, considering the variety of classes present in the image.


III. Model Operations
<figure>
  <img src="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure23.png" alt="Figure 23" style="width:80%">
  <figcaption><a href="https://github.com/ssegovba/identifying-deforestation/blob/main/segmentation/figures/figure23.png">Figure 23. MLOps Pipeline</a></figcaption>
</figure>

<p></p>

To deploy our locally trained model and utilize it for predicting new satellite images of the Amazon rainforest, we will use AWS services to create a robust and scalable architecture visualized in Figure 23. First, we will upload our saved model to an Amazon S3 bucket dedicated to storing model artifacts. This bucket will serve as the source for the model during deployment. Using AWS SageMaker, we will create a model object that points to the model stored in the S3 bucket and deploy this model to a SageMaker endpoint, allowing real-time predictions. We will also create a separate S3 bucket to collect new satellite images as they become available. To automate the transfer of these images, we will use AWS DataSync to schedule and automate the syncing of new satellite images from local storage to the S3 bucket, ensuring that the latest data is always available without manual intervention. To streamline the deployment process, we will set up a CI/CD pipeline using AWS CodePipeline and CodeCommit, ensuring any validated code changes are seamlessly integrated and deployed. AWS is the best option for this due to its ability to handle large-scale data with services like S3 and SageMaker, which provide seamless integration for storing and processing high-resolution images. Additionally, 

For model maintenance and parameter updates with untagged new data, we will implement an active learning approach. We will run new untagged images collected in S3 through the deployed model to generate initial predictions and use Amazon SageMaker Ground Truth to have human experts review and correct a subset of these predictions, creating a labeled dataset. This labeled data will be stored in a third S3 bucket dedicated to tagged data. AWS Lambda functions will automate the retraining process, triggering SageMaker to retrain the model when new labeled data is available in the tagged data S3 bucket. During the initial training and retraining processes, we will select appropriate GPU instances such as ml.p3.8xlarge with 4 NVIDIA V100 GPUs and ml.p4d.24xlarge with 8 NVIDIA A100 GPUs, which provide substantial memory and computational power required for efficientiently processing high resolution satellite images and running complex models. Additionally, if the inference phase requires significant computational power due to the complexity of the models, we will also use GPU instances for the SageMaker endpoint. 

Monitoring will be conducted using Amazon CloudWatch to track computational statistics Amazon SageMaker Model Monitor to track statistical metrics. Computational monitoring will include CPU/GPU usage and latency, ensuring continuous assessment of the model’s resource utilization. Statistical monitoring will track the input data distributions, output predictions, precision, recall, and F1 scores, helping identify any potential data drift or model performance degradation. This active learning loop, combined with robust automation tools like DataSync for continuous data transfer and Lambda for real-time processing, ensures the model is regularly updated and retrained to maintain accuracy and performance.

To ensure the robustness of our model updates, we will utilize shadow deployment. In a shadow deployment, new data is sent simultaneously to both the new and old models, and their results are compared to ensure the new model performs as expected. This approach eliminates the risk of deploying a new model that might not work correctly, as any discrepancies between the models can be identified and addressed before fully switching to the new model. Although shadow deployment uses double the resources since both models run in parallel, it provides a crucial safety net. Monitoring environmental changes in the Amazon rainforest requires high accuracy and reliability, as any errors can lead to misinformed decisions about conservation and resource management. Shadow deployment allows us to rigorously validate model updates, ensuring that our predictions remain accurate and trustworthy.

