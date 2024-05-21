# Identifying Deforestation in the Amazon Basin Using Satellite Imagery

Kieran Martin, Ivanna Rodríguez, Santiago Segovia

According to Global Forest Watch, the world loses an area of forest the size of 48 football fields every minute. Deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. Better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively. 

In this project, we investigated the use of deep learning models to identify deforestation in the Amazon Basin using satellite imagery from Kaggle's "Planet: Understanding the Amazon from Space" competition. By leveraging a dataset containing approximately 40,000 labeled images, we aimed to classify various atmospheric conditions and land cover/land use phenomena to better understand and monitor deforestation. The dataset included challenging factors such as cloud cover and imbalanced class distribution, for which we accounted for through careful preprocessing and model selection.

We trained several models, including custom architectures and fine-tuned pre-trained models like VGG16 and InceptionV3. Our custom models, Contender and Champion, demonstrated the effectiveness of deeper architectures and techniques like batch normalization in improving classification performance. The VGG16 model, known for its simplicity and depth, generally outperformed the InceptionV3 model in recognizing distinct classes like 'clear' (images with no cloud cover) and 'primary' (tree-covered land/forest). The F1 score was utilized to measure performance due to the multi-label nature of the data and the class imbalance. While both models showed improving trends in their composite F1 scores, VGG16 exhibited better generalization across epochs.

We explored segmentation tasks using the YOLOv8n-segmentation model and a custom U-Net model. The segmentation models aimed to identify deforestation by classifying forest and non-forest areas in satellite images. The U-Net model, with its encoder-decoder architecture, performed well in segmenting forest coverage, especially in clear and partly cloudy conditions. Despite some challenges with cloud-covered images, the U-Net model showed potential for accurate deforestation monitoring.

To enhance model robustness and address class imbalances, we proposed future work focusing on improving data quality, exploring additional architectures, and implementing advanced techniques like active learning. By continuously refining our models and leveraging more high-resolution labeled data, we can support more accurate and timely monitoring of deforestation in the Amazon Basin. Our work shows the potential of deep learning in environmental conservation, providing valuable insights for governments and stakeholders to respond more effectively to deforestation threats.

Our Jupyter notebooks with results can be found under classification and segmentation folders, respectively.


