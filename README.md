Very simple PyTorch Re-implementation of a  "Learning to Resize Images for Computer Vision Tasks" using the Imagenette and/Or ImageWoof dataset. The project was edited and run through Google Colab.


Original Paper: https://colab.research.google.com/drive/1HqHG9ZUBeYzfTjMj6GevgGom6Fc3HAMt#scrollTo=vFX_o357b8q2

Project Synopsis:
For this project, I explored the complexity of image resizing in computer vision, reimplementing the paper “Learning To Resize Images for Computer Vision Tasks”. My goal was to understand how advanced resizing techniques can enhance model training by creating optimized image inputs for machine learning algorithms.

My reimplementation focused on building a bilinear resizer with skip connections, addressing challenges in standard image resizing methods (such as bilinear and bicubic interpolation), which can limit model performance and increase training time. I used a ResNet-50 classifier and trained on a smaller dataset, ImageNette, due to resource constraints.

Results showed sustained loss and minor accuracy improvements over traditional static resizing, though with notable resource demands. To further understand the architecture, I studied supporting papers: “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows” and “Injecting the 3D World into Large Language Models”, providing insights into spatial resolution and feature extraction in vision tasks.

This project highlights how advanced resizing can marginally enhance model training but may require considerable resources for practical gains.
