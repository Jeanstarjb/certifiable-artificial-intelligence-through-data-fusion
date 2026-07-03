# Certifiable Artificial Intelligence Through Data Fusion

## Overview

This repository contains a Python/PyTorch implementation of the key concepts discussed in the research paper [Certifiable Artificial Intelligence Through Data Fusion](https://arxiv.org/pdf/2111.02001v1) by Erik Blasch, Junchi Bin, and Zheng Liu. The paper delves into the challenges and opportunities surrounding the certification of AI systems, focusing on the fusion of image data to enhance object recognition certifiability under various operational conditions. The implementation provided here demonstrates a proof-of-concept for integrating image data fusion into an AI model and analyzing its performance in a controlled setting.

## Core Concept

The paper addresses the following key points:
1. **Challenges in AI Certification**: There is a growing need for certifiable AI systems that can reliably perform under real-world conditions. Certification involves determining performance bounds and ensuring the system meets operational requirements.
2. **Data Fusion for Enhanced Certifiability**: Data fusion involves combining information from multiple sources to improve decision-making. In this context, the fusion of image data can enhance object recognition by leveraging complementary information, thereby improving certifiability.
3. **Performance Evaluation**: The paper introduces a notional use case where precision and object distance are considered to evaluate the certifiability of AI systems. 

This repository implements a simplified version of the proposed data fusion framework, focusing on fusing image data and analyzing its effect on object recognition performance.

## Features

- **Image Data Fusion Module**: Combines multiple input images to create a single, enriched representation for object recognition tasks.
- **Object Recognition Pipeline**: Demonstrates the effect of data fusion on object detection precision and performance across different conditions.
- **Evaluation Metrics**: Measures precision and other metrics relative to object distance, as highlighted in the paper.

## Repository Structure

```
├── data/
│   ├── raw/              # Raw input images
│   ├── processed/        # Processed and fused images
├── fusion/
│   ├── fusion_module.py  # Core implementation of the data fusion logic
│   ├── utils.py          # Utility functions for preprocessing and visualization
├── models/
│   ├── object_recognition.py  # Object recognition model implementation
│   ├── evaluation.py          # Performance evaluation scripts
├── notebooks/
│   ├── data_fusion_demo.ipynb # Jupyter Notebook demonstrating the pipeline
├── tests/
│   ├── test_fusion_module.py  # Unit tests for the fusion module
│   ├── test_models.py         # Unit tests for the object recognition model
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── run_pipeline.py            # Main script to run the full pipeline
```

## Getting Started

Follow these steps to set up and run the implementation.

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- Jupyter Notebook (optional, for interactive exploration)

To install the required Python packages, run:
```bash
pip install -r requirements.txt
```

### Quickstart

1. **Prepare the Data**:
   - Place raw input images in the `data/raw/` directory.

2. **Run the Data Fusion Pipeline**:
   - Execute the main script:
     ```bash
     python run_pipeline.py
     ```

3. **Explore Results**:
   - Open the Jupyter Notebook in `notebooks/data_fusion_demo.ipynb` for a step-by-step walkthrough of the pipeline and visualization of results.

### Example Command
```bash
python run_pipeline.py --input_dir data/raw/ --output_dir data/processed/ --evaluate
```

## Code Details

### 1. Data Fusion Module
The `fusion/fusion_module.py` file contains the implementation of the image fusion logic. The module combines multiple input images (e.g., different perspectives or sensor modalities) into a single output image that is used for downstream tasks like object recognition.

### 2. Object Recognition Model
The `models/object_recognition.py` file includes a simple convolutional neural network (CNN) architecture designed for object recognition. This model takes the fused image as input and predicts object classes.

### 3. Evaluation
The `models/evaluation.py` script evaluates the performance of the object recognition model. It computes precision and other metrics across varying object distances, demonstrating the relationship between data fusion and certifiability.

## Results

The results of the implementation align with the findings presented in the paper:
- **Improved Precision**: Fusing complementary image data enhances object recognition precision, especially for objects at greater distances.
- **Certifiability Insights**: The pipeline provides measurable performance bounds, which can be used to assess certifiability in real-world applications.

Sample results and visualizations are included in the `notebooks/data_fusion_demo.ipynb`.

## Contributing

Contributions are welcome! If you have ideas to enhance this implementation or find any issues, feel free to open an issue or submit a pull request.

## References

- Erik Blasch, Junchi Bin, Zheng Liu. "Certifiable Artificial Intelligence Through Data Fusion." arXiv preprint arXiv:2111.02001, 2021. [Link to Paper](https://arxiv.org/pdf/2111.02001v1)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.