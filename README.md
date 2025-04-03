# NASA-Blue Origin Lunar DDL Anomaly Detection

## Overview

This project implements Random Cut Forest (RCF) anomaly detection on NASA and Blue Origin's demonstration of lunar Deorbit, Descent, and Landing Sensors (BODDL-TP) data using Amazon SageMaker. The analysis focuses on detecting anomalies in spacecraft dynamics data, including positions, velocities, and quaternion orientations.

## Features

- Data preprocessing and cleaning
- Random Cut Forest model training using Amazon SageMaker
- Batch anomaly detection for large datasets
- Visualization of results with highlighted anomalies
- S3 integration for data storage and plot uploads

## Prerequisites

- AWS Account with appropriate permissions
- Amazon SageMaker access
- Python 3.7 or later
- Boto3, Pandas, Matplotlib, NumPy, and SageMaker Python SDK

## Installation

1. Clone the repository:

`git clone https://github.com/aws-samples/sample-sagemaker-ai-rcf-anomaly-detection-lunar-spacecraft.git`
`cd sample-sagemaker-ai-rcf-anomaly-detection-lunar-spacecraft`


3. Install required packages:
`bash pip install -r requirements.txt'

## Architecture

![Architecture Diagram](/DDL_RCF_architecture.png)

This architecture implements anomaly detection for NASA-Blue Origin Lunar DDL data using Amazon SageMaker's Random Cut Forest algorithm. Here's how it works:

1. **Data Flow**:
   - Public DDL Data is stored in an S3 bucket
   - JupyterLab accesses this data through a SageMaker Domain
   - A JupyterLab Notebook processes the data and implements anomaly detection

2. **Processing Pipeline**:
   - JupyterLab Notebook trains a Random Cut Forest (RCF) model
   - The model is deployed to a SageMaker Endpoint
   - Anomaly detection is performed on the data
   - Results and visualizations are stored in S3

3. **Output**:
   - Anomaly data and plots are saved to S3
   - Training and model output data is preserved in S3
   - The system maintains VPC security throughout the process

## Usage

1. Configure AWS credentials and region.

2. Update the `bucket_name` and `file_name` variables in the script with your S3 bucket and data file names.

3. Run the script:
`bash python nasa_ddl_anomaly_detection.py

4. The script will:
- Load and preprocess the data
- Train and deploy a Random Cut Forest model
- Detect anomalies in the data
- Generate and upload plots to S3

## Code Structure

- `AnomalyDetector` class: Main class for data processing, model training, and anomaly detection
- `load_and_prepare_data`: Data loading and preprocessing
- `train_and_deploy_model`: RCF model training and deployment
- `predict_anomalies`: Anomaly detection using the trained model
- `plot_results`: Visualization of results
- `upload_plot_to_s3`: Uploading generated plots to S3

## Configuration

Adjust the following parameters in the script as needed:
- `threshold_percentile`: Threshold for anomaly classification
- RCF hyperparameters in `train_and_deploy_model`
- `batch_size` in `predict_anomalies` for large datasets

## Data

The script uses public NASA-Blue Origin Demo of Lunar Deorbit, Descent, and Landing Sensors (BODDL-TP) data (https://data.nasa.gov/Aerospace/Blue-Origin-Demo-of-Deorbit-Descent-and-Landing-Se/nj3a-8wq3/about_data). Ensure your data is in the correct format with columns for timestamps, positions, velocities, and quaternions.

## Results

The script generates plots for:
- Positions (CON_ECEF)
- Velocities (CON_ECEF)
- Quaternions (CON2ECEF)

Anomalies are highlighted in red on the plots. Plots are saved to the specified S3 bucket.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
