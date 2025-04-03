"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This sample code is made available under the MIT-0 license. See the LICENSE file.

This sample code demonstrates Random Cut Forest (RCF) anomaly detection using Amazon SageMaker
on NASA-Blue Origin's lunar Deorbit, Descent, and Landing Sensors (BODDL-TP) data.

Data source: https://techport.nasa.gov/view/116144
"""

import tempfile
import os
import io
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from botocore.exceptions import EndpointConnectionError, ClientError
from datetime import datetime

import boto3
from botocore.config import Config

# Configure S3 client with explicit endpoint configuration
config = Config(
    s3={'addressing_style': 'virtual'},
    region_name='us-east-1',  # replace with your region if different
    retries = dict(
        max_attempts = 10
    )
)

s3 = boto3.client('s3', config=config)

# Try to access bucket and list contents for analysis
bucket_name = 'sample-nasa-ddl-data'
try:
    response = s3.list_objects_v2(Bucket=bucket_name)
    print(f"\nContents of {bucket_name}:")
    for obj in response.get('Contents', []):
        print(f"  {obj['Key']}")
except Exception as e:
    print(f"Error accessing bucket: {str(e)}")
    
class AnomalyDetector:
    def __init__(self, bucket_name, csv_file):
        self.bucket_name = bucket_name
        self.csv_file = csv_file
        self.temp_dir = tempfile.mkdtemp(prefix='rcf_analysis_')
        self.local_path = os.path.join(self.temp_dir, 'truth.csv')        
        self.s3_client = boto3.client('s3')
        self.session = Session()
        self.role = get_execution_role()
        self.df_cleaned = None
        self.train_data = None
        self.rcf_inference = None
        
    def load_and_prepare_data(self):
        """Load and prepare the data for analysis"""
        print("Loading data")
        
        try:
            # Download data from S3
            self.s3_client.download_file(self.bucket_name, self.csv_file, self.local_path)
            
            # Create and clean dataframe
            df = pd.read_csv(self.local_path)

            # Clean column names by removing trailing spaces
            df.columns = df.columns.str.strip()
            
            print("\nCleaned columns in the dataset:")
            print(df.columns.tolist())

            self.df_cleaned = df.dropna()
            
            # Prepare training data
            data = self.df_cleaned.values.astype('float32')
            self.train_data = data.reshape(-1, 10)
            
            return True
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            return False

    def train_and_deploy_model(self):
        """Train and deploy the Random Cut Forest model"""
        print("Training the model and deploying")
        
        try:
            # Create and write to temporary file
            tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
            pd.DataFrame(self.train_data).to_csv(tmp_file, header=False, index=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_file.close()
            train_data_path = tmp_file.name  # Get name after closing

            try:
                train_s3_prefix = 'sagemaker/rcf'
                train_s3_uri = self.session.upload_data(train_data_path, 
                                                      bucket=self.bucket_name, 
                                                      key_prefix=train_s3_prefix)
                
                # Setup RCF model
                rcf_container = get_image_uri(boto3.Session().region_name, 'randomcutforest')
                rcf = sagemaker.estimator.Estimator(
                    rcf_container,
                    self.role,
                    instance_count=1,
                    instance_type='ml.m5.4xlarge',
                    output_path=f's3://{self.bucket_name}/{train_s3_prefix}/output',
                    sagemaker_session=self.session
                )

                # Set hyperparameters
                rcf.set_hyperparameters(
                    num_samples_per_tree=512,
                    num_trees=100,
                    feature_dim=10
                )

                # Train model
                rcf.fit({'train': sagemaker.inputs.TrainingInput(
                    s3_data=train_s3_uri,
                    content_type='text/csv;label_size=0',
                    distribution='ShardedByS3Key'
                )})

                # Deploy model
                self.rcf_inference = rcf.deploy(
                    initial_instance_count=1,
                    instance_type='ml.m5.4xlarge'
                )
                
                return True

            finally:
                # Clean up temporary file
                if os.path.exists(train_data_path):
                    os.unlink(train_data_path)

        except Exception as e:
            print(f"Error in model training/deployment: {str(e)}")
            return False


    def predict_anomalies(self, threshold_percentile=0.90):
        """Predict anomalies using the deployed model"""

        print("Predicting anomalies")
        
        try:
            predictor = Predictor(
                endpoint_name=self.rcf_inference.endpoint_name,
                serializer=CSVSerializer(),
                deserializer=JSONDeserializer()
            )

            # Make predictions
            anomaly_scores = []
            for i in range(len(self.train_data)):
                result = predictor.predict(self.train_data[i].reshape(1,-1))
                anomaly_scores.append(result['scores'][0])

            # Process results
            self.df_cleaned['anomaly_score'] = [x['score'] if isinstance(x, dict) else x 
                                              for x in anomaly_scores]

            # Identify anomalies
            threshold = self.df_cleaned['anomaly_score'].quantile(threshold_percentile)
            self.df_cleaned['anomaly'] = self.df_cleaned['anomaly_score'] > threshold
            
            return self.df_cleaned[self.df_cleaned['anomaly'] == True]
            
        except Exception as e:
            print(f"Error in anomaly prediction: {str(e)}")
            return None
            
    def upload_plot_to_s3(self, plt_figure, plot_name):
        """Upload a matplotlib figure to S3"""
        print("Uploading plots to S3")
        
        try:
            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt_figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)

            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f'visualizations/{plot_name}_{timestamp}.png'

            # Upload to S3
            self.s3_client.upload_fileobj(
                buf,
                'sample-nasa-ddl-data',  # Using the specified bucket
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            print(f"Successfully uploaded plot to s3://sample-nasa-truth-data/{s3_key}")
            return f"s3://sample-nasa-truth-data/{s3_key}"
        except Exception as e:
            print(f"Error uploading plot to S3: {str(e)}")
            return None
            
    def plot_results(self, anomalies, data_type='positions'):
        """Plot the results with anomalies highlighted and upload to S3"""
        print("Plotting results")
        
        try:
            # Create figure
            fig = plt.figure(figsize=(18, 10))
            
            # Define column patterns and labels based on data type
            if data_type == 'positions':
                columns = [col for col in self.df_cleaned.columns if 'truth_pos' in col]
                title = 'Positions'
                y_label = 'Position (CON_ECEF)'
            elif data_type == 'velocities':
                columns = [col for col in self.df_cleaned.columns if 'truth_vel' in col]
                title = 'Velocities'
                y_label = 'Velocity (CON_ECEF)'
            elif data_type == 'quaternions':
                columns = [col for col in self.df_cleaned.columns if 'truth_quat' in col]
                title = 'Quaternions'
                y_label = 'Quaternion (CON2ECEF)'
            
            # Check columns
            if not columns:
                print(f"No columns found for data_type: {data_type}")
                return
                
            print(f"\nPlotting columns for {data_type}:")
            print(columns)
            
            # Create main axis for data
            ax1 = plt.gca()
            
            colors = ['blue', 'green', 'yellow', 'cyan']
            
            # Plot time series
            for col, color in zip(columns, colors):
                ax1.plot(self.df_cleaned.index/100, self.df_cleaned[col], label=col, color=color)
                ax1.scatter(anomalies.index/100, anomalies[col], color='red', marker='x')
            
            # Set labels for main axis
            ax1.set_xlabel('Time (cs)')
            ax1.set_ylabel(y_label)
            
            # Create secondary axis for anomaly scores
            ax2 = ax1.twinx()
            ax2.plot(self.df_cleaned.index/100, self.df_cleaned['anomaly_score'], 
                    label='Anomaly Scores', color='purple', linestyle='--')
            ax2.set_ylabel('Anomaly Score')
    
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
            plt.title(f'Anomaly Detection using Random Cut Forest (RCF) for {title}')
            
            # Upload plot to S3
            s3_uri = self.upload_plot_to_s3(fig, f'rcf_anomaly_{data_type.lower()}')
            
            # Display plot
            plt.show()
            plt.close()
            
            return s3_uri
            
        except Exception as e:
            print(f"Error in plotting results: {str(e)}")
            print(f"Data type: {data_type}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            # Delete SageMaker endpoint if it exists
            if self.rcf_inference:
                self.rcf_inference.delete_endpoint()
                print("Endpoint deleted successfully")
    
            # Clean up temporary directory and its contents
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    # Walk through directory and remove all files and subdirectories
                    for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                        for name in files:
                            os.unlink(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    
                    # Remove the temp directory itself
                    os.rmdir(self.temp_dir)
                    print("Temporary directory cleaned up successfully")
                
                except Exception as e:
                    print(f"Error during directory cleanup: {str(e)}")
    
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def main():
    try:     
        bucket_name = 'sample-nasa-ddl-data'

        print("Running main")
        
        # Downloaded public sample data
        # NASA-Blue Origin Demo of Lunar Deorbit, Descent, and Landing Sensors (BODDL-TP)
        # https://techport.nasa.gov/view/116144
        file_name = 'lunar_ddl_truth_80k.csv'
        
        # Initialize detector
        detector = AnomalyDetector(bucket_name, file_name)
        
        # Run analysis
        if detector.load_and_prepare_data():
            # Print DataFrame info to check columns
            print("\nDataFrame Info:")
            print(detector.df_cleaned.info())
            
            if detector.train_and_deploy_model():
                anomalies = detector.predict_anomalies()
                if anomalies is not None:
                    for data_type in ['positions', 'velocities', 'quaternions']:
                        detector.plot_results(anomalies, data_type)
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()