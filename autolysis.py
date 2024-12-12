# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "requests",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "statsmodels",
#   "scikit-learn"
# ]
# ///

import os
import requests
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import json
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import re
import base64

warnings.filterwarnings("ignore", category=UserWarning)


# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <dataset>")
    sys.exit(1)

# Extract the file path
file_path = sys.argv[1]

# The name of folder where outputs are to be saved
folder_name = sys.argv[1].split('.')[0]

# URL of AI-Proxy
api_url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN') # AI Proxy Token taken from environment
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Get the folder ready to save outputs
if not os.path.exists(folder_name):
    os.mkdir(folder_name)


def load_csv(file_path: str):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file to be loaded.

    Returns:
        pd.DataFrame or None: Returns the loaded DataFrame if successful, or None if an error occurs.
    """
    try:
        # Attempt to load with UTF-8 encoding
        data = pd.read_csv(file_path, encoding="utf-8")
        return data
    except UnicodeDecodeError:
        print("Warning: UTF-8 decoding failed. Trying with 'latin1' encoding.")
        try:
            # Fallback to Latin-1 encoding
            data = pd.read_csv(file_path, encoding="latin1")
            return data
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            return None

def check_numeric_data_type(data: pd.DataFrame):
    """
    Analyzes numeric columns in a DataFrame to determine whether they are discrete or continuous.

    Args:
        data (pd.DataFrame): DataFrame containing numeric columns to analyze.

    Returns:
        dict: A dictionary with column names as keys and their data type (Discrete/Continuous) as values.
    """
    results = {}

    # Iterate over numeric columns
    for column in data.select_dtypes(include=['number']).columns:
        column_info = {}

        # Check if the column contains only integer values (discrete)
        if data[column].apply(lambda x: x.is_integer()).all():
            column_info["type"] = "Discrete (Integer-based)"
        else:
            column_info["type"] = "Continuous"

        results[column] = column_info

    return results

def correlation(numeric_cols: pd.DataFrame, api_url: str, AIPROXY_TOKEN: str, path: str):
    """
    Computes the correlation matrix for selected numeric columns and saves a heatmap to the specified path.
    
    Args:
        numeric_cols (pd.DataFrame): DataFrame containing numeric columns.
        api_url (str): URL for the AI proxy API to get meaningful columns.
        AIPROXY_TOKEN (str): API token for authentication.
        file_path (str): Directory path where the heatmap image will be saved.
        
    Returns:
        pd.DataFrame: Correlation matrix for the selected numeric columns.
    """
    
    def get_meaningful_columns(content: str, api_url: str, AIPROXY_TOKEN: str):
        """
        Helper function to interact with the AI model and get meaningful columns for the correlation matrix.
        
        Args:
            content (str): Comma-separated column names as input for the AI model.
            api_url (str): URL for the AI proxy API to get meaningful columns.
            AIPROXY_TOKEN (str): API token for authentication.
            file_path (str): Directory path where the heatmap image will be saved.  
        
        Returns:
            str: Comma-separated meaningful columns suggested by the model.
        """
        
        # Define API Prompt
        prompt = {
            'model': 'gpt-4o-mini',
            "response_format": { "type": "text" },
            'messages': [
                {'role': 'system', 'content': 'Return comma-separated  2,3 or 4 column names (Avoid the initial columns), for a meaningful correlation matrix'},
                {'role': 'user', 'content': content}
            ]
        }

        try:
            # Call the API
            response = requests.post(
                api_url, 
                headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
                json=prompt
            )
            response.raise_for_status() # Check for successful execution
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        
        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return ''

    def compute_correlation_matrix(columns: str):
        """
        Helper function to compute the correlation matrix of the specified columns.
        
        Args:
            columns (str): Comma-separated column names to include in the correlation matrix.
        
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        column_list = columns.split(', ')
        filtered_cols = numeric_cols[column_list]
        return filtered_cols.corr()

    numeric_cols = numeric_cols.dropna()

    # Get all column names from the numeric columns
    all_columns = ', '.join(numeric_cols.columns)

    # Retrieve meaningful columns using the AI model
    meaningful_columns = get_meaningful_columns(all_columns, api_url, AIPROXY_TOKEN)

    # If no meaningful columns returned, retry with a different prompt format
    if not meaningful_columns:
        meaningful_columns = get_meaningful_columns(f'Columns: {all_columns}', api_url, AIPROXY_TOKEN)

    # Compute the correlation matrix using either meaningful or all columns
    if meaningful_columns:
        correlation_matrix = compute_correlation_matrix(meaningful_columns)
    else:
        correlation_matrix = compute_correlation_matrix(all_columns)

    # Create and save the correlation heatmap
    plt.figure(figsize=(5.2, 5.2), dpi=100)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, ha="right", fontsize=10) 
    plt.title("Correlation Matrix Heatmap", fontsize=14)
    plt.tight_layout()

    # Define the path to save the heatmap
    heatmap_path = os.path.join(path, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    print(f"Correlation heatmap saved to {heatmap_path}")
    
    return correlation_matrix


def basic_analysis(api_url: str, AIPROXY_TOKEN: str, path: str, data: pd.DataFrame):
    """
    Performs basic analysis on a dataset, including summary statistics, 
    correlation matrix, outlier detection, missing value analysis, 
    and numeric data type checks.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        api_url (str): URL for the AI proxy API to get meaningful columns.
        AIPROXY_TOKEN (str): API token for authentication.
        file_path (str): Directory path where the heatmap image will be saved.

    Returns:
        dict: 
            - Summary statistics (pd.DataFrame)
            - Correlation matrix (pd.DataFrame or None)
            - Outliers data (dict or None)
            - Missing values (dict)
            - Numeric data types (dict)
    """
    print('Starting basic analysis...')
    # Initialize outputs
    correlation_matrix = None
    outliers_data = None
    numeric_data_types = None

    analysis_dict = {}
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 1:  # Only calculate if multiple numeric columns exist
        correlation_matrix = correlation(numeric_cols, api_url, AIPROXY_TOKEN, path)
        analysis_dict['Correlation'] = correlation_matrix

        outliers_data = {col: int((numeric_cols[col] > (numeric_cols[col].mean() + 3 * numeric_cols[col].std())).sum() +
                                   (numeric_cols[col] < (numeric_cols[col].mean() - 3 * numeric_cols[col].std())).sum())
                         for col in numeric_cols.columns}
        analysis_dict['Number of Outliers'] = outliers_data

        numeric_data_types = check_numeric_data_type(numeric_cols)
        analysis_dict['Numeric Data Types (Discrete or Continuous)'] = numeric_data_types

    # Analyze missing values
    missing_values = {col: int(data[col].isnull().sum()) for col in data.columns if data[col].isnull().sum() > 0}
    analysis_dict['Missing Values'] = missing_values

    # Summary statistics
    summary_stats = data.describe(include='all').transpose()
    analysis_dict['Data Description'] = summary_stats

    return analysis_dict

def convert_to_json(analysis_dict: dict):
    """
    Converts the analysis dictionary into a JSON-compatible format.

    Args:
        analysis_dict (dict): Dictionary containing various analysis outputs 

    Returns:
        dict: Combined JSON-compatible dictionary with all analysis outputs.
    """
    def to_json(value, orient='index'):
        """Safely convert data (DataFrame or dict) to JSON format."""
        if isinstance(value, pd.DataFrame):
            return json.loads(value.to_json(orient=orient))
        elif isinstance(value, dict):
            return value  # Already JSON-compatible
        else:
            raise ValueError(f"Unsupported data type for conversion: {type(value)}")
    
    # Convert each analysis result in the dictionary to JSON-compatible format
    json_data = {}
    
    # Iterate over each key-value pair in the analysis_dict and convert values
    for key, value in analysis_dict.items():
        if isinstance(value, pd.DataFrame):  # If value is DataFrame, convert to JSON
            json_data[key] = to_json(value, orient='index')
        elif value == None:
            continue
        else:
            json_data[key] = value  # Already JSON-compatible (dict, int, etc.)
    
    return json_data

def Regression_analysis(dependent_variable: str, independent_variables: str, dataset: pd.DataFrame, path: str = folder_name):
    """
    Performs regression analysis to analyze relationships between a dependent variable and independent variables.

    Args:
        dataset (pd.DataFrame): Dataset containing variables.
        dependent_variable (str): Column for the dependent variable.
        independent_variables (str): Column for independent variable.
        path (str) : Path where graphics are saved.

    Returns:
        dict: Regression summary including coefficients, R-squared and p-values.
    """
    # Drop rows with missing values
    dataset = dataset.dropna()
    
    # Define independent and dependent variables
    X = dataset[independent_variables]
    y = dataset[dependent_variable]
    
    # Add a constant (intercept) to the model
    X = sm.add_constant(X)
    
    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    # Plotting the regression
    plt.figure(figsize=(5.2, 3.12), dpi = 100)

    # Scatter plot for the dependent variable vs independent variables
    sns.regplot(
        x = dataset[independent_variables[-1]], 
        y = dataset[dependent_variable], 
        scatter=True, 
        scatter_kws={'color': 'blue', 'label': 'Data Points'},
        line_kws={'color': 'red', 'label': 'Regression Line'},
        )

    # Labels and title
    plt.xlabel(independent_variables[-1])
    plt.ylabel(dependent_variable)
    plt.title(f'Regression Analysis: {dependent_variable} vs {independent_variables[-1]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'Regression Analysis: {dependent_variable} vs {independent_variables[-1]}'))
    print(f" Saved Regression Analysis: {dependent_variable} vs {independent_variables[-1]} in {path}")
    plt.close()
    # Return the regression results in a dictionary
    return {'Regression Analysis': {
            "coefficients": model.params.to_dict(),
            "r_squared": model.rsquared,
            "p_values": model.pvalues.to_dict(),
            }}

def Time_series_analysis(dataset: pd.DataFrame, time_column: str, column_to_analyze: str, moving_average_window: int = 7, path: str = folder_name):
    """
    Analyzes time series data, detecting trends, patterns, and seasonality. It includes trend analysis, moving averages,
    and seasonality detection.

    Args:
        dataset (pd.DataFrame): Dataset containing time series data.
        time_column (str): Column containing time values.
        columns_to_analyze (list): List of numeric columns to analyze for trends and seasonality.
        moving_average_window (int): Window size for moving average (default is 7).
        path (str) : Path where graphics are saved.

    Returns:
        dict: A dictionary containing trend information, moving averages, and seasonality details.
    """
    # Ensure the time column is in datetime format
    dataset[time_column] = pd.to_datetime(dataset[time_column])

    # Initialize the result dictionary
    analysis_result = {
        "moving_averages": {},
        "seasonality": {}
    }

    # Loop through the columns to analyze

    if column_to_analyze not in dataset.columns:
        print(f'{column_to_analyze} not found in Dataset for Time-Series Analysis.')
        return None

    # Calculate moving average (e.g., 7-day window)
    moving_avg = dataset[column_to_analyze].rolling(window=moving_average_window).mean()
    analysis_result["moving_averages"][column_to_analyze] = moving_avg.tolist()

    # Plotting: Time Series with Moving Average
    plt.figure(figsize=(5.2, 3.12), dpi = 100)
    sns.lineplot(x=dataset[time_column], y=dataset[column_to_analyze], label=column_to_analyze, color='blue')
    sns.lineplot(x=dataset[time_column], y=moving_avg, label=f'{moving_average_window}-Day Moving Average', color='red')
    plt.title(f"Time Series and Moving Average for {column_to_analyze}")
    plt.xlabel("Time")
    plt.ylabel(f"{column_to_analyze} Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{column_to_analyze}_time_series_trends.png"))
    print(f" Saved {column_to_analyze}_time_series_trends.png in {path}")
    plt.close()

    return {'Time series analysis' : analysis_result}



def Clustering_analysis(dataset: pd.DataFrame, columns_to_cluster: str, number_of_clusters: int = 3, path: str = folder_name):
    """
    Groups rows using clustering and visualizes the result, applying transformations as needed.

    Args:
        dataset (pd.DataFrame): Dataset to cluster.
        columns_to_cluster (dict): Columns to be considered as features for clustering with transformations.
        number_of_clusters (int): Number of clusters.
        path (str): Folder path to save the clustering results.

    Returns:
        dict: Cluster labels and cluster centroids.
    """
    # Apply transformations to the columns as specified in columns_to_cluster
    transformed_data = dataset.copy().dropna()
    columns_list = []
    column_transformations = {}
    for feature in columns_to_cluster.split(','):
        if ':' in feature:
            column, transformation = feature.split(':')[0].strip(), feature.split(':')[1].strip()
            column_transformations[column] = transformation
        else:
            column = feature.strip()
            transformation = None
        
        columns_list.append(column)

        if transformation == 'Onehot_encoding':
            # One-hot encode the column
            encoder = OneHotEncoder(sparse=False, drop='first')
            transformed = encoder.fit_transform(transformed_data[[column]])
            transformed_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([column]))
            transformed_data = transformed_data.drop(column, axis=1).join(transformed_df)
        
        elif transformation == 'Ordinal_encoding':
            # Ordinal encode the column (assumes a simple ordinal mapping)
            encoder = OrdinalEncoder()
            transformed_data[column] = encoder.fit_transform(transformed_data[[column]])

    # Perform clustering using the transformed dataset
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
    transformed_data['cluster'] = kmeans.fit_predict(transformed_data[columns_list])

    # Draw clustering plot
    plt.figure(figsize=(5.2, 3.12), dpi = 100)
    for cluster_id in range(number_of_clusters):
        cluster_data = transformed_data[transformed_data['cluster'] == cluster_id]
        plt.scatter(
            cluster_data[columns_list[0]],
            cluster_data[columns_list[1]],
            label=f'Cluster {cluster_id}'
        )
    plt.scatter(
        kmeans.cluster_centers_[:, 0], 
        kmeans.cluster_centers_[:, 1], 
        s=200, c='red', marker='X', label='Centroids'
    )
    plt.title('Clustering Results')
    plt.xlabel(columns_list[0])
    plt.ylabel(columns_list[1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'Clustering_analysis.png'))
    print(f' Saved Clustering_analysis.png in {path}')
    plt.close()

    return {'Clustering Analysis': {
        "cluster_labels": kmeans.labels_.tolist(),
        "cluster_centroids": kmeans.cluster_centers_.tolist(),
        'column_transformations': column_transformations
    }}

def convert_dates_to_iso(data):
    """
    Converts datetime values in a DataFrame or dictionary to ISO format.
    """
    if isinstance(data, pd.DataFrame):
        # Convert datetime columns to ISO format in DataFrame
        for column in data.select_dtypes(include=[pd.Timestamp]):
            data[column] = data[column].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
    elif isinstance(data, dict):
        # Recursively convert datetime values in a dictionary to ISO format
        for key, value in data.items():
            if isinstance(value, pd.Timestamp):
                data[key] = value.isoformat()
            elif isinstance(value, dict):
                convert_dates_to_iso(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, pd.Timestamp):
                        value[i] = item.isoformat()
                    elif isinstance(item, dict):
                        convert_dates_to_iso(item)
    return data

def LLM_analysis(path: str, data: pd.DataFrame):
    """
    Requests Gpt-4o-Mini to generate a function that performs enhanced analysis on given data and saves graphical outputs.

    Args:
        json_file (dict): The metadata in JSON format for analysis.
        path (str): The path where the generated plot should be saved.
        data (pd.DataFrame): The dataset to be analyzed.

    Returns:
        dict: Result of the analysis.
    """
    print('Starting LLM Based Analysis...')

    # User-input for LLM
    input_json_string = json.dumps({
        'Analysis': {
            'Data Description': convert_dates_to_iso(data.describe().to_dict()),
            'Data Sample': convert_dates_to_iso(data.head(3).to_dict())
        },
        'Graphics folder': path,
        'Note': 'Start code directly with "def LLM_generated_code(data):"; End with return json.dumps(output)',
        'Avoid': 'Correlation, Linear regression, Clustering and Time-Series Analysis'
    })

    prompt = {
        "model": "gpt-4o-mini",
        "response_format" : {"type" : "text"},
        "messages": [
            {
                "role": "system",
                "content": """
                Using the given metadata in JSON format, follow the steps below to select and perform an appropriate analysis from the options provided:
                (a) Network Analysis  
                (b) Hierarchy Analysis   
                (c) Anomaly Detection
                (d) Pareto Analysis
                (e) Some other meaningful analysis with good visual  

                ### Instructions:
                1. Based on the metadata, **determine the most suitable type of analysis** from the list above.  
                2. Write a **Python function** to perform the chosen analysis. This function should:
                - Accept the entire original data as input(DataFrame) having columns names same as given sample data.  
                - Conduct the analysis as needed, and return JSON output having Analysis Name.  
                - Produce and **save exactly one graphical output** using `seaborn` (`sns`). Keep the Graphic file name as the analysis name.  
                3. Adhere strictly to the following libraries:
                - `os`
                - `sys`
                - `pandas` (as `pd`)  
                - `numpy` (as `np`)  
                - `matplotlib.pyplot` (as `plt`)  
                - `seaborn` (as `sns`)  
                - `statsmodels.api` (as `sm`)  
                - `json`  

                **Key Notes:**
                - Ensure modularity and code readability, making it easy for future extensions or analysis modifications.
                - Handle errors gracefully, such as invalid metadata or unsupported analysis types.
                - Ensure the final output is visually clear, well-labeled, and meaningful.
                - No other text. Just give code.
                - Figure dimensions : dpi = 100 , maximum dimensions (5.2, 5.2)
                """

            },
            {
                "role": "user",
                "content": f'{input_json_string}'
            }
        ]
    }
    
    # Request GPT-4o-Mini response for generating the analysis function
    ai_proxy = requests.post('https://aiproxy.sanand.workers.dev/openai/v1/chat/completions', 
                            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
                            json=prompt)
    response = ai_proxy.json()
    try:
        ai_proxy.raise_for_status()
        # Extract the function code from the response
        function_code = response['choices'][0]['message']['content']
        # Regular expression to remove 'python' (if present) and all 'import' lines
        function_code = re.sub(r'(import .*\n|python)', '', function_code)
        function_code = re.sub(r'^.*?(def LLM_generated_code\(data\):)', r'\1', function_code, flags=re.DOTALL)
        function_code = re.sub(r'(return json.dumps\(output\))\s*.*', r'\1', function_code)
    except Exception as e:
        print(f'Error {e}')

    try:
        # Execute the generated code to run the analysis
        exec(function_code, globals())
        analysis_output = LLM_generated_code(data)
        return analysis_output
    except Exception as e:
        print(f"Error executing generated function. The code generated by LLM gave error.")
        print('Skipping LLM Based Analysis. This was an extra experimental part, meant to generate any extra analysis which had been missed so far.')
        return None

def perform_advanced_analysis(api_url: str, AIPROXY_TOKEN: str, data: pd.DataFrame):
    """
    Executes advanced analytical functions based on the provided data and API configuration.

    Args:
        data (object): The data to be analyzed.
        api_url (str): The URL of the API to call.
        AIPROXY_TOKEN (str): The API token for authentication.

    Returns:
        object: The result of the analysis performed by the selected function.
    """
    
    # Defining the available analytical functions
    functions = [
        {
            "type": "function",
            "function": {
                "name": "Regression_analysis",
                "description": "Performs regression to analyze relationships between a dependent variable and an independent variable.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "object",
                            "description": "Dataset used"
                        },
                        "dependent_variable": {
                            "type": "string",
                            "description": "Column for the dependent variable."
                        },
                        "independent_variables": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Column for independent variable."
                        }
                    },
                    "required": ["dataset", "dependent_variable", "independent_variables"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Time_series_analysis",
                "description": "Detects trends, moving_averages, and seasonality in time series data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "object",
                            "description": "Dataset used(name: data)"
                        },
                        "time_column": {
                            "type": "string",
                            "description": "Column with time values."
                        },
                        "column_to_analyze": {
                            "type": "string",
                            "description": "One numeric column to analyze."
                        }
                    },
                    "required": ["dataset", "time_column", "columns_to_analyze"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Clustering_analysis",
                "description": "Clusters rows based on selected columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "object",
                            "description": "Dataset used(name: data)"
                        },
                        "columns_to_cluster": {
                            "type": "string",
                            "description": """columns to be used as features for clustering, with adequate transformation('Onehot_encoding', 'Ordinal_encoding', None)
                                            E.g. 'columns 1 : transformaton 1, columns 2: transformation 2'"""
                        },
                        "number_of_clusters": {
                            "type": "integer",
                            "description": "Number of clusters.",
                            "default": 3
                        }
                    },
                    "required": ["dataset", "columns_to_cluster"],
                    "additionalProperties": False
                }
            }
        }
    ]
    
    # Preparing the API request payload
    prompt = {
        "model": 'gpt-4o-mini',
        "response_format": { "type": "json_object" },
        'tools': functions,
        'tool_choice': 'required',
        'messages': [{
            "role": "system",
            "content": 'Do meaningful analysis only, given the dataset sample.'
        },
        {
            "role" : "user",
            'content' : f'Give JSON of feasible function and columns for analysis. Sample : {convert_dates_to_iso(data.head(3).to_dict())}'
        }]
    }
    
    # Sending the request to the API
    response = requests.post(api_url, 
                             headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"}, 
                             json=prompt)

    print('Initiating Advanced Analysis...')
    response_data = response.json()
    # Handling the API response
    if response.status_code == 200:
        try:
            tool_calls = response_data['choices'][0]['message']['tool_calls']
            output = {}
            for i in range(len(tool_calls)):
                function_name = tool_calls[i]['function']['name']
                function_args = json.loads(tool_calls[i]['function']['arguments'])
                function_args['dataset'] = data
                # Dynamically invoking the selected function
                function = globals().get(function_name)
                if function:
                    try:
                        analysis_output = function(**function_args)
                        output[function_name] = analysis_output
                    except Exception as e:
                        print(f"Error in {function_name}. Skipping this part of analysis.")
                else:
                    print(f'Invalid function : {function} called by LLM. Skipping this part of analysis.')
        except Exception as e:
            print(f'Stopped processing due to {e}')
            return None
    else:
        print(f'Error in calling LLM.')
        return None
    return output

def Plot_Summary(json_file_input: dict, additional_analysis, folder_name: str, data: pd.DataFrame):
    """
    Generate a comprehensive summary of statistical analysis and related visuals.

    This function:
    1. Summarizes the dataset and analysis results using an AI model.
    2. Appends results and insights to a README file.
    3. Processes and describes up to 5 visualizations in the folder.

    Args:
        json_file_input (dict): Metadata about the dataset and analysis.
        additional_analysis (dict): Results from advanced analysis.
        folder_name (str): Path where generated visuals and README are stored.
        data (pd.DataFrame): Dataset used for analysis.

    Returns:
        None
    """
    print('Generating Summary....')
    def call_LLM(sys_content:str, user_content:str, user_input_type:str = 'text'):
        """
        Helper function to send requests to the AI model and retrieve responses.

        Args:
            sys_content (str): System instructions for the AI.
            user_content (str): User-provided content for analysis.
            user_input_type (str): Type of input ('text' or 'image_url').

        Returns:
            response (Response): The response object from the AI API.
        """
        prompt = {
            "model": "gpt-4o-mini",
            "response_format": { "type": "text" },
            "messages": [
                {
                    "role": "system",
                    "content": sys_content
                },
                {
                    "role": "user",
                    "content": [{
                      'type' : user_input_type,
                      user_input_type : user_content 
                    }]
                }
            ],
        } 
        response = requests.post('https://aiproxy.sanand.workers.dev/openai/v1/chat/completions', 
                            headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
                            json=prompt)
        return response    
    
    try:
        # Generate dataset summary using AI
        user_content = f'{json.dumps(json_file_input)}'
        sys_content = f'''Using the metadata and analysis results provided for the data:
                        1. Give a summary of the data (which includes its dimensions, column names and their datatypes etc.).
                        2. Explain the analysis done with proper justification. Show the analyses results.
                        3. Give useful insights derived from the analysis.
                        4. Give Proper headings to each section. 

                        Analysis Output Related points (Mention at appropriate places, only if applicable):
                        1. The 'Advanced Analysis output' was generated using "Function Calling" in Ai proxy.
                        2. The unknown/Nan entries were dropped during analysis.
                        3. If Clustering Analysis: It used K-Means Algo.
                        4. The 'LLM based analysis' was completely done by LLM from choosing analysis type, to its coding.

                        Dataset Sample: 
                        {convert_dates_to_iso(data.head(3).to_dict())}'''
        response = call_LLM(sys_content, user_content)    
        response.raise_for_status()
        summary = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    except requests.exceptions.RequestException as e:
        print(response.json())
        print(f"Error in API request: {e}")
        return None
    else:
        # Append summary to README file
        with open(os.path.join(folder_name, 'README.md'), "a") as file:
            file.write(summary)
            file.write("\n")
            # Process additional analysis if available
            if additional_analysis is not None:
                user_content = f'{json.dumps(additional_analysis)}'
                user_input_type = 'text'
                sys_content = 'Give a summary with justification for the folowing "LLM based analysis"'
                llm_summary = call_LLM(sys_content, user_content, user_input_type)
                try:
                    llm_summary.raise_for_status()
                    llm_summary_data = llm_summary.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                    file.write(f"\n\n### LLM Generated Analysis\n")
                    file.write(llm_summary_data)
                except Exception as e:
                    print(f'llm_summary Error : {e}')

    def encode_image(image_path):
        """
        Encodes an image in base64 format.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Process images in the folder
    images = {}
    for file in os.listdir(folder_name):
        if '.png' in file:
            new_file_name = file.replace(" ", "_").replace(";", "_").replace(":", "_")
            os.rename(os.path.join(folder_name, file), os.path.join(folder_name, new_file_name))
            image_path = os.path.join(folder_name, new_file_name)
            images[os.path.splitext(new_file_name)[0]] = encode_image(image_path)
            if len(images) >= 5:
                break
    
    # Append images and their descriptions to README
    with open(os.path.join(folder_name, 'README.md'), "a") as file:
        user_input_type = 'image_url'
        for image_name, image_data in images.items():
            file.write(f"\n\n### Image {image_name}\n")
            file.write(f"![{image_name}]({image_name.replace(" ", "_").replace(";", "_").replace(":", "_")}.png)\n\n")

            sys_content = f'Write a description on the {image_name} image in 200-250 words'
            user_content ={"url":  f"data:image/jpeg;base64,{image_data}",
                           "detail" : "low"}
            image_description = call_LLM(sys_content, user_content, user_input_type)
            try:
                image_description.raise_for_status()
                file.write(image_description.json().get('choices', [{}])[0].get('message', {}).get('content', ''))
            except Exception as e:
                print(f'Image Description Error : {e}')
                print(image_description.json())
                continue
    print('Analysis Completed')
    return 

def Statistical_Analysis(data):
    # Initiate basic analysis
    analysis_dict = basic_analysis(api_url, AIPROXY_TOKEN, folder_name, data)

    # Perform 2nd Level of Analysis by using LLM Function Calling
    advanced_analysis_output = perform_advanced_analysis(api_url, AIPROXY_TOKEN, data)
    if advanced_analysis_output is not None: 
        analysis_dict['Advanced Analysis output'] = advanced_analysis_output

    # Compress the derived outputs into a single JSON file
    json_file_input = convert_to_json(analysis_dict)

    # Miscellaneous Analysis using LLM 
    additional_analysis = LLM_analysis(folder_name, data)

    # Draft summary in README.md
    Plot_Summary(json_file_input, additional_analysis, folder_name, data)

# Call the function to start analysis
Statistical_Analysis(load_csv(file_path))
