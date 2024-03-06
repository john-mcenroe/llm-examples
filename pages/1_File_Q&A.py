import streamlit as st
import pandas as pd
import numpy as np
import openai

# Set a default value for the OpenAI API key for testing purposes
DEFAULT_API_KEY = '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'

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    st.markdown("[Fetch an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("[View the source code](https://github.com/streamlit/llm-examples)")
    st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

st.title("📝 File Q&A with OpenAI")

# File uploader and question input
uploaded_file = st.file_uploader("Upload an article", type=("csv"))
question = st.text_input(
    "What question do you want to answer about the data?",
    placeholder="Give me some insights about churn",
    disabled=not uploaded_file,
)

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def categorize_data(input_df):
    float_cols = input_df.select_dtypes(include=['float']).columns
    summary_data = input_df[float_cols].describe(percentiles=[.25, .5, .75])
    categorized_data = input_df.copy()

    for column in float_cols:
        percentiles = summary_data[column]
        bins = [-np.inf, percentiles['25%'], percentiles['50%'], percentiles['75%'], np.inf]
        labels = ['<=25p', '>25p and <=50p', '>50p and <=75p', '>75p']
        categorized_data[f"{column}_Category"] = pd.cut(input_df[column], bins=bins, labels=labels, include_lowest=True)
    return categorized_data[[col for col in categorized_data.columns if "_Category" in col]]

def subset_and_analyze_dimensions(input_df, user_query):
    metric_distributions = input_df.describe().to_dict()

    # Set the API key for OpenAI
    openai.api_key = openai_api_key

    # Prepare the context for GPT, focusing on identifying dimensions and considering metric distributions
    context = f"""
    Purpose:
    - Analyze (1) an input dataframe and (2) a user query to idetify the relevant (a) dimensions, (b) metrics, and (c) weighting metric to analyze the data.
    - Return the response in JSON format so it can be parsed by a downstream web app. The web app will use the output to run follow up analysis on the dataset.
    - The analysis that will be done downstream is that the metrics (b) will be calculated over the dimensions (a). The weighting metric (c) will be used to assess scale (e.g. if the metric is a ratio then having a weight will be important)
    Utilize your expertise to identify which dimensions and metrics should be included for a detailed analysis of the dataset to answer the user's query.

    Processing instructions:
    - (1) input dataframe
      - (a) Dimensions:
        - Look for the relevant dimensions in the input dataset that can be used to answer the user query (2).
        - All columns in the input dataset are dimensions so just return the column names.
        - Only include the most relevant dimensions. (4-6 dimemsions total is the max)
        - The LLM should be quite selective about the dimensions to include. Make sure to think through it deeply why they're relevant.
        - NB: the input dataset also contains metric_CATEGORY values for dimensions which split the metrics up into the quartiles they belong to. Please use these too as dimension if they are relevant for the analysis.
      - (b) Metrics:
        - Outline the metrics that can be used to answer the query.
        - Only return the top top priority metrics so 1-3 MAX.
        - The metrics will be aggreated over the dimensions. We'll use functions like median, 75th percentile to look at the distribution of metric values.
        - Never return _CATEGORY columns as the metric. These are dimensions. Only return data of float type.
      - (c) Weighting Metric:
        - The weighting metric should indicate the size of the dimension when summed. It will be used to order the rows by the relative sizes.
        - This is the most relevant metric for ranking the rows in the dataset by their volume (e.g. revenue, count etc..)
        - This column is needed because not all metrics will have weights (e.g. ratios or fractions).
        - Only return 1 weighting metric.
        - If there is no approprtiate weighting metric in the dataset then return COUNT as the value here.
        - Make sure to think deeply about the weighting metric.
        - Good examples are: revenue, count, number of users etc.. etc.... see these are metrics that clearly indicate size when summed.

    Parameters:
    - input_df (pd.DataFrame): A summary of the columns in the input data frame.
    - user_query (str): The question the user wants to answer with the data.

    Returns:
    - A JSON with suggested (a) dimensions, (b) metrics and (c) weighting metric from the input dataset.
    - Only return the JSON with the values. NOTHING ELSE.

    Given this dataset: {metric_distributions}, and aiming to answer this question: '{user_query}'. Please return a JSON with the relevant (a) dimensions, (b) metrics, and (c) weighting metric for further analysis.

    Example output:

    {{
      Dimensions: ["Dimension_A", "Dimension_B", "Dimension_C"]
      Metrics: ["Metric_A", "Metric_B"]
      Weighting_Metric: "Weighting_Metric_A"
    }}
    """
    # Existing context will have the '...' replaced with the actual content
    messages = [
        {"role": "system", "content": "You are a advanced business analysis tool with deep understanding of what types of dimensions and metrics are best suited to identifying insights in datasets. You are helping create a JSON from the input data for downstream analysis in a web app."},
        {"role": "user", "content": context}
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo", # "gpt-4-1106-preview",  # or "gpt-3.5-turbo", "gpt-4"
        messages=messages,
        max_tokens=4000,
        response_format={"type": "json_object"},
        n=1,
        stop=None,
        temperature=0.8,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )

    return response.choices[0].message.content

# Main app logic
if uploaded_file and question:
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
    else:
        input_df = pd.read_csv(uploaded_file)

        # Convert columns that contain only numeric values to integers
        for column in input_df.columns:
            if all(is_integer(item) for item in input_df[column].astype(str)):
                input_df[column] = input_df[column].astype(float)

        # Categorize numeric columns
        categorized_df = categorize_data(input_df)

        # Append categorized_df columns to input_df
        input_df = pd.concat([input_df, categorized_df], axis=1)

        # Call the function to analyze the dataset
        analysis_result = subset_and_analyze_dimensions(input_df, question)

        # Display the result
        st.json(analysis_result)