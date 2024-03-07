import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import itertools

# Fixed API key input
openai_api_key = "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"

st.title("DashGen - Text to Analysis")

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
        categorized_data[f"{column}_CATEGORY"] = pd.cut(input_df[column], bins=bins, labels=labels, include_lowest=True)
    return categorized_data[[col for col in categorized_data.columns if "_CATEGORY" in col]]

def parse_fields(data):
    parsed_output = {
        "Dimensions": [],
        "Metrics": [],
        "Weighting_Metric": ""
    }
    if "Dimensions" in data:
        parsed_output["Dimensions"] = data["Dimensions"]
    if "Metrics" in data:
        parsed_output["Metrics"] = data["Metrics"]
    if "Weighting_Metric" in data:
        parsed_output["Weighting_Metric"] = data["Weighting_Metric"]

    return parsed_output

import pandas as pd
import numpy as np
import itertools

def aggregate_metrics_and_format(df, dimension_columns, measure_column):
    # Check if the measure column exists in the DataFrame.
    if measure_column not in df.columns:
        st.error(f"Measure column '{measure_column}' does not exist in the DataFrame")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error

    # Convert the measure column to numeric, coercing errors to NaN.
    df[measure_column] = pd.to_numeric(df[measure_column], errors='coerce')

    # If the entire measure_column is NaN, there's no point in continuing.
    if df[measure_column].isna().all():
        st.error(f"Measure column '{measure_column}' contains only NaNs after conversion to numeric.")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error

    results = []

    # Iterate through combinations of dimension columns.
    for r in range(1, len(dimension_columns) + 1):
        for subset in itertools.combinations(dimension_columns, r):
            # Select the current subset of dimension columns and the measure column.
            temp_df = df[list(subset) + [measure_column]].dropna().copy()
            
            # If temp_df is empty, continue to the next combination
            if temp_df.empty:
                st.write(f"No data available for dimensions {subset} after dropping NA.")
                continue

            # Group the data by the current subset of dimension columns.
            grouped = temp_df.groupby(list(subset), as_index=False)

            # Aggregate the groups using median and count.
            try:
                aggregated = grouped.agg(measure_median=(measure_column, 'median'),
                                         measure_count=(measure_column, 'count'))
            except Exception as e:
                st.error(f"Aggregation failed for dimensions {subset}: {e}")
                continue

            # Debugging output
            st.write(f'Aggregated DataFrame for dimensions {subset}:', aggregated)

            # Concatenate the dimension column names for identification.
            aggregated['dimension_combination'] = ' X '.join(subset)

            # Append the aggregated data to the results list.
            results.append(aggregated)

    # If results is empty, there's nothing to concatenate
    if not results:
        st.error("No aggregated data was produced. Please check your data and dimension columns.")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error

    # Combine all aggregated results into a single DataFrame.
    try:
        final_result = pd.concat(results, ignore_index=True)
    except Exception as e:
        st.error(f"Concatenation of aggregated data failed: {e}")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error


    # Filter the final results based on the median count threshold.
    count_threshold = final_result['measure_count'].median()
    final_result_filtered = final_result[final_result['measure_count'] > count_threshold]

    # Calculate the median of the 'measure_median' column across the filtered results.
    overall_median = final_result_filtered['measure_median'].median()

    # Filter out rows with a measure count of 5 or less.
    final_result_filtered = final_result_filtered[final_result_filtered['measure_count'] > 5]

    # Calculate the difference between each row's median and the overall median.
    final_result_filtered['median_difference'] = (final_result_filtered['measure_median'] - overall_median).round(3)

    # Sort the filtered results by 'measure_median' in descending order.
    final_result_filtered.sort_values(by='measure_median', ascending=False, inplace=True)
    st.write('Final sorted DataFrame:', final_result_filtered)

    return final_result_filtered

def subset_and_analyze_dimensions(input_df, user_query):
    # Generate a summary of the DataFrame with additional type information
    metric_distributions = input_df.describe(include='all').to_dict()  # include='all' will describe all columns of the DataFrame
    column_types = input_df.dtypes.apply(lambda x: str(x)).to_dict()  # Get the data types of all columns

    # Combine distributions and types into a single dictionary
    combined_info = {
        'distributions': metric_distributions,
        'types': column_types
    }

    # Set the API key for OpenAI
    openai.api_key = openai_api_key

    # Prepare the context for GPT, focusing on identifying dimensions and considering metric distributions
    context = f"""
    Purpose:
    - Analyze (1) an input dataframe and (2) a user query to idetify the relevant (a) dimensions, (b) metrics, and (c) weighting metric to analyze the data.
    - Return the response in JSON format so it can be parsed by a downstream web app. The web app will use the output to run follow up analysis on the dataset.
    - The analysis that will be done downstream is that the metrics (b) will be calculated over the dimensions (a). The weighting metric (c) will be used to assess scale (e.g. if the metric is a ratio then having a weight will be important)
    - NB: Make sure to consider including both the dimensions and the metric_CATEGORY values in the output. Both are relevant. 
    - Utilize your expertise to identify which dimensions and metrics should be included for a detailed analysis of the dataset to answer the user's query.

    Processing instructions:
    - (1) input dataframe
      - (a) Dimensions:
        - Look for the relevant dimensions in the input dataset that can be used to answer the user query (2).
        - All columns in the input dataset are dimensions so just return the column names.
        - Only include the most relevant dimensions. (4-6 dimemsions total is the max)
        - The LLM should be quite selective about the dimensions to include. Make sure to think through it deeply why they're relevant.
        - NB: the input dataset also contains metric_CATEGORY values for dimensions which split the metrics up into the quartiles they belong to. Please use these too as dimension if they are relevant for the analysis.
        - The dimensions will be used to group the data and calculate the metrics over. Therefore they should be string values, not numeric. Float or numeric value columns should never be returned as dimensions. 
        - NB: only include dimensions that look like they will have higher cardinality. Things like username level or account id level should not be included as they are too specific. 
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

    Given this dataset with distributions: {combined_info['distributions']}, and types: {combined_info['types']}, and aiming to answer this question: '{user_query}'. Please return a JSON with the relevant (a) dimensions, (b) metrics, and (c) weighting metric for further analysis.

    Example output:

    {{
      "Dimensions": ["Dimension_A", "Dimension_B", "Dimension_C_CATEGORY"],
      "Metrics": ["Metric_A", "Metric_B"],
      "Weighting_Metric": "Weighting_Metric_A"
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
    input_df = pd.read_csv(uploaded_file)
    for column in input_df.columns:
        if all(is_integer(item) for item in input_df[column].astype(str)):
            input_df[column] = input_df[column].astype(float)
    categorized_df = categorize_data(input_df)
    input_df = pd.concat([input_df, categorized_df], axis=1)
    st.dataframe(input_df.head(20))
    analysis_result = subset_and_analyze_dimensions(input_df, question)
    st.text(analysis_result)
    # Parse the LLM output
    parsed_llm_data = parse_fields(json.loads(analysis_result))
    
    # Define columns for aggregation
    dimension_columns = parsed_llm_data['Dimensions']
    st.dataframe(dimension_columns)
    metric_to_aggregate = parsed_llm_data['Metrics'][0]
    st.text(metric_to_aggregate)
    
    # Aggregate metrics and format the result
    result_df = aggregate_metrics_and_format(input_df, dimension_columns, metric_to_aggregate)
    
    # Visualize the first 20 rows of the output data frame
    st.dataframe(result_df.head(20))