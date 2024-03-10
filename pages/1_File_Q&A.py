import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import itertools
from tabulate import tabulate
import plotly.graph_objects as go

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

def gen_table(dimension_columns, metric_to_aggregate, weighting_metric):
    # Creating a DataFrame from the data
    table_df = pd.DataFrame({
        "Type": ["Dimensions", "Metric", "Weighting Metric"],
        "Value": [", ".join(dimension_columns), metric_to_aggregate, weighting_metric]
    })
    
    return table_df

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

def aggregate_metrics_and_format(df, dimension_columns, measure_column):
    # Check if the measure column exists in the DataFrame.
    if measure_column not in df.columns:
#        st.error(f"Measure column '{measure_column}' does not exist in the DataFrame")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error

    # Convert the measure column to numeric, coercing errors to NaN.
    df[measure_column] = pd.to_numeric(df[measure_column], errors='coerce')

    # If the entire measure_column is NaN, there's no point in continuing.
    if df[measure_column].isna().all():
 #       st.error(f"Measure column '{measure_column}' contains only NaNs after conversion to numeric.")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error

    results = []

    # Iterate through combinations of dimension columns.
    for r in range(1, len(dimension_columns) + 1):
        for subset in itertools.combinations(dimension_columns, r):
            # Select the current subset of dimension columns and the measure column.
            temp_df = df[list(subset) + [measure_column]].dropna().copy()
            
            # If temp_df is empty, continue to the next combination
            if temp_df.empty:
#                st.write(f"No data available for dimensions {subset} after dropping NA.")
                continue

            # Group the data by the current subset of dimension columns.
            grouped = temp_df.groupby(list(subset), as_index=False)

            # Aggregate the groups using median and count.
            try:
                aggregated = grouped.agg(measure_median=(measure_column, 'median'),
                                         measure_count=(measure_column, 'count'))
            except Exception as e:
#                st.error(f"Aggregation failed for dimensions {subset}: {e}")
                continue

            # Debugging output
#            st.write(f'Aggregated DataFrame for dimensions {subset}:', aggregated)

            # Concatenate the dimension column names for identification.
            aggregated['dimension_combination'] = ' X '.join(subset)

            # Append the aggregated data to the results list.
            results.append(aggregated)

    # If results is empty, there's nothing to concatenate
    if not results:
#        st.error("No aggregated data was produced. Please check your data and dimension columns.")
        return pd.DataFrame()  # Return empty DataFrame as a signal of error

    # Combine all aggregated results into a single DataFrame.
    try:
        final_result = pd.concat(results, ignore_index=True)
    except Exception as e:
#        st.error(f"Concatenation of aggregated data failed: {e}")
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

    return final_result_filtered

def format_and_display_dataset(df):
    # Add an index column for row numbering
    df = df.reset_index(drop=True)
    df.index.name = '#'
    df.reset_index(inplace=True)

    # Automatically identify dimension columns (excluding 'dimension_combination')
    # Assuming dimension columns are of type 'object' (typically string-like data in pandas)
    dimensions = [col for col in df.columns if col != 'dimension_combination' and df[col].dtype == 'object']

    # Define the function for concatenating dimensions
    def concatenate_dimensions(row, dimensions):
        values = [str(row[dim]) if pd.notnull(row[dim]) else '' for dim in dimensions]
        return ' || '.join(filter(None, values))  # Filter out empty strings to avoid unnecessary delimiters

    # Add a column for concatenated dimension values
    df['concatenated_dimensions'] = df.apply(lambda row: concatenate_dimensions(row, dimensions), axis=1)

    # Specify the order of columns, starting with the index, 'dimension_combination', etc.
    specified_columns = ['#', 'dimension_combination', 'concatenated_dimensions', 'measure_median', 'median_difference', 'measure_count']
    
    # Append the rest of the columns that were not specified
    new_column_order = specified_columns + [col for col in df.columns if col not in specified_columns]
    
    # Reorder the DataFrame columns
    df = df[new_column_order]

    # Replace NaN values with an empty string
    df = df.fillna('')

    return df

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

def derive_insights_from_dataset(input_df, user_query):
    # Check if the DataFrame is empty or has no columns
    if input_df.empty or input_df.columns.empty:
        # Return a message or handle the empty DataFrame scenario
        # For example, return a default response or log a message
        return json.dumps({"insights": ["No data available for analysis."]})

    # Proceed with the function if the DataFrame is not empty
    summary_statistics = input_df.describe(include='all').to_json()
    top_records = input_df.head(50).to_json()

    context = f"""
    Goal: 
    - You are an automated AI analyzer helping a founder identify insights about data to answer their question.
    - You take an aggregated dataset and a user's question and provide a ranked list of insights about the dataset to help the user answer their question.
        

    What you do: You analyze an aggregated dataset of user data with theqeustion they want to answer about that data.
    - You are trying to derive insigths about the data to help user answer their question.
    - Focus on insights that are nuanced and unique and opinonated. Not things that can be easily seen in the data table which is also visible to the user.
    - You are trying to provoke engagement from the user and help them identify unique insighs to answer their question.
    - Make the user who analyzes the data with these insights a master of coming up with hypotheses and understanding their data.
    - The output will be shown directly inside an AI analysis app where users ask questions about their data and review the responses.
    - Keep the insights to one liners. Which both call out the insight and say why.

    Other guidelines:
    - Insights should be short, snappy and opinionated.
    - Focus on insights across the data or extreme values. Things not easy to spot.
    - Be opinionated. That is what makes the difference.
    - Provoke user thought and explortion of the data table and follow ups.

    Parameters:
    - input_df (pd.DataFrame): The aggregated dataset after applying the aggregate_metrics function.
    - user_query (str): The user's query or the question they want to answer with the data.

    Returns:
    - A JSON with insights derived from the dataset, tailored to the user's query.
    - Make sure the insights are self contained and actionable.
    - Only return the insights in the JSON. No other information should be included with them.
    - There should be 5-10 insights returned in the JSON. 
    
    Sample JSON output: 
    "insights": [
        "The highest sales volumes occur in Q4, suggesting seasonal promotions drive significant revenue.",
        "Sales in the Northeast region outperform other regions by 15%, indicating regional preferences or distribution strengths.",
        "Product Category B has the highest return rate at 20%, highlighting a potential quality or customer satisfaction issue.",
        "Repeat customers contribute to 40% of total sales, emphasizing the importance of customer retention strategies.",
        "Sales performance correlates with marketing spend, with a 0.8 correlation coefficient, suggesting effective marketing.",
        "The average sales cycle length has increased by 10 days year-over-year, potentially indicating longer decision-making processes.",
        "Products priced between $50-$100 have the highest conversion rate, suggesting a sweet spot for pricing strategies.",
        "Customer reviews have a strong influence on sales, with products rated 4 stars or higher seeing a 25% increase in sales.",
        "Sales reps with more than 5 years of experience have a 30% higher sales quota attainment, highlighting the value of experience.",
        "The use of discount codes increases cart size by an average of 15%, but also correlates with a 5% lower profit margin per sale."
    ]

    Given the user's query: "{user_query}", analyze the summarized data and the top records to provide tailored insights. The summarized statistics are as follows: {summary_statistics}. The top records from the dataset are: {top_records}.
    - How to analyze: Based on the dataset and focusing on the user's query, identify key insights about the dataset, especially focusing on dimensions, measure averages, counts, and any notable trends observed. Rank these insights from high to low based on their importance and potential impact.
    """

    messages = [
        {"role": "system", "content": "You are an automated AI analyzer helping a founder identify insights about data to answer their question."},
        {"role": "user", "content": context}
    ]

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview", # "gpt-4-1106-preview",  # or "gpt-3.5-turbo", "gpt-4"
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

     # Analyze the dataset to get dimensions, metrics, and weighting metric
    analysis_result = subset_and_analyze_dimensions(input_df, question)
    parsed_llm_data = parse_fields(json.loads(analysis_result))
    
    # Define columns for aggregation
    dimension_columns = parsed_llm_data['Dimensions']
    metric_to_aggregate = parsed_llm_data['Metrics'][0]
    weighting_metric = parsed_llm_data['Weighting_Metric']
    
    table = gen_table(dimension_columns, metric_to_aggregate, weighting_metric)

    # Display the table using Streamlit's markdown capability for better formatting
    st.write("## Analysis Column Summary")
    st.dataframe(table)
    
    # Aggregate metrics and format the result
    result_df = aggregate_metrics_and_format(input_df, dimension_columns, metric_to_aggregate)

    # Derive insights from the dataset
    insights = derive_insights_from_dataset(result_df, question)
    
    # Display insights and result_df in the app
    try:
        insights_json = json.loads(insights)  # Parse JSON string to dictionary
        insights_list = insights_json.get('insights', [])
        insights_df = pd.DataFrame({'Insight': insights_list})

        # Convert DataFrame to HTML, omit the header and index
        insights_html = insights_df.to_html(header=False, index=False)

        # Display insights using Streamlit, without headers
        st.markdown("## Insight Summary", unsafe_allow_html=True)
        st.markdown(insights_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to parse insights: {e}")

    # Display the Top Drivers Table using Streamlit's native method
    st.write("## Top Drivers Table", unsafe_allow_html=True)
    # Replace NaN or None with an empty string
    clean_df = result_df.replace({np.nan: '', None: ''})
    formatted_df = format_and_display_dataset(clean_df)

    # Display DataFrame without highlighting nulls
    st.dataframe(formatted_df)