import streamlit as st
import pandas as pd
import sqlite3
import re
from langchain_community.utilities import SQLDatabase
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from dateutil import parser

def robust_date_parsing(date_str):
    try:
        return pd.to_datetime(date_str)
    except (ValueError, TypeError):
        try:
            return parser.parse(date_str, fuzzy=True)
        except (ValueError, TypeError):
            return pd.NaT  # Return NaT for invalid dates

def convert_date_columns(df):
    datetime_patterns = ['date', 'year', 'month', 'day', 'time', 'hour', 'minute', 'second']
    for column in df.columns:
        if any(re.search(pattern, column, re.IGNORECASE) for pattern in datetime_patterns):
            df[column] = df[column].apply(robust_date_parsing)
            st.write(f"Converted {column} to datetime.")
    return df


# Streamlit App Title
st.title("Streamlit CSV to SQL Chatbot")

# File upload section
st.header("Step 1: Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(data.head())

    # Convert any date and time-related columns to datetime
    data = convert_date_columns(data)

    # Create a SQLite database connection and load the database
    conn = sqlite3.connect('uploaded_data.db')
    data.to_sql('uploaded_table', conn, if_exists='replace', index=False)
    st.success("Data has been successfully loaded into the SQL database.")

    db = SQLDatabase.from_uri('sqlite:///uploaded_data.db')

    # Schema extraction function
    def get_schema():
        return db.get_table_info()

    schema = get_schema()


    # Setup NL2SQL with Few-Shot Learning
    st.header("Step 2: Ask a Question")
    user_input = st.text_input("Enter your query:")

    if user_input:
        # Few-shot examples
        examples = [
            {"input": "What is the state with most fraud?",
             "output": "SELECT state, COUNT(is_fraud) AS fraud_count FROM uploaded_table WHERE is_fraud = 1 GROUP BY state ORDER BY fraud_count DESC LIMIT 1;"},
            {"input": "Which year had more frauds?",
             "output": "SELECT strftime('%Y', trans_date_trans_time) AS year, COUNT(is_fraud) AS fraud_count FROM uploaded_table WHERE is_fraud = 1 GROUP BY year ORDER BY fraud_count DESC LIMIT 1;"},
            {"input": "Which category had more fraud in 2019?",
             "output": "SELECT category, COUNT(is_fraud) AS fraud_count FROM uploaded_table WHERE is_fraud = 1 AND strftime('%Y', trans_date_trans_time) = '2019' GROUP BY category ORDER BY fraud_count DESC LIMIT 1;"},
            {"input": "Which merchant had most fraud?",
             "output": "SELECT merchant, COUNT(is_fraud) AS fraud_count FROM uploaded_table WHERE is_fraud = 1 GROUP BY merchant ORDER BY fraud_count DESC LIMIT 1;"},
            {"input": "Which merchant had more fraud in 2019 in Wales?",
             "output": "SELECT merchant, COUNT(is_fraud) AS fraud_count FROM uploaded_table WHERE is_fraud = 1 AND strftime('%Y', trans_date_trans_time) = '2019' AND city = 'Wales' GROUP BY merchant ORDER BY fraud_count DESC LIMIT 1;"},
            {"input": "Month with most Fraud",
             "output": "SELECT strftime('%m', trans_date_trans_time) AS month, COUNT(is_fraud) AS fraud_count FROM uploaded_table WHERE is_fraud = 1 GROUP BY month ORDER BY fraud_count DESC LIMIT 1;"},
            {"input": "What is the year of release for Wall-E?",
             "output": "SELECT strftime('%Y', year) AS year FROM uploaded_table WHERE Film = 'Wall-E';"}
        ]

        # Define the example prompt template
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}"
        )

        # Create the few-shot prompt template
        prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Based on the table schema below, write a SQL query that would answer the user's question:\n{schema}",
            suffix="Input: {input}\nOutput:",
            input_variables=["input", "schema"]
        )

        # Initialize the LLM with OpenAI's Key
        llm = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

        # Create the rephrasing chain
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
        )

        rephrase_answer = answer_prompt | llm

        # Define the chain
        sql_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Generate SQL query using the schema
        generated_query = sql_chain.run({"input": user_input, "schema": schema})
        st.write("Generated SQL Query:")
        st.code(generated_query)

        # Execute the generated SQL query
        try:
            result = pd.read_sql_query(generated_query, conn)
            st.write("Query Result:")
            st.write(result)

            if result.empty:
                st.write(
                    "The query did not return any results. Please ensure the column names and data format are correct.")
            else:
                # Rephrase the answer
                rephrased_answer = rephrase_answer.invoke({
                    "question": user_input,
                    "query": generated_query,
                    "result": result.to_dict(orient='records')
                })
                st.write("Answer:")
                st.write(rephrased_answer)

        except Exception as e:
            st.write(f"An error occurred: {e}")

    conn.close()
