# NL2SQL

## Introduction
This Streamlit application converts CSV data into SQL and includes a chatbot that translates natural language into SQL queries, allowing users to ask questions about the dataset and receive answers.

## Background
Driven by the necessity of a faster analytical tool, I developed this application to analyse CSV files and get simple queries answered almost immediately.

Dataset used to develop the app is from https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data/data

## Tools I Used
To build this application efficiently and ensure reliable performance, the following tools and technologies were used:

- **Python:** The core programming language used to develop the application, chosen for its flexibility, rich ecosystem, and strong data-processing capabilities.
- 
- **Streamlit:** Used to create the interactive web interface, enabling rapid development of a user-friendly data application.
- 
- **VS Code**: Code editor used for development, testing, and project management.

- **Git & GitHub**: Used for version control, repository hosting, and project documentation.

## Code Overview

This application combines data processing, SQL querying, and natural language interaction to create an intuitive data exploration experience.

### Data Handling & Preprocessing

The app uses Pandas to read the uploaded CSV file into a DataFrame.
To improve query reliability, the code includes a custom date-parsing mechanism:

- robust_date_parsing()
Attempts to convert values into datetime format using pandas.to_datetime().
If that fails, it falls back to dateutil.parser, allowing flexible parsing of irregular date formats.

- convert_date_columns()
Automatically detects columns likely to contain temporal data (e.g., names containing date, time, year, etc.) using regex pattern matching.
Identified columns are converted into datetime objects.

This step ensures compatibility with SQL date functions such as strftime().

### Database Layer

The application uses SQLite as a lightweight relational database:

- The processed DataFrame is written into a SQL table (uploaded_table)

- SQLite enables structured querying without requiring an external database server

- LangChain’s SQLDatabase utility extracts the table schema, which is later provided to the language model

### Natural Language → SQL Conversion

The core intelligence of the app is powered by LangChain + OpenAI:

- Few-Shot Prompting
A set of example questions and corresponding SQL queries guide the model’s behavior, improving accuracy and reducing hallucinations.

- FewShotPromptTemplate
Injects:

Table schema

User question

Example mappings

- LLMChain
Uses the OpenAI language model to generate a SQL query from the user’s natural language input.

### Query Execution & Response Generation

Once the SQL query is generated:

1. **Execution**
The query is run against SQLite using Pandas (read_sql_query)

2. **Validation**
If no results are returned, the user receives feedback

3. **Answer Rephrasing**
A secondary prompt transforms:

- User question
- SQL query
- SQL result

into a human-readable answer.

This creates a conversational experience rather than exposing raw SQL output.

### Technologies & Concepts Applied

This project demonstrates practical use of:

- Data preprocessing & cleaning
- Automated datetime normalization
- SQL database integration
- Schema-aware prompting
- Few-shot learning
- Natural Language to SQL (NL2SQL)
- LLM response post-processing
- Interactive data applications with Streamlit

## Conclusion
This project demonstrates how modern data tools and language models can be combined to create a more intuitive way of interacting with structured data.

By integrating **Streamlit, Pandas, SQLite, LangChain,** and **OpenAI,** the application transforms a traditional SQL-based workflow into a conversational experience. Users can explore datasets without requiring deep SQL knowledge, lowering the technical barrier to data analysis.

Beyond its functional purpose, this project highlights important concepts in contemporary data applications, including:

- Data preprocessing and normalization
- Schema-aware prompt engineering
- Few-shot learning for controlled LLM behavior
- Natural Language to SQL (NL2SQL) systems
- Human-readable response generation

The result is an interactive analytical tool that bridges the gap between raw data and user-friendly insights.

This approach reflects a broader trend in data science and analytics: **making complex systems more accessible, efficient, and user-centric through AI-driven interfaces.**
