#Importing Required Libraries
import os 
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import random
import streamlit as st
#load env file
load_dotenv()
if "GROQ_API_KEYS" in st.secrets:
    API_KEYS = st.secrets["GROQ_API_KEYS"]
else:
    API_KEYS = [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2")
    ]
random.shuffle(API_KEYS)

#Defining the embedding model which can break words into vectors 
model=SentenceTransformer(
        "all-MiniLM-L6-v2"
    ) 

def search_web(topic):
    results = []
    try:
        with DDGS() as ddgs:
            search_results = list(
                ddgs.text(
                    f"{topic} statistics trends data report",
                    max_results=10
                )
            )
            for result in search_results:
                results.append(
                    f"""
TITLE: {result.get('title', '')}
CONTENT:
{result.get('body', '')}
SOURCE:
{result.get('href', '')}
"""
                )
    except Exception as e:
        print("SEARCH ERROR:", e)
    return "\n\n".join(results)

## Using Rag System for text retrieval 
def rag_system(text):
    chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    embedding = model.encode(chunks).astype("float32")
    index=faiss.IndexFlatL2(embedding.shape[1])
    index.add(np.array(embedding))
    return index,chunks
def retrieve(query,index,chunks):
    query_embed = model.encode([query]).astype("float32")
    distance,indices=index.search(np.array(query_embed),k=3)
    context=[]
    for i in indices[0]:
        context.append(chunks[i])
    return "\n\n".join(context)
#Generate report in english
def report_generate(topic):
    web_data=search_web(topic)
    web_data = web_data[:7000]
    for key in API_KEYS:
        client = Groq(api_key=key)
        prompt = f"""
You are a world-class AI Research Analyst, Data Analyst, and Technical Writer.

Your task is to generate a highly detailed, professional, analytical, and insightful research report STRICTLY using the provided research data.

TOPIC:
{topic}

RESEARCH DATA:
{web_data}

==================================================
IMPORTANT INSTRUCTIONS
==================================================

1. Use ONLY the provided research data.
2. Never invent facts, statistics, percentages, references, rankings, measurements, or values.
3. If information is missing, clearly mention the limitation.
4. Write in a professional research-report style.
5. Generate detailed explanations and deep analysis.
6. Expand sections properly with meaningful insights.
7. Use headings, subheadings, bullet points, and structured formatting.
8. Include:
   - trends
   - comparisons
   - advantages
   - disadvantages
   - opportunities
   - challenges
   - future scope
   - practical applications
   whenever relevant.

==================================================
DATA ANALYSIS & VISUALIZATION RULES
==================================================

If the research data contains:
- numbers
- statistics
- percentages
- rankings
- financial values
- measurements
- survey results
- timelines
- growth metrics
- comparisons
- scientific data

Then you MUST:

1. Extract the numerical data.
2. Analyze trends and relationships.
3. Compare values when possible.
4. Explain insights from the data.
5. Generate visualization-ready JSON.
6. Select the best chart automatically:
   - bar
   - line
   - pie
   - scatter
   - area

IMPORTANT:
You MUST ALWAYS return a VISUALIZATION_JSON section.

==================================================
VISUALIZATION JSON FORMAT
==================================================

If numerical/statistical data exists, return EXACTLY this format:

VISUALIZATION_JSON:
[
    {{
        "title": "Chart Title",
        "chart_type": "bar",
        "x": ["A", "B", "C"],
        "y": [10, 20, 30]
    }}
]

Rules:
- Return ONLY valid JSON.
- Use ONLY real extracted values.
- Never invent data.
- Arrays must be properly aligned.
- Multiple charts are allowed.
- Do not write explanations inside JSON.

If no valid numerical/statistical data exists, return EXACTLY:

VISUALIZATION_JSON:
[]

IMPORTANT:
Do NOT write only the heading "Visualization Data".
You MUST return actual VISUALIZATION_JSON.

==================================================
THEORETICAL TOPIC RULES
==================================================

If the topic is:
- theoretical
- conceptual
- philosophical
- descriptive
- historical
- educational

Then:
- Focus on explanation and analysis.
- Do NOT invent numerical data.
- Return:

VISUALIZATION_JSON:
[]

==================================================
OUTPUT FORMAT
==================================================

# Title

# Abstract

# Introduction

# Background / Context

# Key Concepts

# Detailed Analysis

# Key Findings

# Trend Analysis
(If applicable)

# Comparative Analysis
(If applicable)

# Statistical / Numerical Insights
(If applicable)

# Visualization Data

# Numerical Examples & Calculations
(Only if sufficient numerical data exists)

# Advantages

# Challenges / Limitations

# Future Scope

# Practical Applications

# Expert Insights

# Conclusion

# References

==================================================
FINAL RULES
==================================================

- Never hallucinate information.
- Never create fake statistics.
- Never create fake references.
- Never create fictional charts.
- Maintain professional research quality.
- Produce deep, meaningful, and useful analysis.
"""
        try:
            response = client.chat.completions.create(model="llama-3.3-70b-versatile",temperature=0.3,max_tokens=4000,messages=[{'role':'user','content':prompt}])
            result = response.choices[0].message.content
            if result:
                return result
        except Exception as e:
            print("key failed:",key[:10],e)
            continue
    return "All API keys exhausted"

#A function for question answer round
def question_answers(question, context):

    prompt = f"""
You are an intelligent AI research assistant.

Answer the question ONLY using the provided context.

If the answer is not available in the context, say:
"Answer not found in the provided report."

CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Be accurate
- Be concise
- Use bullet points if useful
- Do not hallucinate
"""

    for key in API_KEYS:
        try:

            client = Groq(api_key=key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2000,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Key failed: {e}")

    return "All API keys exhausted"
