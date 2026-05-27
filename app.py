import streamlit as st 
from research import question_answers,report_generate,rag_system,retrieve
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import json
import re
import pandas as pd
from xml.sax.saxutils import escape
st.set_page_config(layout="wide")
st.markdown("""
<h1 style="
    text-align:center;
    font-size:46px;
    font-weight:800;
    background: linear-gradient(90deg, #4B0082, #1F3A93);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
    AI Research Assistant
</h1>
""", unsafe_allow_html=True)

@st.cache_data
def cached_report(topic):
    return report_generate(topic)

if "report1" not in st.session_state:
    st.session_state.report1=None
if "index" not in st.session_state:
    st.session_state.index=None
if "chunks" not in st.session_state:
    st.session_state.chunks=None
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

def generate_pdf(report_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []
    # Title
    content.append(Paragraph("AI Research Report", styles["Title"]))
    content.append(Spacer(1, 12))
    # Body
    for line in report_text.split("\n"):
        safe_line = escape(line)
        content.append(Paragraph(safe_line, styles["BodyText"]))
        content.append(Spacer(1, 6))
    doc.build(content)
    buffer.seek(0)
    return buffer

def extract_graphs(report):

    try:
        if "VISUALIZATION_JSON:" not in report:
            return []

        json_text = report.split("VISUALIZATION_JSON:")[1].strip()

        # remove markdown blocks
        json_text = json_text.replace("```json", "")
        json_text = json_text.replace("```", "")

        # find first complete json array
        start = json_text.find("[")
        end = json_text.rfind("]") + 1

        json_text = json_text[start:end]

        return json.loads(json_text)

    except Exception as e:
        print("Graph Extraction Error:", e)
        return []

with st.form('AI Research Assistant'):
    topic=st.text_input('Enter Research Topic')
    submit=st.form_submit_button('Generate Report')

if (submit):
    if topic.strip()!="":    #strip is used to remove spaces in a string
        report1=cached_report(topic)
        st.session_state.report1 = report1 
        st.session_state.index=None  
        st.session_state.chunks=None
        st.session_state.qa_cache = {}

    else:
        st.error("Please Enter Query")
if st.session_state.report1 and st.session_state.index is None:
    with st.spinner("Building knowledge base..."):
        index, chunks = rag_system(st.session_state.report1)
        st.session_state.index = index
        st.session_state.chunks = chunks
if(st.session_state.report1):
    left,right=st.columns([4,1]) 
    with left:
        st.subheader("Reasearch Report")      
        pdf = generate_pdf(st.session_state.report1)
        st.download_button(label="Download Report PDF",data=pdf,file_name="ai_report.pdf",mime="application/pdf")
        if "VISUALIZATION_JSON:" in st.session_state.report1:
            clean_report = st.session_state.report1.split("VISUALIZATION_JSON:")[0]
        else:
            clean_report = st.session_state.report1
        st.markdown(clean_report)
        graphs = extract_graphs(st.session_state.report1)
        if graphs:
            st.subheader("📊 Visualizations")
            for graph in graphs:
                if not graph["x"] or not graph["y"]:
                    continue
                df = pd.DataFrame({
                    "Category": graph["x"],
                    "Value": graph["y"]
                    })
                chart_type = graph["chart_type"].lower()
                st.write(graph["title"])
                print(chart_type)
                if chart_type == "bar":
                    st.bar_chart(df.set_index("Category"))
                elif chart_type == "line":
                    st.line_chart(df.set_index("Category"))
                elif chart_type == "area":
                    st.area_chart(df.set_index("Category"))
                elif chart_type == "pie":
                    st.pyplot(df.set_index("Category").plot.pie(y="Value",autopct="%1.1f%%").figure)
                else:
                    st.dataframe(df)
        st.divider()   
    with right:
        question=st.text_input("Ask Questions",key="question_input")
        if(st.button("Ask")):
            if question.strip()!="":
                if st.session_state.index is not None:
                    if question in st.session_state.qa_cache:
                        answers = st.session_state.qa_cache[question]
                    else:
                        with st.spinner("Finding answer...."):  
                            context=retrieve(question,st.session_state.index,st.session_state.chunks)
                            answers=question_answers(question,context)
                        st.session_state.qa_cache[question] = answers
                    st.write(answers)
                else:
                    st.warning("Knowledge base not ready yet")
            else:
                st.warning("First type your query")
    
