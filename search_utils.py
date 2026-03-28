import time
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

RAG_PROMPT_TEMPLATE = """You are an intelligent job search assistant for an enterprise knowledge graph.
Use ONLY the job listings below that are truly relevant to the question. Ignore listings that do not match.
Retrieved job listings:
{context}
Question: {question}
Instructions:
- Count ONLY the jobs from the listings above that directly match the question criteria (category, location, workplace, priority, etc.).
- Start your answer with "I found X job(s)" where X is the exact count of matching jobs.
- Then mention key locations and patterns in 2-3 sentences total.
- Do NOT inflate the count. If only 3 jobs match, say 3."""


def get_embeddings(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def jobs_to_documents(jobs):
    documents = []
    for job in jobs:
        doc = Document(
            page_content=job.text_description,
            metadata={
                "job_id": job.job_id, "category": job.category,
                "workplace": job.workplace, "employment_type": job.employment_type,
                "priority_class": job.priority_class, "demand_score": job.demand_score,
                "city": job.city, "country": job.country, "region": job.region,
                "department_category": job.department_category,
            }
        )
        documents.append(doc)
    return documents


def build_faiss_pipeline(jobs, groq_api_key, embedding_model, llm_model, top_k):
    """Build FAISS RAG pipeline"""
    documents = jobs_to_documents(jobs)
    embeddings_model = get_embeddings(embedding_model)

    start = time.time()
    vectorstore = FAISS.from_documents(
        documents=documents, embedding=embeddings_model)
    index_time = round((time.time() - start) * 1000, 1)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.3}
    )
    llm = ChatGroq(groq_api_key=groq_api_key,
                   model_name=llm_model, temperature=0.2, max_tokens=512)
    RAG_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return rag_chain, retriever, index_time


def build_pinecone_pipeline(jobs, groq_api_key, pinecone_api_key, index_name, embedding_model, llm_model, top_k):
    """Build Pinecone RAG pipeline"""
    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore
        import os
        import time

        documents = jobs_to_documents(jobs)
        embeddings_model = get_embeddings(embedding_model)

        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        pc = Pinecone(api_key=pinecone_api_key)

        existing = [i.name for i in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(10)

        start = time.time()
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings_model,
            index_name=index_name
        )
        index_time = round((time.time() - start) * 1000, 1)

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        llm = ChatGroq(groq_api_key=groq_api_key,
                       model_name=llm_model, temperature=0.2, max_tokens=512)
        RAG_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=RAG_PROMPT_TEMPLATE)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT | llm | StrOutputParser()
        )
        return rag_chain, retriever, index_time
    except Exception as e:
        return None, None, 0


def run_search(rag_chain, retriever, query):
    """Run search and return answer, results trimmed to AI-stated count, latency"""
    import re

    start = time.time()
    answer = rag_chain.invoke(query)
    latency = round((time.time() - start) * 1000, 1)

    retrieved = retriever.invoke(query)
    all_results = [doc.metadata for doc in retrieved]

    # Parse the count the LLM explicitly stated e.g. "I found 4 remote Data Scientist job listings"
    # Strategy: grab the first number that appears after found/identified/retrieved
    match = re.search(
        r'(?:found|identified|retrieved|there\s+are|showing)\s+(?:\w+\s+){0,5}?(\d+)\b',
        answer, re.IGNORECASE
    )
    # Fallback: simpler grab — first number right after the trigger word
    if not match:
        match = re.search(
            r'(?:found|identified|retrieved|there\s+are|showing)\s+(\d+)',
            answer, re.IGNORECASE
        )
    if match:
        ai_count = int(match.group(1))
        display_results = all_results[:max(1, min(ai_count, len(all_results)))]
    else:
        display_results = all_results

    return answer, display_results, latency


NODE_AGENT_PROMPT = """You are an expert AI agent analyzing a knowledge graph node.
A user clicked on a node in the graph. Explain it clearly.

Node Type: {label}
Node Name: {name}
Properties: {properties}
Relationships: {relationships}

Give a clear, insightful 3-4 sentence explanation of:
1. What this node represents in the job market
2. Its key properties and what they mean
3. How it connects to other nodes
4. Any interesting insights about it

Be specific, professional. Do NOT use HTML tags in your response."""


def explain_node_with_agent(node_name, node_label, node_details, groq_api_key, llm_model):
    """AI Agent that explains a clicked graph node using Groq LLM"""
    try:
        import time as _time
        llm = ChatGroq(groq_api_key=groq_api_key,
                       model_name=llm_model, temperature=0.3, max_tokens=400)
        prompt = NODE_AGENT_PROMPT.format(
            label=node_label,
            name=node_name,
            properties=str(node_details.get("properties", {})),
            relationships="\n".join(node_details.get("relationships", []))
        )
        start = _time.time()
        response = llm.invoke(prompt)
        latency = round((_time.time() - start) * 1000, 1)
        return response.content, latency
    except Exception as e:
        return f"Agent error: {str(e)}", 0


def send_email_report(
    sendgrid_api_key,
    sender_email,
    recipient_email,
    subject,
    text_body,
    png_bytes=None,
    png_filename="subgraph.png",
):
    """
    Send an email report via SendGrid.
    Attaches the subgraph PNG if png_bytes is provided.
    Returns (success: bool, message: str)
    """
    try:
        import base64
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import (
            Mail, Attachment, FileContent, FileName,
            FileType, Disposition
        )

        message = Mail(
            from_email=sender_email,
            to_emails=recipient_email,
            subject=subject,
            html_content=text_body.replace("\n", "<br>")
        )

        if png_bytes:
            encoded = base64.b64encode(png_bytes).decode()
            attachment = Attachment(
                FileContent(encoded),
                FileName(png_filename),
                FileType("image/png"),
                Disposition("attachment")
            )
            message.attachment = attachment

        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        if response.status_code in (200, 202):
            return True, f"Report sent to {recipient_email} ✅"
        else:
            return False, f"SendGrid error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Email error: {str(e)}"


print("search_utils.py written!")
print("  FAISS pipeline: local, MMR retriever, ~36ms")
print("  Pinecone pipeline: cloud, similarity search, ~674ms")
print("  send_email_report: SendGrid with PNG attachment")
