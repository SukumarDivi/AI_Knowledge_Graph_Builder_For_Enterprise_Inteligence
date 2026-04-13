import time
import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Prompt ────────────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an intelligent job search assistant for an enterprise knowledge graph.
Use ONLY the job listings below that are truly relevant to the question. Ignore listings that do not match.

Retrieved job listings:
{context}

Question: {question}

Instructions:
- Count ONLY the jobs from the listings above that directly match the question criteria
  (category, location, country, workplace, employment type, priority, etc.).
- Start your answer with "I found X job(s)" where X is the exact count of matching jobs.
- If 0 jobs match the criteria, say "I found 0 jobs matching..." and explain why.
- Then briefly mention key locations and patterns in 2-3 sentences.
- Do NOT inflate or deflate the count. Be accurate."""


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Document builder ──────────────────────────────────────────────────────────
def jobs_to_documents(jobs) -> list:
    """
    Convert Job objects to LangChain Documents.

    FIX: Repeat country/city/workplace/category keywords in page_content so
    the embedding model weights them heavily. This is why India jobs were
    not being retrieved — the single mention was too weak for cosine similarity.
    """
    documents = []
    for job in jobs:
        city      = (job.city      or "Unknown").strip()
        country   = (job.country   or "Unknown").strip()
        region    = (job.region    or "Unknown").strip()
        workplace = (job.workplace or "Unknown").strip()
        category  = (job.category  or "Unknown").strip()

        enriched = (
            f"{job.text_description}\n\n"
            f"country:{country} country:{country} "
            f"city:{city} city:{city} "
            f"region:{region} "
            f"workplace:{workplace} workplace:{workplace} "
            f"category:{category} category:{category}"
        )

        documents.append(Document(
            page_content=enriched,
            metadata={
                "job_id":              job.job_id,
                "category":            category,
                "workplace":           workplace,
                "employment_type":     job.employment_type,
                "priority_class":      job.priority_class,
                "demand_score":        job.demand_score,
                "city":                city,
                "country":             country,
                "region":              region,
                "department_category": job.department_category,
            },
        ))
    return documents


# ── FAISS pipeline ────────────────────────────────────────────────────────────
def build_faiss_pipeline(jobs, groq_api_key, embedding_model, llm_model, top_k):
    """
    Build FAISS RAG pipeline.

    FIX: Removed score_threshold=0.3 which silently dropped India/Remote jobs.
    Now uses plain similarity so all top_k results are returned for filtering.
    """
    documents = jobs_to_documents(jobs)
    embeddings_model = get_embeddings(embedding_model)

    start = time.time()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings_model)
    index_time = round((time.time() - start) * 1000, 1)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=llm_model,
        temperature=0.1,
        max_tokens=512,
    )
    RAG_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever, index_time


# ── Pinecone pipeline ─────────────────────────────────────────────────────────
def build_pinecone_pipeline(
    jobs, groq_api_key, pinecone_api_key, index_name,
    embedding_model, llm_model, top_k
):
    """
    Build Pinecone RAG pipeline.

    FIX: Only create the index if it does not exist. Never recreate it —
    this was causing dimension-mismatch errors and duplicate vectors.
    """
    try:
        import os
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        documents = jobs_to_documents(jobs)
        embeddings_model = get_embeddings(embedding_model)

        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        pc = Pinecone(api_key=pinecone_api_key)

        existing_names = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_names:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(15)

        start = time.time()
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings_model,
            index_name=index_name,
        )
        index_time = round((time.time() - start) * 1000, 1)

        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.1,
            max_tokens=512,
        )
        RAG_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        return rag_chain, retriever, index_time

    except Exception as e:
        print(f"[Pinecone] pipeline error: {e}")
        return None, None, 0


# ── Keyword extraction ────────────────────────────────────────────────────────
# FIX: stop words must NOT include country/location/workplace/role words.
# These are the most important filter terms!
_STOP_WORDS = {
    "show", "me", "find", "get", "list", "give", "search",
    "listings", "listing", "positions", "position", "roles", "role",
    "the", "a", "an", "and", "or", "with", "that",
    "are", "is", "of", "to", "from", "on", "by", "all", "any", "some",
    "what", "which", "who", "where", "how", "many", "please",
}


def _extract_query_keywords(query: str) -> list:
    """Extract meaningful filter words from the query."""
    words = re.findall(r"[a-zA-Z]+", query.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def _job_matches_query(job_meta: dict, keywords: list) -> bool:
    """Return True if any keyword appears in the job metadata."""
    if not keywords:
        return True
    searchable = " ".join(str(v).lower() for v in job_meta.values())
    return any(kw in searchable for kw in keywords)


# ── Main search runner ────────────────────────────────────────────────────────
def run_search(rag_chain, retriever, query: str):
    """
    Run RAG search and return (answer, display_results, latency_ms).

    FIX 1: keyword post-filter keeps country/workplace words now.
    FIX 2: if LLM says 0 but retriever found keyword-matching jobs, trust retriever.
    FIX 3: FAISS and Pinecone now both go through same post-filter so counts match.
    """
    start = time.time()
    answer = rag_chain.invoke(query)
    latency = round((time.time() - start) * 1000, 1)

    retrieved = retriever.invoke(query)
    all_results = [doc.metadata for doc in retrieved]

    # Step 1 — keyword post-filter
    keywords = _extract_query_keywords(query)
    filtered = [r for r in all_results if _job_matches_query(r, keywords)]
    if not filtered:
        filtered = all_results   # fallback: nothing survived → show all

    # Step 2 — parse LLM-stated count
    count_match = re.search(
        r"(?:found|identified|retrieved|there\s+are|showing)\s+(\d+)",
        answer, re.IGNORECASE,
    )
    if not count_match:
        count_match = re.search(
            r"(?:found|identified|retrieved|there\s+are|showing)\s+(?:\w+\s+){0,5}?(\d+)\b",
            answer, re.IGNORECASE,
        )

    if count_match:
        ai_count = int(count_match.group(1))
        if ai_count == 0 and filtered:
            # LLM under-counted — trust the retriever's keyword-filtered results
            display_results = filtered
        else:
            display_results = filtered[: max(1, min(ai_count, len(filtered)))]
    else:
        display_results = filtered

    return answer, display_results, latency


# ── Node AI agent ─────────────────────────────────────────────────────────────
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

Be specific and professional. Do NOT use HTML tags in your response."""


def explain_node_with_agent(node_name, node_label, node_details, groq_api_key, llm_model):
    """Use Groq LLM to explain a clicked graph node."""
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.3,
            max_tokens=400,
        )
        prompt = NODE_AGENT_PROMPT.format(
            label=node_label,
            name=node_name,
            properties=str(node_details.get("properties", {})),
            relationships="\n".join(node_details.get("relationships", [])),
        )
        start = time.time()
        response = llm.invoke(prompt)
        latency = round((time.time() - start) * 1000, 1)
        return response.content, latency
    except Exception as e:
        return f"Agent error: {str(e)}", 0


# ── Email report ──────────────────────────────────────────────────────────────
def send_email_report(
    sendgrid_api_key,
    sender_email,
    recipient_email,
    subject,
    text_body,
    png_bytes=None,
    png_filename="subgraph.png",
):
    """Send email report via SendGrid with optional PNG attachment."""
    try:
        import base64
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import (
            Mail, Attachment, FileContent, FileName, FileType, Disposition,
        )

        message = Mail(
            from_email=sender_email,
            to_emails=recipient_email,
            subject=subject,
            html_content=text_body.replace("\n", "<br>"),
        )

        if png_bytes:
            encoded = base64.b64encode(png_bytes).decode()
            attachment = Attachment(
                FileContent(encoded),
                FileName(png_filename),
                FileType("image/png"),
                Disposition("attachment"),
            )
            message.attachment = attachment

        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        if response.status_code in (200, 202):
            return True, f"Report sent to {recipient_email} ✅"
        return False, f"SendGrid error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Email error: {str(e)}"