"""
Advanced RAG Patterns
Multi-query, self-query, compression, hybrid search
"""

# MultiQueryRetriever: instead of sending your question as-is,
# it generates MULTIPLE versions of your question to retrieve more relevant docs
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# ContextualCompressionRetriever: after fetching docs, it compresses them
# to only keep the parts relevant to your question (removes noise)
from langchain_classic.retrievers import ContextualCompressionRetriever

# LLMChainExtractor: the actual compressor — uses an LLM to extract
# only the relevant sentences from each retrieved document
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# EnsembleRetriever: combines multiple retrievers (e.g. keyword + semantic)
# and merges their results with weighted scoring
from langchain_classic.retrievers import EnsembleRetriever

# BM25Retriever: classic keyword-based search algorithm (no embeddings needed)
# BM25 = "Best Match 25" — counts word frequency to find relevant docs
# Great for exact keyword matches like "PostgreSQL" or "ACID"
from langchain_community.retrievers import BM25Retriever

# ParentDocumentRetriever: uses SMALL chunks to search (more precise)
# but returns the LARGE parent chunk as context (more information)
# Solves the tradeoff between search precision and answer quality
from langchain_classic.retrievers import ParentDocumentRetriever

# InMemoryStore: simple in-memory key-value store
# used by ParentDocumentRetriever to store the parent (large) chunks
from langchain_classic.storage import InMemoryStore

# Chroma: a local vector database that stores embeddings on disk
# used to store and search document embeddings
from langchain_chroma import Chroma

# ChatOpenAI: wrapper around OpenAI's chat models (e.g. gpt-4o-mini)
# OpenAIEmbeddings: converts text into embedding vectors using OpenAI's model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Document: LangChain's data class for a piece of text + its metadata
# e.g. Document(page_content="...", metadata={"source": "file.pdf"})
from langchain_core.documents import Document

# RecursiveCharacterTextSplitter: splits long text into smaller chunks
# tries to split on paragraph → sentence → word boundaries in order
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChatPromptTemplate: creates reusable prompt templates with variables
# e.g. "Answer this question: {question} using this context: {context}"
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser: parses the LLM's response object into a plain string
from langchain_core.output_parsers import StrOutputParser

# RunnablePassthrough: passes the input unchanged to the next step in the chain
# used when you want one part of the chain to receive the original input
from langchain_core.runnables import RunnablePassthrough

# python-dotenv: loads environment variables from a .env file
# your OPENAI_API_KEY lives in .env so it's not hardcoded in code
from dotenv import load_dotenv

# Python's built-in logging module — used to print info/debug messages
import logging

# Actually loads the .env file into environment variables
# Must be called before any OpenAI API calls so the key is available
load_dotenv()

# Configure logging so we can see what's happening inside LangChain
# format="%(name)s - %(message)s" shows the logger name + the message
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

# Specifically turn on INFO logs for the multi-query retriever
# so we can see what alternative queries it generates
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# INFO_BURIED: documents where relevant info is buried inside long text
# used to demonstrate why contextual compression is useful —
# the LLM needs to dig out just the relevant part from a wall of text
INFO_BURIED = [
    Document(
        # Long mixed document — has company history, tech stack, AND LangChain info
        # The LangChain info is buried in the middle (hard to find without compression)
        page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK
...""",
        # metadata tells us where this document came from
        metadata={"source": "acme_company_overview.pdf"},
    ),
    Document(
        page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4
...""",
        metadata={"source": "technical_docs_v2.4.pdf"},
    ),
]

# TECH_DOCS: our main knowledge base — 8 clean documents on different topics
# These are the documents we'll embed, store in Chroma, and retrieve from
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language...",
        # metadata lets us filter documents by topic, difficulty, etc.
        # e.g. only retrieve "beginner" difficulty docs
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web...",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI...",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications...",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications...",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications...",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database...",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant...",
        metadata={
            "topic": "database",
            "type": "vector",
            "difficulty": "intermediate",
        },
    ),
]


def create_base_vectorstore():
    """Create a basic vector store for demos."""

    # Chroma.from_documents does 3 things in one call:
    # 1. Takes each Document's page_content
    # 2. Sends it to OpenAI's embedding model → gets a vector (list of numbers)
    # 3. Stores both the vector AND the original text in a local Chroma DB
    # text-embedding-3-small is OpenAI's fast, cheap embedding model
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )


def demo_multi_query_retriever():
    """Multi-Query Retriever generates multiple query perspectives."""

    print("=" * 60)
    print("MULTI-QUERY RETRIEVER")
    print("Generates multiple perspectives on your question")
    print("=" * 60)

    # Build the vector store with all our tech docs embedded
    vectorstore = create_base_vectorstore()

    # temperature=0.3 adds slight creativity — good for generating
    # varied alternative phrasings of the same question
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # MultiQueryRetriever wraps a regular retriever with LLM-powered query expansion
    # How it works:
    # 1. Takes your original question
    # 2. Uses the LLM to generate 3-5 alternative phrasings of that question
    # 3. Runs ALL versions through the base retriever
    # 4. Deduplicates and merges all results
    # This catches documents that match some phrasings but not others
    retriever = MultiQueryRetriever.from_llm(
        # base retriever: standard semantic search, return top 2 docs per query
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm  # LLM used to generate alternative queries
    )

    # Original question — deliberately vague to show how multi-query helps
    query = "What tools can I use to build AI applications?"

    print(f"\nOriginal Query: {query}")
    print("\nThe retriever will generate multiple query variations...")
    # The generated alternative queries will appear in the INFO logs
    # e.g. "What frameworks are available for LLM development?"
    #      "Which Python libraries help build AI systems?"
    print("(Check INFO logs above for generated queries)\n")

    # invoke() triggers the full multi-query process:
    # generate alternatives → retrieve for each → deduplicate → return
    docs = retriever.invoke(query)

    # We get MORE unique docs than a single query would return
    # because different query phrasings hit different documents
    print(f"Retrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        # doc.metadata.get('topic', 'N/A') safely gets the topic or 'N/A' if missing
        # [:100] shows first 100 characters to keep output readable
        print(
            f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )


def demo_contextual_compression():
    """Contextual Compression extracts only relevant parts."""

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extracts only query-relevant content from documents")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    # temperature=0 for compression — we want deterministic extraction,
    # not creative variation. Always extract the same relevant sentences.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # LLMChainExtractor is the compression component
    # It takes each retrieved document and asks the LLM:
    # "Given this question, which sentences in this document are actually relevant?"
    # It returns ONLY those relevant sentences, discarding the rest
    compressor = LLMChainExtractor.from_llm(llm)

    # ContextualCompressionRetriever wraps a base retriever with the compressor
    # Pipeline:
    # 1. base_retriever fetches k=4 documents (full text)
    # 2. compressor filters each doc down to only relevant content
    # 3. You get shorter, more focused chunks back
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        # fetch 4 docs so after compression we still have enough useful content
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exist for building LLM applications?"

    print(f"\nQuery: {query}")

    # First, show what you get WITHOUT compression (full chunks)
    # as_retriever() converts the vectorstore into a retriever object
    # search_kwargs={"k": 2} means return top 2 most similar documents
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        # Shows the full length of each retrieved document
        print(f"Length: {len(doc.page_content)} chars")
        # Shows first 150 chars — you'll see a lot of irrelevant info included
        print(f"Content: {doc.page_content[:150]}...\n")

    # Now show what you get WITH compression (relevant parts only)
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        # Much shorter — only the sentences that actually answer the question
        print(f"Length: {len(doc.page_content)} chars")
        # Print full compressed content — should be much more focused
        print(f"Content: {doc.page_content}\n")


def demo_ensemble_hybrid_search():
    """Hybrid search combining keyword (BM25) and semantic search."""

    print("=" * 60)
    print("ENSEMBLE/HYBRID RETRIEVER")
    print("Combines keyword (BM25) + semantic search")
    print("=" * 60)

    # Build semantic vector store
    vectorstore = create_base_vectorstore()

    # BM25Retriever: pure keyword matching — no embeddings, no AI
    # Works like a traditional search engine (counts word occurrences)
    # Great for: exact technical terms, product names, acronyms
    # Bad for: understanding meaning, synonyms, paraphrasing
    bm25_retriever = BM25Retriever.from_documents(TECH_DOCS)
    bm25_retriever.k = 3  # return top 3 keyword matches

    # Semantic retriever: embedding-based similarity search
    # Great for: understanding meaning, paraphrasing, concepts
    # Bad for: exact keyword matching (might miss specific terms)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # EnsembleRetriever merges both retrievers using Reciprocal Rank Fusion (RRF)
    # RRF combines the ranked lists from each retriever into one final ranking
    # weights=[0.4, 0.6] means:
    #   40% influence from BM25 (keyword)
    #   60% influence from semantic search
    # You tune weights based on your use case
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
    )

    # Test 3 different query types to show when each retriever shines
    queries = [
        "ACID transactions",                              # keyword-heavy → BM25 wins
        "How do I store AI model outputs for later retrieval?",  # conceptual → semantic wins
        "fast similarity lookup for embeddings",          # mixed → ensemble wins
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        # Run each retriever independently so we can compare results
        bm25_results = bm25_retriever.invoke(query)
        semantic_results = semantic_retriever.invoke(query)
        ensemble_results = ensemble_retriever.invoke(query)

        # Show only the TOP result from each retriever for easy comparison
        # [0] = first (best) result, [:60] = first 60 chars
        print(f"BM25 top result:     {bm25_results[0].page_content[:60]}...")
        print(f"Semantic top result: {semantic_results[0].page_content[:60]}...")
        print(f"Ensemble top result: {ensemble_results[0].page_content[:60]}...")
        # Key insight: ensemble result should be the best of both worlds


def demo_parent_document_retriever():
    """Parent Document Retriever: small chunks for search, large for context."""

    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("=" * 60)

    # Long document to demonstrate the parent/child split
    # Real-world equivalent: a long PDF, article, or documentation page
    long_doc = Document(
        page_content="""
# Complete Guide to Building AI Agents
...
        """,
        metadata={"source": "ai_agents_guide.md"},
    )

    # parent_splitter: creates LARGE chunks (800 chars) — these are the "parents"
    # These become what's returned to the user — enough context to answer questions
    # chunk_overlap=100 means adjacent chunks share 100 chars (prevents losing info at boundaries)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # child_splitter: creates SMALL chunks (200 chars) — these are the "children"
    # These get embedded and stored in the vector store for precise searching
    # Small chunks = more precise embedding = better search accuracy
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Chroma vectorstore stores the SMALL child chunk embeddings
    # This is what gets searched when a query comes in
    vectorstore = Chroma(
        collection_name="parent_child_demo",  # name this collection in Chroma
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    # InMemoryStore stores the LARGE parent chunks by ID
    # When a child chunk is found, we use its parent_id to fetch the full parent
    store = InMemoryStore()

    # ParentDocumentRetriever ties it all together:
    # - Splits docs into parents (large) and children (small)
    # - Stores children in vectorstore (for search)
    # - Stores parents in docstore (for retrieval)
    # - At query time: search children → find matching parent → return parent
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,        # where child embeddings live
        docstore=store,                 # where parent full text lives
        child_splitter=child_splitter,  # how to create small searchable chunks
        parent_splitter=parent_splitter, # how to create large context chunks
    )

    # add_documents: splits the long doc into parents and children,
    # embeds the children, stores everything in the right places
    retriever.add_documents([long_doc])

    query = "What is LangGraph used for?"

    print(f"\nQuery: {query}")

    # similarity_search directly on the vectorstore returns the RAW CHILD chunk
    # This is what a normal retriever would return — small and potentially missing context
    child_docs = vectorstore.similarity_search(query, k=1)
    print(f"\n--- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")  # will be ~200 chars
    print(f"Content: {child_docs[0].page_content}")

    # ParentDocumentRetriever.invoke() does the full process:
    # 1. Search vectorstore → find best matching child chunk
    # 2. Look up that child's parent_id in the docstore
    # 3. Return the full PARENT chunk instead
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")  # will be ~800 chars
    # [:300] preview since parent is much larger
    print(f"Content preview: {parent_docs[0].page_content[:300]}...")
    # Key insight: search was precise (child), answer has full context (parent)


def demo_advanced_rag_chain():
    """Complete RAG chain with advanced retrieval."""

    print("=" * 60)
    print("COMPLETE ADVANCED RAG CHAIN")
    print("Multi-query + Compression + RAG")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    # temperature=0 for the answer LLM — we want factual, consistent answers
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Layer 1: MultiQueryRetriever
    # Expands one question into multiple → casts a wider net → better recall
    # k=3 means fetch top 3 docs per generated query variation
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )

    # Layer 2: Contextual Compression on top of Multi-Query
    # After multi-query fetches many docs, compression removes irrelevant sentences
    # So we get: broad retrieval → focused content
    compressor = LLMChainExtractor.from_llm(llm)
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever  # compression wraps multi-query
    )

    # RAG prompt template
    # {context} will be filled with retrieved (and compressed) document content
    # {question} will be filled with the user's original question
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the following context. Be specific and cite which technologies you're referring to.

Context:
{context}

Question: {question}

Answer:"""
    )

    # Helper function: takes a list of Document objects and
    # joins their text with double newlines into one big string
    # This is what gets inserted into the {context} slot of the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the complete RAG chain using LangChain Expression Language (LCEL)
    # The | operator chains steps together like Unix pipes
    #
    # Chain breakdown:
    # Step 1: Input is the user's question (a string)
    # Step 2: Two things happen in PARALLEL (the dict):
    #   - "context": question → advanced_retriever → list of docs → format_docs → string
    #   - "question": RunnablePassthrough() → question passes through unchanged
    # Step 3: The dict {"context": "...", "question": "..."} goes into the prompt template
    # Step 4: Formatted prompt string goes to the LLM
    # Step 5: LLM response object goes to StrOutputParser → plain string
    rag_chain = (
        {
            # advanced_retriever takes the question, returns relevant compressed docs
            # format_docs converts the list of docs into one context string
            "context": advanced_retriever | format_docs,

            # RunnablePassthrough() just forwards the question as-is
            # so it's available in the {question} slot of the prompt
            "question": RunnablePassthrough()
        }
        | prompt          # fills {context} and {question} into the template
        | llm             # sends completed prompt to GPT-4o-mini
        | StrOutputParser() # converts LLM response object to plain string
    )

    # Test questions that require understanding from the knowledge base
    questions = [
        "What options do I have for building AI agents?",
        "How can I store and search embeddings?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        # invoke() runs the full chain end-to-end:
        # question → multi-query retrieval → compression → prompt → LLM → string answer
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")


if __name__ == "__main__":
    # Only demo_advanced_rag_chain() is uncommented — it runs when you execute this file
    # The others are commented out — uncomment to run individual demos
    # demo_multi_query_retriever()   # try this first to understand query expansion
    # demo_contextual_compression()  # then this to understand compression
    # demo_ensemble_hybrid_search()  # then this for hybrid search
    # demo_parent_document_retriever() # then this for parent/child chunking
    demo_advanced_rag_chain()  # this combines multi-query + compression into one chain