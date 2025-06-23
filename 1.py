# app.py - Enhanced PDF Research Assistant
# Last Updated: Tuesday, June 25, 2024
# Location: Pune, Maharashtra, India

import streamlit as st
import spacy
from string import punctuation
from collections import Counter
from pyvis.network import Network
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64
import matplotlib.pyplot as plt
from streamlit_pdf_viewer import pdf_viewer

# ---[ 1. NLP MODEL LOADING ]---
# Use a more advanced spaCy model with word vectors for improved semantic understanding.
@st.cache_resource
def load_nlp_model():
    """
    Loads the 'en_core_web_md' spaCy model. This model includes word vectors,
    which significantly improves the accuracy of semantic similarity tasks like Q&A.
    """
    try:
        nlp = spacy.load("en_core_web_md")
        return nlp
    except OSError:
        st.error(
            "🚨 SpaCy model 'en_core_web_md' not found. "
            "Please run the following command in your terminal to download it:\n"
            "`python -m spacy download en_core_web_md`"
        )
        st.stop()

nlp = load_nlp_model()


# ---[ 2. CORE PDF & DATA PROCESSING FUNCTIONS ]---

@st.cache_data(show_spinner=False)
def extract_data_from_pdf(pdf_file):
    """
    Extracts text, images, and metadata from each page of the uploaded PDF.
    It now also stores the raw PDF bytes in the session state for the viewer.
    """
    try:
        # Read the uploaded file's bytes and store for the PDF viewer
        pdf_bytes = pdf_file.read()
        st.session_state.pdf_bytes = pdf_bytes
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_data = []

        progress_bar = st.progress(0, text="Analyzing pages...")
        total_pages = len(doc)

        for i, page in enumerate(doc):
            progress_bar.progress((i + 1) / total_pages, text=f"Analyzing page {i+1}/{total_pages}")
            
            text = page.get_text()
            
            # Extract high-resolution image of the page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            pages_data.append({
                "page_num": i + 1,
                "text": text,
                "image": img,
                "word_count": len(text.split()),
                "char_count": len(text)
            })

        progress_bar.empty()
        doc.close()
        return pages_data
    except Exception as e:
        st.error(f"❌ An error occurred while reading the PDF: {e}")
        return []

# ---[ 3. AI-POWERED ANALYSIS FUNCTIONS ]---

def get_enhanced_summary(full_text):
    """
    Generates a summary using a TF-IDF approach to identify the most important sentences.
    """
    if not full_text:
        return "No text available to summarize.", [], {}

    doc = nlp(full_text)
    sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 15]

    if len(sentences) < 3:
        return "The document is too short for a meaningful summary.", [], {}

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
         return "Could not process the text for summarization due to its content.", [], {}

    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    # Summarize with the top 5 sentences or 20% of sentences, whichever is smaller
    num_summary_sentences = min(5, len(sentences) // 5)
    if num_summary_sentences < 2:
        num_summary_sentences = min(5, len(sentences))

    top_indices = sentence_scores.argsort()[-num_summary_sentences:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_indices)] # Sort to maintain original order
    paragraph_summary = " ".join(top_sentences)

    stats = {
        "total_sentences": len(sentences),
        "total_words": len(full_text.split()),
        "summary_compression": round((1 - len(paragraph_summary) / len(full_text)) * 100, 1) if full_text else 0
    }
    return paragraph_summary, top_sentences, stats

def get_enhanced_entities(text):
    """
    Extracts named entities from the text using the spaCy model.
    """
    if not text:
        return {}

    doc = nlp(text)
    entities_by_label = {}
    entity_descriptions = {
        "PERSON": "👤 People", "ORG": "🏢 Organizations", "GPE": "🌍 Geopolitical Entities",
        "LOC": "📍 Locations", "PRODUCT": "📦 Products", "EVENT": "🎉 Events",
        "DATE": "📅 Dates", "MONEY": "💰 Monetary Values", "NORP": "👥 Nationalities/Groups",
        "FAC": "🏗️ Facilities", "WORK_OF_ART": "🎨 Artworks", "LAW": "⚖️ Legal Documents"
    }

    for ent in doc.ents:
        if ent.label_ in entity_descriptions and len(ent.text.strip()) > 2:
            label = entity_descriptions[ent.label_]
            if label not in entities_by_label:
                entities_by_label[label] = []
            
            # Add entity if not already present in the list for that label
            if not any(d['text'] == ent.text.strip() for d in entities_by_label[label]):
                entities_by_label[label].append({'text': ent.text.strip()})

    return entities_by_label

def enhanced_question_answering(pages_data, question):
    """
    Answers questions using semantic similarity. The 'en_core_web_md' model's
    word vectors provide a more accurate understanding of context.
    """
    if not pages_data:
        return "Please upload and analyze a document first.", None, None, 0

    all_sentences = []
    sentence_page_map = []
    for page_data in pages_data:
        doc = nlp(page_data["text"])
        for sent in doc.sents:
            if len(sent.text.strip()) > 10:
                all_sentences.append(sent)
                sentence_page_map.append({
                    'page_num': page_data['page_num'],
                    'image': page_data['image']
                })

    if not all_sentences:
        return "No suitable content was found in the document for answering questions.", None, None, 0

    question_doc = nlp(question)
    # Use spaCy's built-in similarity, which is highly effective with 'md' models.
    similarities = np.array([question_doc.similarity(sent) for sent in all_sentences])
    
    # A confidence threshold of 0.5 helps filter out irrelevant answers.
    if np.max(similarities) < 0.5:
        return "I could not find a relevant answer in the document. Please try rephrasing your question.", None, None, 0

    best_indices = similarities.argsort()[-3:][::-1]
    answer_sentences = [all_sentences[i].text for i in best_indices if similarities[i] > 0.5]
    answer = "\n\n".join(answer_sentences)

    best_match_info = sentence_page_map[best_indices[0]]
    confidence = round(similarities[best_indices[0]] * 100, 1)

    return answer, best_match_info['page_num'], best_match_info['image'], confidence

# ---[ 4. VISUALIZATION FUNCTIONS ]---

def create_knowledge_graph(text):
    """
    Generates an interactive knowledge graph visualizing relationships between entities.
    """
    if not text:
        return False
        
    doc = nlp(text)
    entities = [ent for ent in doc.ents if ent.label_ in 
                ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "NORP"]]
    
    if len(entities) < 2:
        return False
    
    net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white",
                  notebook=True, cdn_resources='remote')
    
    color_map = {
        "PERSON": "#ff6b6b", "ORG": "#4ecdc4", "GPE": "#45b7d1",
        "LOC": "#96ceb4", "PRODUCT": "#feca57", "EVENT": "#ff9ff3", "NORP": "#54a0ff"
    }
    
    added_nodes = set()
    for ent in entities:
        if ent.text not in added_nodes:
            color = color_map.get(ent.label_, "#cccccc")
            net.add_node(ent.text, label=ent.text, title=f"{ent.text} ({ent.label_})",
                         color=color, size=25)
            added_nodes.add(ent.text)
    
    for sent in doc.sents:
        sent_ents = list(set([ent.text for ent in sent.ents if ent.text in added_nodes]))
        if len(sent_ents) > 1:
            for i in range(len(sent_ents)):
                for j in range(i + 1, len(sent_ents)):
                    net.add_edge(sent_ents[i], sent_ents[j], width=2)
    
    net.set_options("""
    var options = {
      "physics": { "enabled": true, "stabilization": {"iterations": 100} },
      "nodes": { "borderWidth": 2, "borderColor": "#ffffff", "font": {"size": 14, "color": "#ffffff"}, "shadow": true },
      "edges": { "color": {"color": "#848484"}, "smooth": true, "shadow": true }
    }
    """)
    
    try:
        net.save_graph("knowledge_graph.html")
        return True
    except Exception as e:
        st.error(f"Error creating knowledge graph: {e}")
        return False

def create_static_analytics_charts(pages_data, entities):
    """
    Creates static analytics charts using Matplotlib for a clean, non-interactive look.
    """
    if not pages_data:
        return None, None
    
    # --- Page Statistics ---
    page_stats = [{'Page': p['page_num'], 'Words': p['word_count']} for p in pages_data]
    df_pages = pd.DataFrame(page_stats)

    # --- Entity Distribution ---
    entity_counts = {label.split(' ')[1].replace('/Groups',''): len(items)
                     for label, items in entities.items()}

    # --- Create Visualizations using Matplotlib ---
    plt.style.use('dark_background')

    # Bar Chart: Word Count per Page
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df_pages['Page'], df_pages['Words'], color='#4ecdc4', zorder=2)
    ax1.set_title('📊 Word Count per Page', fontsize=16, color='white')
    ax1.set_xlabel('Page Number', fontsize=12, color='white')
    ax1.set_ylabel('Word Count', fontsize=12, color='white')
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    ax1.tick_params(colors='white')
    fig1.tight_layout()

    # Pie Chart: Entity Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if entity_counts:
        wedges, texts, autotexts = ax2.pie(entity_counts.values(), labels=None, autopct='%1.1f%%',
                startangle=140, colors=plt.cm.viridis(np.linspace(0, 1, len(entity_counts))),
                pctdistance=0.85)
        plt.setp(autotexts, size=10, weight="bold", color="white")
        ax2.legend(wedges, entity_counts.keys(), title="Entities", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax2.set_title('🎯 Entity Distribution', fontsize=16, color='white')
    fig2.tight_layout()

    return fig1, fig2


# ---[ 5. STREAMLIT UI SETUP ]---

st.set_page_config(page_title="🚀 NexusAI PDF Assistant", layout="wide",
                   initial_sidebar_state="expanded", page_icon="🚀")

# --- Modern CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 { color: #FFFFFF; }

    h1 {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .answer-box {
        background: rgba(255, 255, 255, 0.95);
        color: #1e3c72;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #4ecdc4;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: white;
        font-weight: 600;
        padding: 1rem 1.5rem;
        margin-right: 0.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    }
    
    .entity-badge {
        display: inline-block;
        background: #4ecdc4;
        color: white;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>🚀 NexusAI PDF Assistant</h1>", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'pages_data' not in st.session_state:
    st.session_state.pages_data = []
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""
if 'summary_para' not in st.session_state:
    st.session_state.summary_para = ""
if 'summary_stats' not in st.session_state:
    st.session_state.summary_stats = {}
if 'entities' not in st.session_state:
    st.session_state.entities = {}
if 'qa_result' not in st.session_state:
    st.session_state.qa_result = (None, None, None, 0)
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.markdown("### 📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file to begin analysis.", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"**File:** `{uploaded_file.name}`\n\n**Size:** `{file_size_mb:.2f} MB`")

    st.markdown("---")

    if st.button("🚀 Analyze Document", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            st.session_state.pages_data = extract_data_from_pdf(uploaded_file)
            if st.session_state.pages_data:
                with st.spinner('Performing AI analysis... This may take a moment.'):
                    st.session_state.full_text = "\n".join([page['text'] for page in st.session_state.pages_data])
                    
                    # Perform all analyses
                    para, _, stats = get_enhanced_summary(st.session_state.full_text)
                    st.session_state.summary_para = para
                    st.session_state.summary_stats = stats
                    st.session_state.entities = get_enhanced_entities(st.session_state.full_text)
                    
                    # Reset states for Q&A and Graph
                    st.session_state.qa_result = (None, None, None, 0)
                    st.session_state.graph_generated = False
                    
                st.balloons()
                st.success("🎉 Analysis Complete!")
            else:
                st.error("❌ Could not process the PDF. It might be empty or corrupted.")
        else:
            st.warning("⚠️ Please upload a PDF file first.")

# --- Main Content Area ---
if not st.session_state.pages_data:
    # Welcome screen
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.1); 
                     border-radius: 20px; backdrop-filter: blur(10px);">
            <h2>🎯 Welcome to NexusAI</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
                Upload a PDF document in the sidebar and click 'Analyze' to unlock:
            </p>
            <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">📊</div><p style="color: white;">Analytics</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">💬</div><p style="color: white;">Q&A</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 3rem;">🕸️</div><p style="color: white;">Knowledge Graphs</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # --- Analytics Overview Metrics ---
    st.markdown("### 📈 Document Overview")
    total_words = st.session_state.summary_stats.get('total_words', 0)
    reading_time = round(total_words / 200, 1)
    total_entities = sum(len(items) for items in st.session_state.entities.values())
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("📄 Pages", len(st.session_state.pages_data))
    m_col2.metric("📝 Words", f"{total_words:,}")
    m_col3.metric("🎯 Entities", total_entities)
    m_col4.metric("⏱️ Read Time", f"~{reading_time} min")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Tabs for different features ---
    tab_analytics, tab_summary, tab_qa, tab_graph, tab_reader = st.tabs([
        "📊 Analytics", "📝 Summary & Entities", "💬 Q&A", "🕸️ Knowledge Graph", "📖 PDF Reader"
    ])

    with tab_analytics:
        st.markdown("### 📊 Static Analytics Dashboard")
        fig1, fig2 = create_static_analytics_charts(st.session_state.pages_data, st.session_state.entities)
        
        if fig1 and fig2:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
        else:
            st.info("Not enough data to generate analytics charts.")
            
    with tab_summary:
        st.markdown("### 🎯 AI-Generated Summary")
        if st.session_state.summary_para:
            st.markdown(f"<div class='answer-box'>{st.session_state.summary_para}</div>", unsafe_allow_html=True)
        
        st.markdown("### 🏷️ Extracted Entities")
        if st.session_state.entities:
            for label, items in st.session_state.entities.items():
                with st.expander(f"{label} ({len(items)} found)"):
                    st.markdown(" ".join(f"<span class='entity-badge'>{item['text']}</span>" for item in items), unsafe_allow_html=True)
        else:
            st.info("No entities were found in the document.")

    with tab_qa:
        st.markdown("### 💬 Intelligent Q&A System")
        question = st.text_input("🤔 Ask anything about your document:", 
                                 placeholder="e.g., What were the main conclusions of the study?")
        
        if st.button("💡 Get Answer", use_container_width=True) and question:
            with st.spinner('🧠 Searching for the best answer...'):
                answer, page_num, page_image, confidence = enhanced_question_answering(
                    st.session_state.pages_data, question)
                st.session_state.qa_result = (answer, page_num, page_image, confidence)
        
        answer, page_num, page_image, confidence = st.session_state.qa_result
        if answer:
            st.markdown(f"#### 🎯 Answer (Confidence: {confidence}%)")
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            
            if page_num and page_image:
                with st.expander(f"📄 View Source: Page {page_num}"):
                    st.image(page_image, caption=f"Content found on page {page_num}", use_column_width=True)

    with tab_graph:
        st.markdown("### 🕸️ Interactive Knowledge Graph")
        st.markdown("Visualize connections between key entities in your document.")
        
        if st.button("🎨 Generate Knowledge Graph", use_container_width=True):
            with st.spinner("🛠️ Building interactive knowledge graph..."):
                if create_knowledge_graph(st.session_state.full_text):
                    st.session_state.graph_generated = True
                    st.success("✅ Knowledge graph is ready!")
                else:
                    st.error("❌ Could not generate the graph. The document may have too few recognized entities.")
        
        if st.session_state.graph_generated and os.path.exists("knowledge_graph.html"):
            with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=650, scrolling=False)

    with tab_reader:
        st.markdown("### 📖 Full Document Reader")
        st.markdown("Read, search, and navigate your PDF document directly.")
        if st.session_state.pdf_bytes:
            pdf_viewer(st.session_state.pdf_bytes, height=700)
        else:
            st.warning("Please upload and analyze a document to use the PDF reader.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
    <p>🚀 Built with NexusAI - Transforming Documents into Intelligence</p>
</div>
""", unsafe_allow_html=True)
