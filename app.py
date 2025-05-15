import streamlit as st
import pandas as pd
import openai
import os
import numpy as np

# --- Configuration & Secrets ---
# Azure OpenAI setup using Streamlit secrets
openai.api_type = "azure"
openai.api_key = st.secrets.get("AZURE_OPENAI_API_KEY")
openai.api_base = st.secrets.get("AZURE_API_BASE")
openai.api_version = st.secrets.get("AZURE_API_VERSION") # Example: "2023-07-01-preview" or "2024-02-15-preview"

# Ensure secrets are loaded
if not all([openai.api_key, openai.api_base, openai.api_version]):
    st.error("❌ Critical Azure OpenAI secrets are missing. Please configure AZURE_OPENAI_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION in your Streamlit secrets.")
    st.stop()

chat_deployment_name = st.secrets.get("AZURE_DEPLOYMENT_NAME") # Your chat model deployment (e.g., gpt-4, gpt-35-turbo)
embedding_deployment_name = st.secrets.get("AZURE_EMBEDDING_DEPLOYMENT") # Your embedding model deployment (e.g., text-embedding-ada-002)

if not chat_deployment_name:
    st.error("❌ AZURE_CHAT_DEPLOYMENT_NAME is missing from Streamlit secrets.")
    st.stop()
if not embedding_deployment_name:
    st.error("❌ AZURE_EMBEDDING_DEPLOYMENT_NAME is missing from Streamlit secrets.")
    st.stop()


# --- Helper Functions ---
def get_embedding(text, engine):
    """Generates an embedding for the given text using Azure OpenAI."""
    try:
        response = openai.Embedding.create(engine=engine, input=text)
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"❌ Error generating embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def chunk_dataframe(df, sheet_name, chunk_size=5):
    """Splits a DataFrame into textual chunks."""
    chunks = []
    df_string = df.to_string(index=False) # Full dataframe string for context if needed later
    
    # Add column headers to each chunk for context
    header = ", ".join(df.columns.tolist())
    
    for i in range(0, len(df), chunk_size):
        chunk_df = df[i:i + chunk_size]
        # More descriptive chunk text
        chunk_text = f"Sheet: {sheet_name}\nColumns: {header}\nRows {i+1} to {i+len(chunk_df)}:\n"
        chunk_text += chunk_df.to_string(index=False, header=False) # Don't repeat header in each row part
        chunks.append({"text": chunk_text, "sheet": sheet_name, "rows": f"{i+1}-{i+len(chunk_df)}"})
    return chunks

# --- Streamlit App ---
st.set_page_config(page_title="📊 P&L Analyzer", layout="wide") # Changed to wide for better display
st.title("📊 P&L Analyzer with Azure OpenAI & Embeddings")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'file_data' not in st.session_state: # Stores raw DataFrames by sheet name
    st.session_state.file_data = {}
if 'chunks_with_embeddings' not in st.session_state: # Stores text chunks and their embeddings
    st.session_state.chunks_with_embeddings = []
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None


# --- File Upload and Processing ---
uploaded_file = st.file_uploader("Upload your P&L file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # If a new file is uploaded, clear previous chunked data and conversation
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.file_data = {}
        st.session_state.chunks_with_embeddings = []
        st.session_state.conversation_history = [] # Optionally reset history for new file
        st.session_state.current_file_name = uploaded_file.name
        st.info("New file detected. Previous analysis context cleared.")

    if not st.session_state.file_data: # Only process if not already processed for this file
        try:
            uploaded_file.seek(0) # Reset file pointer
            filename = uploaded_file.name.lower()
            
            temp_file_data = {}
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                temp_file_data = {"Sheet1": df}
            elif filename.endswith(".xlsx"):
                xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                temp_file_data = {
                    sheet_name: xls.parse(sheet_name)
                    for sheet_name in xls.sheet_names
                }
            else:
                raise ValueError("Unsupported file format.")
            
            st.session_state.file_data = temp_file_data
            st.success(f"✅ File '{uploaded_file.name}' loaded with {len(st.session_state.file_data)} sheet(s).")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.file_data = {} # Clear on error

# --- Sheet Selector and Embedding Generation ---
if st.session_state.file_data and not st.session_state.chunks_with_embeddings: # Only process if file loaded and chunks not yet created
    sheet_names = list(st.session_state.file_data.keys())
    st.markdown("---")
    st.subheader("STEP 1: Select Sheets and Generate Embeddings")
    
    # Use a form for sheet selection and processing to avoid re-runs on widget interaction
    with st.form(key='sheet_selection_form'):
        selected_sheets_for_embedding = st.multiselect(
            "📌 Select sheet(s) to process for analysis:",
            options=sheet_names,
            default=sheet_names if sheet_names else [] # Default to all sheets
        )
        chunk_size_option = st.slider("Rows per chunk for embeddings:", min_value=1, max_value=50, value=10, step=1)
        process_button = st.form_submit_button(label='⚙️ Process Selected Sheets')

    if process_button and selected_sheets_for_embedding:
        with st.spinner(f"Processing {len(selected_sheets_for_embedding)} sheet(s) and generating embeddings... This may take a while."):
            all_chunks_with_embeddings = []
            for sheet_name in selected_sheets_for_embedding:
                if sheet_name in st.session_state.file_data:
                    df = st.session_state.file_data[sheet_name]
                    if df.empty:
                        st.warning(f"Sheet '{sheet_name}' is empty. Skipping.")
                        continue
                    
                    st.markdown(f"#### 📄 Preview: `{sheet_name}` (First 5 rows)")
                    st.dataframe(df.head(5))

                    text_chunks = chunk_dataframe(df, sheet_name, chunk_size=chunk_size_option)
                    
                    for i, chunk_info in enumerate(text_chunks):
                        st.write(f"Generating embedding for chunk {i+1}/{len(text_chunks)} from sheet '{sheet_name}'...")
                        embedding = get_embedding(chunk_info["text"], engine=embedding_deployment_name)
                        if embedding:
                            all_chunks_with_embeddings.append({
                                "text": chunk_info["text"],
                                "embedding": embedding,
                                "sheet": chunk_info["sheet"],
                                "rows": chunk_info["rows"]
                            })
            st.session_state.chunks_with_embeddings = all_chunks_with_embeddings
            if st.session_state.chunks_with_embeddings:
                st.success(f"✅ Successfully processed and embedded {len(st.session_state.chunks_with_embeddings)} chunks from selected sheets!")
            else:
                st.warning("⚠️ No data chunks were processed or embedded. Ensure sheets have data and are selected.")
    elif process_button and not selected_sheets_for_embedding:
        st.warning("⚠️ Please select at least one sheet to process.")


# --- System Prompt and Chat Logic ---
SYS_PROMPT = "" \
"DO THE DATA PROCESSING AND ANALYSIS YOURSELF LIKE CLEANING DATA, PROCESSING IN RIGHT FORMAT, etc. " \
"MTD (Month till date) Cost refers to cost till date. It is NOT just 1 day. So don't do a mistake of extracting only the first column. " \
"Calculate MTD like this summating the cost of all days until now. For example till wherever the current month data is available that is MTD."\
"You are provided the AWS Cost Dataset. It has daily level cost account wise. It contains MTD i.e daily costs for the current month," \
"corresponding costs for the same data for previous month. And previous month day level costs are also having columns for each day. Don't consider just the first column. " \
"You should also be able to compute the projections on the basis of current MTD costs. " \
"MTD (Month till date) Cost refers to cost till date. It is NOT just 1 day. So don't do a mistake of extracting only the first column. " \
"If the user specifically asks for any particular day then mention insights only for that day." \
"Apply the projection formula carefully. Whole month projection cannot be less than the projected cost" \
"Carefully analyze each column and its details. " \
"Don't get confused by the date column names. It is just the day wise cost. All the costs are in USD." \
"Also gather insights like which AWS Accounts are having highest spends on, least spends on, sudden high spikes basis previous day and the percentages." \
"Don't just say that you can do the analysis of this and that. Povide actual insights and analysis of the data." \
"And if you are mentioning HODs name that he had the highest cost, instead prioritize individual AWS Accounts like which all are must to observe on. (You can mention though that this account comes under the HOD, category, etc.)" \
"For example if you are analyzing May, then each day of may wil have separate column. And far from it will be each day columns for Apr. Be intelligent enough to locate." \
"Most important: Don't read the data beyond row 190." \
"Output should be strictly in tabular format with columns: 1. AWS Account Name 2. MTD Cost of the particular month 3. Cost for the same number of days in previous month" \
"4. Projection for current month 5. Growth/decline percentage from the previous month 6. Reason for the insight if any"

if st.session_state.chunks_with_embeddings:
    st.markdown("---")
    st.subheader("STEP 2: Ask Questions About Your Data")

    # Display chat history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    user_question = st.chat_input("Ask a question about your P&L data...")

    if user_question:
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("🔍 Thinking and retrieving relevant data..."):
            try:
                question_embedding = get_embedding(user_question, engine=embedding_deployment_name)
                if not question_embedding:
                    st.error("❌ Could not generate embedding for the question. Please try again.")
                    st.stop()

                # Find relevant chunks
                relevant_chunks = []
                for chunk_data in st.session_state.chunks_with_embeddings:
                    similarity = cosine_similarity(question_embedding, chunk_data["embedding"])
                    relevant_chunks.append({"data": chunk_data, "similarity": similarity})
                
                # Sort by similarity and get top N (e.g., top 5)
                relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
                top_n = 20
                context_chunks = relevant_chunks[:top_n]

                if not context_chunks or context_chunks[0]['similarity'] < 0.7: # Threshold for relevance
                     st.warning("⚠️ Could not find highly relevant data chunks for your question. The answer might be very general or indicate that the information is not available in the processed data.")
                
                context_text = "\n\n---\n\n".join([
                    f"Relevant Chunk from Sheet: {chunk['data']['sheet']}, Rows: {chunk['data']['rows']}\nSimilarity Score: {chunk['similarity']:.4f}\nContent:\n{chunk['data']['text']}" 
                    for chunk in context_chunks
                ])
                
                # Debug: Show context being sent
                # with st.expander("Context sent to AI (Top relevant chunks)"):
                # st.text(context_text)

                messages = [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"Here is the relevant data chunks based on my question:\n{context_text}\n\nMy question is: {user_question}"}
                ]
                
                # Add previous conversation for context (optional, can hit token limits quickly)
                # for msg in st.session_state.conversation_history[-3:]: # last 3 exchanges
                #     if msg["role"] == "user":
                #         messages.insert(-1, {"role": "user", "content": msg["content"]})
                #     elif msg["role"] == "assistant":
                #         messages.insert(-1, {"role": "assistant", "content": msg["content"]})


                response = openai.ChatCompletion.create(
                    engine=chat_deployment_name,
                    messages=messages,
                    temperature=0.2, # Lower temperature for more factual financial analysis
                    max_tokens=1500 # Adjust as needed
                )
                reply = response["choices"][0]["message"]["content"]

                st.session_state.conversation_history.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)

            except openai.error.InvalidRequestError as e:
                st.error(f"❌ OpenAI API Error (Invalid Request): {e}. This might be due to exceeding token limits even with chunking if the question or retrieved context is too large. Try a shorter question or processing with smaller chunk sizes.")
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
else:
    if st.session_state.file_data:
         st.info("☝️ Please select sheets and click 'Process Selected Sheets' to enable chat.")
    else:
        st.info("👋 Welcome! Please upload your P&L file to begin analysis.")


# Optional: show all processed chunks (for debugging or transparency)
if st.session_state.chunks_with_embeddings:
    with st.expander("🔬 View All Processed Data Chunks and Embeddings (for debugging)"):
        st.write(f"Total chunks processed: {len(st.session_state.chunks_with_embeddings)}")
        for i, chunk_data in enumerate(st.session_state.chunks_with_embeddings):
            st.markdown(f"**Chunk {i+1} (Sheet: {chunk_data['sheet']}, Rows: {chunk_data['rows']})**")
            st.text(chunk_data["text"][:300] + "...") # Preview of chunk text
            # st.write(f"Embedding vector (first 10 dims): {chunk_data['embedding'][:10]}") # Don't display full embedding
            st.markdown("---")

