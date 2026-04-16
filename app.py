import streamlit as st
import requests
import os

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="NORTHSTAR BANK - Credit Card Assistant", page_icon="💳", layout="wide")

# ==========================================
# SIDEBAR / TOGGLE
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["User Mode", "Admin Mode"])

# ==========================================
# ADMIN MODE: FILE UPLOAD
# ==========================================
if app_mode == "Admin Mode":
    st.title(" Admin: Document Ingestion")
    st.markdown("Upload multiple PDF documents to update the knowledge base.")

    uploaded_files = st.file_uploader(
        "Select PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if st.button("Upload and Ingest Documents"):
        if not uploaded_files:
            st.warning("Please select at least one PDF file to upload.")
        else:
            with st.spinner("Uploading and processing documents... This may take a while."):
                try:
                    files_payload = [
                        ("files", (file.name, file.getvalue(), "application/pdf")) 
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(f"{API_URL}/admin/upload", files=files_payload)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Successfully processed {res_data.get('files_processed')} files!")
                        with st.expander("View Ingestion Details"):
                            st.json(res_data)
                    else:
                        st.error(f"Failed to upload. Server responded with status code: {response.status_code}")
                        st.write(response.text)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Is your FastAPI server running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ==========================================
# USER MODE: QUERY
# ==========================================
elif app_mode == "User Mode":
    st.title("NORTHSTAR BANK - Credit Card Assistant")
    st.markdown("Analyze your spending patterns and verify bank policy compliance in real-time.")

    # --- THE LAZY HACK: SECRET LOGIN SELECTION ---
    st.sidebar.divider()
    st.sidebar.subheader(" Login")
    users = {
    "James Mitchell": "CC-881001",
    "Sarah Thompson": "CC-882001",
    "Robert Clarke": "CC-884001",
    "Emily Watson": "CC-885001",
    "Daniel Foster": "CC-883001",
    "Laura Bennett": "CC-886001"
}
    active_user = st.sidebar.selectbox("Select User", list(users.keys()))
    secret_card_id = users[active_user]
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- CHAT HISTORY LOOP ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show image if it exists in history
            if message["role"] == "assistant" and message.get("image_path"):
                if os.path.exists(message["image_path"]):
                    st.image(message["image_path"], caption="Retrieved Context Image", use_container_width=True)
                else:
                    st.warning(f"Image file missing from disk: {message['image_path']}")
            
            # Show metadata if it exists in history
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Sources & Metadata", expanded=False):
                    meta = message["metadata"]
                    st.markdown(f"** Document:** {meta.get('Document Name') or 'N/A'}")
                    st.markdown(f"** Page(s):** {meta.get('Page No') or 'N/A'}")
                    st.markdown(f"** Citation:** {meta.get('Citations') or 'N/A'}")
                    
                    if meta.get("SQL Query Executed"):
                        st.markdown("** SQL Query Executed:**")
                        st.code(meta["SQL Query Executed"], language="sql")
                    
                    if meta.get("Source Chunks"):
                        st.markdown("---")
                        st.markdown("** Retrieved Context (Raw Chunks):**")
                        # Join chunks with double newlines for vertical clarity
                        raw_text = "\n\n".join([f"{chunk}" for chunk in meta["Source Chunks"]])
                        st.code(raw_text, language="text")

    # --- LIVE QUERY LOOP ---
    if prompt := st.chat_input("Ask a question..."):
        # Display the normal prompt so the user doesn't see our hack
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            try:
                
                card_id_selected = f"{prompt} ( card_id = '{secret_card_id}')"
                payload = {"query": card_id_selected}
                # -----------------------------------

                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    raw_answer = data.get("answer", "No answer provided.")
                    
                    clean_answer = raw_answer.replace("\\n", "\n").replace("`", "")
                    
                    # GET THE IMAGE PATH STRING FROM FASTAPI
                    image_path_str = data.get("image_path")
                    
                    metadata = {
                        "Document Name": data.get("document_name"),
                        "Page No": data.get("page_no"),
                        "Citations": data.get("citation"),
                        "SQL Query Executed": data.get("sql_query_executed"),
                        "Source Chunks": data.get("source_chunks") 
                    }
                    
                    with st.chat_message("assistant"):
                        with st.container():
                            # 1. Print the text answer
                            st.write(clean_answer)

                            # 2. Display the image directly from the file path
                            if image_path_str:
                                if os.path.exists(image_path_str):
                                    st.image(image_path_str, caption="Retrieved Context Image", use_container_width=True)
                                else:
                                    st.warning(f"Image found in database, but file is missing from disk: {image_path_str}")
                        
                        # 3. Display the metadata expander
                        with st.expander("Sources & Metadata", expanded=False):
                            st.markdown(f" Document: {metadata['Document Name'] or 'N/A'}")
                            st.markdown(f" Page(s): {metadata['Page No'] or 'N/A'}")
                            st.markdown(f" Citation: {metadata['Citations'] or 'N/A'}")
                            
                            if metadata["SQL Query Executed"]:
                                st.markdown("SQL Query Executed:")
                                st.code(metadata["SQL Query Executed"], language="sql")
                            
                            if metadata["Source Chunks"]:
                                st.markdown("---")
                                st.markdown("Retrieved Chunks:")
                                raw_chunks = "\n\n".join([f"{chunk}" for chunk in metadata["Source Chunks"]])
                                st.code(raw_chunks, language="text")

                    # Save the response to session state history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": clean_answer,
                        "metadata": metadata,
                        "image_path": image_path_str  # Save path so it renders when scrolling
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure your FastAPI server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")