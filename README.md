## 📝 Project Overview and Features
This project consists of a Retrieval-Augmented Generation chatbot build with LangChain and Streamlit for 
interactive experiementation. It performs retrieval of local PDFs stored in a Chroma vector database
and integrates live economic data source from the FRED API, allowing for a hybrid retrieval data workflow.
<br>
<br>
## 🚀 Features

### 🔍 Retrieval-Augmented Generation
<br>
Uses Chroma as the vector database to index and retrieve text from locally stored economic reports (PDFs).
<br>
The FRED API is called when a LLM assistant determines external data is required to answer user's query. LLM can dynamically build the request to the API, ensuring only the required economic data is requested.

<br>
<br>

### 🧩 Interactive UI - The UI allows for a experimental usage to understand how parameters affect the system output.

Switch between different LLM models.

Adjust temperature and top_k chunks retrieved.

Re-run local embeddings with a different chunk and overlap size.

View back-end logs directly from the testing area.
<br>
<br>
### 🗣️ Query Augmentation

Optional query rewriting feature that generates three semantically similar queries to improve retrieval quality.

#### Example of Query Augmentation

<img width="1853" height="538" alt="image" src="https://github.com/user-attachments/assets/e0e25feb-e5b9-4973-a0cf-bce0a349c04e" />

<br>
<br>

## 💻 Application Demo


#### Memory functionality
<img width="1763" height="656" alt="image" src="https://github.com/user-attachments/assets/eb7cc8d1-fcc8-4e6d-bdc0-94c68295b441" />
<br>
<img width="1438" height="675" alt="image" src="https://github.com/user-attachments/assets/aff8ca2d-c2b7-414d-9a16-219c88ca2aa6" />
<br>
<br>

### Context Printing

<img width="1879" height="826" alt="image" src="https://github.com/user-attachments/assets/09631fc8-b17d-425e-be11-fdf1bef25300" />

