# AI Research Assistant Chatbot

Welcome to the AI Research Assistant Chatbot repository! This powerful tool is designed to assist researchers in navigating the vast landscape of academic literature by providing intelligent responses to research paper-related queries. Leveraging cutting-edge technologies in natural language processing (NLP) and document parsing, this chatbot aims to streamline the research process and empower scholars with instant access to relevant information.

## Features

### 1. Document Format Support

Our chatbot is equipped with the capability to handle various document formats, including:

- PDF
- DOCX
- PPTX
- LaTeX

### 2. Intelligent Text Extraction

Utilizing advanced libraries such as PyPDF2, python-docx, and pptx, the chatbot seamlessly extracts text from uploaded documents, ensuring a smooth analysis process.

### 3. Natural Language Understanding

Powered by Hugging Face Transformers and NLTK, the chatbot comprehends complex natural language queries, enabling users to ask questions in a conversational manner.

### 4. Optical Character Recognition (OCR)

PyTesseract integrates OCR functionality, allowing the chatbot to extract text even from images embedded within documents, enhancing its versatility in handling diverse content.

### 5. Deep Learning Capabilities

With PyTorch and Transformers, our chatbot employs state-of-the-art deep learning models for tasks such as question-answering and image captioning, ensuring accurate and informative responses.

### 6. Interactive Web Interface

Built on Streamlit, the chatbot offers an intuitive and user-friendly web interface, making it accessible to researchers of all levels of technical expertise.

### 7. Question-Answer History

Users can conveniently download the question-answer history in a .txt format, facilitating documentation and review of past interactions with the chatbot.

## Tech Stack

- **Python**: Core programming language for development.
- **Streamlit**: UI development tool for creating interactive web applications.
- **PyPDF2, python-docx, pptx**: Libraries for parsing and extracting text from document formats.
- **pylatexenc**: Library for converting LaTeX equations into human-readable text.
- **PyTesseract**: Optical character recognition (OCR) tool for extracting text from images.
- **Transformers (Hugging Face)**: Library for NLP tasks, including question-answering and image captioning.
- **PyTorch**: Deep learning framework for tasks such as image captioning using pre-trained models.
- **NLTK**: Toolkit for NLP tasks like tokenization and part-of-speech tagging.
- **Aspose.Slides**: Library for extracting images from PPTX files.
- **OpenAI GPT-3**: Language model for generating responses to user queries and conducting conversations.
- **Langchain**: Library for advanced text processing tasks such as text splitting, embeddings, and conversational chains.

### 8. Langchain: Advanced Text Processing

Langchain is a powerful library designed to handle advanced text processing tasks, enhancing the capabilities of our AI Research Assistant Chatbot. Here's how Langchain contributes to the functionality of the chatbot:

- **Text Splitting**: Langchain offers sophisticated algorithms for splitting text into meaningful segments, allowing the chatbot to analyze documents at a granular level and extract key information effectively.

- **Embeddings**: Langchain facilitates the generation of text embeddings, which encode the semantic meaning of words and sentences into numerical vectors. These embeddings enable the chatbot to understand the context of research papers and provide accurate responses to user queries.

- **Conversational Chains**: By leveraging Langchain's conversational chain functionality, the chatbot can maintain context across multiple interactions with users, ensuring a seamless and engaging conversational experience.

With Langchain, our AI Research Assistant Chatbot is equipped to handle complex text processing tasks, enabling it to deliver intelligent and contextually relevant responses to researchers' queries. This integration enhances the chatbot's overall performance and user satisfaction, making it an indispensable tool in the research workflow.

## Workflow Description

The AI Research Assistant Chatbot follows a comprehensive workflow to ensure accurate and efficient processing of research papers and user queries. Here's an overview of the workflow:

1. **Ensuring Proper Functioning of Document Formats**: The chatbot ensures proper functioning across all document formats, including edge cases such as two-column formats. It is tested on various sizes and orientations, such as A4 and A5.

2. **Input File Handling**: Users can upload research papers in PDF, DOCX, PPTX, or LaTeX formats. The chatbot performs chunking of the given text for optimized representation.

3. **Word Embedding Model**: The chatbot utilizes a word embedding model, such as Ada v2 or Instructure XL, to generate context-based embeddings. These embeddings are stored in a dense vector space format.

4. **User Query Processing**: When a user submits a query, the chatbot converts the query into a dense vector representation.

5. **Ranked Retrieval**: The chatbot performs ranked retrieval from the vector space, using the dense vectors generated from the embeddings.

6. **Language Model Integration**: Utilizing Langchain, the chatbot employs a Language Model (LLM), such as OpenAI's GPT-3 or LLaMA 2 17B, to retrieve relevant outputs based on the user query.

7. **Output Generation**: Finally, the chatbot generates an output based on the retrieved information, providing users with contextually relevant answers to their queries.

This structured workflow ensures that the AI Research Assistant Chatbot delivers accurate and informative responses, enhancing the research experience for users.

## Features

The AI Research Assistant Chatbot boasts a multifaceted set of features, categorized into different levels for a comprehensive understanding of its capabilities:

### Level-1: Textual Understanding

The chatbot excels at textual understanding, providing users with intelligent responses based on the content of research papers.

### Level-2: Explanation of Tables

The chatbot offers satisfactory explanations of tables present in documents, enhancing the user's understanding of tabular data.

### Level-3: Mathematical Equations

- **Identification**: The chatbot is capable of identifying mathematical equations present in documents.
- **Cross-Format Equation Retrieval**: It can find equations from all document formats.
- **Explanation**: The chatbot provides satisfactory explanations of the identified equations, elucidating the logic and mathematics used.

### Level-4: Image Extraction

- **Extraction**: The chatbot can extract images from documents.
- **Image Count**: It outputs the number of images present in the document.

### Level-5: Advanced Functionality

- **Seamless Question-Answering**: The chatbot seamlessly answers both logical and factual questions.
- **Source Citation**: It cites the origin source of the information retrieved, indicating precisely where the output is generated from.
- **Downloadable Chat History**: Users can download the entire chat history in a question-answer format for further reference.
- **OCR from Images**: The chatbot performs OCR from images, generating meaningful brief explanations using nltk-pos_tagging.
- **Image Descriptions**: Using GPT-2-based vision transformers, the chatbot generates short descriptions of images.
- **User-Friendly Interface**: The chatbot features a user-friendly interface for a seamless and intuitive experience.
- **Prompt-Control Chain**: An effective prompt-control chain is implemented to prevent hallucination or misleading responses.
- **Test Case Generation**: The chatbot has the capability to generate test cases for model performance evaluation.

These features collectively make the AI Research Assistant Chatbot a powerful and versatile tool for researchers, providing a holistic solution for document analysis and information retrieval.


## External Dependencies

Ensure the following dependencies are installed externally:

- **Tesseract OCR**: Install Tesseract OCR for PyTesseract to work properly.
  - Install via: `pip install tesseract-ocr`
- **TeX/LaTeX Distribution**: Install a TeX/LaTeX distribution for pylatexenc.
  - Example: TeX Live - [Download Link](https://www.tug.org/texlive/)

## Installation Instructions

1. Create a virtual environment with Python 3.9:
    
    ```bash
    conda create -p venv python==3.9
    ```

2. Activate the virtual environment:
    
    ```bash
    conda activate venv
    ```

3. Install the required Python packages using pip:
    
    ```bash
    pip install -r requirements.txt
    ```

4. Install external dependencies:
   - Install Tesseract OCR:
    
    ```bash
    pip install install tesseract-ocr
    ```

## Usage

1. Run the Streamlit app:
    
    ```bash
    streamlit run chatbot_AiReS.py
    ```

2. Upload research papers in PDF, DOCX, PPTX, or LaTeX formats.
3. Click on the "Process" button to analyze the uploaded research papers.
4. Ask questions related to the content of the research papers in the text input box.
5. The chatbot will analyze the documents and provide answers based on the content.
6. Download the question-answer history in the form of a .txt file by clicking the download button.

## Note

- Ensure that the uploaded documents contain relevant research papers with readable text. The effectiveness of the chatbot depends on the quality and relevance of the content provided.
- For better performance, it is recommended to provide clear and concise questions related to the content of the research papers.

## Conclusion

The AI Research Assistant Chatbot revolutionizes the way researchers interact with academic literature, offering unparalleled accessibility and intelligence in navigating complex research domains. With its comprehensive feature set and advanced technologies, this chatbot serves as an indispensable tool for scholars seeking to expedite their research endeavors and unlock new insights within the vast repository of scholarly knowledge. Experience the future of academic research assistance today!

---
Please find the demo video and performance evolution [here](https://drive.google.com/drive/folders/1D5eJtdLSYhUJ0dgnw1IMhDGQmNjO505t)

**Project Details:**

- **Hackathon:** MINeD'24
- **Organized by:** Nirma University and SUNY BIGHMIENJFDB University
- **Dates:** February 29th to March 2nd, 2024
- **Problem Statement Provider:** Cactus Communication
- **Team Name:** The Pandavas
- **Achievements:**
  - **Cactus Track:** 2nd Prize
  - **Overall MINeD Hackathon:** 1st Prize
- **Prize:** $500+ USD
