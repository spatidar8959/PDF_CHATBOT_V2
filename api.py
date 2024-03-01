from flask import Flask, request, jsonify
from flask_cors import CORS  
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
import os
import fitz
from PIL import Image
import pytesseract
from langchain.memory import ConversationBufferMemory
import io
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.environ["OPENAI_API_KEY"]

# Initialize global variables
language = 'English'
pdf_path = None
pdf_document = None
chat_buffer = ConversationBufferMemory(memory_key="messages", return_messages=True)
embeddings = None
db = None
llm_chain = None
pdf_processing_complete = False

# Define language model and LLMChain
template = """
you are a helpful assistant which reads pdf content with all attention and gives an answer to the user's query based on
pdf context. You generate the answers in the format given in context. If the context is in table format, generate the answer in table format. If you don't know the answer, tell the user "I don't understand your question".

# Context: {context}
# Question: {query}
# Answer: """

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)
language_model = OpenAI()

llm_chain = LLMChain(
    llm=language_model,
    prompt=prompt_template,
    verbose=True,
)

ALLOWED_EXTENSIONS = {'pdf','txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/set_language', methods=['POST'])
def set_language():
    global language

    language = request.form.get('language', '')

    return jsonify({'message': f'Language set to {language}'})

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global language, pdf_path, pdf_document, chat_buffer, embeddings, db, pdf_processing_complete

    try:
        # Check if a PDF file is provided in the request
        pdf_file = request.files.get('pdf_file')

        if not pdf_file or pdf_file.filename == '':
            return jsonify({'error': 'Please provide a PDF file in the "pdf_file" form field.'}), 400
        
        if not allowed_file(pdf_file.filename):
            return jsonify({'error': 'Only PDF and TEXT files are allowed.'}), 400

        # Save the PDF file temporarily
        pdf_path = pdf_file.filename
        pdf_file.save(pdf_path)

        # Load PDF and extract text
        pdf_document = fitz.open(pdf_path)
        extracted_text = ""

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            page_text = page.get_text()
            extracted_text += page_text
            images = page.get_images(full=True)
            for img_index, img_info in enumerate(images):
                img_index += 1
                image_index = img_info[0]
                base_image = pdf_document.extract_image(image_index)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                image_text = pytesseract.image_to_string(image)
                extracted_text += f"\n[Image {img_index}]\n{image_text}"

        # Process extracted text
        raw_text = extracted_text.strip()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=10,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_text(raw_text)

        # Create FAISS database
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings)
        pdf_processing_complete = True

        return jsonify({'message': 'Document processed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global language, pdf_path, pdf_document, chat_buffer, embeddings, db, llm_chain, pdf_processing_complete

    if not pdf_processing_complete:
        return jsonify({'error': 'Document processing is not complete. Please process the Document first.'}), 400

    user_question = request.form.get('question', '')

    if not user_question:
        return jsonify({'error': 'Please provide a question in the "question" form field.'}), 400

    query = f"give answer of the given question in {language} language. question : {user_question}?"
    docs = db.similarity_search(query, k=3)
    response = llm_chain.predict(context=docs, query=query)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
