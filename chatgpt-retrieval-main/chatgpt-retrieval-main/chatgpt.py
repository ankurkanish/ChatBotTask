import os
import sys
import PyPDF2
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Function to concatenate the content from multiple files
def concatenate_content(loaders):
    combined_content = ""
    for loader in loaders:
        if isinstance(loader, PyPDFLoader):
            content = "\n".join(page.extractText() for page in loader.pdf.pages)
        elif isinstance(loader, TextLoader):
            content = loader.load()
        else:
            content = ""  # Handle other loaders if needed
        combined_content += content + "\n"  # Separate content with a newline
    return combined_content

# Function to get files whose title starts with the user's query
def get_matching_files(folder_path, query):
    matching_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a PDF and create a PyPDFLoader for it
        if filename.endswith(".pdf"):
            title = os.path.splitext(filename)[0]
        else:
            title = filename
        # Split the title into words and compare the first word with the query
        if title.split()[0].lower() == query.lower():
            matching_files.append(file_path)
    return matching_files

if len(sys.argv) > 1:
    query = sys.argv[1]
else:
    query = input("Enter your query: ")

# Folder path that contains multiple files
folder_path = "C:/Users/Ankur Sharma/Desktop/datachatbot"

# Get matching files based on the user's query
matching_files = get_matching_files(folder_path, query)

# Create a list to hold all the loaders
loaders = []

# Loop through all matching files and create loaders for each file
for file_path in matching_files:
    # Check if the file is a PDF and create a PyPDFLoader for it
    if file_path.endswith(".pdf"):
        pdf_loader = PyPDFLoader(file_path=file_path)
        loaders.append(pdf_loader)
    else:
        text_loader = TextLoader(file_path=file_path)
        loaders.append(text_loader)

# Concatenate the content from all files
combined_content = concatenate_content(loaders)

# Create the index from the combined content
index = VectorstoreIndexCreator().from_loaders([TextLoader(text=combined_content)])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    # Add hidden post prompts to guide the model's behavior
    post_prompt = "You are an assistant providing information from multiple files. Please answer the user's question based on the combined content of the matching files."
    full_query = post_prompt + " " + query

    try:
        result = chain({"question": full_query, "chat_history": chat_history})
        print(result['answer'])
    except IndexError as e:
        print(f"An error occurred: {e}")

    chat_history.append((query, result['answer']))
    query = None
