import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Set your OpenAI API key

def search_website(question, website_url, chat_openai_model):
    # Prepare the prompt with the question and website URL
    prompt = f"What is the answer to the question '{question}' on the website {website_url}?"

    # Create a QA chain using the provided ChatOpenAI model
    chain = load_qa_chain(chat_openai_model, chain_type="stuff")

    # Call the QA chain to answer the question
    with get_openai_callback() as cb:
        response = chain.run(input_text=prompt)

    # Get the answer from the response
    answer = response.strip()

    return answer

# Load the ChatOpenAI model with the specified model name
chat_openai_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# Example usage
website_url = "https://en.wikipedia.org/wiki/France"  # Replace with the actual website URL
question = "What is the capital of France?"  # Replace with your question

answer = search_website(question, website_url, chat_openai_model)
print("Answer:", answer)

