import sys
import os
import re
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
# from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
# from langchain.utilities import GoogleSearchAPIWrapper
# from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader

# search = GoogleSearchAPIWrapper()


# def cut_dialogue_history(history_memory, keep_last_n_words=500):
#     tokens = history_memory.split()
#     n_tokens = len(tokens)
#     print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
#     if n_tokens < keep_last_n_words:
#         return history_memory
#     else:
#         paragraphs = history_memory.split('\n')
#         last_n_tokens = n_tokens
#         while last_n_tokens >= keep_last_n_words:
#             last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
#             paragraphs = paragraphs[1:]
#         return '\n' + '\n'.join(paragraphs)


def on_token_change(user_token, state):
    print(user_token)
    openai.api_key = user_token or os.environ.get("OPENAI_API_KEY")
    state["user_token"] = user_token
    return state
        

class ConversationBot:
    def __init__(self):
        print("Initializing ChatGPT")
        # self.llm = OpenAI(temperature=0)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        # self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True)
        # self.toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        # self.agent = initialize_agent(
        #     self.toolkit.get_tools(),
        #     self.llm,
        #     agent="conversational-react-description",
        #     memory=self.memory,
        #     verbose=True)
        loader = TextLoader("./state_of_the_union.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(documents, embeddings)
        self.chat_history = []
        self.agent = ConversationalRetrievalChain.from_llm(
            self.llm, 
            self.vectorstore.as_retriever(),
            # return_source_documents=True,
            memory=self.memory)

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"question": text})
        # res = self.agent({"question": text, "chat_history": self.chat_history})
        chat_history = [(text, res["answer"])]
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['answer'])
        # print(f"res['source_documents'][0]: {res['source_documents'][0]}")
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

        
if __name__ == '__main__':
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Matsuo-Ken ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.8):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            with gr.Column(scale=0.2, min_width=0):
                clear = gr.Button("Clear️")
        with gr.Row():
            with gr.Column():
                gr.Markdown("Enter your own OpenAI API Key to try out more than 5 times. You can get it [here](https://platform.openai.com/account/api-keys).")
                user_token = gr.Textbox(placeholder="OpenAI API Key", type="password", show_label=False)

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        user_token.change(on_token_change, inputs=[user_token, state], outputs=[state])
        demo.launch(server_name="0.0.0.0", server_port=8080)