import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

from llm_engine import get_llm_answer

st.set_page_config(page_title='Langchain x syllabus')
st.title('Langchain x syllabus')

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# chroma db呼び出し
persist_directory = "./docs/chroma"
db = Chroma(collection_name="langchain_store", persist_directory=persist_directory, embedding_function=embedding)

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

# 質問文
query = st.chat_input('やりたいことを入力')

# チャットログセッションの初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if query:
    docs = db.similarity_search(query, k=2)
    answer_meta = docs[0].metadata
    answer_title = answer_meta['title']
    answer_url = answer_meta['url']
    answer_content = docs[0].page_content
    answer = get_llm_answer(query, answer_title, answer_content)

    # ログの表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # chat表示
    with st.chat_message(USER_NAME):
        st.write(query)

    with st.chat_message(ASSISTANT_NAME):
        # st.write(answer)
        st.write(answer + f"[{answer_title}]({answer_url})")

    # with st.chat_message(ASSISTANT_NAME):
    #     st.write(answer_title)

    # data表示
    with st.expander('docs', expanded=False):
        st.write(docs)

    # セッションにログを追加
    st.session_state.chat_log.append({"name" : USER_NAME, "msg" : query})
    st.session_state.chat_log.append({"name" : ASSISTANT_NAME, "msg" : answer})