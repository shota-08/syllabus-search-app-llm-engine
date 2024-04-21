from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain.chains import LLMChain
from typing import List
from pydantic import BaseModel, Field
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()  # read local .env file

# 会話履歴クラス
class ChatMessageHistory(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages = []

# LLMモデルの設定
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
history = ChatMessageHistory()

def get_llm_summary(text):
    prompt_template = """
    # 命令
    あなたは履修登録のアドバイザーです。
    履修登録を検討している大学生に対して、この講義で学べる題材は何か、50文字以内で簡潔に説明してください。
    一度大きく深呼吸をしてステップバイステップで考えてください。

    # 制約条件:
    ・50文字以内で簡潔に答える
    ・講義で扱うジャンルや年代や題材のキーワードは説明に含めること
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・複数の要素がある場合、重要度の最も高い要素のみを説明すること

    # 講義内容:
    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm
    answer = chain.invoke({"text": text})
    return answer.content

def pass_text(query, title, text):
    query_with_text = f"""
    # 命令
    あなたは履修相談のアドバイザーです。
    学生の問い合わせに親切に回答する過程で、講座名と講座概要を参照しながら、おすすめの講座を簡潔に紹介してください。
    学生の問い合わせに対して、なぜその講座をお勧めしたか理由を明確に伝えてください。
    一度大きく深呼吸をしてステップバイステップで考えてください。

    # 制約条件:
    ・おすすめした理由と一緒に、講座名と講座概要を簡潔に伝えること
    ・ジャンルや年代や題材などの講座概要にある具体的なキーワードを含めること
    ・講座概要に書いていないことは伝えてはいけない
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・会話が続くように、文章の最後に「他にどんなことを学びたいですか？」など付けること
    ・100文字以内で簡潔に答える
    ・学生との直前の会話を参照すること

    # 講座名:
    {title}

    # 講座概要:
    {text}

    # 学生の問い合わせ:
    {query}

    # 直前の会話
    """
    return query_with_text

def get_llm_answer(query, title, text):
    query_with_text = pass_text(query, title, text)
    template = query_with_text + "\n{chat_history}"

    prompt = PromptTemplate(
        template = "{template}",
        input_variables = ["template"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
    )
    answer = conversation.invoke({"template": template, "chat_history": history})

    history.add_user_message(query)
    history.add_ai_message(answer["text"])

    # prompt = PromptTemplate(
    #     template = "{template}",
    #     input_variables = ["template"]
    # )
    # query_with_text = pass_text(query, title, text)
    # chain = prompt | llm
    # answer = chain.invoke({"template": query_with_text})
    return answer