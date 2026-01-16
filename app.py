import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os

# 페이지 설정
st.set_page_config(page_title="PDF 챗봇", page_icon="📚", layout="wide")

# API 키 설정
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# PDF 텍스트 추출
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# 벡터 스토어 생성
@st.cache_resource
def create_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 대화 체인 생성
def create_conversation_chain(vectorstore):
    # 환각 방지 프롬프트
    prompt_template = """당신은 제공된 문서의 내용만을 기반으로 답변하는 AI 어시스턴트입니다.

중요한 규칙:
1. 반드시 제공된 문서(Context)의 내용만을 사용하여 답변하세요.
2. 문서에 없는 내용에 대한 질문을 받으면 "죄송합니다. 제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
3. 추측하거나 일반적인 지식으로 답변하지 마세요.
4. 확실하지 않으면 모른다고 솔직히 말하세요.

Context: {context}

Chat History: {chat_history}

질문: {question}

답변:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,  # 낮은 temperature로 환각 방지
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # 더 많은 문맥 제공
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return conversation_chain

# 메인 UI
st.title("📚 PDF 기반 AI 챗봇")
st.markdown("**Gemini 2.0 Flash** 모델 | 문서 내용만 참조하여 답변합니다")

# 사이드바 - 파일 업로드
with st.sidebar:
    st.header("📄 문서 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="최대 200MB까지 업로드 가능합니다"
    )
    
    use_default = st.checkbox("기본 test.pdf 사용", value=False)
    
    if st.button("문서 처리 시작", type="primary"):
        with st.spinner("문서를 분석 중입니다..."):
            try:
                # 파일 선택
                if use_default and os.path.exists("test.pdf"):
                    pdf_file = open("test.pdf", "rb")
                elif uploaded_file:
                    pdf_file = uploaded_file
                else:
                    st.error("파일을 업로드하거나 기본 파일을 선택하세요")
                    st.stop()
                
                # 텍스트 추출
                text = extract_text_from_pdf(pdf_file)
                
                if len(text) < 100:
                    st.error("PDF에서 충분한 텍스트를 추출할 수 없습니다")
                    st.stop()
                
                # 벡터 스토어 생성
                vectorstore = create_vectorstore(text)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = create_conversation_chain(vectorstore)
                st.session_state.messages = []
                
                st.success(f"✅ 문서 처리 완료! ({len(text):,}자)")
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화"):
        if "messages" in st.session_state:
            st.session_state.messages = []
            if "conversation" in st.session_state:
                st.session_state.conversation.memory.clear()
            st.rerun()

# 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("문서에 대해 질문하세요..."):
    if "conversation" not in st.session_state:
        st.warning("⚠️ 먼저 사이드바에서 PDF 문서를 업로드하고 처리하세요")
        st.stop()
    
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                response = st.session_state.conversation({"question": prompt})
                answer = response["answer"]
                
                st.markdown(answer)
                
                # 참조 문서 표시
                if response.get("source_documents"):
                    with st.expander("📖 참조한 문서 부분 보기"):
                        for i, doc in enumerate(response["source_documents"][:3]):
                            st.markdown(f"**참조 {i+1}:**")
                            st.text(doc.page_content[:400] + "...")
                            st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"오류 발생: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 하단 정보
st.sidebar.markdown("---")
st.sidebar.info("""
**사용 방법:**
1. PDF 파일 업로드 또는 test.pdf 선택
2. '문서 처리 시작' 버튼 클릭
3. 채팅창에서 질문 입력

**특징:**
- 문서 내용만 참조하여 답변
- 문서에 없는 정보는 "모른다"고 답변
- 환각(Hallucination) 방지

**모델:** Gemini 2.0 Flash Experimental
""")

# 예시 질문
if "conversation" in st.session_state and len(st.session_state.messages) == 0:
    st.markdown("### 💡 예시 질문")
    st.markdown("""
    - 이 문서의 주요 내용을 요약해주세요
    - [특정 주제]에 대해 문서에서 무엇이라고 말하나요?
    - 문서에서 언급된 핵심 키워드는 무엇인가요?
    """)
