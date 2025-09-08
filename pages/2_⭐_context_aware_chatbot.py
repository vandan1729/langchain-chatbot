import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import utils
from streaming import StreamHandler

st.set_page_config(page_title="Context aware chatbot", page_icon="⭐")
st.header("Context aware chatbot")
st.write("Enhancing Chatbot Interactions through Context Awareness")
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/vandan1729/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)"
)


class ContextChatbot:
    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()

    @st.cache_resource
    def setup_chain(_self):
        memory = ConversationBufferMemory()
        chain = ConversationChain(llm=_self.llm, memory=memory, verbose=False)
        return chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke({"input": user_query}, {"callbacks": [st_cb]})

                # Use streamed text if available, otherwise fall back to result response
                streamed_response = st_cb.get_final_text()
                if streamed_response.strip():
                    response = streamed_response
                else:
                    response = result["response"]
                    st.markdown(response)

                # Get current LLM model and store with response
                utils.add_assistant_message_to_history(response)
                utils.print_qa(ContextChatbot, user_query, response)


if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
