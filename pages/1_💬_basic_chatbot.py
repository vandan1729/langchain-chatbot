import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chains import ConversationChain

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')
st.write('Allows users to interact with the LLM')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/vandan1729/langchain-chatbot/blob/master/pages/1_%F0%9F%92%AC_basic_chatbot.py)')

class BasicChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
    
    def setup_chain(self):
        chain = ConversationChain(llm=self.llm, verbose=False)
        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input":user_query},
                    {"callbacks": [st_cb]}
                )
                
                # Use streamed text if available, otherwise fall back to result response
                streamed_response = st_cb.get_final_text()
                if streamed_response.strip():
                    response = streamed_response
                else:
                    response = result["response"]
                    # If no streaming occurred, display the response
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(BasicChatbot, user_query, response)

if __name__ == "__main__":
    obj = BasicChatbot()
    obj.main()