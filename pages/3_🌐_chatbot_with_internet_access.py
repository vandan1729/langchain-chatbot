import utils
import streamlit as st
from streaming import StreamHandler

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

st.set_page_config(page_title="ChatNet", page_icon="🌐")
st.header('Chatbot with Internet Access')
st.write('Equipped with internet access, enables users to ask questions about recent events')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/vandan1729/langchain-chatbot/blob/master/pages/3_%F0%9F%8C%90_chatbot_with_internet_access.py)')

class InternetChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()

    # @st.cache_resource(show_spinner='Connecting..')
    def setup_agent(_self):
        # Define tool
        ddg_search = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions",
            )
        ]

        # Get the prompt - can modify this
        prompt = hub.pull("hwchase17/react-chat")

        # Setup LLM and Agent
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent = create_react_agent(_self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)
        return agent_executor, memory

    @utils.enable_chat_history
    def main(self):
        agent_executor, memory = self.setup_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = agent_executor.invoke(
                    {"input": user_query, "chat_history": memory.chat_memory.messages},
                    {"callbacks": [st_cb]}
                )
                
                # Use streamed text if available, otherwise fall back to result response
                streamed_response = st_cb.get_final_text()
                if streamed_response.strip():
                    response = streamed_response
                else:
                    response = result["output"]
                    st.markdown(response)
                
                # Get current LLM model and store with response
                current_llm = utils.get_current_llm_model()
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "llm_model": current_llm
                })
                utils.print_qa(InternetChatbot, user_query, response)


if __name__ == "__main__":
    obj = InternetChatbot()
    obj.main()