from langchain_core.callbacks import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.complete = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)
    
    def on_llm_end(self, response, **kwargs):
        # Mark as complete and ensure final text is displayed
        self.complete = True
        if self.text:
            self.container.markdown(self.text)
    
    def get_final_text(self):
        return self.text
    
    def has_content(self):
        return bool(self.text.strip())