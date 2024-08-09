from langchain_openai import ChatOpenAI
from abc import ABC, abstractmethod


class BaseConversationChain(ABC):

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    @abstractmethod  # 구현 필요
    def create_prompt(self):
        pass

    def create_model(self):
        model = ChatOpenAI(model_name=self.model_name)
        return model

    @abstractmethod
    def create_output_parser(self):
        pass

    def create_chain(self):  # 구현 불필요
        prompt = self.create_prompt()
        model = self.create_model()
        output_parser = self.create_output_parser()
        chain = prompt | model | output_parser
        return chain