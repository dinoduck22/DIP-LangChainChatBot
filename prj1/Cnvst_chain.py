from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from base_chain import BaseConversationChain


# 틀 만들어 주기
class EnglishConversationChain(BaseConversationChain):

    def create_prompt(self):
        template = """
            당신은 영어를 가르치는 10년차 영어 선생님입니다. 주어진 상황에 맞는 영어 회화를 작성해 주세요.
            양식은 [FORMAT]을 참고하여 작성해 주세요.

            #상황:
            {question}

            #FORMAT:
            ***영어 회화***
            - A: That sounds interesting! What do we need?
            - B: We’ll need oats, peanut butter, cocoa powder, and honey.
            - A: Great! How do we make them?
            ...

            ***한글 해석***
            - B: 아니, 없어. 하지만 오븐 없이 만드는 쿠키 레시피를 찾았어!
            - A: 흥미롭다! 어떤 재료가 필요해?
            - B: 우리는 귀리, 땅콩버터, 코코아 가루, 그리고 꿀이 필요해.
            ...

            """
        prompt = PromptTemplate.from_template(template)
        return prompt

    def create_output_parser(self):
        return StrOutputParser()


class SummaryChain(BaseConversationChain):
    def create_prompt(self):
        template = """
        you are a professional translator and you are good at summarizing the context.
        you are going to translate the user's input and summarize it within 1 to 3 sentences. 
        한 문장 뒤에 엔터로 줄바꿈 해 줘
        
        #Context:
        {question}
        """

        prompt = PromptTemplate.from_template(template)
        return prompt

    def create_output_parser(self):
        return StrOutputParser()
