# coding:utf-8

from langchain_core.prompts import PromptTemplate


def get_prompt_template():
    prompt = PromptTemplate.from_template("""
    你是一个聪明的智能客服，你能回答我的问题。
    Question: {input}
    """)
    return prompt
