## langchain rag demo system

#### 简介
```
包含
rag、ensembel_retriever混合检索
mcp server、mcp client等
supervisor agent主管agent架构
```

#### 技术栈
``` 
model：
    - qwen3-max (千问)
    - xop3qwen1b7 (科大讯飞)
    - GLM-4.5-Flash (智谱)
    
Vectordb：
    - Pinecorn
    - Chroma
   
embeddings：
    - HuggingFace的 sentence-transformers/all-MiniLM-L6-v2
    
tools：
    - duckduckgo
```

#### 调试
```
pip install requirements.txt

langgarph dev
```
#### .env.example改.env 添加自己的api key