from pydantic import BaseSettings, BaseModel, Field

class GraphServiceConfig(BaseModel):
    storage_file: str = Field("data/knowledge_graph/graph_data.json", env="RAG_GRAPH_STORAGE_FILE")

class VectorServiceConfig(BaseModel):
    storage_file: str = Field("data/vector_store/embeddings.pkl", env="RAG_VECTOR_STORAGE_FILE")
    dimension: int = Field(768, env="RAG_VECTOR_DIMENSION")

class LLMServiceConfig(BaseModel):
    model_name: str = Field("mock-llm", env="RAG_LLM_MODEL_NAME")
    embedding_dimension: int = Field(768, env="RAG_LLM_EMBEDDING_DIM")
    response_delay: float = Field(0.1, env="RAG_LLM_DELAY")

class QueryAgentConfig(BaseModel):
    max_entities: int = Field(10, env="RAG_QUERY_MAX_ENTITIES")
    confidence_threshold: float = Field(0.7, env="RAG_QUERY_CONF_THRESH")

class SearchAgentConfig(BaseModel):
    max_results: int = Field(20, env="RAG_SEARCH_MAX_RESULTS")
    similarity_threshold: float = Field(0.5, env="RAG_SEARCH_SIM_THRESH")

class GraphAgentConfig(BaseModel):
    max_path_length: int = Field(5, env="RAG_GRAPH_MAX_PATH")
    max_neighbors: int = Field(10, env="RAG_GRAPH_MAX_NEIGHBORS")

class ReasoningAgentConfig(BaseModel):
    max_reasoning_steps: int = Field(5, env="RAG_REASONING_MAX_STEPS")
    confidence_threshold: float = Field(0.6, env="RAG_REASONING_CONF_THRESH")

class ContextAgentConfig(BaseModel):
    max_context_size: int = Field(1000, env="RAG_CONTEXT_MAX_SIZE")
    session_timeout: int = Field(3600, env="RAG_CONTEXT_TIMEOUT")

class AnswerAgentConfig(BaseModel):
    max_answer_length: int = Field(500, env="RAG_ANSWER_MAX_LEN")
    include_reasoning: bool = Field(True, env="RAG_ANSWER_INC_REASON")
    include_confidence: bool = Field(True, env="RAG_ANSWER_INC_CONF")

class RAGConfig(BaseSettings):
    graph_service: GraphServiceConfig = GraphServiceConfig()
    vector_service: VectorServiceConfig = VectorServiceConfig()
    llm_service: LLMServiceConfig = LLMServiceConfig()
    query_agent: QueryAgentConfig = QueryAgentConfig()
    search_agent: SearchAgentConfig = SearchAgentConfig()
    graph_agent: GraphAgentConfig = GraphAgentConfig()
    reasoning_agent: ReasoningAgentConfig = ReasoningAgentConfig()
    context_agent: ContextAgentConfig = ContextAgentConfig()
    answer_agent: AnswerAgentConfig = AnswerAgentConfig()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"