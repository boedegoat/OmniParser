from enum import StrEnum

class LLM_Provider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    GEMINI = "gemini"

LLM_MODELS = {
    "OmniParser + GPT 4o": {
        "name": "gpt-4o",
        "providers": [LLM_Provider.OPENAI],
        "abilities": ["text", "image"],
        "price_per_1m_tokens": 2.5
    },
    "OmniParser + GPT o1": {
        "name": "gpt-o1",
        "providers": [LLM_Provider.OPENAI],
        "abilities": ["text", "thinking"],
        "price_per_1m_tokens": 15
    },
    "OmniParser + GPT o3 Mini": {
        "name": "gpt-o3-mini",
        "providers": [LLM_Provider.OPENAI],
        "abilities": ["text", "thinking"],
        "price_per_1m_tokens": 1.1
    },
    "OmniParser + Gemini 2.0 Flash": {
        "name": "gemini-2.0-flash",
        "providers": [LLM_Provider.GEMINI],
        "abilities": ["text", "image"],
        "price_per_1m_tokens": 15
    },
    "OmniParser + Gemini 2.5 Flash": {
        "name": "gemini-2.5-flash-preview-04-17",
        "providers": [LLM_Provider.GEMINI],
        "abilities": ["text", "image"]
    },
    "OmniParser + Gemini 2.5 Flash Thinking": {
        "name": "gemini-2.5-flash-preview-04-17",
        "providers": [LLM_Provider.GEMINI],
        "abilities": ["text", "image", "thinking"]
    }
}

class LLM:
    def __init__(self):
        self.model_labels = list(LLM_MODELS.keys())

        self.default_model_label = "OmniParser + Gemini 2.0 Flash"

        if self.default_model_label not in LLM_MODELS:
             # Fallback to the first available label if the default isn't found
             self.default_model_label = list(LLM_MODELS.keys())[0]

        self.default_model = LLM_MODELS[self.default_model_label]


    def get_model_data(self, label: str = None, name: str = None):
        if label:
            model_data = LLM_MODELS.get(label)
            if not model_data:
                raise ValueError(f"No model found with label: {label}")
            return model_data
        elif name:
            for model_data in LLM_MODELS.values():
                if model_data["name"] == name:
                    return model_data
            raise ValueError(f"No model found with name: {name}")
        else:
            raise ValueError("Either label or name must be provided")
        
    def count_cost(self, model, token_usage):
        return token_usage * model["price_per_1m_tokens"] / 1000000


llm = LLM()