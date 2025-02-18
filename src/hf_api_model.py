import os
from types import SimpleNamespace
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

class APIModelClient:
    def __init__(self, config: Dict[str, Any]):
        print(f"APIModelClient config: {config}")
        self.device = config.get("device", "cpu")
        self.model_name = config.get("model")
        self.client = InferenceClient(model=self.model_name, token=os.environ.get("HF_TOKEN"))
        print(f"Loaded model {self.model_name} to {self.device}")

    def create(self, params: Dict[str, Any]) -> SimpleNamespace:
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")
        
        num_of_responses = params.get("n", 1)
        response = SimpleNamespace()
        response.choices = []
        response.model = self.model_name

        for _ in range(num_of_responses):
            text = self.client.chat_completion(
                params["messages"],
                max_tokens=params["params"]["max_new_tokens"],
                stream=params["params"]["stream"],
                temperature=params["params"]["temperature"]
            )
            choice = SimpleNamespace(
                message=SimpleNamespace(content=text.choices[0].message.content, function_call=None)
            )
            response.choices.append(choice)
        
        return response

    @staticmethod
    def message_retrieval(response: SimpleNamespace) -> List[str]:
        """Retrieve the messages from the response."""
        return [choice.message.content for choice in response.choices]

    @staticmethod
    def cost(response: SimpleNamespace) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response: SimpleNamespace) -> Optional[Dict[str, Any]]:
        """
        Returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        if usage needs to be tracked, else None
        """
        return {}