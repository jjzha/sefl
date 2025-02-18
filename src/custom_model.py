from types import SimpleNamespace
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import huggingface_hub
from dotenv import load_dotenv, find_dotenv
import os


class CustomModelClient:
    def __init__(self, config: Dict[str, Any], **kwargs):
        print(f"CustomModelClient config: {config}")
        self._login()
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = config["model"]
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", config["params"]["max_new_tokens"])
        print(f"Loaded model {self.model_name} to {self.device}")

    def create(self, params: Dict[str, Any]) -> SimpleNamespace:
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")

        num_of_responses = params.get("n", 1)
        response = SimpleNamespace()
        inputs = self._prepare_inputs(params["messages"])
        inputs_length = inputs.shape[-1]
        generation_config = self._create_generation_config(inputs_length)

        response.choices = []
        response.model = self.model_name

        for _ in range(num_of_responses):
            outputs = self.model.generate(inputs, generation_config=generation_config)
            text = self.tokenizer.decode(
                outputs[0, inputs_length:], skip_special_tokens=True
            )
            choice = SimpleNamespace(
                message=SimpleNamespace(content=text, function_call=None)
            )
            response.choices.append(choice)

        return response

    def _prepare_inputs(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        return inputs

    def _create_generation_config(self, inputs_length: int) -> GenerationConfig:
        return GenerationConfig(
            max_length=self.max_length + inputs_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    def _login(self) -> None:
        load_dotenv(find_dotenv())
        huggingface_hub.login(token=os.environ.get("HF_TOKEN"))

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
        """Get usage statistics for the response."""
        return None  # Implement if usage tracking is needed
