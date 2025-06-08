from langchain_ollama import ChatOllama


class InterfaceLangchain:
    def __init(self):
        pass

    def get_llm(
        self,
        model_name: str = "llama3.2:3b",
        temperature: float = 0,
        num_predict: int = 256,
    ):
        """
        Initialize the LLM with the specified parameters.

        :param model_name: Name of the model to use.
        :param temperature: Sampling temperature for the model.
        :param num_predict: Number of tokens to predict.
        :return: An instance of ChatOllama configured with the provided parameters.
        """
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
            # other params can be added here as needed
        )
