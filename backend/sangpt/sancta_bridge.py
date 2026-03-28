from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class SanctaLangChainLLM(LLM):
    """Wrapper for the SanctaGPT engine to work with LangChain."""
    
    # Configuration for the Sancta Engine
    temperature: float = 0.7
    max_tokens: int = 120

    @property
    def _llm_type(self) -> str:
        return "sancta_gpt"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """The core logic that runs the SanctaGPT engine."""
        # CRITICAL FIX: Import inside the function to break circular recursion
        # This prevents sancta_bridge and sancta_gpt from loading each other in a loop
        from sancta_gpt import get_engine
        
        # Ensure the engine is initialized
        engine = get_engine()
        
        # Generate the response using your engine's logic
        response = engine.generate(
            prompt=prompt, 
            max_tokens=self.max_tokens, 
            temperature=self.temperature
        )
        
        # LangChain uses stop sequences to truncate output
        if stop is not None:
            for sequence in stop:
                if sequence in response:
                    response = response.split(sequence)[0]
        
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Export internal parameters for LangChain tracking."""
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}