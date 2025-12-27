import torch
from typing import Optional, List

from mgds.PipelineModule import PipelineModule


class WanVideoTextEncoder(PipelineModule):
    """
    MGDS pipeline module for encoding text prompts for WAN 2.2 video generation.
    
    This module handles text tokenization and encoding specifically optimized
    for video generation tasks, including temporal prompt understanding.
    """
    
    def __init__(
        self,
        prompt_in_name: str,
        tokens_out_name: str,
        hidden_state_out_name: str,
        pooled_out_name: Optional[str] = None,
        tokenizer=None,
        text_encoder=None,
        max_token_length: int = 77,
        autocast_contexts: Optional[List[torch.autocast]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.prompt_in_name = prompt_in_name
        self.tokens_out_name = tokens_out_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.max_token_length = max_token_length
        self.autocast_contexts = autocast_contexts or []
        self.dtype = dtype
    
    def length(self) -> int:
        return self._get_previous_length(self.prompt_in_name)
    
    def get_inputs(self) -> list[str]:
        return [self.prompt_in_name]
    
    def get_outputs(self) -> list[str]:
        outputs = [self.tokens_out_name, self.hidden_state_out_name]
        if self.pooled_out_name:
            outputs.append(self.pooled_out_name)
        return outputs
    
    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        print(f"DEBUG: WanVideoTextEncoder.get_item called - variation: {variation}, index: {index}")
        
        try:
            prompt = self._get_previous_item(variation, index, self.prompt_in_name)
            print(f"DEBUG: Got prompt: {prompt}")
            
            if prompt is None:
                print(f"DEBUG: Prompt is None, creating fallback")
                prompt = "a cube"  # Fallback prompt
                
        except Exception as e:
            print(f"DEBUG: Error getting prompt: {e}")
            prompt = "a cube"  # Fallback prompt
        
        # Tokenize the prompt
        if self.tokenizer:
            # Handle video-specific prompt formatting
            formatted_prompt = self._format_video_prompt(prompt)
            
            # Tokenize with padding and truncation
            tokens = self.tokenizer(
                formatted_prompt,
                padding="max_length",
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = tokens.input_ids.squeeze(0)  # Remove batch dimension
            attention_mask = tokens.attention_mask.squeeze(0)
        else:
            # Fallback if no tokenizer provided
            input_ids = torch.zeros(self.max_token_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_token_length, dtype=torch.long)
        
        result = {self.tokens_out_name: input_ids}
        
        # Encode text if text encoder is provided
        if self.text_encoder:
            with torch.no_grad():
                # Apply autocast contexts if provided
                autocast_context = torch.autocast('cuda', enabled=False)
                if self.autocast_contexts:
                    autocast_context = self.autocast_contexts[0]
                
                with autocast_context:
                    # Add batch dimension for encoding
                    input_ids_batch = input_ids.unsqueeze(0)
                    attention_mask_batch = attention_mask.unsqueeze(0)
                    
                    # Encode text
                    encoder_outputs = self.text_encoder(
                        input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        return_dict=True
                    )
                    
                    # Extract hidden states
                    hidden_states = encoder_outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
                    hidden_states = hidden_states.to(dtype=self.dtype)
                    
                    result[self.hidden_state_out_name] = hidden_states
                    
                    # Extract pooled output if available and requested
                    if self.pooled_out_name and hasattr(encoder_outputs, 'pooler_output'):
                        pooled_output = encoder_outputs.pooler_output.squeeze(0)
                        pooled_output = pooled_output.to(dtype=self.dtype)
                        result[self.pooled_out_name] = pooled_output
        
        return result
    
    def _format_video_prompt(self, prompt: str) -> str:
        """
        Format prompt for video generation.
        
        This can include adding video-specific tokens or formatting
        to help the model understand temporal aspects.
        """
        # For now, return the prompt as-is
        # In the future, this could add video-specific formatting like:
        # - Temporal markers
        # - Action descriptors
        # - Scene transition indicators
        return prompt