# src/tilegen/integrations/vllm_pspkv/__init__.py
#
# vllm-pspkv (PSP-KV vLLM fork) integration package: registers the
# TILEGEN_PSPKV attention backend (see attention_backend.py). The module is
# deliberately NOT imported here: registration happens through the vLLM OOT
# plugin (tilegen.integrations.vllm.platform_plugin), because importing it
# requires a vLLM 0.34-based environment.
