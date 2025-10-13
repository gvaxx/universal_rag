"""Gradio UI scaffold for Universal RAG System."""

from __future__ import annotations

import gradio as gr

from config import AppConfig


def _fake_answer(query: str) -> str:
    # Placeholder to keep UI functional
    return f"Echo: {query}"


def create_interface(config: AppConfig) -> gr.Blocks:
    """Create and return a minimal Gradio interface."""
    with gr.Blocks(title="Universal RAG") as demo:
        gr.Markdown("# Universal RAG System")
        inp = gr.Textbox(label="Вопрос")
        out = gr.Textbox(label="Ответ")
        btn = gr.Button("Спросить")
        btn.click(_fake_answer, inputs=inp, outputs=out)
    return demo


