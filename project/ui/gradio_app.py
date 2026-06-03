import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem
import locale_fa as L
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

def create_gradio_ui():
    rag_system = RAGSystem()
    needs_reindex = rag_system.initialize()

    doc_manager = DocumentManager(rag_system)
    if needs_reindex:
        indexed = doc_manager.reindex_all()
        if indexed:
            gr.Info(L.UI_REINDEXED.format(count=indexed))

    chat_interface = ChatInterface(rag_system)
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return L.UI_NO_DOCUMENTS
        return "\n".join([f"{f}" for f in files])
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
            
        added, skipped = doc_manager.add_documents(
            files, 
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(L.UI_ADDED_INFO.format(added=added, skipped=skipped))
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info(L.UI_CLEARED_INFO)
        return format_file_list()
    
    def chat_handler(msg, hist):
        for chunk in chat_interface.chat(msg, hist):
            yield chunk
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    with gr.Blocks(title=L.UI_TITLE) as demo:
        
        with gr.Tab(L.UI_TAB_DOCUMENTS, elem_id="doc-management-tab"):
            gr.Markdown(L.UI_UPLOAD_TITLE)
            gr.Markdown(L.UI_UPLOAD_DESC)
            
            files_input = gr.File(
                label=L.UI_FILE_LABEL,
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False
            )
            
            add_btn = gr.Button(L.UI_ADD_BTN, variant="primary", size="md")
            
            gr.Markdown(L.UI_CURRENT_DOCS)
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines = 7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button(L.UI_REFRESH_BTN, size="md")
                clear_btn = gr.Button(L.UI_CLEAR_BTN, variant="stop", size="md")
            
            add_btn.click(upload_handler, [files_input], [files_input, file_list], show_progress="corner")
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab(L.UI_TAB_CHAT):
            chatbot = gr.Chatbot(
                height=720, 
                placeholder=L.UI_CHAT_PLACEHOLDER,
                show_label=False,
                avatar_images=(None, os.path.join(ASSETS_DIR, "chatbot_avatar.png")),
                layout="bubble"
            )
            chatbot.clear(clear_chat_handler)
            
            gr.ChatInterface(fn=chat_handler, chatbot=chatbot)
    
    return demo
