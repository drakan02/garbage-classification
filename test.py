import gradio as gr

def test(x):
    return x

with gr.Blocks() as demo:
    gr.Markdown("## Test Gradio Blocks")
    inp = gr.Textbox()
    out = gr.Textbox()
    inp.change(test, inp, out)

demo.launch()
