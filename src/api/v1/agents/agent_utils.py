# FORMATTING OUTPUT
def format_llm_output(res):
    content = res.content

    # dict 
    if isinstance(content, dict):
        return content.get("text", str(content))

    

    return str(content)

