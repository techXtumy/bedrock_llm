import sys
import os
import asyncio
import platform

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Add for print console with color
from termcolor import cprint
from datetime import datetime

from src.bedrock_llm.client import LLMClient, ModelName, RetryConfig, ModelConfig, MessageBlock

client = LLMClient(
    region_name="us-west-2",
    model_name=ModelName.CLAUDE_3_5_SONNET,
    retry_config=RetryConfig(
        max_retries=3,
        retry_delay=1,
        exponential_backoff=True
    )
)

model_config = ModelConfig(
    max_tokens=1024,
    temperature=0.5,
    top_p=0.9,
    top_k=60
)

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""

tools=[
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1080,
        "display_height_px": 1080,
        "display_number": 1
    },
    {
        "type": "text_editor_20241022",
        "name": "str_replace_editor"
    },
    {
        "type": "bash_20241022",
        "name": "bash"
    },
]


memory = []


async def get_input(
    placeholder: str
) -> str:
    return input(placeholder)


async def main():
    while True:
        
        user_input = await get_input("Enter a prompt: ")
        
        if user_input.lower() == "/bye":
            break
        
        memory.append(MessageBlock(
            role="user",
            content=user_input
            ).model_dump()
        )
        
        try:
            async for chunks, stop_reason in client.generate(
                prompt=memory,
                config=model_config
            ):
                if isinstance(chunks, str):  
                    cprint(chunks, "green", end="", flush=True)
                if stop_reason:
                    memory.append(chunks.model_dump())
                    cprint(f"\n{stop_reason}", "red", flush=True)
                    break
        except Exception as e:
            cprint(e, "red")


if __name__ == "__main__":
    asyncio.run(main())