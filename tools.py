"""PrivAI - Day 5: Tools + a small agent loop.

The model can ask us to run these tools. For safety, every tool can ONLY
touch files inside the 'workspace' folder (no escaping with ../ tricks).
"""
import csv
from pathlib import Path

import ollama
from docx import Document
from openpyxl import Workbook
from openpyxl.styles import Font
from pypdf import PdfReader

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)
MAX_READ_CHARS = 6000


def _safe_path(filename: str) -> Path:
    """Turn a filename into a path, but refuse anything outside workspace/."""
    root = WORKSPACE.resolve()
    filename = filename.replace("\\", "/")      # treat ..\x like ../x on every OS
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Access denied: path is outside the workspace folder")
    return path


# ------------------------- TOOLS -------------------------
# (Type hints + docstrings matter: the model reads them to learn how to use each tool.)

def list_files() -> str:
    """List all files available in the workspace folder."""
    files = [p.name for p in WORKSPACE.iterdir() if p.is_file()]
    return "\n".join(files) if files else "The workspace is empty."


def read_file(filename: str) -> str:
    """Read a file from the workspace (txt, md, csv, pdf or docx).

    Args:
        filename: Name of the file inside the workspace, e.g. notes.txt
    """
    try:
        path = _safe_path(filename)
        if not path.exists():
            return f"Error: {filename} does not exist."
        ext = path.suffix.lower()
        if ext == ".pdf":
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif ext == ".docx":
            text = "\n".join(p.text for p in Document(path).paragraphs)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n...[truncated]"
        return text or "(file is empty)"
    except Exception as e:
        return f"Error: {e}"


def write_file(filename: str, content: str) -> str:
    """Write plain text to a file in the workspace (creates or overwrites it).

    Args:
        filename: Name of the file to create, e.g. summary.txt
        content: The text to save inside the file
    """
    try:
        _safe_path(filename).write_text(content, encoding="utf-8")
        return f"Saved {filename}"
    except Exception as e:
        return f"Error: {e}"


def create_word_report(filename: str, title: str, content: str) -> str:
    """Create a Word (.docx) report in the workspace.

    Args:
        filename: Name of the Word file, e.g. report.docx
        title: Title shown at the top of the report
        content: Report text. Use '## ' at the start of a line for a heading
            and '- ' for bullet points. Separate paragraphs with blank lines.
    """
    try:
        if not filename.endswith(".docx"):
            filename += ".docx"
        doc = Document()
        doc.add_heading(title, level=0)
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("## "):
                doc.add_heading(line[3:], level=1)
            elif line.startswith("- "):
                doc.add_paragraph(line[2:], style="List Bullet")
            else:
                doc.add_paragraph(line)
        doc.save(_safe_path(filename))
        return f"Created Word report {filename}"
    except Exception as e:
        return f"Error: {e}"


def create_excel(filename: str, data: str) -> str:
    """Create an Excel (.xlsx) file in the workspace from comma-separated rows.

    Args:
        filename: Name of the Excel file, e.g. marks.xlsx
        data: Rows of data, one row per line, values separated by commas.
            The first line is the header row.
    """
    try:
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        wb = Workbook()
        ws = wb.active
        for r, row in enumerate(csv.reader(data.strip().splitlines()), start=1):
            ws.append(row)
            if r == 1:
                for cell in ws[1]:
                    cell.font = Font(bold=True)
        wb.save(_safe_path(filename))
        return f"Created Excel file {filename}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [list_files, read_file, write_file, create_word_report, create_excel]
TOOL_MAP = {f.__name__: f for f in TOOLS}

AGENT_PROMPT = (
    "You are PrivAI, a private offline assistant that can use tools to work "
    "with files in a workspace folder. Use a tool only when the task needs it. "
    "If you are unsure which files exist, call list_files first. "
    "When finished, briefly say what you did and name any files you created."
)


# ------------------------- AGENT LOOP -------------------------
def run_agent(model, user_prompt, history=None, max_steps=6):
    """Let the model think -> call tools -> see results -> repeat until done.

    Returns (final_answer, steps) where steps is a list of tool calls made.
    """
    messages = [{"role": "system", "content": AGENT_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": user_prompt})
    steps = []

    for _ in range(max_steps):
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        msg = response.message
        messages.append(msg)

        if not msg.tool_calls:                    # no more tools -> final answer
            return msg.content, steps

        for call in msg.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            if name in TOOL_MAP:
                try:
                    result = TOOL_MAP[name](**args)
                except TypeError as e:            # model sent wrong arguments
                    result = f"Error: bad arguments ({e})"
            else:
                result = f"Error: unknown tool {name}"
            steps.append({"tool": name, "args": args, "result": str(result)})
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "I stopped because I reached the step limit.", steps