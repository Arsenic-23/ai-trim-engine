import os
import sys
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.shell import BashLexer
from rich.console import Console
import typer

from trim_engine.config import PROJECTS_DIR

console = Console()

class TrimCommandCompleter(Completer):
    def __init__(self):
        self.commands = {
            '/ingest': 'Run full ingestion pipeline on a video file',
            '/edit': 'Run the edit pipeline (intent → retrieve → plan → critic → render)',
            '/status': 'Show ingestion status, coverage, and cost for a video',
            '/ask': "Ask a question about the video's knowledge base",
            '/suite': 'Run the full 25-prompt regression suite',
            '/clear': 'Clear the terminal screen',
            '/exit': 'Exit the interactive shell',
            '/quit': 'Exit the interactive shell'
        }
    
    def get_video_ids(self):
        if not PROJECTS_DIR.exists():
            return []
        return [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir() and (d / "project.db").exists()]
        
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        words = text.split()
        
        # Complete commands
        if len(words) == 0 or (len(words) == 1 and not text.endswith(' ')):
            for cmd, desc in self.commands.items():
                if cmd.startswith(words[0] if words else ''):
                    yield Completion(cmd, start_position=-len(words[0]) if words else 0, display_meta=desc)
        
        # Complete video IDs for commands that need it
        elif len(words) == 1 and text.endswith(' '):
            if words[0] in ['/edit', '/status', '/ask', '/suite']:
                for vid in self.get_video_ids():
                    yield Completion(vid, start_position=0)
        elif len(words) == 2 and not text.endswith(' '):
            if words[0] in ['/edit', '/status', '/ask', '/suite']:
                for vid in self.get_video_ids():
                    if vid.startswith(words[1]):
                        yield Completion(vid, start_position=-len(words[1]))

from prompt_toolkit.shortcuts import CompleteStyle

style = Style.from_dict({
    'prompt': '#00ffff bold',
    'input': '#ffffff',
    'completion-menu': 'bg:#111111 #ffffff',
    'completion-menu.completion': 'bg:#111111 #ffffff',
    'completion-menu.completion.current': 'bg:#00ffff #000000 bold',
    'completion-menu.meta.completion': 'bg:#111111 #888888',
    'completion-menu.meta.completion.current': 'bg:#00ffff #000000',
})

def run_shell():
    # Deferred imports to avoid circular dependency
    from trim_engine.cli import ingest, edit, status, ask, suite
    
    session = PromptSession(
        completer=TrimCommandCompleter(),
        style=style,
        lexer=PygmentsLexer(BashLexer),
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
    )
    
    while True:
        try:
            text = session.prompt("\n craon ❯ ")
        except (KeyboardInterrupt, EOFError):
            break
            
        text = text.strip()
        if not text:
            continue
            
        if text in ['/exit', '/quit', 'exit', 'quit']:
            break
            
        if text == '/clear':
            os.system('clear')
            continue
            
        parts = text.split()
        cmd = parts[0]
        args = parts[1:]
        
        try:
            if cmd == '/ingest':
                if len(args) != 1:
                    console.print("[red]Usage: /ingest <video_path>[/red]")
                    continue
                ingest(args[0])
            elif cmd == '/edit':
                if len(args) < 2:
                    console.print("[red]Usage: /edit <video_id> <prompt>[/red]")
                    continue
                edit(args[0], " ".join(args[1:]), revise=None, yes=False)
            elif cmd == '/status':
                if len(args) != 1:
                    console.print("[red]Usage: /status <video_id>[/red]")
                    continue
                status(args[0])
            elif cmd == '/ask':
                if len(args) < 2:
                    console.print("[red]Usage: /ask <video_id> <question>[/red]")
                    continue
                ask(args[0], " ".join(args[1:]))
            elif cmd == '/suite':
                if len(args) != 1:
                    console.print("[red]Usage: /suite <video_id>[/red]")
                    continue
                suite(args[0])
            elif cmd.startswith('/'):
                console.print(f"[red]Unknown command: {cmd}[/red]")
            else:
                console.print("[yellow]Please use a slash command like /edit, /status, or /ask[/yellow]")
                
        except typer.Exit as e:
            # Typer exits when a command completes or fails via raise typer.Exit()
            pass
        except SystemExit:
            pass
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
