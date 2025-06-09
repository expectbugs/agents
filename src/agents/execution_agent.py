#!/usr/bin/env python3
"""
Execution Agent - Performs concrete tasks and executes plans
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio
import os
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from ..orchestrator import BaseAgent, MultiAgentState
from ..config import Config
from ..logging_config import get_logger

logger = get_logger(__name__)

class ExecutionAgent(BaseAgent):
    """Agent responsible for executing tasks and running code"""
    
    def __init__(self):
        super().__init__("executor")
        self.workspace_dir = Config.WORKSPACE_DIR
        self.workspace_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the execution agent"""
        await super().initialize()
        logger.info("Execution agent initialized")
    
    async def process(self, state: MultiAgentState) -> Dict[str, Any]:
        """Process execution request"""
        logger.info("Execution agent processing...")
        
        # Get current task from queue
        task_queue = state.get("task_queue", [])
        current_task = None
        
        if task_queue:
            # Find next pending task
            for task in task_queue:
                if task.get("status") == "pending":
                    current_task = task
                    break
        
        if not current_task:
            # No pending tasks, check if we should create one from messages
            return await self._handle_direct_execution(state)
        
        # Execute the current task
        result = await self._execute_task(current_task)
        
        # Create updated task for state reducer
        updated_task = {
            **current_task,
            "status": "completed" if result["success"] else "failed",
            "completed_at": datetime.now().isoformat(),
            "result": result
        }
        
        # Save execution state
        await self.save_state({
            "last_task": current_task,
            "execution_result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create response message
        status_emoji = "✅" if result["success"] else "❌"
        response_content = f"{status_emoji} **Task Execution**\n\n"
        response_content += f"**Task:** {current_task.get('title', 'Unknown')}\n"
        response_content += f"**Result:** {result['message']}\n"
        
        if result.get("output"):
            response_content += f"**Output:**\n```\n{result['output'][:500]}...\n```"
        
        response_message = AIMessage(content=response_content)
        
        # Determine next agent
        remaining_tasks = [t for t in task_queue if t.get("status") == "pending"]
        next_agent = "executor" if remaining_tasks else "end"
        
        return {
            "messages": [response_message],
            "current_agent": next_agent,
            "task_queue": [updated_task],  # Send updated task to reducer
            "context": {
                **state.get("context", {}),
                "last_execution": result,
                "execution_timestamp": datetime.now().isoformat()
            }
        }
    
    async def _handle_direct_execution(self, state: MultiAgentState) -> Dict[str, Any]:
        """Handle direct execution requests without a task queue"""
        messages = state.get("messages", [])
        
        if not messages:
            return {
                "messages": [AIMessage(content="No task to execute.")],
                "current_agent": "orchestrator"
            }
        
        # Get the last human message as the task
        task_content = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'content') and 'execute' in getattr(msg, 'content', '').lower()):
                task_content = getattr(msg, 'content', str(msg))
                break
        
        if not task_content:
            return {
                "messages": [AIMessage(content="No clear execution task found.")],
                "current_agent": "orchestrator"
            }
        
        # Create a temporary task
        temp_task = {
            "title": task_content[:50] + "..." if len(task_content) > 50 else task_content,
            "description": task_content,
            "agent": "executor",
            "type": "direct_execution"
        }
        
        result = await self._execute_task(temp_task)
        
        response_message = AIMessage(
            content=f"🔧 **Direct Execution**\n\n{result['message']}"
        )
        
        return {
            "messages": [response_message],
            "current_agent": "orchestrator",
            "context": {
                **state.get("context", {}),
                "direct_execution": result
            }
        }
    
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task"""
        task_title = task.get("title", "").lower()
        task_description = task.get("description", "").lower()
        
        # Determine task type and execute accordingly
        if "install" in task_title or "install" in task_description:
            return await self._execute_installation(task)
        elif "create project" in task_title or "setup" in task_title:
            return await self._execute_project_setup(task)
        elif "run" in task_title or "execute" in task_title:
            return await self._execute_command(task)
        elif "test" in task_title or "validate" in task_title:
            return await self._execute_testing(task)
        else:
            return await self._execute_generic(task)
    
    async def _execute_installation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute package installation tasks using async subprocess"""
        description = task.get("description", "")
        
        # Extract package names from description
        packages = []
        if "requests" in description:
            packages.append("requests")
        if "beautifulsoup4" in description:
            packages.append("beautifulsoup4")
        if "scrapy" in description:
            packages.append("scrapy")
        if "pandas" in description:
            packages.append("pandas")
        if "numpy" in description:
            packages.append("numpy")
        
        if not packages:
            packages = ["requests"]  # Default fallback
        
        try:
            cmd = ["pip", "install"] + packages
            
            # Use async subprocess to avoid blocking the event loop
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=Config.ASYNC_TIMEOUT
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "message": f"Installation timed out after {Config.ASYNC_TIMEOUT} seconds"
                }
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            if process.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully installed packages: {', '.join(packages)}",
                    "output": stdout_text,
                    "packages_installed": packages
                }
            else:
                return {
                    "success": False,
                    "message": f"Installation failed: {stderr_text}",
                    "output": stderr_text
                }
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return {
                "success": False,
                "message": f"Installation error: {str(e)}"
            }
    
    async def _execute_project_setup(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute project setup tasks"""
        project_name = "web_scraper_project"
        project_dir = self.workspace_dir / project_name
        
        try:
            # Create project directory
            project_dir.mkdir(exist_ok=True)
            
            # Create basic project structure
            (project_dir / "src").mkdir(exist_ok=True)
            (project_dir / "tests").mkdir(exist_ok=True)
            (project_dir / "data").mkdir(exist_ok=True)
            
            # Create basic files
            (project_dir / "README.md").write_text("# Web Scraper Project\n\nCreated by AI agent.\n")
            (project_dir / "requirements.txt").write_text("requests\nbeautifulsoup4\nlxml\n")
            (project_dir / ".gitignore").write_text("__pycache__/\n*.pyc\ndata/\n.env\n")
            
            return {
                "success": True,
                "message": f"Project structure created at {project_dir}",
                "project_path": str(project_dir),
                "files_created": ["README.md", "requirements.txt", ".gitignore"]
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Project setup failed: {str(e)}"
            }
    
    async def _execute_command(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command-line tasks with async subprocess"""
        try:
            # Extract command from task description
            description = task.get("description", "")
            
            # For security, only allow specific safe commands
            safe_commands = Config.ALLOWED_COMMANDS
            
            # Find a matching safe command
            command_to_run = None
            for safe_cmd in safe_commands:
                if safe_cmd.lower() in description.lower():
                    command_to_run = safe_cmd.split()
                    break
            
            if not command_to_run:
                return {
                    "success": False,
                    "message": f"Command not allowed. Safe commands: {', '.join(safe_commands)}"
                }
            
            if not Config.ENABLE_COMMAND_EXECUTION:
                return {
                    "success": True,
                    "message": f"Command execution simulated: {' '.join(command_to_run)}",
                    "note": "Set ENABLE_COMMAND_EXECUTION=true to enable real execution"
                }
            
            # Execute command asynchronously
            process = await asyncio.create_subprocess_exec(
                *command_to_run,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30  # Shorter timeout for commands
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "message": "Command execution timed out"
                }
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            return {
                "success": process.returncode == 0,
                "message": f"Command executed: {' '.join(command_to_run)}",
                "output": stdout_text,
                "error": stderr_text if stderr_text else None
            }
            
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "success": False,
                "message": f"Command execution error: {str(e)}"
            }
    
    async def _execute_testing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing tasks"""
        try:
            # Check if project structure exists
            project_dir = self.workspace_dir / "web_scraper_project"
            
            if project_dir.exists():
                test_results = {
                    "project_exists": True,
                    "structure_valid": (project_dir / "src").exists() and (project_dir / "tests").exists(),
                    "files_present": len(list(project_dir.glob("*")))
                }
                
                return {
                    "success": True,
                    "message": "Project validation completed",
                    "test_results": test_results
                }
            else:
                return {
                    "success": False,
                    "message": "Project not found for testing"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Testing error: {str(e)}"
            }
    
    async def _execute_generic(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic tasks"""
        return {
            "success": True,
            "message": f"Task processed: {task.get('title', 'Unknown task')}",
            "note": "Generic execution - implement specific logic for this task type"
        }

# Example standalone test
async def test_execution_agent():
    """Test the execution agent"""
    import asyncio
    
    agent = ExecutionAgent()
    await agent.initialize()
    
    # Test with a task queue
    test_tasks = [
        {
            "id": "task_1",
            "title": "Install required libraries",
            "description": "Install required libraries (requests, beautifulsoup4)",
            "agent": "executor",
            "status": "pending"
        },
        {
            "id": "task_2", 
            "title": "Set Up Development Environment",
            "description": "Create project structure",
            "agent": "executor",
            "status": "pending"
        }
    ]
    
    test_state = MultiAgentState(
        messages=[HumanMessage(content="Execute the installation task")],
        current_agent="executor",
        current_model="",
        gpu_memory={},
        context={},
        task_queue=test_tasks,
        agent_states={},
        error_count=0,
        metadata={}
    )
    
    # Execute first task
    result1 = await agent.process(test_state)
    print("First execution result:")
    print(result1['messages'][0].content)
    
    # Update state manually (MultiAgentState doesn't have update() method)
    if result1.get("messages"):
        test_state.messages.extend(result1["messages"])
    if result1.get("context"):
        test_state.context.update(result1["context"])
    if result1.get("task_queue"):
        test_state.task_queue.extend(result1["task_queue"])
    if result1.get("current_agent"):
        test_state.current_agent = result1["current_agent"]
    result2 = await agent.process(test_state)
    print("\nSecond execution result:")
    print(result2['messages'][0].content)
    
    await agent.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_execution_agent())