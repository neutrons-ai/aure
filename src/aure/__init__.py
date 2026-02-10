"""
Agentic Reflectivity Modeling Workflow.

This module provides an intelligent assistant for analyzing neutron reflectivity data
using LangGraph for workflow management and refl1d for optimization.

Main components:
- workflow: LangGraph state machine for the analysis pipeline
- state: State definitions for the workflow
- tools: LangChain tools for data loading, feature extraction, and model building
- database: Material SLD database
- cli: Click-based command-line interface
- mcp_server: FastMCP server for AI assistant integration

Usage:
    # Python API
    from aure import run_analysis
    result = run_analysis("data.dat", "100 nm polystyrene on silicon")
    
    # CLI
    python -m aure.cli analyze data.dat "100 nm polystyrene on silicon"
    
    # MCP Server
    python -m aure.cli mcp-server
"""

import warnings

# Suppress compatibility warnings for Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

from .state import ReflectivityState, create_initial_state
from .workflow import create_workflow, run_analysis

__all__ = [
    'ReflectivityState', 
    'create_initial_state',
    'create_workflow',
    'run_analysis',
]
