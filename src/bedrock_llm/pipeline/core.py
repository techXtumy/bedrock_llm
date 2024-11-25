import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from rx import operators as ops
from rx.subject import Subject

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class NodeContext:
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any]
    status: NodeStatus


class PipelineNode:
    def __init__(
        self,
        name: str,
        func: Union[Callable, None] = None,
        is_async: bool = False
    ):
        self.name = name
        self.func = func
        self.is_async = is_async
        self.input_subject = Subject()
        self.output_subject = Subject()
        self.downstream_nodes: List['PipelineNode'] = []
        self.context = NodeContext(
            inputs={},
            outputs={},
            metadata={},
            status=NodeStatus.PENDING
        )

    async def process(self, data: Any) -> Any:
        try:
            self.context.status = NodeStatus.RUNNING
            self.context.inputs = data

            if self.func is None:
                result = data
            elif self.is_async:
                result = await self.func(data)
            else:
                with ThreadPoolExecutor() as executor:
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, self.func, data
                    )

            self.context.outputs = result
            self.context.status = NodeStatus.COMPLETED
            return result

        except Exception as e:
            self.context.status = NodeStatus.FAILED
            logger.error(f"Error in node {self.name}: {str(e)}")
            raise

    def connect(self, node: 'PipelineNode') -> 'PipelineNode':
        self.downstream_nodes.append(node)
        self.output_subject.pipe(
            ops.filter(lambda x: x is not None)
        ).subscribe(node.input_subject)
        return node


class Pipeline:
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, PipelineNode] = {}
        self.start_nodes: List[PipelineNode] = []

    def add_node(self, node: PipelineNode) -> PipelineNode:
        self.nodes[node.name] = node
        return node

    def set_start_node(self, node: PipelineNode):
        if node.name in self.nodes:
            self.start_nodes.append(node)
        else:
            raise ValueError(f"Node {node.name} not found in pipeline")

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        results = {}

        async def process_node(node: PipelineNode, data: Any):
            try:
                result = await node.process(data)
                results[node.name] = result

                # Process downstream nodes
                for downstream in node.downstream_nodes:
                    await process_node(downstream, result)

            except Exception as e:
                logger.error(f"Error executing node {node.name}: {str(e)}")
                raise

        # Start execution from start nodes
        for start_node in self.start_nodes:
            await process_node(start_node, input_data)

        return results

    def get_node(self, name: str) -> Optional[PipelineNode]:
        return self.nodes.get(name)

    def get_pipeline_status(self) -> Dict[str, NodeStatus]:
        return {name: node.context.status for name, node in self.nodes.items()}
