# SPDX-License-Identifier: Apache-2.0

import os
import time
from collections.abc import Mapping
from typing import Optional, List, Tuple
from contextlib import AbstractContextManager

from vllm.logger import init_logger
from vllm.utils import run_once
from vllm.sequence import SequenceGroup, SequenceGroupMetadata, ExecuteModelRequest


TRACE_HEADERS = ["traceparent", "tracestate"]

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import (
        SpanKind,
        Tracer,
        Span,
        use_span,
        set_tracer_provider,
        get_current_span,
        set_span_in_context,
    )
    from opentelemetry.context import attach, detach, get_current
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.propagate import inject

    try:
        from opentelemetry.trace import SpanLink  # ≥ 1.24
    except ImportError:
        from opentelemetry.trace import Link as SpanLink  # ≤ 1.23

        logger.warning("Using deprecated 'Link' class instead of 'SpanLink'")

    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass


# Global tracer
vllm_instance_global_tracer: dict = dict()


def init_tracer_globally(
    tracer_name: str, otlp_traces_endpoint: str
) -> Optional[Tracer]:
    if tracer_name in vllm_instance_global_tracer:
        logger.warning(
            f"Tracer with name '{tracer_name}' is already initialized. "
            "Returning the existing tracer."
        )
        return vllm_instance_global_tracer[tracer_name]
    else:
        tracer = init_tracer(tracer_name, otlp_traces_endpoint)
        if tracer is not None:
            vllm_instance_global_tracer[tracer_name] = tracer
            logger.info(f"Initialized global tracer '{tracer_name}'")
            return tracer
        else:
            logger.error(
                f"Failed to initialize tracer '{tracer_name}'. "
                "Ensure OpenTelemetry packages are installed."
            )


def get_tracer_globally(tracer_name: str) -> Optional[Tracer]:
    if tracer_name in vllm_instance_global_tracer:
        return vllm_instance_global_tracer[tracer_name]
    else:
        raise ValueError(
            f"Tracer with name '{tracer_name}' is not initialized. "
            "Call `init_tracer_globally` to initialize it first."
        )


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(
    instrumenting_module_name: str, otlp_traces_endpoint: str
) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}"
        )
    trace_provider = TracerProvider()

    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )  # type: ignore
    else:
        raise ValueError(f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:

    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")


class BatchedSpanManager(AbstractContextManager):
    """
    - 只创建 1 个 batch-span（代表 GPU-kernel / prefill / decode …）
    - 用 SpanLink 把这一批所有请求的 SpanContext 挂进来
    - （可选）把 batch-span 当作 parent 继续向下游 worker 传播
    """

    def __init__(
        self, 
        tracer, 
        seq_group_metadata_list: Optional[List] = None, 
        scheduled_seq_groups: Optional[List] = None
    ):
        """
        Args:
            tracer: OTel trace provider
            seq_group_metadata_list: 从调度器输出的当前调度轮次的用户请求的meta_data，这部分数据要传给Worker
            scheduled_seq_groups: 从调度器输出的当前调度轮次的用户请求
        """
        self.tracer = tracer
        self.seq_group_metadata_list = seq_group_metadata_list
        self.scheduled_seq_groups = scheduled_seq_groups

        self.batch_span = None
        self._attach_token = None

    def __enter__(self):
        links = []  # List[SpanLink]
        rep_ctx = None  # 选一条请求的 ctx 作为代表
        step_type = None  # ‘p’ or ‘d’
        span_name = None
        call_mode = None
        
        # 0）确定调用BatchedSpanManager的上下文
        if self.scheduled_seq_groups is not None: # vLLM主控侧
            seq_group = next(iter(self.scheduled_seq_groups)).seq_group
            step_type = "prefill" if seq_group.is_prefill() else "decode"
            seq_group.step_cnt += 1
            span_name = f"{step_type}_{seq_group.step_cnt}"
            call_mode = "controller"
        else: # vLLM Worker侧
            span_name = f"worker_execute_model"
            call_mode = "worker"
            
        # 1) 收集所有请求的 SpanContext → SpanLink
        for idx, seq_group_meta in enumerate(self.seq_group_metadata_list):
            
            if call_mode == "controller" and self.scheduled_seq_groups[idx].seq_group.is_finished():
                    continue
            
            # 解析 trace headers -> Context -> SpanContext
            if call_mode == "controller":
                req_ctx = extract_trace_context(self.scheduled_seq_groups[idx].seq_group.trace_headers)
            else:
                req_ctx = extract_trace_context(seq_group_meta.trace_headers_variant)
                
            parent_span = get_current_span(req_ctx)  # NonRecordingSpan
            links.append(SpanLink(parent_span.get_span_context()))
            
            # 选第一条请求的 ctx 做代表，以便 batch-span 与它同 trace-id
            if rep_ctx is None:
                rep_ctx = req_ctx

        # 如果这一批里所有请求都已结束，直接 return self
        if rep_ctx is None:
            return self

        # 2) 创建一个batch-span，挂上所有 links
        self.batch_span = self.tracer.start_span(
            name=span_name,
            context=rep_ctx,  # 把代表 ctx 放 parent 位
            links=links,  # 把其余请求挂 Link
            kind=SpanKind.SERVER,
            start_time=time.time_ns(),
            attributes={
                "batch.size": len(links)
            }
        )
        
        self._attach_token = attach(set_span_in_context(self.batch_span)) # 激活 batch-span 上下文

        # 3) 把新的 trace headers，写回 execute_model_req，传递给下游 Worker
        # 注意，这里会把所有的请求的父span设置为当前的batch-span，然后发送给Worker
        for _, seq_group_meta in enumerate(self.seq_group_metadata_list):
            new_headers = {}
            inject(new_headers)  # 默认从“当前”Context 取 parent= batch-span
            seq_group_meta.trace_headers_variant = new_headers

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.batch_span:
            self.batch_span.end(end_time=time.time_ns())

        # 注意，在vLLM主控侧恢复ContextVar stack没有意义:
        # 1. vLLM主控循环有多层嵌套的函数调用，但是我们只关心execute_model这一层，所以BatchedSpanManager不会被嵌套调用
        # 2. 在__enter__函数中，是从seq_group获取的trace_headers，而不是从inject
        if self._attach_token:
            detach(self._attach_token)


class BatchedSpanManagerAuto(AbstractContextManager):
    def __init__(self, tracer, span_name: str):
        self.tracer = tracer
        self.span_name = span_name
        
        self.batch_span = None
        self._attach_token = None

    def __enter__(self):
        
        # 1）获取当前的父span的trace context
        parent_span_ctx = get_current()
        self.batch_span = self.tracer.start_span(
            name=self.span_name,
            context=parent_span_ctx,  # 把代表 ctx 放 parent 位
            kind=SpanKind.INTERNAL,
            start_time=time.time_ns()
        )
        
        # 2）激活 batch-span 上下文
        self._attach_token = attach(set_span_in_context(self.batch_span))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.batch_span:
            self.batch_span.end(end_time=time.time_ns())

        if self._attach_token:
            detach(self._attach_token)
