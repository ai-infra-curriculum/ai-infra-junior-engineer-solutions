"""
OpenTelemetry tracing configuration.

Provides distributed tracing with:
- Automatic FastAPI instrumentation
- Manual span creation for custom operations
- Trace context propagation
- OTLP export to Jaeger/Tempo
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from typing import Optional


def setup_tracing(
    service_name: str,
    service_version: str,
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False
) -> TracerProvider:
    """
    Configure OpenTelemetry tracing.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint (e.g., http://jaeger:4318)
        enable_console_export: Enable console exporter for debugging

    Returns:
        Configured TracerProvider
    """
    # Define service resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint is provided
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"{otlp_endpoint}/v1/traces"
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Add console exporter for debugging
    if enable_console_export:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Instrument logging to add trace context
    LoggingInstrumentor().instrument()

    return provider


def instrument_fastapi(app):
    """
    Instrument FastAPI application with OpenTelemetry.

    This automatically creates spans for all HTTP requests.

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually __name__)

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def create_span(
    name: str,
    attributes: Optional[dict] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
) -> trace.Span:
    """
    Create a new span.

    Args:
        name: Span name
        attributes: Span attributes
        kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)

    Returns:
        Started span
    """
    tracer = get_tracer()
    span = tracer.start_span(name, kind=kind)

    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    return span


# Context manager for easier span creation
class TracedOperation:
    """
    Context manager for creating traced operations.

    Usage:
        with TracedOperation("model_inference", {"model": "resnet50"}):
            # Your code here
            result = model.predict(image)
    """

    def __init__(
        self,
        name: str,
        attributes: Optional[dict] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL
    ):
        self.name = name
        self.attributes = attributes or {}
        self.kind = kind
        self.span: Optional[trace.Span] = None

    def __enter__(self):
        tracer = get_tracer()
        self.span = tracer.start_span(self.name, kind=self.kind)

        # Set attributes
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)

        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # Record exception if one occurred
            if exc_type is not None:
                self.span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exc_val))
                )
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(trace.Status(trace.StatusCode.OK))

            self.span.end()


def add_span_event(name: str, attributes: Optional[dict] = None):
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Event attributes
    """
    span = trace.get_current_span()
    if span != trace.INVALID_SPAN:
        span.add_event(name, attributes or {})


def set_span_attribute(key: str, value):
    """
    Set an attribute on the current span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    span = trace.get_current_span()
    if span != trace.INVALID_SPAN:
        span.set_attribute(key, value)


def set_span_error(error: Exception):
    """
    Mark the current span as error and record the exception.

    Args:
        error: Exception instance
    """
    span = trace.get_current_span()
    if span != trace.INVALID_SPAN:
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        span.record_exception(error)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a hex string.

    Returns:
        Trace ID or None if no active span
    """
    span = trace.get_current_span()
    if span != trace.INVALID_SPAN:
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID as a hex string.

    Returns:
        Span ID or None if no active span
    """
    span = trace.get_current_span()
    if span != trace.INVALID_SPAN:
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.span_id, "016x")
    return None
