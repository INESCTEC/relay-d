import time

from relay_d.utils.coloring_logger import logger


def start_demo(recorder, topic_subscribers, containers, dataset_name, demo_name=None):
    """The per-Start steps the GUI and AppAPI share: subscribe every input again (re-detects the message type and QoS,
    resets custom TF buffers; a broken custom TF topic raises TfTopicSubscriptionError), then start the recorder (it
    flushes the data from before Start and first retries an unsaved demo, see ``recorder.retried_file``).
    Returns the demo name, or None if the recording did not start. An input that could not be subscribed refuses the
    Start: it would be missing from the demo with nothing to show it until conversion."""
    results = topic_subscribers.create_subscribers_for_containers(containers)
    logger.info(f"Subscribers created: {sum(results.values())}/{len(results)}")
    failed = sorted(name for name, ok in results.items() if not ok)
    if failed:
        logger.error(f"Recording not started: could not subscribe {', '.join(failed)}. Check that the topics are published.")
        return None
    demo_name = demo_name or f"{dataset_name}_{time.time_ns() // 1_000_000}"  # ms: two Starts in one second must not clash ("w-")
    return demo_name if recorder.start_recording(topic_subscribers, demo_name) else None
