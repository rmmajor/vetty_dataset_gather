import asyncio
import logging
from datetime import datetime

from telegram import Bot

from cua.pipeline.registry import RepoRegistry
from cua.settings import NOTIFY

logger = logging.getLogger(__name__)


def get_bot_config() -> tuple[str, str]:
    token, chat_id = NOTIFY.telegram_bot_token, NOTIFY.telegram_chat_id
    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment (or .env file)")
    return token, chat_id


def format_report(
    registry: RepoRegistry,
    run_duration_seconds: float,
    budget_used: float,
    budget_limit: float,
    stage_results: dict[int, dict],
    rate_limit_info=None,
) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    hours = int(run_duration_seconds // 3600)
    minutes = int((run_duration_seconds % 3600) // 60)

    lines = [
        f"📊 Pipeline Overnight Report ({date_str})",
        "",
        f"Run duration: {hours}h {minutes:02d}m",
        f"Budget used: ${budget_used:.2f} / ${budget_limit:.2f}",
        "",
    ]

    if rate_limit_info is not None:
        util = getattr(rate_limit_info, "utilization", None)
        util_str = f"{util * 100:.0f}%" if util is not None else "unknown"
        rl_type = getattr(rate_limit_info, "rate_limit_type", None) or "?"
        status = getattr(rate_limit_info, "status", None) or "?"
        lines.append(f"Claude usage: {util_str} of limit ({rl_type}, status={status})")
        resets_at = getattr(rate_limit_info, "resets_at", None)
        if resets_at:
            resets_str = datetime.fromtimestamp(resets_at).strftime("%Y-%m-%d %H:%M")
            lines.append(f"Usage resets at: {resets_str}")
        lines.append("")

    stage_names = {
        3: "Score",
        4: "Analyze",
        5: "Trial",
        6: "Generate",
    }

    for stage_num in sorted(stage_results.keys()):
        result = stage_results[stage_num]
        name = stage_names.get(stage_num, f"Stage {stage_num}")
        processed = result.get("processed", 0)
        passed = result.get("passed", 0)
        rejected = result.get("rejected", 0)
        extra = ""
        if stage_num == 6:
            trajectories = result.get("trajectories", 0)
            extra = f", {trajectories} new trajectories"
        lines.append(f"Stage {stage_num} ({name}): {processed} processed, {passed} passed, {rejected} rejected{extra}")
        # A crashed stage starves everything downstream, so the morning report has to lead with it.
        if result.get("error"):
            lines.append(f"  🚨 STAGE CRASHED: {result['error'][:200]}")

    lines.append("")

    stats = registry.get_stats()
    lines.append("Registry totals:")
    status_display_order = [
        ("discovered", "Discovered"),
        ("filtered", "Filtered"),
        ("scored", "Scored"),
        ("analyzed", "Analyzed"),
        ("validated", "Validated"),
        ("completed", "Completed"),
    ]
    for status_key, display_name in status_display_order:
        count = stats.get(status_key, 0)
        extra = ""
        if status_key == "completed":
            total_traj = registry.count_total_trajectories()
            extra = f" ({total_traj} trajectories total)"
        lines.append(f"  {display_name}: {count:,}{extra}")

    rejected_total = sum(v for k, v in stats.items() if k.startswith("rejected_"))
    lines.append(f"  Rejected: {rejected_total:,}")
    lines.append("")

    pending = registry.get_pending_counts()
    lines.append("Pending for next run:")
    for stage_num in [3, 4, 5, 6]:
        count = pending.get(stage_num, 0)
        lines.append(f"  Stage {stage_num}: {count:,} repos")
    lines.append("")

    rejection_reasons = registry.get_rejection_reasons(limit=5)
    if rejection_reasons:
        lines.append("Top rejection reasons:")
        for i, (reason, count) in enumerate(rejection_reasons, 1):
            lines.append(f'  {i}. "{reason}" ({count:,})')

    return "\n".join(lines)


async def send_telegram_message(token: str, chat_id: str, message: str):
    bot = Bot(token=token)
    # Telegram max message length is 4096
    if len(message) > 4000:
        message = message[:4000] + "\n... (truncated)"
    await bot.send_message(chat_id=chat_id, text=message)
    logger.info("Telegram report sent successfully")


async def send_report(
    registry: RepoRegistry,
    run_duration_seconds: float,
    budget_used: float,
    budget_limit: float,
    stage_results: dict[int, dict],
    rate_limit_info=None,
):
    try:
        token, chat_id = get_bot_config()
    except ValueError as e:
        logger.warning(f"Telegram not configured: {e}")
        return

    report = format_report(
        registry,
        run_duration_seconds,
        budget_used,
        budget_limit,
        stage_results,
        rate_limit_info=rate_limit_info,
    )
    logger.info(f"Report:\n{report}")

    try:
        await send_telegram_message(token, chat_id, report)
    except Exception as e:
        logger.error(f"Failed to send Telegram report: {e}")


async def test_notifier():
    token, chat_id = get_bot_config()
    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text="🧪 Pipeline notifier test — this message confirms the bot works!")
    print("Test message sent!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if "--test" in sys.argv:
        asyncio.run(test_notifier())
