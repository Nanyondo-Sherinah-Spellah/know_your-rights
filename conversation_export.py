
import csv
import io
import json
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def export_conversation_json(conversation, messages):
    """Export a conversation and its messages as JSON."""
    data = {
        "conversation": conversation,
        "messages": messages,
        "exported_at": datetime.now().astimezone().isoformat(),
    }

    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def export_conversation_csv(messages):
    """Export conversation messages as CSV."""
    output = io.StringIO(newline="")

    fieldnames = [
        "id",
        "role",
        "content",
        "model",
        "sources",
        "created_at",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )
    writer.writeheader()

    for message in messages:
        row = dict(message)
        row["sources"] = json.dumps(
            row.get("sources", []),
            ensure_ascii=False,
            default=str,
        )
        writer.writerow(row)

    return output.getvalue()


def export_conversation_pdf(conversation, messages):
    """Export a readable PDF containing a conversation and its messages."""
    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=conversation.get("title", "Conversation"),
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ConversationTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#34495E"),
        spaceAfter=12,
    )

    body_style = ParagraphStyle(
        "MessageBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6,
        wordWrap="CJK",
    )

    story = [
        Paragraph(
            "Know Your Rights AI",
            title_style,
        ),
        Paragraph(
            conversation.get("title", "Conversation"),
            styles["Heading2"],
        ),
        Paragraph(
            f"Exported: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
            styles["Normal"],
        ),
        Spacer(1, 10),
    ]

    for message in messages:
        role = str(message.get("role", "unknown")).title()
        timestamp = str(message.get("created_at", ""))
        model = message.get("model")
        content = str(message.get("content", ""))

        story.append(
            Paragraph(
                f"<b>{role}</b> — {timestamp}",
                styles["Heading4"],
            )
        )

        if model:
            story.append(
                Paragraph(
                    f"<i>Model: {model}</i>",
                    styles["Normal"],
                )
            )

        # Escape user/model text before placing it in ReportLab paragraphs.
        from xml.sax.saxutils import escape

        for line in content.splitlines() or [""]:
            story.append(
                Paragraph(
                    escape(line) or " ",
                    body_style,
                )
            )

        sources = message.get("sources", [])
        if sources:
            story.append(Paragraph("Sources", styles["Heading5"]))

            for source in sources:
                if isinstance(source, dict):
                    label = (
                        source.get("article")
                        or source.get("title")
                        or source.get("document")
                        or "Reference"
                    )
                    detail = (
                        source.get("text")
                        or source.get("content")
                        or ""
                    )
                    text = f"{label}: {detail}" if detail else str(label)
                else:
                    text = str(source)

                story.append(
                    Paragraph(escape(text), body_style)
                )

        story.append(Spacer(1, 8))

    document.build(story)

    return output.getvalue()
