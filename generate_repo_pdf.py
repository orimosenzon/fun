#!/usr/bin/env python3
"""
Generate a beautiful PDF overview of all projects in the fun/ repository.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

OUTPUT_PATH = "/home/ori/fun/repository_overview.pdf"

# ── Colors ────────────────────────────────────────────────────────────────────
C_BG        = colors.HexColor("#F8F9FA")
C_NAVY      = colors.HexColor("#1A2B4A")
C_BLUE      = colors.HexColor("#2563EB")
C_CYAN      = colors.HexColor("#0EA5E9")
C_PURPLE    = colors.HexColor("#7C3AED")
C_GREEN     = colors.HexColor("#059669")
C_ORANGE    = colors.HexColor("#D97706")
C_RED       = colors.HexColor("#DC2626")
C_PINK      = colors.HexColor("#DB2777")
C_GRAY      = colors.HexColor("#6B7280")
C_LGRAY     = colors.HexColor("#E5E7EB")
C_WHITE     = colors.white
C_DARK      = colors.HexColor("#111827")

# ── Category colors ───────────────────────────────────────────────────────────
CAT_COLORS = {
    "Web Apps":         C_BLUE,
    "Machine Learning": C_PURPLE,
    "Reinforcement Learning": C_GREEN,
    "AI Tools":         C_CYAN,
    "Math & Science":   C_ORANGE,
    "Games":            C_PINK,
    "Classic Projects": C_GRAY,
}

# ── Custom Flowables ──────────────────────────────────────────────────────────

class ColoredRect(Flowable):
    """A filled rectangle used as a decorative element."""
    def __init__(self, width, height, color, radius=4):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
        self.radius = radius

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height,
                            self.radius, fill=1, stroke=0)


class CategoryBadge(Flowable):
    """A pill-shaped badge with category label."""
    def __init__(self, text, color, font_size=8):
        Flowable.__init__(self)
        self.text = text
        self.color = color
        self.font_size = font_size
        self.pad_x = 8
        self.pad_y = 3
        self.width = len(text) * font_size * 0.6 + self.pad_x * 2
        self.height = font_size + self.pad_y * 2

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.roundRect(0, 0, self.width, self.height, self.height / 2, fill=1, stroke=0)
        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", self.font_size)
        c.drawCentredString(self.width / 2, self.pad_y + 1, self.text)


class AsciiDiagram(Flowable):
    """Renders a preformatted ASCII diagram box."""
    def __init__(self, lines, width, box_color, text_color=None):
        Flowable.__init__(self)
        self.lines = lines
        self.width = width
        self.box_color = box_color
        self.text_color = text_color or C_WHITE
        self.font_size = 7.5
        self.line_h = self.font_size * 1.35
        self.pad = 10
        self.height = len(lines) * self.line_h + self.pad * 2

    def draw(self):
        c = self.canv
        c.setFillColor(self.box_color)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        c.setFillColor(self.text_color)
        c.setFont("Courier", self.font_size)
        y = self.height - self.pad - self.font_size
        for line in self.lines:
            c.drawString(self.pad, y, line)
            y -= self.line_h


class TechStack(Flowable):
    """Horizontal pills for tech stack."""
    def __init__(self, techs, pill_color):
        Flowable.__init__(self)
        self.techs = techs
        self.pill_color = pill_color
        self.font_size = 7
        self.pad_x = 7
        self.pad_y = 3
        self.gap = 5
        pill_w = [len(t) * self.font_size * 0.58 + self.pad_x * 2 for t in techs]
        self.width = sum(pill_w) + self.gap * (len(techs) - 1)
        self.height = self.font_size + self.pad_y * 2
        self.pill_widths = pill_w

    def draw(self):
        c = self.canv
        x = 0
        for tech, pw in zip(self.techs, self.pill_widths):
            c.setFillColor(self.pill_color)
            c.roundRect(x, 0, pw, self.height, self.height / 2, fill=1, stroke=0)
            c.setFillColor(C_WHITE)
            c.setFont("Helvetica-Bold", self.font_size)
            c.drawCentredString(x + pw / 2, self.pad_y + 0.5, tech)
            x += pw + self.gap


# ── Styles ────────────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        fontName="Helvetica-Bold",
        fontSize=28,
        textColor=C_WHITE,
        spaceAfter=4,
        leading=34,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica",
        fontSize=13,
        textColor=colors.HexColor("#CBD5E1"),
        spaceAfter=0,
        leading=18,
    )
    section_style = ParagraphStyle(
        "Section",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=C_NAVY,
        spaceBefore=6,
        spaceAfter=8,
        leading=18,
    )
    project_title_style = ParagraphStyle(
        "ProjectTitle",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=C_DARK,
        spaceBefore=2,
        spaceAfter=4,
        leading=15,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#374151"),
        spaceAfter=4,
        leading=13,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#374151"),
        leftIndent=12,
        spaceBefore=1,
        spaceAfter=1,
        leading=12,
        bulletIndent=0,
    )
    mono_style = ParagraphStyle(
        "Mono",
        fontName="Courier",
        fontSize=8,
        textColor=colors.HexColor("#6B7280"),
        spaceBefore=2,
        spaceAfter=2,
        leading=11,
    )
    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "section": section_style,
        "project_title": project_title_style,
        "body": body_style,
        "bullet": bullet_style,
        "mono": mono_style,
    }


# ── Cover Page ────────────────────────────────────────────────────────────────

class CoverPage(Flowable):
    def __init__(self, page_width, page_height):
        Flowable.__init__(self)
        self.page_width = page_width
        self.page_height = page_height
        self.width = page_width
        self.height = page_height

    def draw(self):
        c = self.canv
        w, h = self.page_width, self.page_height

        # Background gradient effect (layered rects)
        c.setFillColor(C_NAVY)
        c.rect(0, 0, w, h, fill=1, stroke=0)

        # Decorative circles
        c.setFillColor(colors.HexColor("#2563EB30"))
        c.circle(w * 0.85, h * 0.75, 120, fill=1, stroke=0)
        c.circle(w * 0.1, h * 0.2, 80, fill=1, stroke=0)

        c.setFillColor(colors.HexColor("#7C3AED20"))
        c.circle(w * 0.7, h * 0.3, 90, fill=1, stroke=0)

        # Top accent bar
        c.setFillColor(C_BLUE)
        c.rect(0, h - 6, w, 6, fill=1, stroke=0)

        # Main title
        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 36)
        c.drawCentredString(w / 2, h * 0.62, "ori's fun/ repository")

        # Subtitle
        c.setFillColor(colors.HexColor("#94A3B8"))
        c.setFont("Helvetica", 15)
        c.drawCentredString(w / 2, h * 0.55, "A Tour of Projects, Experiments & Ideas")

        # Divider line
        c.setStrokeColor(colors.HexColor("#334155"))
        c.setLineWidth(1)
        c.line(w * 0.2, h * 0.50, w * 0.8, h * 0.50)

        # Stats row
        stats = [
            ("20+", "Projects"),
            ("8", "Categories"),
            ("Python · JS · TS", "Languages"),
            ("2016 – 2026", "Timespan"),
        ]
        col_w = w / len(stats)
        for i, (num, label) in enumerate(stats):
            cx = col_w * i + col_w / 2
            c.setFillColor(C_CYAN)
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(cx, h * 0.44, num)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.setFont("Helvetica", 9)
            c.drawCentredString(cx, h * 0.41, label)

        # Footer
        c.setFillColor(colors.HexColor("#475569"))
        c.setFont("Helvetica", 8)
        c.drawCentredString(w / 2, 30, "Generated March 2026  •  github.com/orimosenzon/fun")


# ── Project card builder ──────────────────────────────────────────────────────

def project_card(styles, name, path, category, description, bullets,
                 techs, diagram_lines=None, cat_color=None):
    """Build a list of Flowables for one project card."""
    if cat_color is None:
        cat_color = CAT_COLORS.get(category, C_GRAY)

    elements = []

    # Card header: colored left border + title
    header_data = [[
        ColoredRect(4, 36, cat_color),
        Paragraph(f"<b>{name}</b>", styles["project_title"]),
        Paragraph(f'<font color="#9CA3AF" size="7">{path}</font>', styles["body"]),
    ]]
    header_table = Table(header_data, colWidths=[8, 280, 140])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("ROUNDEDCORNERS", [4, 4, 0, 0]),
    ]))
    elements.append(header_table)

    # Body
    body_elements = [Paragraph(description, styles["body"])]
    for b in bullets:
        body_elements.append(Paragraph(f"• {b}", styles["bullet"]))

    body_elements.append(Spacer(1, 6))
    body_elements.append(TechStack(techs, cat_color))

    if diagram_lines:
        body_elements.append(Spacer(1, 8))
        diag_color = colors.HexColor("#1E293B")
        body_elements.append(AsciiDiagram(diagram_lines, 420, diag_color))

    body_data = [[col] for col in body_elements]
    body_table = Table([[e] for e in body_elements], colWidths=[432])
    body_table.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), C_WHITE),
    ]))
    elements.append(body_table)

    # Bottom border
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=C_LGRAY, spaceAfter=14))
    return KeepTogether(elements)


# ── Section header builder ────────────────────────────────────────────────────

def section_header(styles, title, emoji, color, description=""):
    elements = []
    elements.append(Spacer(1, 10))

    # Background band
    band_data = [[
        Paragraph(f"{emoji}  {title}", ParagraphStyle(
            "SH", fontName="Helvetica-Bold", fontSize=13,
            textColor=C_WHITE, leading=16
        )),
    ]]
    band = Table(band_data, colWidths=[432])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    elements.append(band)
    if description:
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(description, ParagraphStyle(
            "SHDesc", fontName="Helvetica", fontSize=8.5,
            textColor=C_GRAY, leftIndent=4, leading=12
        )))
    elements.append(Spacer(1, 10))
    return elements


# ── TOC page ─────────────────────────────────────────────────────────────────

def toc_page(styles):
    elements = []

    elements.append(Paragraph("Table of Contents", ParagraphStyle(
        "TOCTitle", fontName="Helvetica-Bold", fontSize=20,
        textColor=C_NAVY, spaceAfter=16, leading=24
    )))
    elements.append(HRFlowable(width="100%", thickness=2,
                               color=C_BLUE, spaceAfter=16))

    toc_items = [
        ("1", "Web Applications",         "Full-stack apps & SaaS products",       C_BLUE),
        ("2", "Machine Learning & AI",     "Deep learning, LLMs, fine-tuning",      C_PURPLE),
        ("3", "Reinforcement Learning",    "Agents, environments, DQN",             C_GREEN),
        ("4", "AI Tools & Utilities",      "Voice, transcription, API wrappers",    C_CYAN),
        ("5", "Math & Science Notebooks",  "Calculus, physics simulations",         C_ORANGE),
        ("6", "Games & Interactive",       "Browser games, Scratch, Sudoku",        C_PINK),
        ("7", "Classic & Legacy",          "Old but gold Python/JS experiments",    C_GRAY),
    ]

    for num, title, desc, color in toc_items:
        row_data = [[
            Paragraph(f"<b>{num}</b>", ParagraphStyle(
                "Num", fontName="Helvetica-Bold", fontSize=14,
                textColor=color, leading=18
            )),
            Paragraph(f"<b>{title}</b><br/>"
                      f'<font color="#6B7280" size="8">{desc}</font>',
                      ParagraphStyle("TRow", fontName="Helvetica",
                                     fontSize=11, leading=16)),
        ]]
        row_table = Table(row_data, colWidths=[28, 404])
        row_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, C_LGRAY),
        ]))
        elements.append(row_table)

    return elements


# ── Main content ─────────────────────────────────────────────────────────────

def draw_cover(canvas, doc):
    """Draw cover page directly on canvas."""
    w, h = A4
    canvas.saveState()

    # Background
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)

    # Decorative circles
    canvas.setFillColor(colors.HexColor("#2563EB30"))
    canvas.circle(w * 0.85, h * 0.75, 120, fill=1, stroke=0)
    canvas.circle(w * 0.1, h * 0.2, 80, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#7C3AED20"))
    canvas.circle(w * 0.7, h * 0.3, 90, fill=1, stroke=0)

    # Top accent bar
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, h - 6, w, 6, fill=1, stroke=0)

    # Main title
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 36)
    canvas.drawCentredString(w / 2, h * 0.62, "ori's fun/ repository")

    # Subtitle
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.setFont("Helvetica", 15)
    canvas.drawCentredString(w / 2, h * 0.55, "A Tour of Projects, Experiments & Ideas")

    # Divider
    canvas.setStrokeColor(colors.HexColor("#334155"))
    canvas.setLineWidth(1)
    canvas.line(w * 0.2, h * 0.50, w * 0.8, h * 0.50)

    # Stats row
    stats = [
        ("20+", "Projects"),
        ("7", "Categories"),
        ("Python  JS  TS", "Languages"),
        ("2016 - 2026", "Timespan"),
    ]
    col_w = w / len(stats)
    for i, (num, label) in enumerate(stats):
        cx = col_w * i + col_w / 2
        canvas.setFillColor(C_CYAN)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(cx, h * 0.44, num)
        canvas.setFillColor(colors.HexColor("#94A3B8"))
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(cx, h * 0.41, label)

    # Footer
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(w / 2, 30, "Generated March 2026  -  github.com/orimosenzon/fun")

    canvas.restoreState()


def build_content(styles):
    story = []

    # Page 1: cover drawn via on_first_page callback; PageBreak moves story to page 2
    story.append(PageBreak())


    # ── TOC ──
    story.extend(toc_page(styles))

    # ════════════════════════════════════════════
    # 1. WEB APPLICATIONS
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "Web Applications", "🌐", C_BLUE,
        "Full-stack web apps built for real users — from ceramics studios to hotel management."
    ))

    story.append(project_card(
        styles,
        name="Inbal — Ceramics Studio Manager",
        path="vibe_coding/inbal/",
        category="Web Apps",
        description=(
            "A full-featured SaaS application for managing a ceramics studio. "
            "Students can view the weekly schedule, register for wheel/hand-building slots, "
            "manage a standby list, and handle transfers. Admins control the entire schedule."
        ),
        bullets=[
            "Student self-registration & standby queue",
            "Admin weekly/day calendar with drag-and-drop slot management",
            "Automatic standby notifications (WhatsApp planned)",
            "Authentication via Google / Facebook / Email",
            "Deployed on Vercel with Supabase PostgreSQL (Mumbai region)",
        ],
        techs=["Next.js 14", "TypeScript", "Prisma", "Supabase", "Vercel", "Tailwind"],
        diagram_lines=[
            "  Student                Admin               Database",
            "    │                     │                      │",
            "    ├── View Schedule ────►│                      │",
            "    ├── Register Slot ────►├── Validate ─────────►│",
            "    │   (or join standby)  │◄── Confirm ──────────┤",
            "    │◄── Confirmation ─────┤                      │",
            "    │                     ├── Send Notification ──►│",
        ],
        cat_color=C_BLUE,
    ))

    story.append(project_card(
        styles,
        name="Road to Recovery — Ride Coordination",
        path="vibe_coding/road-to-recovery/",
        category="Web Apps",
        description=(
            "A ride coordination system for a non-profit rehabilitation association. "
            "Volunteers offer rides, patients request trips, and coordinators match them. "
            "Built with a focus on simplicity and accessibility."
        ),
        bullets=[
            "Patient ride requests with pickup / destination",
            "Volunteer availability and ride offer management",
            "Coordinator dashboard for matching and oversight",
            "Mobile-first responsive design",
        ],
        techs=["Next.js 14", "TypeScript", "Prisma", "PostgreSQL", "Tailwind"],
        cat_color=C_BLUE,
    ))

    story.append(project_card(
        styles,
        name="Uri's Hotel Management",
        path="uri/hotel/",
        category="Web Apps",
        description=(
            "A hotel management web application with rooms, guests, and reservations. "
            "Built as a monorepo with a React frontend and REST API backend."
        ),
        bullets=[
            "Room status tracking (available, occupied, maintenance)",
            "Guest profiles with search",
            "Reservation management dashboard",
            "Monorepo structure with shared packages",
        ],
        techs=["React", "TypeScript", "Vite", "TanStack Query", "Tailwind"],
        cat_color=C_BLUE,
    ))

    story.append(project_card(
        styles,
        name="Oti's ChatGPT Web App",
        path="oti/chatgpt_api_example/",
        category="Web Apps",
        description=(
            "A Flask web application that wraps the OpenAI API with a simple chat interface. "
            "Built on OpenAI's quickstart example, extended with session history and personality prompts."
        ),
        bullets=[
            "Stateful conversation via Flask sessions",
            "Personality and summarization prompt helpers",
            "Clean HTML/CSS frontend",
        ],
        techs=["Python", "Flask", "OpenAI API"],
        cat_color=C_BLUE,
    ))

    story.append(project_card(
        styles,
        name="Pipin — Middle Earth Adventure Game",
        path="vibe_coding/pipin/",
        category="Web Apps",
        description=(
            "A browser-based text adventure set in Tolkien's Middle Earth. "
            "The player controls Pippin (Peregrin Took) in a world faithful to the lore. "
            "A dynamic narrative engine powered by Gemini generates story responses. "
            "Every location has a unique AI-generated illustration."
        ),
        bullets=[
            "~30 hand-placed locations with individual artwork",
            "Dynamic narrative via Gemini 1.5 Pro — story diverges from the books",
            "Hebrew RTL interface with a modern, bright design",
            "AI image generation for locations and NPC portraits",
            "World graph: locations, connections, items, characters",
        ],
        techs=["HTML/CSS/JS", "Gemini API", "Nano Banana", "RTL"],
        diagram_lines=[
            "  Player Action",
            "       │",
            "       ▼",
            "  game.js ──► world.js (locations / NPCs / items)",
            "       │",
            "       ▼",
            "  prompts.js ──► api.js ──► Gemini 1.5 Pro",
            "                               │",
            "       ◄───────── story response + choices ────────",
        ],
        cat_color=C_PINK,
    ))

    # ════════════════════════════════════════════
    # 2. MACHINE LEARNING & AI
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "Machine Learning & AI", "🧠", C_PURPLE,
        "Deep learning models, transformers, GANs, fine-tuning, and LLM-powered tools."
    ))

    story.append(project_card(
        styles,
        name="Transformer from Scratch",
        path="ml/transformer/",
        category="Machine Learning",
        description=(
            "A clean, educational implementation of the original Transformer architecture "
            "('Attention Is All You Need', Vaswani et al. 2017) in PyTorch. "
            "Built alongside Aladdin Persson's YouTube tutorial series."
        ),
        bullets=[
            "Multi-head self-attention with scaled dot-product",
            "Encoder / Decoder blocks with residual connections",
            "Positional encoding, feed-forward layers",
            "Mask support for decoder (causal masking)",
        ],
        techs=["Python", "PyTorch", "Transformer"],
        diagram_lines=[
            "  Input Tokens",
            "       │  Positional Encoding",
            "       ▼",
            "  ┌─────────────────────┐",
            "  │  Encoder Block × N  │  ← Multi-Head Attention + FFN",
            "  └────────┬────────────┘",
            "            │",
            "  ┌─────────▼────────────┐",
            "  │  Decoder Block × N  │  ← Masked Attention + Cross-Attention + FFN",
            "  └────────┬────────────┘",
            "            ▼",
            "       Output Tokens",
        ],
        cat_color=C_PURPLE,
    ))

    story.append(project_card(
        styles,
        name="Llama 2 Fine-tuning",
        path="ofer/myenv/uri_lamma2_finetune.py",
        category="Machine Learning",
        description=(
            "Fine-tuning Meta's Llama 2 language model using QLoRA (quantized LoRA), "
            "PEFT, and the TRL library. Includes a Gradio interface for streaming inference. "
            "Built with Ofer for a specific instruction-following task."
        ),
        bullets=[
            "4-bit quantization via BitsAndBytes (QLoRA)",
            "LoRA adapters via PEFT — efficient parameter tuning",
            "SFTTrainer from TRL for supervised fine-tuning",
            "W&B integration for experiment tracking",
            "Streaming output via Gradio UI",
        ],
        techs=["Python", "PyTorch", "Llama 2", "PEFT", "QLoRA", "Gradio", "W&B"],
        cat_color=C_PURPLE,
    ))

    story.append(project_card(
        styles,
        name="GAN — Generative Adversarial Network",
        path="with_uri/gan_uri_ori.ipynb",
        category="Machine Learning",
        description=(
            "A GAN implementation on the Fashion MNIST dataset, built collaboratively "
            "with Uri. Uses TensorFlow 2 with GPU acceleration and TensorBoard for monitoring "
            "training dynamics (loss curves, generated samples)."
        ),
        bullets=[
            "Generator and Discriminator networks in TF2/Keras",
            "Trained on Fashion MNIST (28×28 grayscale clothing images)",
            "TensorBoard integration for real-time loss visualization",
            "GPU training with tf.device('/device:GPU:0')",
        ],
        techs=["Python", "TensorFlow 2", "Keras", "TensorBoard", "Fashion MNIST"],
        diagram_lines=[
            "  Random Noise ──► Generator ──► Fake Image",
            "                                     │",
            "  Real Image  ──────────────────► Discriminator ──► Real / Fake?",
            "                                     │",
            "  ◄──── Generator Loss ───────────────┘",
            "  ◄──── Discriminator Loss ─────────────────────────────",
        ],
        cat_color=C_PURPLE,
    ))

    story.append(project_card(
        styles,
        name="Cat vs Dog Classifier",
        path="ml/ron_course/",
        category="Machine Learning",
        description=(
            "A binary image classifier (cat vs. dog) built during Ron's deep learning course. "
            "Includes dataset preparation scripts that resize images to 200×200 "
            "and organize them into balanced train sets."
        ),
        bullets=[
            "Dataset preprocessing: resize to 200×200, balance classes",
            "Subset extraction: 100 cats + 100 dogs for fast iteration",
        ],
        techs=["Python", "PIL", "CNN", "Image Classification"],
        cat_color=C_PURPLE,
    ))

    story.append(project_card(
        styles,
        name="Avishai's Assignment Grader",
        path="ofer/myenv/Avishai_first_assignment.ipynb",
        category="Machine Learning",
        description=(
            "An AI-powered homework grading system for Avishai's English classes in Israel. "
            "Reads student submissions from Google Sheets (image URLs), processes them "
            "with GPT-4 and Gemini, and writes results to a Google Doc."
        ),
        bullets=[
            "Reads student work URLs from Google Sheets API",
            "Multi-model evaluation: GPT-4o and Gemini 1.5 Pro/Flash",
            "Outputs graded feedback to Google Docs API",
            "Handles image-based submissions (PIL + BytesIO)",
        ],
        techs=["Python", "GPT-4", "Gemini 1.5", "Google Sheets API", "Google Docs API"],
        cat_color=C_PURPLE,
    ))

    story.append(project_card(
        styles,
        name="MNIST Classifiers",
        path="tensorflow/",
        category="Machine Learning",
        description=(
            "Two implementations of the classic handwritten digit classifier on MNIST: "
            "a simple softmax model and a convolutional network (CNN) — "
            "both written in TensorFlow 1.x as learning exercises."
        ),
        bullets=[
            "mnist.py — softmax regression baseline",
            "mnist_convol.py — 2-layer CNN with max pooling",
            "Weight/bias initialization helpers",
            "Interactive TensorFlow session (TF1 style)",
        ],
        techs=["Python", "TensorFlow 1", "CNN", "MNIST"],
        cat_color=C_PURPLE,
    ))

    # ════════════════════════════════════════════
    # 3. REINFORCEMENT LEARNING
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "Reinforcement Learning", "🎮", C_GREEN,
        "Agents learning to play games through interaction — from 2048 to CartPole."
    ))

    story.append(project_card(
        styles,
        name="2048 — RL Agent Suite",
        path="rl/2048/gym2048o/",
        category="Reinforcement Learning",
        description=(
            "A full reinforcement learning environment for the game 2048, "
            "packaged as an OpenAI Gym-compatible environment. "
            "Multiple agents were developed and compared."
        ),
        bullets=[
            "Custom Gym environment with proper reset/step/render API",
            "Random Agent — uniform random baseline",
            "Cross Entropy Agent — policy gradient with experience replay",
            "DQN Agent — Deep Q-Network with epsilon-greedy exploration",
            "Search Agent — tree search heuristic",
            "Tkinter GUI for human play, keyboard UI for testing",
            "TensorBoard integration for training metrics",
        ],
        techs=["Python", "PyTorch", "OpenAI Gym", "DQN", "Cross Entropy", "Tkinter"],
        diagram_lines=[
            "  Agent                 Environment (2048 Grid)",
            "    │                          │",
            "    ├── action (0-3: U/D/L/R) ─►│",
            "    │◄── state, reward, done ───┤",
            "    │                          │",
            "  Neural Net                  │",
            "  Q(s,a) ──► argmax ──────────┘",
            "    │",
            "  Experience Replay Buffer",
        ],
        cat_color=C_GREEN,
    ))

    story.append(project_card(
        styles,
        name="CartPole — Cross Entropy Agent",
        path="rl/2048/gym2048o/cart_pole_cea.py",
        category="Reinforcement Learning",
        description=(
            "A cross entropy method agent applied to OpenAI Gym's CartPole environment — "
            "a simpler testbed used to validate the agent before applying it to 2048."
        ),
        bullets=[
            "Policy network trained on elite episodes",
            "Validates cross entropy method on a well-understood benchmark",
        ],
        techs=["Python", "PyTorch", "OpenAI Gym", "Cross Entropy Method"],
        cat_color=C_GREEN,
    ))

    # ════════════════════════════════════════════
    # 4. AI TOOLS & UTILITIES
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "AI Tools & Utilities", "🛠", C_CYAN,
        "Practical tools that integrate AI APIs into everyday workflows."
    ))

    story.append(project_card(
        styles,
        name="Voice-to-Claude",
        path="vibe_coding/voice_to_claude.py",
        category="AI Tools",
        description=(
            "A hands-free voice input system for Claude Code. "
            "Listens continuously using a microphone, detects speech automatically "
            "with VAD (Voice Activity Detection), transcribes with Whisper, "
            "and pastes directly into the terminal."
        ),
        bullets=[
            "Continuous listening — no button press needed",
            "Voice Activity Detection: starts/stops on speech automatically",
            "Whisper transcription (local, small model)",
            "Hebrew language support (language='he')",
            "Auto-pastes into active terminal via xdotool",
        ],
        techs=["Python", "Whisper", "VAD", "pyaudio", "Hebrew"],
        diagram_lines=[
            "  Microphone",
            "      │",
            "      ▼",
            "  VAD (Voice Activity Detection)",
            "      │  speech detected",
            "      ▼",
            "  Record audio buffer",
            "      │",
            "      ▼",
            "  Whisper (local) ──► Hebrew text",
            "      │",
            "      ▼",
            "  xdotool type ──► paste into Claude Code terminal",
        ],
        cat_color=C_CYAN,
    ))

    story.append(project_card(
        styles,
        name="WhatsApp Voice Transcriber",
        path="vibe_coding/openclaw/whatsapp_trans/",
        category="AI Tools",
        description=(
            "A transcription tool for WhatsApp voice messages, integrated as an OpenClaw "
            "provider. Receives OGG/Opus audio files, converts to WAV using ffmpeg, "
            "and transcribes with Google Cloud Speech-to-Text."
        ),
        bullets=[
            "OGG/Opus → 16kHz mono WAV conversion via ffmpeg",
            "Google Cloud STT for accurate Hebrew/multilingual transcription",
            "OpenClaw CLI provider interface",
            "Voice agent wrapper for full pipeline automation",
        ],
        techs=["Python", "Google Cloud STT", "ffmpeg", "OpenClaw", "WhatsApp"],
        cat_color=C_CYAN,
    ))

    story.append(project_card(
        styles,
        name="Wikipedia Information Extractor",
        path="ie/",
        category="AI Tools",
        description=(
            "A set of scripts for extracting structured information from Wikipedia "
            "via the MediaWiki API. Progressively more complex queries: "
            "categories, revisions, page content extraction."
        ),
        bullets=[
            "query1–4.py: incrementally complex Wikipedia API queries",
            "Category listing with prefix filtering",
            "Page revision and content retrieval",
        ],
        techs=["Python", "Wikipedia API", "urllib"],
        cat_color=C_CYAN,
    ))

    # ════════════════════════════════════════════
    # 5. MATH & SCIENCE
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "Math & Science Notebooks", "📐", C_ORANGE,
        "Interactive explorations of mathematics, physics, and algorithms."
    ))

    story.append(project_card(
        styles,
        name="Infinitesimal Calculus 2 — Technion",
        path="colabs/infi2.ipynb",
        category="Math & Science",
        description=(
            "Interactive Jupyter notebook following the Technion's Calculus 2M course "
            "by Dr. Aviv Tsenzor. Covers topics from sequences and series to multi-variable "
            "calculus, with interactive visualizations."
        ),
        bullets=[
            "Sequences, limits, and series convergence",
            "Multi-variable functions and partial derivatives",
            "Interactive Plotly/Matplotlib visualizations",
            "Hebrew + English mixed notes",
        ],
        techs=["Python", "Jupyter", "NumPy", "Matplotlib", "Plotly"],
        cat_color=C_ORANGE,
    ))

    story.append(project_card(
        styles,
        name="Function Fitter",
        path="colabs/FunctionFitter.ipynb",
        category="Math & Science",
        description=(
            "A notebook for fitting mathematical functions to data points. "
            "Explores curve fitting methods and visualization of residuals."
        ),
        bullets=[
            "Polynomial and non-linear curve fitting",
            "Residual analysis and goodness-of-fit metrics",
        ],
        techs=["Python", "Jupyter", "SciPy", "NumPy", "Matplotlib"],
        cat_color=C_ORANGE,
    ))

    story.append(project_card(
        styles,
        name="ESLA — Physics & Math Simulations",
        path="esla/",
        category="Math & Science",
        description=(
            "A collection of Python simulations from an early period — "
            "physical simulations, 3D visualizations, and mathematical tools. "
            "Originally managed in SVN (pre-Git era)."
        ),
        bullets=[
            "3dFunction.py — 3D surface plotter",
            "vec_comb.py — vector combination visualizer (VPython)",
            "rectInt.py — rectangular integration approximation",
            "gEliminate.py — Gaussian elimination solver",
            "eretz_ir.py — city/country data processor",
            "primes/ — Sieve of Eratosthenes, Ulam spiral, double primes",
            "surf/ — surface rendering experiments",
        ],
        techs=["Python", "VPython", "Matplotlib", "NumPy", "tkinter"],
        cat_color=C_ORANGE,
    ))

    # ════════════════════════════════════════════
    # 6. GAMES & INTERACTIVE
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "Games & Interactive", "🕹", C_PINK,
        "Games, puzzles, and interactive visual projects."
    ))

    story.append(project_card(
        styles,
        name="Sudoku Solver",
        path="soduku/soduku.py",
        category="Games",
        description=(
            "A Sudoku puzzle solver and interactive board built with Python's tkinter. "
            "Supports both manual puzzle entry and automated solving."
        ),
        bullets=[
            "Interactive tkinter canvas UI",
            "Constraint-based solver algorithm",
            "Edit mode for entering custom puzzles",
        ],
        techs=["Python", "tkinter"],
        cat_color=C_PINK,
    ))

    story.append(project_card(
        styles,
        name="Scratch Projects",
        path="scratch/",
        category="Games",
        description=(
            "A collection of visual programming projects from Scratch — "
            "an early exploration of algorithms and interactive graphics."
        ),
        bullets=[
            "Koch Snowflake — recursive fractal",
            "Sierpinski Triangle — recursive fractal",
            "Tower of Hanoi — animated auto-solution",
            "Find Path in Maze — pathfinding visualization",
            "Breakout clone",
            "Big World — scrolling and zooming camera",
            "Bat Shooter, Bouncing Ball, Obstacle Race",
        ],
        techs=["Scratch", "Algorithms", "Fractals", "Pathfinding"],
        cat_color=C_PINK,
    ))

    story.append(project_card(
        styles,
        name="Breakout (Python)",
        path="misc/breakout.py",
        category="Games",
        description=(
            "A Python implementation of the classic Breakout arcade game, "
            "demonstrating event-driven programming and basic game loop concepts."
        ),
        bullets=[
            "Paddle, ball, and brick collision detection",
            "Score tracking and game-over conditions",
        ],
        techs=["Python", "tkinter / pygame"],
        cat_color=C_PINK,
    ))

    story.append(project_card(
        styles,
        name="Empires — History Timeline",
        path="vibe_coding/ori/ and animate/",
        category="Games",
        description=(
            "An interactive browser-based visualization of world empires throughout history. "
            "Shows the rise and fall of civilizations on a timeline with animated transitions. "
            "Available in Hebrew, English, and Spanish."
        ),
        bullets=[
            "SVG/Canvas-based timeline animation",
            "Multilingual: Hebrew (RTL), English, Spanish",
            "Mobile-responsive layout",
            "Search and filter by civilization",
        ],
        techs=["JavaScript", "HTML/CSS", "SVG/Canvas", "RTL"],
        cat_color=C_PINK,
    ))

    # ════════════════════════════════════════════
    # 7. CLASSIC & LEGACY
    # ════════════════════════════════════════════
    story.append(PageBreak())
    story.extend(section_header(
        styles, "Classic & Legacy", "📦", C_GRAY,
        "Older experiments and learning projects that show the journey."
    ))

    story.append(project_card(
        styles,
        name="CheckProj — Django Starter",
        path="checkProj/",
        category="Classic Projects",
        description=(
            "A minimal Django project used to explore Python web development. "
            "Includes models, views, URL routing, and template rendering."
        ),
        bullets=[
            "Django MVC structure (models, views, templates)",
            "Basic URL routing",
        ],
        techs=["Python", "Django"],
        cat_color=C_GRAY,
    ))

    story.append(project_card(
        styles,
        name="Python Features Notebook",
        path="python_features/python_features.ipynb",
        category="Classic Projects",
        description=(
            "A reference notebook exploring Python language features — "
            "decorators, generators, comprehensions, and more."
        ),
        bullets=[
            "Language feature demonstrations with examples",
            "Used as a personal reference / cheatsheet",
        ],
        techs=["Python", "Jupyter"],
        cat_color=C_GRAY,
    ))

    story.append(project_card(
        styles,
        name="Misc Utilities",
        path="misc/",
        category="Classic Projects",
        description=(
            "A grab-bag of standalone Python scripts exploring algorithms and data structures."
        ),
        bullets=[
            "priorotyQueue.py — custom priority queue implementation",
            "gcd.py — greatest common divisor variants",
            "shablulPrimes.py — prime number experiments",
            "writeItself.py — a quine (program that prints its own source)",
        ],
        techs=["Python", "Algorithms", "Data Structures"],
        cat_color=C_GRAY,
    ))

    story.append(project_card(
        styles,
        name="C++ / OpenGL",
        path="cpp/opengl/",
        category="Classic Projects",
        description=(
            "Early experiments with C++ and OpenGL for 3D graphics rendering."
        ),
        bullets=[
            "OpenGL rendering pipeline",
            "3D geometry and transformations",
        ],
        techs=["C++", "OpenGL"],
        cat_color=C_GRAY,
    ))

    # ── Final page ──
    story.append(PageBreak())

    # Summary stats
    story.extend(section_header(styles, "Summary", "📊", C_NAVY))

    summary_data = [
        ["Category", "# Projects", "Primary Tech"],
        ["Web Applications", "5", "Next.js, React, TypeScript"],
        ["Machine Learning & AI", "6", "PyTorch, TensorFlow, LLMs"],
        ["Reinforcement Learning", "2", "PyTorch, OpenAI Gym"],
        ["AI Tools & Utilities", "3", "Python, Whisper, Cloud APIs"],
        ["Math & Science", "3", "Python, Jupyter, NumPy"],
        ["Games & Interactive", "4", "Python, JavaScript, Scratch"],
        ["Classic & Legacy", "4", "Python, Django, C++"],
    ]

    col_widths = [180, 70, 182]
    summary_table = Table(summary_data, colWidths=col_widths)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, colors.HexColor("#F8FAFC")]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), C_DARK),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, C_LGRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)

    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "This repository spans a decade of experimentation — from early Python simulations "
        "and Scratch games to production-grade Next.js applications and LLM fine-tuning. "
        "Each project reflects a different curiosity, collaboration, or real-world need.",
        ParagraphStyle("Closing", fontName="Helvetica", fontSize=9.5,
                       textColor=C_GRAY, leading=14, alignment=TA_CENTER)
    ))

    return story


# ── Page template (footer) ────────────────────────────────────────────────────

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_GRAY)
    canvas.drawCentredString(A4[0] / 2, 20, f"ori's fun/ repository  |  page {doc.page}")
    canvas.restoreState()


def on_first_page(canvas, doc):
    draw_cover(canvas, doc)


# ── Build PDF ─────────────────────────────────────────────────────────────────

def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="ori's fun/ repository overview",
        author="ori mosenzon",
    )

    styles = make_styles()
    story = build_content(styles)

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_page)
    print(f"✓ PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
