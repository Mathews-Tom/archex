"""archex explainer — animated overview (Manim Community v0.20).

Render:
    python3 render_video.py archex_explainer.py ArchexExplainer --quality high \
        --format mp4 --output assets/archex-explainer.mp4

Beats: title -> the problem (grep) -> archex pipeline -> measured bars -> stat tiles -> close.
"""

from __future__ import annotations

from manim import (
    BOLD,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    UR,
    Arrow,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Mobject,
    MoveToTarget,
    Rectangle,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    VGroup,
    Write,
)

# Palette — matches assets/archex-infographic.svg
BG = "#0b0f1e"
PANEL = "#131a30"
TEXT = "#eaf0fb"
MUTED = "#8a93a8"
TEAL = "#2dd4bf"  # archex
SKY = "#56b6ff"
SLATE = "#586079"  # competition
AMBER = "#f4a836"  # grep
GREEN = "#4ade80"
RED = "#f0626a"


class ArchexExplainer(Scene):
    def construct(self):
        self.camera.background_color = BG

        self._beat_title()
        self._beat_problem()
        self._beat_pipeline()
        self._beat_bars()
        self._beat_stats()
        self._beat_close()

    # ---------- helpers ----------
    def _heading(self, label: str) -> Mobject:
        h = Text(label, font_size=26, color=MUTED, weight=BOLD).to_corner(UL, buff=0.6)
        return h

    def _clear(self, *mobjects, run_time: float = 0.6):
        self.play(*[FadeOut(m) for m in mobjects], run_time=run_time)

    # ---------- beat 1: title ----------
    def _beat_title(self):
        brand = Text("archex", font_size=120, weight=BOLD, color=TEXT)
        dot = Text(".", font_size=120, weight=BOLD, color=TEAL)
        wordmark = VGroup(brand, dot).arrange(RIGHT, buff=0.02)
        subtitle = Text("Local code context for agents.", font_size=42, color=TEAL)
        subtitle.next_to(wordmark, DOWN, buff=0.45)
        group = VGroup(wordmark, subtitle).move_to(ORIGIN)

        self.play(FadeIn(wordmark, shift=UP * 0.3), run_time=1.1)
        self.play(Write(subtitle), run_time=0.9)
        self.wait(1.8)
        self.play(FadeOut(group, shift=UP * 0.3), run_time=0.7)

    # ---------- beat 2: the problem ----------
    def _beat_problem(self):
        heading = self._heading("THE PROBLEM")
        self.play(FadeIn(heading), run_time=0.5)

        # agent on the left
        agent_box = RoundedRectangle(
            corner_radius=0.15,
            width=2.6,
            height=1.4,
            stroke_color=SKY,
            stroke_width=2.5,
            fill_color=PANEL,
            fill_opacity=1,
        )
        agent_lbl = Text("AI agent", font_size=30, color=TEXT, weight=BOLD)
        agent = VGroup(agent_box, agent_lbl).move_to([-4.6, 1.4, 0])

        # repo files on the right
        files = (
            VGroup(
                *[
                    RoundedRectangle(
                        corner_radius=0.06,
                        width=1.9,
                        height=0.42,
                        stroke_color=MUTED,
                        stroke_width=1.5,
                        fill_color="#1b2440",
                        fill_opacity=1,
                    )
                    for _ in range(6)
                ]
            )
            .arrange(DOWN, buff=0.16)
            .move_to([4.6, 1.2, 0])
        )
        repo_lbl = Text("repository", font_size=24, color=MUTED).next_to(files, UP, buff=0.25)

        grep = Text("runs grep / glob", font_size=26, color=AMBER, weight=BOLD)
        grep.move_to([0, 2.1, 0])
        grep_arrow = Arrow(
            [-3.2, 1.4, 0],
            [3.4, 1.4, 0],
            color=AMBER,
            stroke_width=4,
            buff=0.2,
            max_tip_length_to_length_ratio=0.06,
        )

        self.play(
            FadeIn(agent),
            LaggedStart(*[FadeIn(f) for f in files], lag_ratio=0.1),
            FadeIn(repo_lbl),
            run_time=1.1,
        )
        self.play(Write(grep), GrowArrow(grep_arrow), run_time=0.9)

        # context window — whole files get dumped in
        ctx = RoundedRectangle(
            corner_radius=0.12,
            width=9.6,
            height=1.5,
            stroke_color=RED,
            stroke_width=2.5,
            fill_color="#1a0f12",
            fill_opacity=1,
        ).move_to([0, -1.4, 0])
        ctx_lbl = Text("context window", font_size=24, color=MUTED).next_to(ctx, UP, buff=0.22)
        self.play(FadeIn(ctx), FadeIn(ctx_lbl), run_time=0.6)

        # whole files flow in and pack the window full
        dumped = VGroup(*[f.copy() for f in files])
        dumped.generate_target()
        dumped.target.arrange(RIGHT, buff=0.14).scale_to_fit_width(9.0).move_to(ctx.get_center())
        self.play(MoveToTarget(dumped), run_time=1.4)
        self.play(Indicate(ctx, color=RED, scale_factor=1.03), run_time=0.6)

        # readout
        recall = Text("recall 1.00", font_size=30, color=GREEN, weight=BOLD).move_to(
            [-3.3, -3.05, 0]
        )
        toneff = Text("token-efficiency 0.00", font_size=30, color=RED, weight=BOLD).move_to(
            [2.4, -3.05, 0]
        )
        self.play(
            FadeIn(recall, shift=UP * 0.2),
            FadeIn(toneff, shift=UP * 0.2),
            run_time=0.8,
        )
        self.wait(2.2)

        self._clear(
            heading, agent, files, repo_lbl, grep, grep_arrow, ctx, ctx_lbl, dumped, recall, toneff
        )

    # ---------- beat 3: pipeline ----------
    def _beat_pipeline(self):
        heading = self._heading("WHAT ARCHEX DOES")
        self.play(FadeIn(heading), run_time=0.5)

        data = [
            ("01", "Parse", "tree-sitter\n25 languages", TEAL),
            ("02", "Index", "BM25F + vectors\n+ SPLADE", SKY),
            ("03", "Retrieve", "fusion + rerank\n+ graph", SKY),
            ("04", "Assemble", "token-budgeted\nbundle", TEAL),
        ]
        boxes = VGroup()
        for n, t, s, c in data:
            rect = RoundedRectangle(
                corner_radius=0.12,
                width=2.7,
                height=1.9,
                stroke_color=c,
                stroke_width=2.5,
                fill_color=PANEL,
                fill_opacity=1,
            )
            num = Text(n, font_size=18, color=c, weight=BOLD)
            title = Text(t, font_size=29, color=TEXT, weight=BOLD)
            sub = Text(s, font_size=17, color=MUTED, line_spacing=0.85)
            inner = VGroup(title, sub).arrange(DOWN, buff=0.14)
            num.move_to(rect.get_corner(UL) + RIGHT * 0.32 + DOWN * 0.3)
            inner.move_to(rect.get_center() + DOWN * 0.08)
            boxes.add(VGroup(rect, num, inner))
        boxes.arrange(RIGHT, buff=0.55).move_to([0, 0.4, 0])

        arrows = VGroup(
            *[
                Arrow(
                    boxes[i].get_right(),
                    boxes[i + 1].get_left(),
                    buff=0.12,
                    color=MUTED,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.25,
                )
                for i in range(len(boxes) - 1)
            ]
        )

        repo_in = (
            Text("repository", font_size=22, color=MUTED)
            .next_to(boxes[0], UP, buff=0.4)
            .shift(LEFT * 0.1)
        )
        bundle_out = (
            Text("context bundle", font_size=22, color=TEAL, weight=BOLD)
            .next_to(boxes[-1], UP, buff=0.4)
            .shift(RIGHT * 0.1)
        )

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in boxes], lag_ratio=0.22),
            run_time=1.8,
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.3), run_time=1.0)
        self.play(FadeIn(repo_in), FadeIn(bundle_out), run_time=0.6)

        tagline = Text("It returns context, not an answer.", font_size=34, color=TEXT, weight=BOLD)
        tagline.move_to([0, -2.6, 0])
        self.play(Write(tagline), run_time=1.0)
        self.wait(2.4)

        self._clear(heading, boxes, arrows, repo_in, bundle_out, tagline)

    # ---------- beat 4: measured bars ----------
    def _beat_bars(self):
        heading = self._heading("MEASURED · 19-TASK HEAD-TO-HEAD")
        self.play(FadeIn(heading), run_time=0.5)

        # legend
        leg_a = VGroup(
            Square(0.28, fill_color=TEAL, fill_opacity=1, stroke_width=0),
            Text("archex", font_size=24, color=TEXT),
        ).arrange(RIGHT, buff=0.18)
        leg_c = VGroup(
            Square(0.28, fill_color=SLATE, fill_opacity=1, stroke_width=0),
            Text("competition", font_size=24, color=MUTED),
        ).arrange(RIGHT, buff=0.18)
        legend = VGroup(leg_a, leg_c).arrange(RIGHT, buff=0.8).to_corner(UR, buff=0.7)
        self.play(FadeIn(legend), run_time=0.5)

        bar_x = -3.2
        unit = 6.2

        def bar(value: float, color: str, y: float) -> Rectangle:
            width = value * unit
            rect = Rectangle(
                width=width,
                height=0.34,
                fill_color=color,
                fill_opacity=1,
                stroke_width=0,
            )
            rect.move_to([bar_x + width / 2, y, 0])
            return rect

        def val(value: float, color: str, b: Rectangle) -> Text:
            return Text(f"{value:.2f}", font_size=24, color=color, weight=BOLD).next_to(
                b, RIGHT, buff=0.18
            )

        groups = [
            ("Recall", 0.95, 0.32, 1.6),
            ("F1", 0.66, 0.31, 0.2),
            ("Token eff.", 0.76, 0.48, -1.2),
        ]
        all_objs = []
        for name, a_val, c_val, yc in groups:
            label = Text(name, font_size=26, color=TEXT).move_to([bar_x - 0.4, yc, 0])
            label.align_to([bar_x - 0.4, 0, 0], RIGHT)
            a_bar = bar(a_val, TEAL, yc + 0.24)
            c_bar = bar(c_val, SLATE, yc - 0.24)
            a_lbl = val(a_val, TEAL, a_bar)
            c_lbl = val(c_val, MUTED, c_bar)
            self.play(FadeIn(label), run_time=0.25)
            self.play(GrowFromEdge(a_bar, LEFT), GrowFromEdge(c_bar, LEFT), run_time=0.9)
            self.play(FadeIn(a_lbl), FadeIn(c_lbl), run_time=0.3)
            all_objs += [label, a_bar, c_bar, a_lbl, c_lbl]
        self.wait(2.6)

        self._clear(heading, legend, *all_objs)

    # ---------- beat 5: stat tiles ----------
    def _beat_stats(self):
        heading = self._heading("THE PAYOFF")
        self.play(FadeIn(heading), run_time=0.5)

        def tile(kicker: str, a_text: str, c_text: str, note: str, pos):
            rect = RoundedRectangle(
                corner_radius=0.14,
                width=5.8,
                height=2.7,
                stroke_color="#232c46",
                stroke_width=2,
                fill_color=PANEL,
                fill_opacity=1,
            )
            k = Text(kicker, font_size=20, color=MUTED, weight=BOLD)
            a = Text(a_text, font_size=52, color=TEAL, weight=BOLD)
            vs = Text("vs", font_size=28, color=MUTED)
            c = Text(c_text, font_size=52, color=SLATE, weight=BOLD)
            row = VGroup(a, vs, c).arrange(RIGHT, buff=0.3)
            n = Text(note, font_size=20, color=MUTED)
            inner = VGroup(k, row, n).arrange(DOWN, buff=0.32)
            inner.move_to(rect.get_center())
            return VGroup(rect, inner).move_to(pos)

        t1 = tile("TOKENS FOR THE AGENT TO FINISH", "922", "11,188", "≈ 12× fewer", [-3.1, 0.2, 0])
        t2 = tile("COLD START", "0 ms", "4,721 ms", "no daemon warm-up", [3.1, 0.2, 0])

        self.play(FadeIn(t1, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(t2, shift=UP * 0.2), run_time=0.7)
        self.wait(2.8)

        self._clear(heading, t1, t2)

    # ---------- beat 6: close ----------
    def _beat_close(self):
        traits = Text(
            "local-first · deterministic · no API key · Apache-2.0", font_size=32, color=MUTED
        )
        traits.move_to([0, 1.7, 0])

        term = RoundedRectangle(
            corner_radius=0.12,
            width=8.4,
            height=1.2,
            stroke_color="#2a3350",
            stroke_width=2,
            fill_color="#0a0d18",
            fill_opacity=1,
        ).move_to([0, 0.1, 0])
        prompt = Text("$", font_size=38, color=TEAL, weight=BOLD)
        cmd = Text("uv tool install archex", font_size=38, color=TEXT, weight=BOLD)
        cmdline = VGroup(prompt, cmd).arrange(RIGHT, buff=0.35).move_to(term.get_center())

        repo = Text("github.com/Mathews-Tom/archex", font_size=30, color=TEAL, weight=BOLD)
        repo.move_to([0, -1.7, 0])

        self.play(FadeIn(traits), run_time=0.7)
        self.play(FadeIn(term), run_time=0.4)
        self.play(Write(cmdline), run_time=1.0)
        self.play(FadeIn(repo, shift=UP * 0.2), run_time=0.6)
        self.wait(3.0)
        self.play(FadeOut(VGroup(traits, term, cmdline, repo)), run_time=0.8)
