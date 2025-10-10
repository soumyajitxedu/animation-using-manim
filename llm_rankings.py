from manim import *
import numpy as np

class LLMRankingsAnimation(Scene):
    def construct(self):
        # Clean white background
        self.camera.background_color = "#ffffff"
        
        # === TOP 10 MODELS ONLY (Cleaner) ===
        models_data = [
            ("Gemini 2.5 Pro", 1452, "Google", "#4285f4"),
            ("Claude Opus 4.1", 1448, "Anthropic", "#d4a373"),
            ("Claude Sonnet 4.5", 1448, "Anthropic", "#c49563"),
            ("ChatGPT-4o", 1442, "OpenAI", "#10a37f"),
            ("GPT-4.5 Preview", 1441, "OpenAI", "#0d9475"),
            ("O3", 1440, "OpenAI", "#0b7a5f"),
            ("GPT-5 High", 1440, "OpenAI", "#096650"),
            ("Claude Opus 4", 1438, "Anthropic", "#b48553"),
            ("Qwen3 Max", 1433, "Alibaba", "#ff6a00"),
            ("GPT-5 Chat", 1426, "OpenAI", "#075240"),
        ]
        
        # === CLEAN TITLE ===
        title = Text(
            "LLM Arena Rankings",
            color="#1a1a1a",
            weight=BOLD,
            font="Arial"
        ).scale(0.85)
        
        subtitle = Text(
            "October 2025 • Top 10 Models",
            color="#666666",
            font="Arial"
        ).scale(0.42)
        
        VGroup(title, subtitle).arrange(DOWN, buff=0.16).to_edge(UP, buff=0.5)
        
        self.play(
            FadeIn(title, shift=DOWN*0.2, run_time=0.8),
            FadeIn(subtitle, shift=DOWN*0.2, run_time=0.8)
        )
        self.wait(0.6)
        
        # === LAYOUT PARAMETERS (CLEAN SPACING) ===
        bar_max_width = 4.8
        bar_height = 0.5
        bar_spacing = 0.65  # HUGE spacing
        start_y = 2.2
        bar_start_x = 0.5  # Bars start here
        
        # Normalize scores
        max_score = 1452
        min_score = 1420
        
        def normalize_score(score):
            return max(0.15, (score - min_score) / (max_score - min_score))
        
        # === CLEAN GRID (VERTICAL ONLY) ===
        grid_lines = VGroup()
        for i in range(5):
            x_pos = bar_start_x + (i * bar_max_width / 4)
            line = Line(
                start=[x_pos, start_y + 0.15, 0],
                end=[x_pos, start_y - len(models_data) * bar_spacing + 0.5, 0],
                color="#f0f0f0",
                stroke_width=1.5
            )
            grid_lines.add(line)
        
        self.play(FadeIn(grid_lines, run_time=0.5))
        self.wait(0.2)
        
        # === ANIMATE EACH BAR INDIVIDUALLY (ZERO CLUTTER) ===
        all_elements = VGroup()
        
        for i, (model, score, org, color) in enumerate(models_data):
            y_pos = start_y - (i * bar_spacing)
            normalized_width = normalize_score(score) * bar_max_width
            
            # === RANK NUMBER (FAR LEFT) ===
            rank = Text(
                f"#{i+1}",
                color=color,
                font="Arial",
                weight=BOLD
            ).scale(0.55)
            rank.move_to([-5.8, y_pos, 0])
            
            # === MODEL NAME (LEFT, CLEAR) ===
            model_name = Text(
                model,
                color="#1a1a1a",
                font="Arial",
                weight=BOLD
            ).scale(0.42)
            model_name.next_to(rank, RIGHT, buff=0.4)
            model_name.align_to([0, y_pos + 0.05, 0], ORIGIN)
            
            # === ORGANIZATION (SMALL, BELOW NAME) ===
            org_tag = Text(
                org,
                color="#999999",
                font="Arial"
            ).scale(0.28)
            org_tag.next_to(model_name, DOWN, buff=0.06, aligned_edge=LEFT)
            
            # === BAR (RIGHT SIDE ONLY) ===
            bar = Rectangle(
                width=normalized_width,
                height=bar_height,
                fill_color=color,
                fill_opacity=0.85,
                stroke_width=2,
                stroke_color=color,
                stroke_opacity=0.3,
                sheen_factor=0.3,
                sheen_direction=RIGHT
            )
            bar.move_to([bar_start_x + normalized_width/2, y_pos, 0])
            
            # === SCORE (END OF BAR) ===
            score_text = Text(
                str(score),
                color="#ffffff",
                font="Arial",
                weight=BOLD
            ).scale(0.45)
            score_text.move_to([bar_start_x + normalized_width - 0.35, y_pos, 0])
            
            # Adjust if bar too short
            if normalized_width < 1.5:
                score_text.set_color(color)
                score_text.move_to([bar_start_x + normalized_width + 0.4, y_pos, 0])
            
            # === DIVIDER LINE (SUBTLE) ===
            divider = Line(
                start=[-6.5, y_pos - bar_spacing/2, 0],
                end=[6, y_pos - bar_spacing/2, 0],
                color="#e8e8e8",
                stroke_width=1
            )
            
            # === ANIMATION: ONE ITEM AT A TIME ===
            # Start invisible
            rank.set_opacity(0).shift(LEFT * 0.5)
            model_name.set_opacity(0).shift(LEFT * 0.5)
            org_tag.set_opacity(0).shift(LEFT * 0.5)
            bar.save_state()
            bar.stretch(0, 0, about_edge=LEFT)
            score_text.set_opacity(0).scale(0.5)
            divider.set_opacity(0)
            
            # Animate in sequence
            self.play(
                rank.animate(run_time=0.25, rate_func=smooth).shift(RIGHT * 0.5).set_opacity(1),
                model_name.animate(run_time=0.3, rate_func=smooth).shift(RIGHT * 0.5).set_opacity(1),
                org_tag.animate(run_time=0.3, rate_func=smooth).shift(RIGHT * 0.5).set_opacity(1),
            )
            
            self.play(
                Restore(bar, run_time=0.65, rate_func=smooth),
                score_text.animate(run_time=0.6, rate_func=smooth).scale(2).set_opacity(1),
            )
            
            if i < len(models_data) - 1:
                self.play(FadeIn(divider, run_time=0.15))
            
            all_elements.add(VGroup(rank, model_name, org_tag, bar, score_text, divider))
        
        self.wait(1.0)
        
        # === CLEAN INFO PANEL ===
        info_box = Rectangle(
            width=5.5,
            height=1.0,
            fill_color="#f8f9fa",
            fill_opacity=1,
            stroke_color="#d0d0d0",
            stroke_width=1.5
        ).to_edge(DOWN, buff=0.6)
        
        info_title = Text(
            "Arena Stats",
            color="#1a1a1a",
            font="Arial",
            weight=BOLD
        ).scale(0.38).next_to(info_box.get_top(), DOWN, buff=0.15)
        
        stat1 = Text("253 models", color="#666666", font="Arial").scale(0.32)
        stat2 = Text("4.2M votes", color="#666666", font="Arial").scale(0.32)
        stat3 = Text("Updated daily", color="#666666", font="Arial").scale(0.32)
        
        stats = VGroup(stat1, stat2, stat3).arrange(RIGHT, buff=0.5)
        stats.next_to(info_title, DOWN, buff=0.12)
        
        self.play(
            FadeIn(info_box, run_time=0.5),
            Write(info_title, run_time=0.4),
            *[FadeIn(stat, shift=UP*0.1, run_time=0.4) for stat in [stat1, stat2, stat3]]
        )
        self.wait(0.8)
        
        # === SOURCE ===
        source = Text(
            "Source: lmarena.ai • October 2025",
            color="#999999",
            font="Arial",
            slant=ITALIC
        ).scale(0.32).next_to(info_box, DOWN, buff=0.3)
        
        self.play(FadeIn(source, run_time=0.5))
        self.wait(2.5)
        
        # === CLEAN FADE OUT ===
        self.play(
            *[FadeOut(mob, run_time=1.0, rate_func=smooth) for mob in self.mobjects]
        )
        self.wait(0.5)
