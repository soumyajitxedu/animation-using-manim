from manim import *
import numpy as np

class IQTestComparison(Scene):
    def construct(self):
        # Clean white background
        self.camera.background_color = "#ffffff"
        
        # === ALL 11 MODELS ===
        models_data = [
            ("GPT-5 Pro", 142, 116, "#10a37f"),        # Highest Mensa
            ("Grok-4", 135, 122, "#1a1a1a"),           # Biggest swing (Offline higher)
            ("Gemini Pro", 134, 117, "#4285f4"),
            ("GPT-5", 128, 116, "#0d9475"),
            ("Claude-4-Sonnet", 121, 107, "#d4a373"),
            ("Claude-4-Opus", 119, 118, "#c49563"),
            ("DeepSeek-R1", 112, 89, "#6366f1"),       # Mensa much higher
            ("Manus", 111, 92, "#8b5cf6"),
            ("Llama-4", 103, 78, "#0467df"),
            ("Perplexity", 97, 97, "#20808d"),          # Equal scores!
            ("Gemini Pro Vision", 93, 102, "#5a8fd8"),
        ]
        
        # === PERSISTENT HEADER ===
        title = Text(
            "AI Model IQ Test Comparison",
            color="#1a1a1a",
            weight=BOLD,
            font="Arial"
        ).scale(0.85).to_edge(UP, buff=0.5)
        
        mensa_header = Text(
            "Mensa Norway",
            color="#4285f4",
            font="Arial",
            weight=BOLD
        ).scale(0.48)
        
        offline_header = Text(
            "Offline Test",
            color="#ff7849",
            font="Arial",
            weight=BOLD
        ).scale(0.48)
        
        header_y = 2.8
        mensa_header.move_to([-2.5, header_y, 0])
        offline_header.move_to([2.5, header_y, 0])
        
        self.play(FadeIn(title, shift=DOWN*0.2, run_time=1.0))
        self.wait(0.5)
        self.play(Write(mensa_header, run_time=0.8), Write(offline_header, run_time=0.8))
        self.wait(0.8)
        
        # === LAYOUT PARAMETERS ===
        bar_max_width = 3.2
        bar_height = 0.42
        bar_spacing = 0.58
        start_y = 2.0
        model_name_x = -5.2
        mensa_bar_x = -2.5
        offline_bar_x = 2.5
        
        max_score = 145
        min_score = 65
        
        def normalize_score(score):
            return (score - min_score) / (max_score - min_score)
        
        # === IQ 100 BASELINE (KEY FEATURE #2) ===
        baseline_iq = 100
        baseline_normalized = normalize_score(baseline_iq)
        
        # Mensa baseline
        mensa_baseline = DashedLine(
            start=[mensa_bar_x + baseline_normalized * bar_max_width, start_y + 0.15, 0],
            end=[mensa_bar_x + baseline_normalized * bar_max_width, 
                 start_y - len(models_data) * bar_spacing - 0.2, 0],
            color="#ff6b6b",
            stroke_width=2.5,
            dash_length=0.08
        )
        
        mensa_baseline_label = Text(
            "IQ 100",
            color="#ff6b6b",
            font="Arial",
            weight=BOLD
        ).scale(0.28).next_to(mensa_baseline.get_top(), UP, buff=0.1)
        
        # Offline baseline
        offline_baseline = DashedLine(
            start=[offline_bar_x + baseline_normalized * bar_max_width, start_y + 0.15, 0],
            end=[offline_bar_x + baseline_normalized * bar_max_width, 
                 start_y - len(models_data) * bar_spacing - 0.2, 0],
            color="#ff6b6b",
            stroke_width=2.5,
            dash_length=0.08
        )
        
        offline_baseline_label = Text(
            "IQ 100",
            color="#ff6b6b",
            font="Arial",
            weight=BOLD
        ).scale(0.28).next_to(offline_baseline.get_top(), UP, buff=0.1)
        
        # Show baselines
        self.play(
            Create(mensa_baseline, run_time=0.8),
            Create(offline_baseline, run_time=0.8),
            FadeIn(mensa_baseline_label, run_time=0.6),
            FadeIn(offline_baseline_label, run_time=0.6)
        )
        self.wait(0.6)
        
        # === GRID ===
        grid_lines = VGroup()
        for x_pos in [mensa_bar_x, offline_bar_x]:
            for i in range(5):
                x = x_pos + (i * bar_max_width / 4)
                line = Line(
                    start=[x, start_y + 0.15, 0],
                    end=[x, start_y - len(models_data) * bar_spacing - 0.2, 0],
                    color="#f0f0f0",
                    stroke_width=1.5
                )
                grid_lines.add(line)
        
        self.play(FadeIn(grid_lines, run_time=0.6))
        self.wait(0.4)
        
        # === ANIMATE MODELS IN GROUPS ===
        all_elements = VGroup()
        delta_annotations = []  # Store delta indicators
        
        for group_start in range(0, len(models_data), 3):
            group_end = min(group_start + 3, len(models_data))
            group_elements = VGroup()
            
            for i in range(group_start, group_end):
                model, mensa_score, offline_score, color = models_data[i]
                y_pos = start_y - (i * bar_spacing)
                
                mensa_width = normalize_score(mensa_score) * bar_max_width
                offline_width = normalize_score(offline_score) * bar_max_width
                
                # Calculate delta (KEY FEATURE #1)
                delta = offline_score - mensa_score
                
                # === MODEL NAME ===
                model_name = Text(
                    model,
                    color="#2a2a2a",
                    font="Arial",
                    weight=BOLD
                ).scale(0.42).move_to([model_name_x, y_pos, 0])
                
                # === BARS ===
                mensa_bar = Rectangle(
                    width=mensa_width,
                    height=bar_height,
                    fill_color="#4285f4",
                    fill_opacity=0.85,
                    stroke_width=0
                ).move_to([mensa_bar_x + mensa_width/2, y_pos, 0])
                
                offline_bar = Rectangle(
                    width=offline_width,
                    height=bar_height,
                    fill_color="#ff7849",
                    fill_opacity=0.85,
                    stroke_width=0
                ).move_to([offline_bar_x + offline_width/2, y_pos, 0])
                
                # === SCORES ===
                mensa_score_text = Text(
                    str(mensa_score),
                    color="#ffffff",
                    font="Arial",
                    weight=BOLD
                ).scale(0.38).move_to([mensa_bar_x + mensa_width - 0.28, y_pos, 0])
                
                offline_score_text = Text(
                    str(offline_score),
                    color="#ffffff",
                    font="Arial",
                    weight=BOLD
                ).scale(0.38).move_to([offline_bar_x + offline_width - 0.28, y_pos, 0])
                
                # === DELTA INDICATOR (KEY FEATURE #1) ===
                delta_group = VGroup()
                
                if abs(delta) >= 10:  # Show delta for significant differences
                    # Connecting line
                    delta_line = Line(
                        start=[mensa_bar_x + mensa_width, y_pos, 0],
                        end=[offline_bar_x, y_pos, 0],
                        color="#ff6b6b" if abs(delta) >= 15 else "#ffd166",
                        stroke_width=2.5
                    )
                    
                    # Delta label
                    delta_text = Text(
                        f"{delta:+d}",  # Show +/- sign
                        color="#ff6b6b" if abs(delta) >= 15 else "#ffd166",
                        font="Arial",
                        weight=BOLD
                    ).scale(0.32)
                    delta_text.move_to([(mensa_bar_x + mensa_width + offline_bar_x) / 2, 
                                       y_pos + 0.25, 0])
                    
                    delta_group = VGroup(delta_line, delta_text)
                    delta_group.set_opacity(0)
                    delta_annotations.append((i, delta_group, delta))
                
                # === DIVIDER ===
                divider = Line(
                    start=[-6, y_pos - bar_spacing/2, 0],
                    end=[6, y_pos - bar_spacing/2, 0],
                    color="#e8e8e8",
                    stroke_width=1
                )
                
                # Setup animation states
                model_name.set_opacity(0).shift(LEFT * 0.3)
                mensa_bar.save_state()
                mensa_bar.stretch(0, 0, about_edge=LEFT)
                offline_bar.save_state()
                offline_bar.stretch(0, 0, about_edge=LEFT)
                mensa_score_text.set_opacity(0)
                offline_score_text.set_opacity(0)
                divider.set_opacity(0)
                
                group_elements.add(VGroup(
                    model_name, mensa_bar, offline_bar,
                    mensa_score_text, offline_score_text, divider, delta_group
                ))
                all_elements.add(group_elements[-1])
            
            # === ANIMATE GROUP ===
            self.play(
                *[group_elements[j][0].animate(run_time=0.6).shift(RIGHT * 0.3).set_opacity(1) 
                  for j in range(len(group_elements))]
            )
            
            animations = []
            for j in range(len(group_elements)):
                animations.extend([
                    Restore(group_elements[j][1], run_time=1.0, rate_func=smooth),
                    Restore(group_elements[j][2], run_time=1.0, rate_func=smooth),
                    FadeIn(group_elements[j][3], run_time=0.8),
                    FadeIn(group_elements[j][4], run_time=0.8),
                    FadeIn(group_elements[j][5], run_time=0.4),
                ])
            
            self.play(*animations)
            self.wait(0.8)
            
            # Show delta annotations
            for idx, delta_grp, delta_val in delta_annotations:
                if group_start <= idx < group_end and delta_grp:
                    self.play(delta_grp.animate(run_time=0.5).set_opacity(1))
            
            self.wait(0.6)
        
        # === FINAL HOLD ===
        self.wait(3.0)
        
        # === HIGHLIGHT TOP PERFORMER (KEY FEATURE #3) ===
        # GPT-5 Pro (index 0) - Highest Mensa score
        highlight_rect = SurroundingRectangle(
            all_elements[0],
            color="#10a37f",
            stroke_width=3.5,
            buff=0.15,
            corner_radius=0.05
        )
        
        winner_label = Text(
            "Highest Mensa Score",
            color="#10a37f",
            font="Arial",
            weight=BOLD
        ).scale(0.38).next_to(highlight_rect, LEFT, buff=0.3)
        
        self.play(
            Create(highlight_rect, run_time=0.8),
            FadeIn(winner_label, shift=RIGHT*0.2, run_time=0.7)
        )
        self.wait(2.0)
        
        # === HIGHLIGHT GROK-4 (Biggest Delta) ===
        grok_highlight = SurroundingRectangle(
            all_elements[1],
            color="#ff6b6b",
            stroke_width=3.5,
            buff=0.15,
            corner_radius=0.05
        )
        
        grok_label = Text(
            "Offline Test Outperformed",
            color="#ff6b6b",
            font="Arial",
            weight=BOLD
        ).scale(0.38).next_to(grok_highlight, LEFT, buff=0.3)
        
        self.play(
            FadeOut(highlight_rect, run_time=0.4),
            FadeOut(winner_label, run_time=0.4)
        )
        
        self.play(
            Create(grok_highlight, run_time=0.8),
            FadeIn(grok_label, shift=RIGHT*0.2, run_time=0.7)
        )
        self.wait(2.5)
        
        # === LEGEND ===
        info_box = Rectangle(
            width=10,
            height=0.95,
            fill_color="#f8f9fa",
            fill_opacity=0.98,
            stroke_color="#d0d0d0",
            stroke_width=1.5
        ).to_edge(DOWN, buff=0.4)
        
        legend_blue = VGroup(
            Dot(color="#4285f4", radius=0.1),
            Text("Mensa Norway (Public)", color="#4285f4", font="Arial", weight=MEDIUM).scale(0.38)
        ).arrange(RIGHT, buff=0.2)
        
        legend_orange = VGroup(
            Dot(color="#ff7849", radius=0.1),
            Text("Offline Test (Proprietary)", color="#ff7849", font="Arial", weight=MEDIUM).scale(0.38)
        ).arrange(RIGHT, buff=0.2)
        
        legends = VGroup(legend_blue, legend_orange).arrange(RIGHT, buff=0.8)
        legends.move_to(info_box)
        
        self.play(
            FadeOut(grok_highlight, run_time=0.5),
            FadeOut(grok_label, run_time=0.5),
            FadeIn(info_box, run_time=0.7),
            *[FadeIn(leg, shift=UP*0.15, run_time=0.6) for leg in legends]
        )
        self.wait(2.0)
        
        # === SOURCE ===
        source = Text(
            "Data: IQ Test Arena • October 2025",
            color="#999999",
            font="Arial",
            slant=ITALIC
        ).scale(0.35).next_to(info_box, DOWN, buff=0.25)
        
        self.play(FadeIn(source, run_time=0.6))
        self.wait(3.0)
        
        # === FADE OUT ===
        self.play(
            *[FadeOut(mob, run_time=1.5, rate_func=smooth) for mob in self.mobjects]
        )
        self.wait(0.5)
