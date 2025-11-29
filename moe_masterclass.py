from manim import *
import numpy as np
import random

# ==========================================
# HELPER CLASSES (3b1b Style Recreation)
# ==========================================

class MOEConfig:
    DENSE_COLOR = BLUE
    SPARSE_COLOR = TEAL
    ACTIVE_COLOR = YELLOW
    INACTIVE_COLOR = GREY_D
    ROUTER_COLOR = PURPLE
    SHARED_COLOR = ORANGE
    TEXT_COLOR = WHITE
    BG_COLOR = "#0F172A" # Slate 900

class NumericEmbedding(VGroup):
    """
    Visualizes a vector as a column of colored rectangles (Heatmap style).
    """
    def __init__(self, length=8, values=None, height=4, width=0.5, **kwargs):
        super().__init__(**kwargs)
        if values is None:
            values = np.random.uniform(-1, 1, length)
        
        self.rects = VGroup()
        self.values = values
        self.length = length
        
        box_height = height / length
        
        for val in values:
            rect = Rectangle(height=box_height, width=width)
            # Color map: Blue (negative) to Red (positive)
            if val < 0:
                color = interpolate_color(BLACK, BLUE, abs(val))
            else:
                color = interpolate_color(BLACK, RED, abs(val))
                
            rect.set_style(fill_color=color, fill_opacity=1, stroke_color=WHITE, stroke_width=1)
            self.rects.add(rect)
            
        self.rects.arrange(DOWN, buff=0)
        self.add(self.rects)
        
        # Add brackets
        self.brackets = Brace(self.rects, LEFT, buff=0.1)
        self.add(self.brackets)

    def set_values(self, new_values):
        """Animate changing values"""
        self.values = new_values
        new_colors = []
        for val in new_values:
            if val < 0:
                new_colors.append(interpolate_color(BLACK, BLUE, abs(val)))
            else:
                new_colors.append(interpolate_color(BLACK, RED, abs(val)))
        
        anims = []
        for rect, color in zip(self.rects, new_colors):
            anims.append(rect.animate.set_fill(color))
        return AnimationGroup(*anims)

class WeightMatrix(VGroup):
    """
    Visualizes a matrix as a grid of colored squares.
    """
    def __init__(self, rows=8, cols=8, height=4, width=4, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        
        self.grid = VGroup()
        self.cells = [] # 2D list
        
        cell_h = height / rows
        cell_w = width / cols
        
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                val = np.random.uniform(-1, 1)
                rect = Rectangle(height=cell_h, width=cell_w)
                
                if val < 0:
                    color = interpolate_color(BLACK, BLUE, abs(val))
                else:
                    color = interpolate_color(BLACK, RED, abs(val))
                
                rect.set_style(fill_color=color, fill_opacity=1, stroke_color=GREY, stroke_width=0.5)
                self.grid.add(rect)
                row_cells.append(rect)
            self.cells.append(row_cells)
            
        self.grid.arrange_in_grid(rows=rows, cols=cols, buff=0)
        self.add(self.grid)
        
        self.brackets = Brace(self.grid, LEFT, buff=0.1)
        self.add(self.brackets)

    def highlight_columns(self, col_indices, color=YELLOW):
        """Highlights specific columns"""
        anims = []
        for j in col_indices:
            for i in range(self.rows):
                rect = self.cells[i][j]
                anims.append(rect.animate.set_stroke(color, width=3))
        return AnimationGroup(*anims)

    def dim_all(self):
        return self.grid.animate.set_opacity(0.2)

    def highlight_cluster(self, row_start, row_end, col_start, col_end, color=YELLOW):
        """Highlights a block (Expert)"""
        anims = []
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                rect = self.cells[i][j]
                anims.append(rect.animate.set_opacity(1).set_stroke(color, width=2))
        return AnimationGroup(*anims)

# ==========================================
# SCENE 1: FOUNDATION (REFINED)
# ==========================================

class Scene1_Foundation(Scene):
    def construct(self):
        self.camera.background_color = MOEConfig.BG_COLOR
        
        # --- INTRO TITLE ---
        title = Text("The Mixture of Experts (MoE)", font_size=48).to_edge(UP)
        subtitle = Text("Scaling Intelligence Efficiently", font_size=32, color=GREY_B).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(subtitle))

        # --- STEP 1: THE TOKEN & EMBEDDING ---
        
        # Text Token
        token_text = Text('"Apple"', font_size=48).move_to(LEFT * 4)
        self.play(Write(token_text))
        self.wait(1)
        
        # Arrow
        arrow1 = Arrow(LEFT, RIGHT).next_to(token_text, RIGHT)
        self.play(GrowArrow(arrow1))
        
        # Embedding Vector
        embedding = NumericEmbedding(length=16, height=4, width=0.6).next_to(arrow1, RIGHT)
        emb_label = Text("Embedding Vector (x)", font_size=24).next_to(embedding, UP)
        
        self.play(FadeIn(embedding), Write(emb_label))
        self.wait(2)
        
        # --- STEP 2: THE DENSE MODEL (THE PROBLEM) ---
        
        # Move embedding to left to make room
        self.play(
            token_text.animate.shift(LEFT * 2).set_opacity(0),
            arrow1.animate.shift(LEFT * 2).set_opacity(0),
            embedding.animate.move_to(LEFT * 5),
            emb_label.animate.next_to(embedding, UP).shift(LEFT * 5), # Fix label pos
            FadeOut(title)
        )
        
        # Create Massive Weight Matrix
        # Shifted slightly down to make room for top text
        dense_matrix = WeightMatrix(rows=16, cols=12, height=5, width=6).move_to(RIGHT * 1 + DOWN * 0.5)
        matrix_label = Text("Dense FFN Parameters (W)", font_size=24).next_to(dense_matrix, UP)
        
        self.play(FadeIn(dense_matrix), Write(matrix_label))
        self.wait(1)
        
        # Equation - Moved to Top Left to avoid overlap with matrix label
        eq = MathTex(r"y = \sigma(W \cdot x)").to_edge(UP).shift(LEFT * 3)
        self.play(Write(eq))
        
        # Animation: Computation (All weights active)
        # Flash the entire matrix to show "Activation"
        self.play(
            dense_matrix.grid.animate.set_stroke(YELLOW, width=2),
            run_time=1
        )
        self.play(
            dense_matrix.grid.animate.set_stroke(GREY, width=0.5),
            run_time=1
        )
        
        problem_text = Text("Every token uses 100% of parameters", font_size=32, color=RED).next_to(dense_matrix, DOWN)
        self.play(Write(problem_text))
        self.wait(3)
        
        self.play(FadeOut(problem_text))
        
        # --- STEP 3: THE SPARSE SOLUTION (MoE) ---
        
        # Transform Matrix into Experts (Clusters)
        # We will visually break the matrix into 4 blocks
        
        expert_1 = WeightMatrix(rows=8, cols=6, height=2, width=2.5).move_to(RIGHT * -1 + UP * 1.5)
        expert_2 = WeightMatrix(rows=8, cols=6, height=2, width=2.5).move_to(RIGHT * 2.5 + UP * 1.5)
        expert_3 = WeightMatrix(rows=8, cols=6, height=2, width=2.5).move_to(RIGHT * -1 + DOWN * 1.5)
        expert_4 = WeightMatrix(rows=8, cols=6, height=2, width=2.5).move_to(RIGHT * 2.5 + DOWN * 1.5)
        
        experts = VGroup(expert_1, expert_2, expert_3, expert_4)
        
        # Animate the split
        self.play(
            ReplacementTransform(dense_matrix, experts),
            matrix_label.animate.set_text("Experts (E1, E2, E3, E4)").next_to(experts, UP)
        )
        self.wait(1)
        
        # Dim all experts initially
        self.play(
            expert_1.dim_all(),
            expert_2.dim_all(),
            expert_3.dim_all(),
            expert_4.dim_all(),
        )
        
        # Introduce the Router
        router_box = RoundedRectangle(corner_radius=0.2, height=1, width=1, color=PURPLE)
        router_label = Text("Router", font_size=20).move_to(router_box)
        router_group = VGroup(router_box, router_label).move_to(LEFT * 2)
        
        self.play(FadeIn(router_group))
        
        # Route logic: x -> Router -> Select E2
        path_line = Line(embedding.get_right(), router_box.get_left(), color=WHITE)
        self.play(Create(path_line))
        
        # Router "Thinking"
        self.play(router_box.animate.set_fill(PURPLE, opacity=0.5), run_time=0.5)
        self.play(router_box.animate.set_fill(PURPLE, opacity=0), run_time=0.5)
        
        # Selection Line
        select_line = Line(router_box.get_right(), expert_2.get_left(), color=YELLOW, stroke_width=4)
        self.play(Create(select_line))
        
        # Activate Expert 2
        self.play(
            expert_2.grid.animate.set_opacity(1).set_stroke(YELLOW, width=2),
            run_time=1.5
        )
        
        efficiency_text = Text("Only 25% Parameters Active!", font_size=32, color=GREEN).next_to(experts, DOWN)
        self.play(Write(efficiency_text))
        self.wait(3)
        
        # Cleanup
        self.play(
            FadeOut(VGroup(embedding, emb_label, experts, router_group, path_line, select_line, efficiency_text, eq, matrix_label, token_text, arrow1))
        )

# ==========================================
# SCENE 2: THE ROUTER (DEEP DIVE & MATH)
# ==========================================

class Scene2_TheRouter(Scene):
    def construct(self):
        self.camera.background_color = MOEConfig.BG_COLOR
        
        title = Text("Part 2: The Gating Network (Math)", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Setup Layout
        # Left: Equations
        # Right: Visuals
        
        # 1. The Gating Equation
        eq_group = VGroup()
        eq_title = Text("1. Gating Score", font_size=28, color=YELLOW).to_edge(LEFT).shift(UP*2)
        
        # h(x) = W_g . x
        eq_1 = MathTex(r"h(x) = W_g \cdot x").next_to(eq_title, DOWN, aligned_edge=LEFT)
        
        self.play(Write(eq_title))
        self.play(Write(eq_1))
        self.wait(2)
        
        # Visualizing Matrix Mult
        matrix_group = VGroup()
        input_vec = Matrix([[0.5], [-0.2], [0.8], [0.1]]).scale(0.6)
        router_w = Matrix([["w_1"], ["w_2"], ["w_3"]]).scale(0.6)
        
        router_w.next_to(eq_1, RIGHT, buff=2)
        input_vec.next_to(router_w, RIGHT)
        
        dot = MathTex(r"\cdot").move_to((router_w.get_right() + input_vec.get_left()) / 2)
        
        self.play(FadeIn(router_w), FadeIn(dot), FadeIn(input_vec))
        self.wait(1)
        
        # Result Logits
        logits = Matrix([[2.5], [0.1], [3.8]]).scale(0.6).next_to(input_vec, RIGHT, buff=1)
        arrow = Arrow(input_vec.get_right(), logits.get_left())
        
        self.play(Create(arrow))
        self.play(TransformFromCopy(VGroup(router_w, input_vec), logits))
        
        logits_label = Text("Logits (Scores)", font_size=20).next_to(logits, DOWN)
        self.play(Write(logits_label))
        self.wait(2)
        
        # 2. Softmax Equation
        eq_title_2 = Text("2. Softmax Probability", font_size=28, color=YELLOW).next_to(eq_1, DOWN, buff=1.5, aligned_edge=LEFT)
        
        # p_i = exp(h_i) / sum(exp(h_j))
        eq_2 = MathTex(r"p_i = \frac{e^{h(x)_i}}{\sum_{j} e^{h(x)_j}}").next_to(eq_title_2, DOWN, aligned_edge=LEFT)
        
        self.play(Write(eq_title_2))
        self.play(Write(eq_2))
        self.wait(2)
        
        # Visualizing Probabilities
        probs_vals = [0.2, 0.05, 0.75]
        prob_group = VGroup()
        
        for i, p in enumerate(probs_vals):
            bar_bg = Rectangle(width=3, height=0.3, color=GREY, fill_opacity=0.2)
            bar_fill = Rectangle(width=3*p, height=0.3, color=GREEN, fill_opacity=1).align_to(bar_bg, LEFT)
            
            label = MathTex(f"p_{i+1} = {p:.2f}").scale(0.7).next_to(bar_bg, LEFT)
            
            row = VGroup(label, bar_bg, bar_fill)
            prob_group.add(row)
            
        prob_group.arrange(DOWN, buff=0.5).next_to(eq_2, RIGHT, buff=2)
        
        self.play(FadeIn(prob_group))
        self.wait(2)
        
        # 3. Top-K Selection
        eq_title_3 = Text("3. Top-K Selection (k=1)", font_size=28, color=YELLOW).next_to(eq_2, DOWN, buff=1.5, aligned_edge=LEFT)
        
        # y = sum(p_i * E_i(x))
        eq_3 = MathTex(r"y = \sum_{i \in TopK} p_i E_i(x)").next_to(eq_title_3, DOWN, aligned_edge=LEFT)
        
        self.play(Write(eq_title_3))
        self.play(Write(eq_3))
        self.wait(2)
        
        # Highlight Winner
        winner_rect = SurroundingRectangle(prob_group[2], color=YELLOW)
        self.play(Create(winner_rect))
        
        winner_text = Text("Expert 3 Selected", font_size=24, color=YELLOW).next_to(winner_rect, RIGHT)
        self.play(Write(winner_text))
        self.wait(3)
        
        self.clear()

# ==========================================
# SCENE 3: EXPERT GRANULARITY (FINE-GRAINED)
# ==========================================

class Scene3_ExpertGranularity(Scene):
    def construct(self):
        self.camera.background_color = MOEConfig.BG_COLOR
        
        title = Text("Part 3: Fine-Grained Experts", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Standard MoE
        std_group = VGroup()
        # Using WeightMatrix for experts now
        std_experts = VGroup(*[WeightMatrix(rows=6, cols=6, height=1.5, width=1.5) for i in range(2)]).arrange(DOWN, buff=0.5)
        std_label = Text("Standard MoE\n(Few Large Experts)", font_size=24).next_to(std_experts, UP)
        std_group.add(std_experts, std_label).shift(LEFT * 3.5)
        
        self.play(FadeIn(std_group))
        self.wait(1)
        
        # Fine-Grained MoE
        fg_group = VGroup()
        # 8 Small experts
        fg_experts = VGroup(*[WeightMatrix(rows=3, cols=3, height=0.6, width=0.6) for i in range(8)])
        fg_experts.arrange_in_grid(rows=4, cols=2, buff=0.2)
        fg_label = Text("Fine-Grained MoE\n(Many Small Experts)", font_size=24).next_to(fg_experts, UP)
        fg_group.add(fg_experts, fg_label).shift(RIGHT * 3.5)
        
        self.play(FadeIn(fg_group))
        self.wait(2)
        
        # Animation: Activation
        # Standard: Activate 1 big expert
        self.play(std_experts[0].grid.animate.set_stroke(YELLOW, 2).set_opacity(1), run_time=1.5)
        
        # Fine-Grained: Activate 4 small experts (flexible combination)
        # This shows "Knowledge Hybridity"
        self.play(
            fg_experts[0].grid.animate.set_stroke(YELLOW, 2).set_opacity(1),
            fg_experts[3].grid.animate.set_stroke(YELLOW, 2).set_opacity(1),
            fg_experts[5].grid.animate.set_stroke(YELLOW, 2).set_opacity(1),
            fg_experts[7].grid.animate.set_stroke(YELLOW, 2).set_opacity(1),
            run_time=1.5
        )
        
        explanation = Text("More flexible combinations!", font_size=32, color=YELLOW).next_to(fg_group, DOWN)
        self.play(Write(explanation))
        self.wait(3)
        
        self.clear()

# ==========================================
# SCENE 4: SHARED EXPERTS (DEEPSEEK STYLE)
# ==========================================

class Scene4_SharedExperts(Scene):
    def construct(self):
        self.camera.background_color = MOEConfig.BG_COLOR
        
        title = Text("Part 4: Shared vs Routed Experts", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Problem: Common knowledge needs to be in EVERY expert?
        # Solution: Have a dedicated "Shared Expert" that is ALWAYS active.
        
        # Shared Expert (Always Active)
        shared_expert = WeightMatrix(rows=12, cols=6, height=4, width=2).move_to(LEFT * 2.5)
        
        # Routed Experts (Selectively Active)
        routed_experts = VGroup(*[WeightMatrix(rows=4, cols=4, height=1, width=1) for i in range(3)]).arrange(DOWN, buff=0.5)
        routed_experts.move_to(RIGHT * 2.5)
        
        label_shared = Text("Always Active", font_size=20, color=MOEConfig.SHARED_COLOR).next_to(shared_expert, UP)
        label_routed = Text("Selectively Active", font_size=20, color=MOEConfig.SPARSE_COLOR).next_to(routed_experts, UP)
        
        self.play(
            FadeIn(shared_expert), Write(label_shared),
            FadeIn(routed_experts), Write(label_routed)
        )
        self.wait(2)
        
        # Highlight Shared Expert (Orange)
        self.play(
            shared_expert.grid.animate.set_stroke(ORANGE, 2).set_opacity(1)
        )
        
        # Highlight Routed Expert 1 (Teal)
        self.play(
            routed_experts[1].grid.animate.set_stroke(TEAL, 2).set_opacity(1)
        )
        
        # Recombination
        sum_op = Text("+", font_size=48).move_to(RIGHT * 5)
        self.play(Write(sum_op))
        
        # Arrows flowing to sum
        arrow_shared = Arrow(shared_expert.get_right(), sum_op.get_left(), color=ORANGE)
        arrow_routed = Arrow(routed_experts[1].get_right(), sum_op.get_left(), color=TEAL)
        
        self.play(GrowArrow(arrow_shared), GrowArrow(arrow_routed))
        
        final_y = Text("Final Output", font_size=24, color=GREEN).next_to(sum_op, RIGHT)
        self.play(Write(final_y))
        
        caption = Text("DeepSeekMoE Architecture", font_size=32, color=BLUE).to_edge(DOWN)
        self.play(Write(caption))
        self.wait(3)
        
        self.clear()

# ==========================================
# SCENE 5: LOAD BALANCING
# ==========================================

class Scene5_LoadBalancing(Scene):
    def construct(self):
        self.camera.background_color = MOEConfig.BG_COLOR
        
        title = Text("Part 5: The Load Balancing Problem", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        experts = VGroup(*[WeightMatrix(rows=6, cols=4, height=1.5, width=1) for i in range(4)]).arrange(RIGHT, buff=0.5)
        self.play(FadeIn(experts))
        
        # Scenario: Collapse
        # All tokens going to Expert 0
        
        tokens = VGroup(*[Dot(color=RED) for _ in range(10)]).arrange(RIGHT, buff=0.2).move_to(UP * 2)
        self.play(FadeIn(tokens))
        self.wait(1)
        
        self.play(
            tokens.animate.move_to(experts[0].get_center()),
            experts[0].grid.animate.set_stroke(RED, 3).set_opacity(1),
            run_time=3
        )
        
        alert = Text("Expert Collapse!", font_size=48, color=RED).next_to(experts, DOWN)
        self.play(Write(alert))
        self.wait(2)
        
        # Solution: Aux Loss
        solution = Text("Solution: Auxiliary Loss", font_size=36, color=GREEN).next_to(alert, DOWN)
        self.play(Write(solution))
        self.wait(2)
        
        # Reset
        self.play(
            FadeOut(tokens),
            FadeOut(alert),
            experts[0].dim_all()
        )
        
        # Balanced Flow
        new_tokens = VGroup(*[Dot(color=GREEN) for _ in range(12)]).arrange(RIGHT, buff=0.2).move_to(UP * 2)
        
        anims = []
        for i, t in enumerate(new_tokens):
            target = experts[i % 4]
            anims.append(t.animate.move_to(target.get_center()))
            
        self.play(LaggedStart(*anims, lag_ratio=0.1), run_time=3)
        self.play(
            experts[0].grid.animate.set_stroke(GREEN, 2).set_opacity(1),
            experts[1].grid.animate.set_stroke(GREEN, 2).set_opacity(1),
            experts[2].grid.animate.set_stroke(GREEN, 2).set_opacity(1),
            experts[3].grid.animate.set_stroke(GREEN, 2).set_opacity(1),
        )
        
        self.wait(3)
