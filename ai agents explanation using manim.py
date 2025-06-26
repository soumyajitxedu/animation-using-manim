from manim import *
class FactorByGrouping(Scene):
    def construct(self):
        # -----------------------------------------------------------------
        # 1. Introduction: Set up the problem
        # -----------------------------------------------------------------
        # Create the title and the initial expression
        title = Tex("Factorise the following expression:").to_edge(UP)
        expression = MathTex("bx + 2b + cx + 2c").next_to(title, DOWN, buff=0.8)

        # Animate the introduction
        self.play(Write(title))
        self.play(Write(expression))
        self.wait(2)

        # -----------------------------------------------------------------
        # 2. Step 1: Group the terms
        # -----------------------------------------------------------------
        step1_text = Tex("Step 1: Group the terms into pairs.").to_edge(UP)
        grouped_expr = MathTex("(bx + 2b)", "+", "(cx + 2c)").next_to(step1_text, DOWN, buff=0.8)

        # Animate the transition to step 1
        self.play(Transform(title, step1_text))
        # TransformMatchingTex smartly animates the original expression into the new grouped one
        self.play(TransformMatchingTex(expression, grouped_expr))
        self.wait(2)

        # -----------------------------------------------------------------
        # 3. Step 2: Factor the first group (bx + 2b)
        # -----------------------------------------------------------------
        step2_text = Tex("Step 2: Factor the GCF from the first group.").to_edge(UP)
        self.play(Transform(title, step2_text))

        # Highlight the first group
        first_group_box = SurroundingRectangle(grouped_expr[0], color=YELLOW)
        self.play(Create(first_group_box))
        
        # Explain the GCF (Greatest Common Factor)
        gcf_b_text = Tex("The common factor is 'b'").next_to(grouped_expr, DOWN)
        self.play(Write(gcf_b_text))
        self.wait(2)

        # Create the new expression with the first part factored
        expr_part_factored = MathTex("b(x + 2)", "+", "(cx + 2c)").next_to(title, DOWN, buff=0.8)
        
        # Animate the transformation of the first group
        self.play(
            TransformMatchingTex(grouped_expr, expr_part_factored),
            FadeOut(first_group_box),
            FadeOut(gcf_b_text)
        )
        self.wait(2)

        # -----------------------------------------------------------------
        # 4. Step 3: Factor the second group (cx + 2c)
        # -----------------------------------------------------------------
        step3_text = Tex("Step 3: Factor the GCF from the second group.").to_edge(UP)
        self.play(Transform(title, step3_text))

        # Highlight the second group
        second_group_box = SurroundingRectangle(expr_part_factored[2], color=BLUE)
        self.play(Create(second_group_box))

        # Explain the GCF
        gcf_c_text = Tex("The common factor is 'c'").next_to(expr_part_factored, DOWN)
        self.play(Write(gcf_c_text))
        self.wait(2)

        # Create the expression with both groups factored
        full_factored_expr = MathTex("b(x + 2)", "+", "c(x + 2)").next_to(title, DOWN, buff=0.8)

        # Animate the transformation of the second group
        self.play(
            TransformMatchingTex(expr_part_factored, full_factored_expr),
            FadeOut(second_group_box),
            FadeOut(gcf_c_text)
        )
        self.wait(2)

        # -----------------------------------------------------------------
        # 5. Step 4: Factor out the common binomial (x + 2)
        # -----------------------------------------------------------------
        step4_text = Tex("Step 4: Factor out the common binomial factor.").to_edge(UP)
        self.play(Transform(title, step4_text))

        # Create a new Tex object and color the common part to highlight it
        highlighted_common_factor = MathTex("b", "(x + 2)", "+", "c", "(x + 2)").next_to(title, DOWN, buff=0.8)
        highlighted_common_factor.set_color_by_tex("(x + 2)", YELLOW)
        
        # Transform the plain expression into the highlighted one
        self.play(Transform(full_factored_expr, highlighted_common_factor))
        self.wait(2)

        # The final answer
        final_answer = MathTex("(x + 2)(b + c)").next_to(title, DOWN, buff=0.8)
        
        # Animate the final factoring step
        self.play(TransformMatchingTex(full_factored_expr, final_answer))
        self.wait(2)

        # -----------------------------------------------------------------
        # 6. Conclusion and Summary
        # -----------------------------------------------------------------
        # Box the final answer in green to signify success
        final_box = SurroundingRectangle(final_answer, color=GREEN)
        self.play(Create(final_box))
        self.wait(3)

        # Clear the screen for the final summary
        self.play(FadeOut(title, final_answer, final_box))

        # Create and animate the summary
        summary_title = Tex("Summary").to_edge(UP)
        original_final = MathTex("bx + 2b + cx + 2c", "=", "(x + 2)(b + c)").scale(1.2)
        
        self.play(Write(summary_title))
        self.play(Write(original_final))
        self.wait(5)
