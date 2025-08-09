from manim import *

# This script creates a Manim animation for solving a system of linear equations.
# It uses the `Text` mobject to render all the text, which means it does not
# require an external LaTeX installation to run.

class Name(Scene):
    def construct(self):
        # We start with the system of equations.
        # We use the basic `Text` mobject instead of `MathTex`
        # to avoid the LaTeX dependency.
        eq1 = Text("2x + 3y = 12", font_size=50)
        eq2 = Text("4x - 3y = 6", font_size=50)

        # Group them together for easy positioning and alignment.
        equation_group = VGroup(eq1, eq2).arrange(DOWN, buff=0.5)
        self.play(Write(equation_group))
        self.wait(1)

        # To solve the system, we add the two equations together.
        # We'll create a plus sign and a horizontal line to represent this operation.
        plus_sign = Text("+", font_size=50).next_to(equation_group, LEFT, buff=0.75)
        line = Line(LEFT, RIGHT).scale(4).next_to(equation_group, DOWN, buff=0.5)

        self.play(
            FadeIn(plus_sign),
            Create(line)
        )
        self.wait(1)

        # Now, we perform the addition: (2x+4x) + (3y-3y) = (12+6)
        # The `3y` and `-3y` terms cancel out.
        cancellation_line_1 = Line(eq1[3].get_center(), eq2[3].get_center())
        cancellation_line_2 = Line(eq1[4].get_center(), eq2[4].get_center())
        cancellation_line_3 = Line(eq1[5].get_center(), eq2[5].get_center())
        cancellation_line_4 = Line(eq1[6].get_center(), eq2[6].get_center())
        cancellation_group = VGroup(cancellation_line_1, cancellation_line_2, cancellation_line_3, cancellation_line_4).set_color(RED)

        # `cancellation_group` is a custom-made mobject that will be used to show
        # the cancellation of the `3y` and `-3y` terms.
        self.play(Create(cancellation_group))
        self.wait(1)

        # Sum of the equations.
        result_eq = Text("6x = 18", font_size=50).next_to(line, DOWN, buff=0.5)
        self.play(TransformFromCopy(VGroup(eq1, eq2), result_eq))
        self.wait(1)

        # Solve for x.
        solution_x = Text("x = 3", font_size=50).next_to(result_eq, DOWN, buff=0.5)
        self.play(Write(solution_x))
        self.wait(1)

        # Now we'll substitute x=3 back into the first equation to solve for y.
        # First, we remove the other mobjects.
        self.play(
            FadeOut(equation_group, plus_sign, line, cancellation_group, result_eq),
            FadeOut(solution_x, target_position=ORIGIN)
        )
        self.wait(1)

        # Show the first equation and the x=3 solution.
        eq1_copy = eq1.copy().scale(1.2).to_edge(UP)
        solution_x_copy = solution_x.copy().next_to(eq1_copy, DOWN)
        self.play(FadeIn(eq1_copy, solution_x_copy))
        self.wait(1)

        # Substitute x=3 into the equation.
        substituted_eq = Text("2(3) + 3y = 12", font_size=50).next_to(eq1_copy, DOWN, buff=1)
        self.play(Transform(eq1_copy, substituted_eq))
        self.wait(1)

        # Solve for y.
        step1 = Text("6 + 3y = 12", font_size=50).next_to(substituted_eq, DOWN)
        step2 = Text("3y = 6", font_size=50).next_to(step1, DOWN)
        step3 = Text("y = 2", font_size=50).next_to(step2, DOWN)

        self.play(Transform(substituted_eq, step1))
        self.wait(1)
        self.play(Transform(substituted_eq, step2))
        self.wait(1)
        self.play(Transform(substituted_eq, step3))
        self.wait(1)

        # Final answer.
        final_solution = Text("Solution: (x, y) = (3, 2)", font_size=50).to_edge(DOWN)
        self.play(Write(final_solution))
        self.wait(3)
